"""Run RRF (Random Rule Forest) on a dataset with the local Ollama model.

RRF's members are YES/NO questions. A shortlist is written once from labelled
training examples, every question is then put to every row, and the answer
vectors are collapsed the same way PI's are — a logistic regression, fitted
out-of-fold on the training split. The learned object is the question list, in
`models/<dataset>/rrf/`.

By default the shipped shortlist for the dataset is reused, so a rerun scores
the same 16 (VCBench) or 14 (Movie) questions that produced `precomputed/`.
Pass `--regenerate-questions` to write a fresh one instead.

Restart-safe twice over: answered `(id, question)` pairs are skipped via
`raw_responses.jsonl`, and any call that does get repeated is a free hit on the
disk cache.

Usage — smoke test against a running Ollama (~5 minutes):
    python run_rrf.py --dataset movie --smoke

Usage — full Movie run (2,188 films x 14 questions; hours, see the banner):
    caffeinate -i python run_rrf.py --dataset movie

Usage — full VCBench run (needs the CSV from vcbench.com):
    VCBENCH_DATA=~/.trl-data/vcbench/vcbench_final_public.csv \\
      caffeinate -i python run_rrf.py --dataset vcbench

Outputs (under --out-dir, default results/rrf_<dataset>/):
  <dataset>_<split>_rrf_scores.csv   id,score[,label] — one row per sample
  metrics.json                       AUCs, LLM call count, wall-clock
  question_metrics.csv               fire rate / precision / recall per question
  raw_responses.jsonl                every answer with its one-line rationale
  llm_cache.jsonl                    the restart cache (never commit it)
"""
from __future__ import annotations
import argparse
import asyncio
import json
import random
import time
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

import _runner_common as rc

METHOD = "rrf"

# Wording from the runs that produced the shipped shortlists.
PROMPTS = {
    "vcbench": {
        "qgen": (
            "You design YES/NO diagnostic questions that discriminate founders whose "
            "companies go on to raise over $500M or exit above $500M from those that "
            "do not, answerable purely from an anonymised founder summary. Make them "
            "concrete, non-redundant, and predictive; avoid questions the summary "
            "cannot possibly answer."
        ),
        "answer": (
            "You are a VC analyst evaluating startup founders. You decide whether a "
            "YES/NO question applies to a founder based ONLY on observable signals in "
            "their anonymised summary. Be deterministic. Do not infer beyond what the "
            "summary states."
        ),
        "template": (
            "Given the following anonymised founder summary, decide whether the "
            "question applies to this founder. Answer with `YES` or `NO`, then provide "
            "one short sentence (<=25 words) citing the specific evidence in the "
            "summary you used. If the summary is silent on what the question asks, "
            "answer `NO`.\n\n"
            "**Question:** {question}\n\n"
            "**Founder summary:**\n{summary}"
        ),
    },
    "movie": {
        "qgen": (
            "You design YES/NO diagnostic questions that discriminate films that earn a "
            "major award nomination from those that don't, answerable purely from a movie "
            "plot summary. Make them concrete, non-redundant, and predictive of "
            "award-worthiness (e.g. thematic ambition, character complexity, distinctive "
            "point of view, social resonance); avoid questions about box office, cast, or "
            "anything not visible from the summary itself."
        ),
        "answer": (
            "You are a film critic evaluating movie plot summaries. You decide whether a "
            "YES/NO question applies to a film based ONLY on what is visible in its plot "
            "summary. Be deterministic. Do not infer beyond what the summary states."
        ),
        "template": (
            "Given the following movie plot summary, decide whether the question applies "
            "to this film. Answer with `YES` or `NO`, then provide one short sentence "
            "(<=25 words) citing the specific evidence in the summary you used. If the "
            "summary is silent on what the question asks, answer `NO`.\n\n"
            "**Question:** {question}\n\n"
            "**Plot summary:**\n{summary}"
        ),
    },
}

SHIPPED_QUESTIONS = {
    "vcbench": rc.EXAMPLE_ROOT / "models" / "vcbench" / "rrf" / "coder14b_shortlist.json",
    "movie": rc.EXAMPLE_ROOT / "models" / "movie" / "rrf" / "movie_rrf_questions_qwen14b.json",
}

ANSWER_CHARS = 3000   # what the shipped runs passed to the model
SMOKE_QUESTIONS = 5   # --smoke asks a handful, not the whole shortlist


class Questions(BaseModel):
    questions: list[str]


class AnswerResponse(BaseModel):
    answer: Literal["YES", "NO"] = Field(..., description="YES or NO.")
    reasoning: str = Field(
        ..., description="One sentence (<=25 words) citing the specific summary evidence.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    rc.add_common_args(ap)
    ap.add_argument("--questions", type=Path, default=None,
                    help="question JSON [{qid,text,expected_direction}]; "
                         "default: the shipped shortlist for the dataset")
    ap.add_argument("--regenerate-questions", action="store_true",
                    help="write a fresh shortlist instead of reusing the shipped one")
    ap.add_argument("--n-questions", type=int, default=14,
                    help="questions to generate with --regenerate-questions")
    return ap.parse_args(argv)


def _prompts(spec: rc.DatasetSpec) -> dict[str, str]:
    if spec.name in PROMPTS:
        return PROMPTS[spec.name]
    return {
        "qgen": (f"You design YES/NO diagnostic questions that discriminate the "
                 f"{spec.positive} class from the {spec.negative} class, answerable "
                 f"purely from a {spec.subject}."),
        "answer": (f"You decide whether a YES/NO question applies to a sample based "
                   f"ONLY on what is visible in its {spec.subject}. Be deterministic."),
        "template": ("Given the following {subject}, decide whether the question applies. "
                     "Answer with `YES` or `NO`, then one short sentence (<=25 words) "
                     "citing the evidence you used. If the text is silent on what the "
                     "question asks, answer `NO`.\n\n"
                     "**Question:** {question}\n\n**Text:**\n{summary}"
                     ).replace("{subject}", spec.subject),
    }


async def generate_questions(llm, args: argparse.Namespace, spec: rc.DatasetSpec,
                             fit_df: pd.DataFrame) -> list[dict]:
    """One call: labelled examples in, a YES/NO shortlist out."""
    rng = random.Random(args.seed)
    pos = fit_df[fit_df["label"] == 1].to_dict("records")
    neg = fit_df[fit_df["label"] == 0].to_dict("records")
    rng.shuffle(pos)
    rng.shuffle(neg)
    sample = pos[:20] + neg[:20]
    rng.shuffle(sample)
    block = "\n\n".join(
        f"[{(spec.positive if r['label'] else spec.negative).upper()}] {r['text'][:500]}"
        for r in sample)
    query = (f"Here are {len(sample)} labelled examples:\n\n{block}\n\n"
             f"Write exactly {args.n_questions} YES/NO diagnostic questions that best "
             f'discriminate the two classes. Return JSON {{"questions": [...]}}.')
    response = await llm.respond(
        query=query, llm_priority=[rc.choice(args.model)], response_format=Questions,
        instructions=_prompts(spec)["qgen"], temperature=0.4)
    questions = response.response.questions if response.response else []
    if not questions:
        raise RuntimeError("Question generation returned nothing.")
    return [{"qid": f"q{i:02d}", "text": q, "expected_direction": 1, "source": args.model}
            for i, q in enumerate(questions)]


def load_questions(args: argparse.Namespace, spec: rc.DatasetSpec, out_dir: Path) -> Path | None:
    """The question file to reuse, or None if one has to be generated."""
    if args.regenerate_questions:
        return None
    for candidate in (args.questions, out_dir / "questions.json", SHIPPED_QUESTIONS.get(spec.name)):
        if candidate and Path(candidate).exists():
            return Path(candidate)
    return None


async def run(args: argparse.Namespace) -> dict:
    t0 = time.time()
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    spec, _df, fit_df, scored, out_dir, cache_path = rc.prepare_run(METHOD, args)
    prompts = _prompts(spec)

    qpath = load_questions(args, spec, out_dir)
    n_q = args.n_questions
    if qpath:
        n_q = len(json.loads(qpath.read_text()))
        if args.smoke:
            n_q = min(n_q, SMOKE_QUESTIONS)
    n_rows = len(set(fit_df["id"]).union(*(set(f["id"]) for f in scored.values())))
    est_calls = rc.estimate_calls(METHOD, len(fit_df), n_rows, n_questions=n_q,
                                  generate=0 if qpath else 1)
    rc.banner(METHOD, spec, args, fit_df, scored, est_calls, out_dir, cache_path)

    llm = rc.build_llm(cache_path, expected_calls=est_calls)

    if qpath:
        qspec = json.loads(qpath.read_text())
        if args.smoke:  # the point is that the plumbing works, not the score
            qspec = qspec[:SMOKE_QUESTIONS]
        rc.logger.info("Using %d questions from %s", len(qspec), qpath)
    else:
        rc.logger.info("Generating %d questions from %d labelled rows (one call)...",
                       args.n_questions, len(fit_df))
        qspec = await generate_questions(llm, args, spec, fit_df)
        (out_dir / "questions.json").write_text(json.dumps(qspec, indent=2))
        for q in qspec:
            rc.logger.info("  [%s] %s", q["qid"], q["text"])
    qids = [q["qid"] for q in qspec]

    # Resume: anything already answered without an error is skipped.
    raw_path = out_dir / "raw_responses.jsonl"
    done: set[tuple[str, str]] = set()
    if raw_path.exists():
        for line in raw_path.open():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not record.get("error"):
                done.add((record["id"], record["qid"]))

    # The fit rows are answered too even when only the held-out split is being
    # scored: the logistic combiner is fitted on their answer vectors.
    rows = pd.concat([f.assign(split=name) for name, f in scored.items()]
                     + [fit_df.assign(split=spec.fit_split)], ignore_index=True)
    rows = rows.drop_duplicates(subset="id", keep="first").reset_index(drop=True)
    pairs = [(row, q) for row in rows.to_dict("records") for q in qspec
             if (row["id"], q["qid"]) not in done]
    rc.logger.info("rows=%d  questions=%d  pairs=%d  already answered=%d  to run=%d",
                   len(rows), len(qspec), len(rows) * len(qspec), len(done), len(pairs))

    llm_priority = [rc.choice(args.model)]
    with raw_path.open("a") as fh:
        for row, q in pairs:
            answer, reasoning, tokens, error = None, None, None, None
            try:
                response = await llm.respond(
                    query=prompts["template"].format(
                        question=q["text"], summary=row["text"][:ANSWER_CHARS]),
                    llm_priority=llm_priority, response_format=AnswerResponse,
                    instructions=prompts["answer"], temperature=0.0)
                parsed = response.response
                if parsed is None:
                    error = "no_parsed_response"
                else:
                    answer = str(parsed.answer).strip().upper()
                    reasoning, tokens = parsed.reasoning, response.total_tokens
            except Exception as exc:  # one bad answer must not end a multi-hour run
                error = repr(exc)[:200]
            fh.write(json.dumps({
                "id": row["id"], "split": row["split"], "label": int(row["label"]),
                "qid": q["qid"], "answer": answer, "binary": 1 if answer == "YES" else 0,
                "reasoning": reasoning, "total_tokens": tokens, "error": error}) + "\n")
            fh.flush()

    # Answer matrix, one row per sample, questions in shortlist order.
    answers = pd.DataFrame([json.loads(line) for line in raw_path.open() if line.strip()])
    answers = answers[answers["error"].isna()]
    matrix = (answers.pivot_table(index="id", columns="qid", values="binary", aggfunc="first")
              .reindex(columns=qids))

    all_scores = rc.combine_answers(matrix, fit_df, scored, seed=args.seed)

    # Per-question diagnostics on the fit split — which questions actually fire.
    from sklearn.metrics import fbeta_score, precision_score, recall_score

    fit_answers = matrix.reindex(fit_df["id"]).fillna(0).to_numpy(dtype=int)
    y_diag = fit_df["label"].to_numpy()
    pd.DataFrame([{
        "qid": qid,
        "fire_rate": round(float(fit_answers[:, j].mean()), 3),
        "precision": round(float(precision_score(y_diag, fit_answers[:, j], zero_division=0)), 3),
        "recall": round(float(recall_score(y_diag, fit_answers[:, j], zero_division=0)), 3),
        "f0_5": round(float(fbeta_score(y_diag, fit_answers[:, j], beta=0.5, zero_division=0)), 3),
        "text": qspec[j]["text"],
    } for j, qid in enumerate(qids)]).to_csv(out_dir / "question_metrics.csv", index=False)

    metrics = rc.base_metrics(METHOD, spec, args, fit_df, started, t0, llm)
    metrics["n_questions"] = len(qids)
    metrics["questions_from"] = str(qpath) if qpath else str(out_dir / "questions.json")
    written = []
    for split, frame in scored.items():
        out = frame[["id", "label"]].copy()
        out["score"] = all_scores[split]
        written.append(rc.write_scores(out_dir, spec.name, split, METHOD, out,
                                       with_label=(split == spec.fit_split)))
        metrics["splits"][split] = {"n": len(out), **rc.score_series(out["label"], out["score"])}
    written.append(out_dir / "question_metrics.csv")
    return rc.finish(out_dir, metrics, written, t0)


def main(argv: list[str] | None = None) -> dict:
    return asyncio.run(run(parse_args(argv)))


if __name__ == "__main__":
    main()
