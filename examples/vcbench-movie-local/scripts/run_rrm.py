"""Run RRM (Reasoned Rule Mining) on a dataset with the local Ollama model.

RRM reasons about labelled examples one at a time, extracts a rule from each
piece of reasoning, drops the rules the model was least confident in (a
perplexity filter over token logprobs), compiles the survivors into a single
written policy, and calibrates a probability on top. Scoring puts the policy to
each row three times and fuses the votes. The learned objects are the rules and
the compiled policy, in `models/<dataset>/rrm/` — notebook 02 prints both.

Two things are specific to this method:

* It needs real token logprobs, which is why the example ships its own LLM
  wrapper rather than using the library's `OpenAILLM` directly.
* Fit-stage reasoning runs at `reason_temperature=1.0`. The disk cache skips
  sampled calls by default — repeated draws are meant to differ — so this
  runner opts back in with `cache_sampled=True`. Without it, a killed fit
  restarts from zero, which on a local model is hours. Scoring votes at
  temperature 0 and is unaffected either way.

Usage — smoke test against a running Ollama (~10 minutes):
    python run_rrm.py --dataset movie --smoke

Usage — full Movie run (fit on 346 films, score 2,188; hours, see the banner):
    caffeinate -i python run_rrm.py --dataset movie

Usage — full VCBench run (needs the CSV from vcbench.com):
    VCBENCH_DATA=~/.trl-data/vcbench/vcbench_final_public.csv \\
      caffeinate -i python run_rrm.py --dataset vcbench

Outputs (under --out-dir, default results/rrm_<dataset>/):
  <dataset>_<split>_rrm_scores.csv   id,score[,label] — one row per sample
  metrics.json                       AUCs, rule count, wall-clock
  rrm_rules.csv, rrm_policy.txt      what the fit learned
  llm_cache.jsonl                    the restart cache (never commit it)
"""
from __future__ import annotations
import argparse
import asyncio
import json
import logging
import math
import random
import time

import pandas as pd

import _runner_common as rc

METHOD = "rrm"

TASK_DESCRIPTIONS = {
    "vcbench": (
        "You are an expert in venture capital tasked with identifying successful "
        "founders from their unsuccessful counterparts. All founders are sourced "
        "from anonymised LinkedIn and Crunchbase profiles of companies that have "
        "raised between $100K and $4M in funding. A successful founder is defined "
        "as one whose company has achieved either a total funding of over $500M or "
        "an exit/IPO valued at over $500M."
    ),
    "movie": (
        "You are an expert film critic and awards judge, identifying which films earn a "
        "major award nomination (YES) versus those that do not (NO), judging purely from "
        "the film's free-text plot summary and its narrative/creative qualities."
    ),
}

# Stratified fit sizes used for the shipped bundles.
FIT_SIZES = {"vcbench": 0, "movie": 346}

MAX_YES_RULES, MAX_NO_RULES = 100, 300


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    rc.add_common_args(ap)
    ap.add_argument("--memory", action="store_true",
                    help="enable RRM's rolling memory; off in every shipped run because "
                         "the growing prompt blows up wall-clock on a local model")
    ap.add_argument("--refit", action="store_true",
                    help="refit even if a fitted bundle is already in the out-dir")
    ap.add_argument("--task", default=None, help="override the task description")
    return ap.parse_args(argv)


def _patch_compile_policy(seed: int) -> None:
    """Subsample the rule list before the policy-compile call.

    Compilation puts every surviving rule in one prompt. On VCBench that is
    thousands of rules, well past qwen2.5-coder:14b's context, and Ollama
    answers with a truncated policy that then votes against almost nothing.
    The shipped runs capped it at 100 YES + 300 NO rules; same wrapper here.
    """
    from think_reason_learn.reasoned_rule_mining import ReasonedRuleMining
    from think_reason_learn.reasoned_rule_mining._prompts import POLICY_COMPILATION_INSTRUCTIONS

    async def compile_policy(self) -> None:
        kept = [r for r in self._rules if r.perplexity <= self.config.perplexity_threshold]
        yes_rules = [r.rule for r in kept if r.outcome == "YES"]
        no_rules = [r.rule for r in kept if r.outcome == "NO"]
        rng = random.Random(seed)
        if len(yes_rules) > MAX_YES_RULES:
            yes_rules = rng.sample(yes_rules, MAX_YES_RULES)
        if len(no_rules) > MAX_NO_RULES:
            no_rules = rng.sample(no_rules, MAX_NO_RULES)
        logging.getLogger("runner").info(
            "Compiling a policy from %d YES + %d NO rules (of %d that passed the "
            "perplexity filter)", len(yes_rules), len(no_rules), len(kept))
        query = (f"Task:\n{self._task_description}\n\n"
                 "Rules for the YES class:\n" + ("\n".join(yes_rules) or "(none)") + "\n\n"
                 "Rules for the NO class:\n" + ("\n".join(no_rules) or "(none)"))
        response = await self._respond(
            query=query, llm_priority=self.reason_llmc, response_format=str,
            instructions=POLICY_COMPILATION_INSTRUCTIONS,
            temperature=self.reason_temperature, caller="ReasonedRuleMining._compile_policy")
        policy = (response.response or "").strip()
        if not policy:
            raise RuntimeError("Compile returned an empty policy.")
        self._policy = policy

    ReasonedRuleMining._compile_policy = compile_policy


async def run(args: argparse.Namespace) -> dict:
    t0 = time.time()
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    if args.fit_size is None and not args.smoke:
        args.fit_size = FIT_SIZES.get(args.dataset or "", 0)
    spec, _df, fit_df, scored, out_dir, cache_path = rc.prepare_run(METHOD, args)
    task = args.task or TASK_DESCRIPTIONS.get(spec.name)
    if not task:
        raise SystemExit("No task description for this dataset — pass --task.")

    n_rows = len(set(fit_df["id"]).union(*(set(f["id"]) for f in scored.values())))
    est_calls = rc.estimate_calls(METHOD, len(fit_df), n_rows)
    rc.banner(METHOD, spec, args, fit_df, scored, est_calls, out_dir, cache_path)

    # cache_sampled=True: see the module docstring. Fit reasons at temperature
    # 1.0, and those calls are the expensive ones to lose.
    llm = rc.build_llm(cache_path, cache_sampled=True, expected_calls=est_calls)

    from think_reason_learn.reasoned_rule_mining import ReasonedRuleMining, RRMConfig

    _patch_compile_policy(args.seed)
    llmc = [rc.choice(args.model)]
    bundle = out_dir / "reasoned_rule_mining.json"

    if bundle.exists() and not args.refit:
        rc.logger.info("Loading the fitted model from %s (pass --refit to redo it)", out_dir)
        rrm = ReasonedRuleMining.load(out_dir)
        rrm._llm_instance = llm
        rrm._llm_semaphore = asyncio.Semaphore(args.concurrency)
        rrm.reason_llmc, rrm.predict_llmc = llmc, llmc
    else:
        rrm = ReasonedRuleMining(
            reason_llmc=llmc, extract_llmc=llmc, predict_llmc=llmc,
            reason_temperature=1.0, predict_temperature=0.0,
            llm_semaphore_limit=args.concurrency,
            config=RRMConfig(use_memory=args.memory, random_state=args.seed),
            save_path=str(out_dir), name=f"rrm_{spec.name}", _llm=llm)
        await rrm.set_task(task)
        X = fit_df[["text"]].rename(columns={"text": spec.text_field})
        y = fit_df["label"].map({1: "YES", 0: "NO"}).tolist()
        rc.logger.info("Fitting RRM on %d rows (reason -> extract -> compile -> calibrate)...",
                       len(fit_df))
        await rrm.fit(X, y)
        rrm.save(out_dir)
        rc.logger.info("Fit done in %s: %d rules", rc.human_time(time.time() - t0), len(rrm._rules))

        pd.DataFrame([{
            "rule": r.rule, "outcome": r.outcome,
            "perplexity": (round(float(r.perplexity), 4) if r.perplexity is not None else None),
        } for r in rrm._rules]).to_csv(out_dir / "rrm_rules.csv", index=False)
        if rrm._policy:
            (out_dir / "rrm_policy.txt").write_text(rrm._policy)

    metrics = rc.base_metrics(METHOD, spec, args, fit_df, started, t0, llm)
    metrics["n_rules"] = len(rrm._rules)
    written = []
    for split, frame in scored.items():
        frame = frame.reset_index(drop=True)
        rc.logger.info("Scoring %s (%d rows, %d votes each)...", split, len(frame),
                       rrm.config.ensemble_size)
        # Resume: rows already written for this split are skipped.
        rows_path = out_dir / f"predictions_{split}.jsonl"
        done: dict[str, float] = {}
        if rows_path.exists():
            for line in rows_path.read_text().splitlines():
                if line.strip():
                    record = json.loads(line)
                    done[record["id"]] = record["score"]
        todo = frame[~frame["id"].isin(done)].reset_index(drop=True)
        if len(todo):
            Xp = todo[["text"]].rename(columns={"text": spec.text_field})
            with rows_path.open("a") as fh:
                async for idx, scores, _label, _tc in rrm.predict(Xp):
                    row_id = str(todo.iloc[int(idx)]["id"])
                    score = float(dict(scores).get("score", math.nan))
                    done[row_id] = score
                    fh.write(json.dumps({"id": row_id, "score": score}) + "\n")
                    fh.flush()

        out = frame[["id", "label"]].copy()
        out["score"] = [done.get(i, math.nan) for i in frame["id"]]
        written.append(rc.write_scores(out_dir, spec.name, split, METHOD, out,
                                       with_label=(split == spec.fit_split)))
        metrics["splits"][split] = {"n": len(out), **rc.score_series(out["label"], out["score"])}

    metrics.update(llm_calls=llm.calls, s_per_call=round(llm.s_per_call, 3),
                   wall_clock_s=int(time.time() - t0))
    return rc.finish(out_dir, metrics, written, t0)


def main(argv: list[str] | None = None) -> dict:
    return asyncio.run(run(parse_args(argv)))


if __name__ == "__main__":
    main()
