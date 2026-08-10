"""Run PolicyInduction end-to-end on a dataset with the local Ollama model.

PI writes a short list of natural-language policies from labelled examples,
asks every policy about every row, and fits a logistic regression over the
resulting YES/NO vectors. The learned object is the policy list — read it in
`models/<dataset>/pi/` or in notebook 02.

Fits on the training split, scores every row, and writes per-sample score CSVs
in the same `id,score[,label]` shape as `precomputed/`, plus a `metrics.json`
with the wall-clock. Restart-safe: every call goes through the disk cache in
the out-dir, so rerunning the identical command after a kill replays finished
work for free.

Usage — smoke test against a running Ollama (~5 minutes):
    python run_pi.py --dataset movie --smoke

Usage — full Movie run (fit 1,461 films, score 2,188; hours, see the banner):
    caffeinate -i python run_pi.py --dataset movie

Usage — full VCBench run (needs the CSV from vcbench.com):
    VCBENCH_DATA=~/.trl-data/vcbench/vcbench_final_public.csv \\
      caffeinate -i python run_pi.py --dataset vcbench

Outputs (under --out-dir, default results/pi_<dataset>/):
  <dataset>_<split>_pi_scores.csv   id,score[,label] — one row per sample
  metrics.json                      AUCs, LLM call count, wall-clock
  pi_policies.csv                   the policies this run induced
  llm_cache.jsonl                   the restart cache (never commit it)
"""
from __future__ import annotations
import argparse, asyncio, time

import numpy as np
import pandas as pd

import _runner_common as rc

METHOD = "pi"

# Same wording as the runs that produced the shipped artifacts.
TASK_DESCRIPTIONS = {
    "vcbench": (
        "You are an expert in venture capital tasked with identifying successful "
        "founders. All founders are sourced from anonymised LinkedIn and "
        "Crunchbase profiles of companies that have raised between $100K and $4M "
        "in funding. A successful founder is defined as one whose company has "
        "achieved either a total funding of over $500M or an exit/IPO valued at "
        "over $500M. Given a founder's profile, predict if the founder is likely "
        "to succeed."
    ),
    "movie": (
        "You are an expert film critic and awards judge tasked with identifying "
        "films that earn a major award nomination. Each film is described by a "
        "free-text plot summary. A nominated film is one that earned a major "
        "award nomination for its narrative and creative qualities. Given a "
        "film's plot summary, predict if the film is likely to be nominated."
    ),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    rc.add_common_args(ap)
    ap.add_argument("--n-policies", type=int, default=None,
                    help="policies to induce (max_policy_length); 10 in every shipped "
                         "run, 4 under --smoke")
    ap.add_argument("--max-samples-as-context", type=int, default=10,
                    help="labelled rows shown per policy-generation call (library default 10)")
    ap.add_argument("--task", default=None, help="override the task description")
    return ap.parse_args(argv)


def _to01(value: object) -> int:
    return 1 if str(value).strip().upper() in ("YES", "1", "TRUE", "1.0") else 0


async def run(args: argparse.Namespace) -> dict:
    t0 = time.time()
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    if args.n_policies is None:
        args.n_policies = 4 if args.smoke else 10
    spec, _df, fit_df, scored, out_dir, cache_path = rc.prepare_run(METHOD, args)
    task = args.task or TASK_DESCRIPTIONS.get(spec.name)
    if not task:
        raise SystemExit("No task description for this dataset — pass --task.")

    n_rows = len(set(fit_df["id"]).union(*(set(f["id"]) for f in scored.values())))
    est_calls = rc.estimate_calls(METHOD, len(fit_df), n_rows, n_policies=args.n_policies)
    rc.banner(METHOD, spec, args, fit_df, scored, est_calls, out_dir, cache_path)

    # PI generates policies from balanced batches and stops as soon as one class
    # cannot fill its half. Fewer than half a batch of positives means zero
    # batches, zero policies, and a failure two stages later that reads as a
    # bug in the model rather than in the sizing.
    context = args.max_samples_as_context
    n_pos = int(fit_df["label"].sum())
    if n_pos < context / 2:
        context = max(2, 2 * n_pos)
        rc.logger.warning(
            "Only %d positive rows in the fit set — showing %d rows per generation "
            "call instead of %d so a balanced batch can be formed",
            n_pos, context, args.max_samples_as_context)

    llm = rc.build_llm(cache_path, expected_calls=est_calls)

    # PolicyInduction reads the module-level `llm` singleton — there is no
    # injection hook — so the module attribute is what has to be replaced. Do
    # it before the class is imported and every call routes through the cache.
    import think_reason_learn.policy_induction._policy_induction as pi_module
    pi_module.llm = llm
    from think_reason_learn.policy_induction import PolicyInduction, WeightTrainerConfig

    pi = PolicyInduction(
        gen_llmc=[rc.choice(args.model)],
        predict_llmc=[rc.choice(args.model)],
        config=WeightTrainerConfig(beta=0.5, cv_folds=5, penalty="l2"),
        gen_temperature=0.0,
        predict_temperature=0.0,
        llm_semaphore_limit=args.concurrency,
        max_policy_length=args.n_policies,
        max_samples_as_context=context,
        save_path=str(out_dir),
        name=f"pi_{spec.name}",
        random_state=args.seed,
        confirm_requests=False,
    )

    X = fit_df[["text"]].rename(columns={"text": spec.text_field})
    y = ["YES" if v == 1 else "NO" for v in fit_df["label"]]

    await pi.set_task(task)
    rc.logger.info("Fitting PolicyInduction on %d rows...", len(fit_df))
    await pi.fit(X, y)
    rc.logger.info("Fit done in %s (%d calls so far)",
                   rc.human_time(time.time() - t0), llm.calls)

    pi.save(out_dir)
    policies = pi._policy_memory["policy"].tolist()
    pd.DataFrame({"policy_id": range(len(policies)), "policy": policies}).to_csv(
        out_dir / "pi_policies.csv", index=False)
    rc.logger.info("Induced %d policies -> %s", len(policies), out_dir / "pi_policies.csv")

    async def answer_matrix(frame: pd.DataFrame, tag: str) -> pd.DataFrame:
        """Ask every policy about every row of `frame`, in row order.

        Rows that were in the fit set were already asked these exact questions
        at temperature 0, so they come back from the cache, not the model.
        PI checkpoints predictions to `save_path` under one fixed filename and
        keys them by row position, so each pass needs its own directory —
        otherwise a resumed run replays the wrong split's answers.
        """
        rc.logger.info("Scoring %s (%d rows)...", tag, len(frame))
        pi.save_path = out_dir / f"predict_{tag}"
        frame = frame.reset_index(drop=True)
        Xp = frame[["text"]].rename(columns={"text": spec.text_field})
        rows: list[tuple[int, list]] = []
        async for idx, vector, _label, _tc in pi.predict(Xp):
            rows.append((int(idx), list(vector)))
        rows.sort(key=lambda r: r[0])
        return pd.DataFrame(
            [[_to01(v) for v in vec] for _, vec in rows], index=frame["id"])

    answers = pd.concat([await answer_matrix(frame, split) for split, frame in scored.items()])
    missing = fit_df[~fit_df["id"].isin(answers.index)]
    if len(missing):
        # --predict-split held-out: the combiner still needs the fit rows'
        # answer vectors. Every one of those calls is a cache hit.
        answers = pd.concat([answers, await answer_matrix(missing, "fit")])
    answers = answers[~answers.index.duplicated()]

    all_scores = rc.combine_answers(answers, fit_df, scored, seed=args.seed)

    metrics = rc.base_metrics(METHOD, spec, args, fit_df, started, t0, llm)
    metrics["n_policies"] = len(policies)
    written = []
    for split, frame in scored.items():
        out = frame[["id", "label"]].copy()
        out["score"] = all_scores[split]
        written.append(rc.write_scores(out_dir, spec.name, split, METHOD, out,
                                       with_label=(split == spec.fit_split)))
        metrics["splits"][split] = {"n": len(out), **rc.score_series(out["label"], out["score"])}
    return rc.finish(out_dir, metrics, written, t0)


def main(argv: list[str] | None = None) -> dict:
    return asyncio.run(run(parse_args(argv)))


if __name__ == "__main__":
    main()
