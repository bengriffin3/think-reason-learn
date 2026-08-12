"""Run GPTree on a dataset with the local Ollama model.

GPTree grows a decision tree whose splits are LLM questions: at each node it
proposes candidate questions, answers them for the samples that reached the
node, and keeps whichever question splits the labels best by gini. Fitting runs
on a stratified subsample of the training split; scoring walks every row down
to a leaf and reads that leaf's class distribution. The learned object is the
tree itself, in `models/<dataset>/gptree.json` — notebook 02 prints it.

Restart-safe on both stages: the fit auto-checkpoints after every node, and the
walk checkpoints every 100 rows, on top of the per-call disk cache. Rerun the
identical command after a kill.

Usage — smoke test against a running Ollama (~10 minutes):
    python run_gptree.py --dataset movie --smoke

Usage — full Movie run (fit on 350 films, score 2,188; hours, see the banner):
    caffeinate -i python run_gptree.py --dataset movie

Usage — full VCBench run (needs the CSV from vcbench.com):
    VCBENCH_DATA=~/.trl-data/vcbench/vcbench_final_public.csv \\
      caffeinate -i python run_gptree.py --dataset vcbench

Outputs (under --out-dir, default results/gptree_<dataset>/):
  <dataset>_<split>_gptree_scores.csv   id,score[,label] — leaf P(positive)
  metrics.json                          AUCs, node count, wall-clock
  gptree_<dataset>/gptree.json          the fitted tree
  llm_cache.jsonl                       the restart cache (never commit it)
"""
from __future__ import annotations
import argparse
import asyncio
import json
import time
from pathlib import Path

import pandas as pd

import _runner_common as rc

METHOD = "gptree"
RETRIES, BACKOFF, CKPT_EVERY = 4, 2.0, 100

TASK_DESCRIPTIONS = {
    "vcbench": (
        "You are an expert in venture capital tasked with identifying successful "
        "founders from their unsuccessful counterparts. All founders are sourced "
        "from anonymised LinkedIn and Crunchbase profiles of companies that have "
        "raised between $100K and $4M in funding. A successful founder is defined "
        "as one whose company has achieved either a total funding of over $500M or "
        "an exit/IPO valued at over $500M. Given a founder's profile, classify "
        "the founder as 'successful' or 'failed'."
    ),
    "movie": (
        "You are an expert film critic and awards judge. Each film is described by a "
        "free-text plot summary. Given a summary, classify the film as 'nominated' "
        "(it earned a major award nomination) or 'not nominated', judging purely from "
        "the narrative and creative qualities visible in the summary."
    ),
}

# Per-dataset tree settings, as used for the shipped trees.
TREE_DEFAULTS = {
    "vcbench": {"fit_size": 0, "min_samples_leaf": 3, "decision_threshold": 0.15},
    "movie": {"fit_size": 350, "min_samples_leaf": 8, "decision_threshold": 0.5},
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    rc.add_common_args(ap)
    ap.add_argument("--max-depth", type=int, default=None, help="default 3, 2 under --smoke")
    ap.add_argument("--max-node-width", type=int, default=3)
    ap.add_argument("--min-samples-leaf", type=int, default=None)
    ap.add_argument("--decision-threshold", type=float, default=None)
    ap.add_argument("--fresh", action="store_true", help="ignore the fit checkpoint")
    ap.add_argument("--predict-only", action="store_true",
                    help="load the tree from --out-dir and only score")
    ap.add_argument("--task", default=None, help="override the task description")
    return ap.parse_args(argv)


def _patch_question_type() -> None:
    """Force question_type=INFERENCE.

    qwen2.5-coder tags nearly every question it writes as "CODE", which GPTree
    treats as a no-op: each node then terminates immediately and the "tree" is
    a bare root. This is the same wrapper the shipped runs used — no library
    edit, and harmless for a model that types its questions correctly.
    """
    from think_reason_learn.gptree import GPTree

    original = GPTree._generate_questions

    async def forced(self, *args, **kwargs):
        result = await original(self, *args, **kwargs)
        for question in result.questions:
            question.question_type = "INFERENCE"
        return result

    GPTree._generate_questions = forced


async def fit_tree(args: argparse.Namespace, spec: rc.DatasetSpec, fit_df: pd.DataFrame,
                   task: str, out_dir: Path, run_name: str):
    from think_reason_learn.gptree import GPTree

    defaults = TREE_DEFAULTS.get(spec.name, {})
    min_leaf = (args.min_samples_leaf if args.min_samples_leaf is not None
                else defaults.get("min_samples_leaf", 3))
    if min_leaf * 2 > len(fit_df):
        # A leaf minimum the fit set cannot satisfy means the root never splits,
        # and a one-node "tree" scores every row at the base rate.
        min_leaf = max(2, len(fit_df) // 4)
        rc.logger.warning("Fitting on only %d rows — min_samples_leaf lowered to %d "
                          "so the tree can actually split", len(fit_df), min_leaf)
    llmc = [rc.choice(args.model)]
    tree = GPTree(
        qgen_llmc=llmc, critic_llmc=llmc, qgen_instr_llmc=llmc, qanswer_llmc=llmc,
        qgen_temperature=0.0, qanswer_temperature=0.0,
        criterion="gini", max_depth=args.max_depth, max_node_width=args.max_node_width,
        min_samples_leaf=min_leaf,
        llm_semaphore_limit=args.concurrency,
        min_question_candidates=3, max_question_candidates=5, n_samples_as_context=15,
        class_ratio="balanced", class_weight="balanced",
        decision_threshold=(args.decision_threshold
                            if args.decision_threshold is not None
                            else defaults.get("decision_threshold", 0.5)),
        use_critic=False, save_path=str(out_dir), name=run_name, random_state=args.seed,
    )

    X = fit_df[["text"]].rename(columns={"text": spec.text_field})
    y = fit_df["label"].map({1: spec.positive, 0: spec.negative}).tolist()
    await tree.set_tasks(task_description=task)
    rc.logger.info("Growing the tree on %d rows (checkpoints after every node)...", len(X))
    async for node in tree.fit(X, y, reset=args.fresh):
        rc.logger.info("Node id=%d gini=%.3f children=%d: %s", node.id, node.gini,
                       len(node.children or []),
                       node.question.value if node.question else "leaf")
    if len(tree._nodes) <= 1:
        raise SystemExit(
            "The tree has one node — every question came back unusable. With a local "
            "model this usually means the question_type patch did not take. Do not "
            "score against this tree.")
    rc.logger.info("Fit done: %d nodes", len(tree._nodes))
    return tree


async def walk(tree, args, spec, frame: pd.DataFrame, tag: str, out_dir: Path) -> list[float | None]:
    """Walk every row of `frame` down to a leaf; return P(positive) per row."""
    from think_reason_learn.core.llms import TokenCounter

    ckpt_path = out_dir / f"predict_checkpoint_{tag}.json"
    leaves: dict[int, int] = {}
    failed: list[int] = []
    if ckpt_path.exists():
        state = json.loads(ckpt_path.read_text())
        leaves = {int(k): v for k, v in state["leaves"].items()}
        failed = list(state.get("failed", []))
        rc.logger.info("Resumed %s checkpoint: %d done, %d failed", tag, len(leaves), len(failed))

    remaining = [i for i in range(len(frame)) if i not in leaves and i not in failed]
    rc.logger.info("Walking %s: %d rows to do (%d already done)", tag, len(remaining), len(leaves))

    counter = TokenCounter()
    semaphore = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    done = [0]

    async def walk_one(i: int) -> None:
        async with semaphore:
            sample = f"{spec.text_field}: {frame.iloc[i]['text']}"
            for attempt in range(RETRIES):
                try:
                    async for record in tree._predict(i, sample, counter):
                        _, question, _, node_id = record
                        if question == "No Question":
                            async with lock:
                                leaves[i] = node_id
                    break
                except Exception:
                    if attempt < RETRIES - 1:
                        await asyncio.sleep(BACKOFF * (2 ** attempt))
                    else:
                        async with lock:
                            failed.append(i)
            async with lock:
                done[0] += 1
                if done[0] % CKPT_EVERY == 0 or done[0] == len(remaining):
                    ckpt_path.write_text(json.dumps({"leaves": leaves, "failed": failed}))

    await asyncio.gather(*[walk_one(i) for i in remaining])
    if failed:
        rc.logger.warning("%s: %d rows failed to reach a leaf and score as NaN", tag, len(failed))
    return [(tree.get_leaf_proba(leaves[i]).get(spec.positive) if i in leaves else None)
            for i in range(len(frame))]


async def run(args: argparse.Namespace) -> dict:
    t0 = time.time()
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    if args.smoke:
        # A tree needs room to split: eight rows can't fill Movie's leaves.
        args.max_depth = args.max_depth or 2
        args.min_samples_leaf = args.min_samples_leaf or 2
    else:
        args.max_depth = args.max_depth or 3
        if args.fit_size is None:
            args.fit_size = TREE_DEFAULTS.get(args.dataset or "", {}).get("fit_size", 0)
    spec, _df, fit_df, scored, out_dir, cache_path = rc.prepare_run(METHOD, args)
    task = args.task or TASK_DESCRIPTIONS.get(spec.name)
    if not task:
        raise SystemExit("No task description for this dataset — pass --task.")

    n_rows = len(set(fit_df["id"]).union(*(set(f["id"]) for f in scored.values())))
    est_calls = rc.estimate_calls(METHOD, len(fit_df), n_rows, max_depth=args.max_depth)
    rc.banner(METHOD, spec, args, fit_df, scored, est_calls, out_dir, cache_path)

    llm = rc.build_llm(cache_path, expected_calls=est_calls)

    # GPTree reads the module-level `llm` singleton, like PolicyInduction.
    import think_reason_learn.gptree._gptree as gptree_module
    gptree_module.llm = llm
    _patch_question_type()

    run_name = f"gptree_{spec.name}"
    if args.predict_only:
        from think_reason_learn.gptree import GPTree

        tree = GPTree.load(out_dir / run_name)
        rc.logger.info("Loaded a %d-node tree from %s", len(tree._nodes), out_dir / run_name)
    else:
        tree = await fit_tree(args, spec, fit_df, task, out_dir, run_name)
    tree.qanswer_llmc = [rc.choice(args.model)]
    tree._llm_semaphore = asyncio.Semaphore(args.concurrency)

    metrics = rc.base_metrics(METHOD, spec, args, fit_df, started, t0, llm)
    metrics["n_nodes"] = len(tree._nodes)
    written = []
    for split, frame in scored.items():
        frame = frame.reset_index(drop=True)
        out = frame[["id", "label"]].copy()
        out["score"] = await walk(tree, args, spec, frame, split, out_dir)
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
