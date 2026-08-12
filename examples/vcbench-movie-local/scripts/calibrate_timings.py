"""Calibrate per-method wall-clock timings on this machine.

Runs each of the four reasoning methods (PI, RRF, GPTree, RRM) on a fresh
cache with a stratified n=100-row slice of the target dataset, measures
wall-clock per method, and extrapolates linearly to the full n to estimate
full-run cost on the current hardware. Results are written to
``precomputed/timings.json`` as::

    {
      "machine": "<arch · OS · cores>",
      "model": "qwen2.5-coder:14b",
      "dataset": "<vcbench|movie>",
      "n_calibration": 100,
      "methods": {
        "<method>": {
          "calibration_wall_s": ...,
          "calls_made": ...,
          "s_per_call": ...,
          "extrapolated_full_run_s": ...,
          "projected_full_calls": ...,      # calls a full run makes, from the
          "projected_full_run_s": ...       # runner's defaults x s_per_call
        },
        ...
      },
      "captured": "<ISO timestamp>"
    }

`s_per_call` is the number to trust: one structured completion, and it hardly
moves between methods. `extrapolated_full_run_s` is the linear-in-rows figure.
`projected_full_run_s` multiplies the measured rate by the call count a full run
would actually make, read from each runner's own defaults — use that one for a
runtime table. A method that could not be measured gets an entry saying so
rather than being left out, so a missing method never reads as a free one.

This replaces guesswork: historical file-timestamp spans include machine
sleep/idle and multi-run appends, so they over- or under-state true
throughput. A short measured run is the only trustworthy basis for the
README's runtime table.

Each method runs into a throwaway directory with an empty cache, so nothing
here reads or writes `precomputed/*_scores.csv` or `models/`. The only file it
touches is `precomputed/timings.json`.

The extrapolation is linear in rows touched. That is exact for RRF (rows x
questions) and close for PI and RRM's scoring stage; it under-states GPTree,
whose fit re-answers questions at every node it opens, and it assumes the model
keeps the same throughput on a longer text. Treat the numbers as the right
order of magnitude for planning, not a promise.

Usage — the four methods on Movie (~30-60 minutes, 100 films each):
    python calibrate_timings.py --dataset movie

Usage — one method, smaller slice:
    python calibrate_timings.py --dataset movie --methods gptree -n 40

Usage — VCBench (needs the CSV from vcbench.com):
    VCBENCH_DATA=~/.trl-data/vcbench/vcbench_final_public.csv \\
      python calibrate_timings.py --dataset vcbench
"""
from __future__ import annotations
import argparse
import json
import os
import platform
import tempfile
import time
from pathlib import Path

import _runner_common as rc

METHODS = ["pi", "rrf", "gptree", "rrm"]
OUT_PATH = rc.EXAMPLE_ROOT / "precomputed" / "timings.json"

# GPTree is the one method a tiny slice cannot exercise: a handful of rows
# cannot be split into leaves, the fit terminates at the root, and the runner
# (rightly) refuses to score against a one-node tree. Give it more rows and a
# smaller leaf minimum. Only its per-call rate is taken from this run — the
# projected call count comes from the runner's own full-run defaults — so the
# looser tree does not distort anything downstream.
SLICE_FLOORS = {"gptree": 20}
EXTRA_ARGV = {"gptree": ["--min-samples-leaf", "3"]}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=sorted(rc.DATASETS), default="movie")
    ap.add_argument("--data", type=Path, default=None,
                    help="records file; Movie defaults to the HuggingFace fetch")
    ap.add_argument("-n", "--n-calibration", type=int, default=100,
                    help="rows in the calibration slice (default 100)")
    ap.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    ap.add_argument("--model", default=rc.DEFAULT_MODEL)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    ap.add_argument("--merge", action="store_true",
                    help="keep entries already in the output file for methods not run now")
    return ap.parse_args(argv)


def machine_label() -> str:
    """What the platform will tell us about the chip this was measured on.

    Deliberately not the hostname the schema sketch suggested: these files get
    committed, and a personal machine name says nothing about throughput. The
    architecture, OS release and core count are what another reader needs to
    know whether your numbers should look like theirs.
    """
    if platform.system() == "Darwin":
        # platform.processor() is just "arm" on Apple silicon, so the machine
        # type and release string are the most it gives without shelling out.
        chip = f"{platform.machine()} · macOS {platform.mac_ver()[0]}"
    else:
        chip = f"{platform.machine()} · {platform.system()} {platform.release()}"
    return f"{chip} · {os.cpu_count()} cores"


def full_run_rows(args: argparse.Namespace, spec: rc.DatasetSpec) -> tuple[int, int]:
    """(rows a full run would fit on, rows it would score) for this dataset."""
    load_args = argparse.Namespace(
        dataset=args.dataset, data=args.data, text_field=None, label_field=None,
        id_field=None, split_field=None, train_value="train", test_value="test",
        max_chars=None, fit_size=0, limit=0, predict_split="both", seed=42)
    df = rc.load_frame(load_args, spec)
    return int((df["split"] == spec.fit_split).sum()), len(df)


def projected_calls(method: str, spec: rc.DatasetSpec, n_fit_full: int, n_rows_full: int) -> int:
    """How many calls a full run of this method would make, at its own defaults.

    Read from each runner's own defaults so this cannot drift from what the
    runners actually do. Multiplied by the measured `s_per_call`, this is a
    steadier estimate than scaling a short run's wall-clock — the per-call rate
    is the stable quantity, the call count is arithmetic.
    """
    import importlib
    import json as _json

    if method == "pi":
        return rc.estimate_calls("pi", n_fit_full, n_rows_full, n_policies=10)
    if method == "rrf":
        shortlist = importlib.import_module("run_rrf").SHIPPED_QUESTIONS.get(spec.name)
        n_q = len(_json.loads(shortlist.read_text())) if shortlist and shortlist.exists() else 14
        return rc.estimate_calls("rrf", n_fit_full, n_rows_full, n_questions=n_q)
    if method == "gptree":
        defaults = importlib.import_module("run_gptree").TREE_DEFAULTS.get(spec.name, {})
        return rc.estimate_calls("gptree", defaults.get("fit_size") or n_fit_full,
                                 n_rows_full, max_depth=3)
    if method == "rrm":
        fit_size = importlib.import_module("run_rrm").FIT_SIZES.get(spec.name, 0)
        return rc.estimate_calls("rrm", fit_size or n_fit_full, n_rows_full)
    raise ValueError(method)


def calibrate_one(method: str, args: argparse.Namespace, touched_full: int) -> dict:
    """Run one method on the slice in a throwaway directory with an empty cache."""
    import importlib

    module = importlib.import_module(f"run_{method}")
    n = max(args.n_calibration, SLICE_FLOORS.get(method, 0))
    with tempfile.TemporaryDirectory(prefix=f"trl_calib_{method}_") as tmp:
        argv = ["--dataset", args.dataset, "--model", args.model,
                "--concurrency", str(args.concurrency),
                "--fit-size", str(n), "--limit", str(n),
                "--out-dir", tmp] + EXTRA_ARGV.get(method, [])
        if args.data:
            argv += ["--data", str(args.data)]
        print(f"\n### calibrating {method} on {n} rows (fresh cache in {tmp})\n", flush=True)
        t0 = time.time()
        metrics = module.main(argv)
        wall = time.time() - t0

    calls = int(metrics.get("llm_calls", 0))
    rows_fit = int(metrics.get("n_fit", 0))
    rows_scored = int(sum(v["n"] for v in metrics.get("splits", {}).values()))
    touched = max(rows_fit + rows_scored, 1)
    return {
        "calibration_wall_s": round(wall, 1),
        "calls_made": calls,
        "s_per_call": round(wall / calls, 3) if calls else None,
        "extrapolated_full_run_s": int(round(wall * touched_full / touched)),
        "calibration_rows_fit": rows_fit,
        "calibration_rows_scored": rows_scored,
    }


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    rc.setup_logging()
    rc.env_guard()
    spec = rc.DATASETS[args.dataset]

    n_fit_full, n_rows_full = full_run_rows(args, spec)
    print(f"{args.dataset}: a full run fits on {n_fit_full:,} rows and scores "
          f"{n_rows_full:,}; calibrating on {args.n_calibration}.")

    results: dict[str, dict] = {}
    if args.merge and args.out.exists():
        results = json.loads(args.out.read_text()).get("methods", {})

    # Rows a full run touches (fit once, score every row), against the rows the
    # calibration actually touched — which each run reports back.
    touched_full = n_fit_full + n_rows_full

    for method in args.methods:
        try:
            entry = calibrate_one(method, args, touched_full)
            calls = projected_calls(method, spec, n_fit_full, n_rows_full)
            entry["source"] = "calibrate_timings.py"
            entry["projected_full_calls"] = calls
            entry["projected_full_run_s"] = (
                int(round(calls * entry["s_per_call"])) if entry["s_per_call"] else None)
            results[method] = entry
            print(f"\n{method}: {entry['calibration_wall_s']:.0f}s for "
                  f"{entry['calls_made']} calls ({entry['s_per_call']}s each) -> "
                  f"{rc.human_time(entry['extrapolated_full_run_s'])} scaled by rows, "
                  f"{rc.human_time(entry['projected_full_run_s'] or 0)} by call count\n")
        except KeyboardInterrupt:
            raise
        # A runner aborting — including the deliberate SystemExit a runner
        # raises when it will not produce usable output — must cost this method
        # its entry, not the whole calibration. The failure is recorded so
        # nobody reads a missing method as a method that takes no time.
        except (Exception, SystemExit) as exc:
            rc.logger.error("%s calibration failed: %s: %s", method, type(exc).__name__, exc)
            results[method] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
        write_payload(args, results, n_fit_full, n_rows_full)  # never lose a finished method

    payload = write_payload(args, results, n_fit_full, n_rows_full)
    print(f"\nWrote {args.out}")
    total = sum(v.get("projected_full_run_s") or 0 for v in results.values())
    if total:
        print(f"All {len(results)} methods, full run, this machine: ~{rc.human_time(total)}")
    return payload


def write_payload(args: argparse.Namespace, results: dict,
                  n_fit_full: int, n_rows_full: int) -> dict:
    """Write timings.json. Called after every method so a later failure cannot
    take a finished measurement down with it."""
    payload = {
        "machine": machine_label(),
        "model": args.model,
        "dataset": args.dataset,
        "n_calibration": args.n_calibration,
        "n_fit_full": n_fit_full,
        "n_rows_full": n_rows_full,
        "methods": results,
        "captured": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "note": (
            "Measured on a fresh cache. `s_per_call` is the reliable number — one "
            "structured completion, and it barely moves between methods. "
            "`extrapolated_full_run_s` scales the calibration wall-clock by rows "
            "touched, which under-states GPTree (its fit re-answers questions at "
            "every node it opens). `projected_full_run_s` is s_per_call times the "
            "call count a full run would make at each runner's own defaults, and is "
            "the better basis for a runtime table."),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    main()
