"""Build the playground records file from your VCBench download.

The playground is a fixed 1,500-founder subsample of the VCBench public split
(``playground/vcbench_playground_ids.csv``) chosen so that the example's core
findings reproduce on it — see the README's playground section for what does
and deliberately does not carry over. Raw prose is not committed to this
repo, so this script slices it out of the full public CSV you request at
vcbench.com:

    python make_playground.py --data "$VCBENCH_DATA"

writes ``results/playground_records.csv`` (gitignored), which every runner
accepts directly:

    python run_gptree.py --dataset vcbench --data ../results/playground_records.csv
    python run_rrm.py    --dataset vcbench --data ../results/playground_records.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
IDS = EXAMPLE_ROOT / "playground" / "vcbench_playground_ids.csv"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True,
                    help="the full VCBench public CSV ($VCBENCH_DATA)")
    ap.add_argument("--out", type=Path,
                    default=EXAMPLE_ROOT / "results" / "playground_records.csv")
    args = ap.parse_args()

    ids = pd.read_csv(IDS)
    full = pd.read_csv(args.data)
    records = full[full["founder_uuid"].isin(ids["founder_uuid"])]
    records = records.reset_index(drop=True)

    missing = len(ids) - len(records)
    if missing:
        raise SystemExit(
            f"{missing} playground founders are not in {args.data} — is this "
            "the full 4,500-row public CSV from vcbench.com?")
    merged = records.merge(ids, on="founder_uuid")
    if not (merged["success"] == merged["label"]).all():
        raise SystemExit("labels in --data disagree with the playground ids "
                         "file — wrong or modified dataset?")

    args.out.parent.mkdir(exist_ok=True)
    records.to_csv(args.out, index=False)
    rate = records["success"].mean()
    print(f"wrote {args.out} — {len(records)} founders, "
          f"{int(records['success'].sum())} positives (base rate {rate:.1%})")


if __name__ == "__main__":
    main()
