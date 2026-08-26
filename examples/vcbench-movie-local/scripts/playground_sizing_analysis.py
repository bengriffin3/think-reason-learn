"""The Monte Carlo behind the playground's size (n=1,500) — rerunnable.

For each candidate size n, draws stratified subsamples of the VCBench public
4,500 (preserving the 9% base rate) and measures how often each full-set
conclusion reproduces, how far each series' ROC/PR-AUC drifts, and what a
full four-method run at that size costs. Reads only ``precomputed/`` — no
LLM, no network, no raw dataset — and is seeded, so it reproduces the numbers
quoted in the README's playground section exactly.

Ensembles are recomputed WITHIN each subsample, exactly as they would be on
the playground: reasoning = pct-rank average of the four methods, traditional
= pct-rank average of the 5-model suite (from the permodel refit CSV),
combined = rank01(traditional) + rank01(reasoning).

Two caveats the numbers carry:

* This simulates *evaluating* on fewer founders. *Fitting* on fewer costs
  extra — the maintainer verification runs measured roughly −0.02 to −0.06
  ROC-AUC per refit model at n=1,500 (``playground/reference.json`` has the
  end-to-end result) — so these curves are an optimistic lower bound.
* Probabilities describe a random draw. The shipped playground is a
  covariate-balanced draw validated to reproduce the core conclusions, so
  these numbers price how well *future* results of similar margin carry, not
  whether the documented ones hold (they were verified directly).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata
from sklearn.metrics import average_precision_score, roc_auc_score

import _runner_common as rc

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
PRECOMPUTED = EXAMPLE_ROOT / "precomputed"
OUT_DIR = EXAMPLE_ROOT / "results" / "playground_sizing"

SIZES = [250, 500, 750, 1000, 1250, 1500, 2000, 3000]
N_DRAWS = 2000
SEED = 42

METHODS = ["pi", "rrf", "gptree", "rrm"]
SUITE = ["lr", "hgb", "rf", "et", "gnb"]
INTERPRETABLE = ["lr", "gnb", "dt", "knn"]
TRAD_ALL = SUITE + ["dt", "knn"]
ENSEMBLES = ["reasoning_ens", "trad_ens", "combined"]
ALL_SERIES = METHODS + TRAD_ALL + ENSEMBLES

VCBENCH_CALL_PARAMS = {"pi": {"n_policies": 10}, "rrf": {"n_questions": 16},
                       "gptree": {"max_depth": 3}, "rrm": {"ensemble_size": 3}}

CORE = [
    "ROC: combined > trad ens",
    "ROC: combined > reasoning ens",
    "ROC: trad ens > reasoning ens",
    "ROC: reasoning ens > every interpretable",
]


def load_matrix() -> tuple[pd.DataFrame, np.ndarray]:
    permodel = pd.read_csv(
        PRECOMPUTED / "vcbench_public_traditional_permodel_scores.csv"
    ).set_index("id")
    df = cast(pd.DataFrame, permodel[TRAD_ALL].copy())
    for m in METHODS:
        s = pd.read_csv(PRECOMPUTED / f"vcbench_public_{m}_scores.csv"
                        ).set_index("id")["score"]
        df[m] = s.reindex(df.index)
    assert not df.isna().to_numpy().any()
    return df, permodel["label"].to_numpy()


def pct_ranks(x: np.ndarray) -> np.ndarray:
    return rankdata(x, axis=0) / x.shape[0]


def series_scores(scores: np.ndarray, cols: dict[str, int]
                  ) -> dict[str, np.ndarray]:
    out = {name: scores[:, cols[name]] for name in METHODS + TRAD_ALL}
    reasoning = pct_ranks(scores[:, [cols[m] for m in METHODS]]).mean(axis=1)
    trad = pct_ranks(scores[:, [cols[m] for m in SUITE]]).mean(axis=1)
    out["reasoning_ens"] = reasoning
    out["trad_ens"] = trad
    out["combined"] = (rankdata(trad) + rankdata(reasoning)) / len(trad)
    return out


def metrics(scores: np.ndarray, y: np.ndarray, cols: dict[str, int]
            ) -> tuple[dict[str, float], dict[str, float]]:
    ss = series_scores(scores, cols)
    roc = {k: float(roc_auc_score(y, v)) for k, v in ss.items()}
    pr = {k: float(average_precision_score(y, v)) for k, v in ss.items()}
    return roc, pr


def conclusions(roc: dict[str, float], pr: dict[str, float]
                ) -> dict[str, bool]:
    return {
        "ROC: combined > trad ens": roc["combined"] > roc["trad_ens"],
        "ROC: combined > reasoning ens":
            roc["combined"] > roc["reasoning_ens"],
        "ROC: trad ens > reasoning ens":
            roc["trad_ens"] > roc["reasoning_ens"],
        "ROC: reasoning ens > every interpretable":
            all(roc["reasoning_ens"] > roc[m] for m in INTERPRETABLE),
        "ROC: PI best reasoning method":
            all(roc["pi"] >= roc[m] for m in METHODS),
        "ROC: GPTree weakest reasoning method":
            all(roc["gptree"] <= roc[m] for m in METHODS),
        "ROC: extra-trees best suite model":
            all(roc["et"] >= roc[m] for m in SUITE),
        "PR: combined > trad ens": pr["combined"] > pr["trad_ens"],
        "PR: combined > reasoning ens": pr["combined"] > pr["reasoning_ens"],
        "PR: trad ens > reasoning ens": pr["trad_ens"] > pr["reasoning_ens"],
        "PR: reasoning ens > every interpretable":
            all(pr["reasoning_ens"] > pr[m] for m in INTERPRETABLE),
    }


def projected_hours(n: int, timings: dict) -> dict[str, float]:
    hours = {}
    for m in METHODS:
        calls = rc.estimate_calls(m, n, n, **VCBENCH_CALL_PARAMS[m])
        hours[m] = calls * timings["methods"][m]["s_per_call"] / 3600
    hours["total"] = sum(hours.values())
    return hours


def main() -> None:
    df, y_all = load_matrix()
    cols = {name: i for i, name in enumerate(df.columns)}
    scores_all = df.to_numpy()
    timings = json.loads((PRECOMPUTED / "timings.json").read_text())
    pos_idx = np.flatnonzero(y_all == 1)
    neg_idx = np.flatnonzero(y_all == 0)
    base_rate = len(pos_idx) / len(y_all)

    full_roc, full_pr = metrics(scores_all, y_all, cols)
    full_concl = conclusions(full_roc, full_pr)
    tracked = [k for k, v in full_concl.items() if v]
    print("Full-set (n=4,500) values with the within-sample operator:")
    for k in ALL_SERIES:
        print(f"  {k:14s} ROC {full_roc[k]:.4f}  PR {full_pr[k]:.4f}")

    full_lb = np.array([full_roc[s] for s in ALL_SERIES])
    rng = np.random.default_rng(SEED)
    rows = []
    for n in SIZES:
        n_pos = int(round(n * base_rate))
        agree = dict.fromkeys(tracked, 0)
        all_core = 0
        taus: list[float] = []
        devs: dict[str, list[float]] = {s: [] for s in ALL_SERIES}
        for _ in range(N_DRAWS):
            idx = np.concatenate([
                rng.choice(pos_idx, n_pos, replace=False),
                rng.choice(neg_idx, n - n_pos, replace=False)])
            roc, pr = metrics(scores_all[idx], y_all[idx], cols)
            c = conclusions(roc, pr)
            for k in tracked:
                agree[k] += c[k]
            all_core += all(c[k] for k in CORE)
            tau = kendalltau(full_lb, [roc[s] for s in ALL_SERIES])
            taus.append(float(cast(float, tau[0])))
            for s in ALL_SERIES:
                devs[s].append(abs(roc[s] - full_roc[s]))
        row: dict[str, float] = {"n": n,
                                 "P(all 4 ROC core)": all_core / N_DRAWS,
                                 "tau_med": float(np.median(taus))}
        for k in tracked:
            row[k] = agree[k] / N_DRAWS
        for s in ENSEMBLES:
            row[f"rocdev90_{s}"] = float(np.quantile(devs[s], 0.90))
        row.update({f"h_{m}": v
                    for m, v in projected_hours(n, timings).items()})
        rows.append(row)
        print(f"n={n:5d}  P(all core)={row['P(all 4 ROC core)']:.3f}  "
              f"tau={row['tau_med']:.3f}  "
              f"rocdev90(comb)={row['rocdev90_combined']:.4f}  "
              f"total h={row['h_total']:.1f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_DIR / "carryover_table.csv", index=False)
    print(f"wrote {OUT_DIR / 'carryover_table.csv'}")


if __name__ == "__main__":
    main()
