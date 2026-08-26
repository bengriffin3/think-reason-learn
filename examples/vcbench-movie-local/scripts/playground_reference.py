"""Recompute and assert the playground's reference numbers, from shipped CSVs.

The playground (``playground/``) is a fixed 1,500-founder subsample of the
VCBench public split (135 positives — the same 9% base rate). This script
rebuilds its two reference views from committed files only and writes
``playground/reference.json``:

* **eval-only** — the full-benchmark reference models (``precomputed/``)
  evaluated on the playground founders. This is the view Stage-1 style
  subsampling sees: same scores, fewer rows.
* **intern** — what you get after actually refitting on the playground:
  GPTree and RRM from the maintainer verification runs
  (``playground/{gptree,rrm}_refit_scores.csv``), the traditional suite from
  ``playground/traditional_refit_permodel_scores.csv`` (3-fold OOF within the
  1,500), and PI/RRF from ``precomputed/`` — those two fit on fixed tiny
  samples (10-row context batches / 40 labelled examples), so their full-set
  scores stand in unchanged.

GPTree leaves some rows unscored (91 here — rows its tree cannot route to a
leaf); they are median-imputed before ensembling, a rank-neutral middle
position. Every number is asserted at 4 dp against the reference run, so the
committed JSON cannot silently drift from the CSVs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
PRECOMPUTED = EXAMPLE_ROOT / "precomputed"
PLAYGROUND = EXAMPLE_ROOT / "playground"

METHODS = ["pi", "rrf", "gptree", "rrm"]
SUITE = ["lr", "hgb", "rf", "et", "gnb"]
INTERPRETABLE = ["lr", "gnb", "dt", "knn"]

# The reference values (maintainer runs, 2026-08-25). Any recomputed number
# that does not match at 4 dp aborts instead of writing.
REFERENCE = {
    "evalonly": {"reasoning_ens": 0.7380, "trad_ens": 0.7524,
                 "combined": 0.7789, "gptree": 0.6305, "rrm": 0.6669},
    "intern": {"reasoning_ens": 0.7075, "trad_ens": 0.7050,
               "combined": 0.7409, "gptree": 0.5800, "rrm": 0.6034},
}


def rank01(x: np.ndarray) -> np.ndarray:
    return rankdata(x) / len(x)


def build_ensembles(view: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    view["reasoning_ens"] = np.mean([rank01(view[m]) for m in METHODS], axis=0)
    view["trad_ens"] = np.mean([rank01(view[m]) for m in SUITE], axis=0)
    view["combined"] = rank01(view["trad_ens"]) + rank01(view["reasoning_ens"])
    return view


def main() -> None:
    idx = pd.Index(
        pd.read_csv(PLAYGROUND / "vcbench_playground_ids.csv")["founder_uuid"],
        name="id")
    permodel = pd.read_csv(
        PRECOMPUTED / "vcbench_public_traditional_permodel_scores.csv"
    ).set_index("id").reindex(idx)
    y = permodel["label"].to_numpy()
    shipped = {m: pd.read_csv(PRECOMPUTED / f"vcbench_public_{m}_scores.csv"
                              ).set_index("id")["score"].reindex(idx)
               for m in METHODS}

    trad_refit = pd.read_csv(
        PLAYGROUND / "traditional_refit_permodel_scores.csv"
    ).set_index("id").reindex(idx)
    gptree_refit = pd.read_csv(PLAYGROUND / "gptree_refit_scores.csv"
                               ).set_index("id")["score"].reindex(idx)
    n_unscored = int(gptree_refit.isna().sum())
    gptree_refit = cast(pd.Series, gptree_refit.fillna(gptree_refit.median()))
    rrm_refit = pd.read_csv(PLAYGROUND / "rrm_refit_scores.csv"
                            ).set_index("id")["score"].reindex(idx)

    views = {
        "evalonly": build_ensembles(
            {m: shipped[m].to_numpy() for m in METHODS}
            | {m: permodel[m].to_numpy() for m in SUITE + ["dt", "knn"]}),
        "intern": build_ensembles(
            {"pi": shipped["pi"].to_numpy(), "rrf": shipped["rrf"].to_numpy(),
             "gptree": gptree_refit.to_numpy(), "rrm": rrm_refit.to_numpy()}
            | {m: trad_refit[m].to_numpy() for m in SUITE + ["dt", "knn"]}),
    }

    out: dict = {"n": int(len(idx)), "n_pos": int(y.sum()),
                 "gptree_unscored_rows": n_unscored,
                 "refit_runs": "GPTree fit 1,500 / RRM fit 350, "
                               "qwen2.5-coder:14b, 2026-08-25"}
    for name, view in views.items():
        roc = {k: round(float(roc_auc_score(y, v)), 4)
               for k, v in view.items()}
        pr = {k: round(float(average_precision_score(y, v)), 4)
              for k, v in view.items()}
        for k, expected in REFERENCE[name].items():
            assert roc[k] == expected, (name, k, roc[k], expected)
        out[name] = {"roc_auc": roc, "pr_auc": pr}

    ri, pi_ = out["intern"]["roc_auc"], out["intern"]["pr_auc"]
    out["conclusions_intern_view"] = {
        "ROC: combined > trad ens": ri["combined"] > ri["trad_ens"],
        "ROC: combined > reasoning ens": ri["combined"] > ri["reasoning_ens"],
        "ROC: reasoning ens > every interpretable":
            all(ri["reasoning_ens"] > ri[m] for m in INTERPRETABLE),
        "ROC: trad ens > reasoning ens (full-benchmark claim)":
            ri["trad_ens"] > ri["reasoning_ens"],
        "PR: combined > trad ens": pi_["combined"] > pi_["trad_ens"],
        "PR: combined > reasoning ens":
            pi_["combined"] > pi_["reasoning_ens"],
    }

    path = PLAYGROUND / "reference.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {path.relative_to(EXAMPLE_ROOT)}")
    for name in ("evalonly", "intern"):
        r = out[name]["roc_auc"]
        print(f"  {name:9s} combined {r['combined']:.4f}  "
              f"reasoning {r['reasoning_ens']:.4f}  trad {r['trad_ens']:.4f}")


if __name__ == "__main__":
    main()
