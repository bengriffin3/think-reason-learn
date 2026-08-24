"""Generate the per-model traditional score CSVs the hero figure draws from.

``run_traditional.py`` ships only its rank-average ensemble in
``precomputed/*_traditional_scores.csv``; the hero figure also draws one line
per individual traditional model, including two interpretable models the suite
does not contain (a shallow decision tree and k-NN). This script fits all
seven — the 5-model suite plus those two — with the suite's exact features,
splits and seed, and writes per-row scores to
``precomputed/{movie_test,vcbench_public}_traditional_permodel_scores.csv``.

Usage (matches ``run_all.sh``'s traditional step):

    python make_permodel_scores.py --dataset movie
    python make_permodel_scores.py --dataset vcbench --data "$VCBENCH_DATA"

Movie fits on the deduplicated temporal train split and scores the test split;
VCBench is 3-fold out-of-fold over the public 4,500 (StratifiedKFold, seed 42).
Row order follows the shipped ``*_traditional_scores.csv`` files.

Scores drift a little across machines and library versions (different BLAS,
sentence-transformers, scikit-learn): regenerating these CSVs elsewhere will
shift third-decimal digits, and ``make_hero_figure.py`` will then refuse to
draw until its reference table is updated to match. The committed CSVs are the
maintainer reference run under the pinned scikit-learn 1.9.0.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import _runner_common as rc
import run_traditional as rt

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
PRECOMPUTED = EXAMPLE_ROOT / "precomputed"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SEED = 42


def all_models(seed: int) -> dict:
    """The run_traditional.py suite plus the two extra interpretable models."""
    models = rt.make_models(seed)
    models["dt"] = lambda: DecisionTreeClassifier(
        max_depth=5, min_samples_leaf=20, class_weight="balanced",
        random_state=seed)
    models["knn"] = lambda: make_pipeline(
        StandardScaler(with_mean=False), KNeighborsClassifier(n_neighbors=50))
    return models


def movie_scores() -> tuple[str, pd.DataFrame]:
    """Fit train → score test on the same rows the reasoning runs used."""
    spec = rc.DATASETS["movie"]
    args = argparse.Namespace(
        dataset="movie", data=None, text_field=None, label_field=None,
        id_field=None, split_field=None, train_value="train",
        test_value="test", max_chars=None)
    df = rc.load_frame(args, spec).rename(columns={"text": "summary"})
    df_train = cast(pd.DataFrame, df[df["split"] == "train"].reset_index(drop=True))
    df_test = cast(pd.DataFrame, df[df["split"] == "test"].reset_index(drop=True))
    X_train, X_test, _ = rt.build_features(
        df_train, df_test, ["summary"], [], EMBED_MODEL, 400)
    y_train = df_train["label"].to_numpy()
    y_test = df_test["label"].to_numpy()

    out = pd.DataFrame(index=pd.Index(df_test["id"], name="id"))
    for name, factory in all_models(SEED).items():
        model = factory()
        model.fit(X_train, y_train)
        out[name] = model.predict_proba(X_test)[:, 1]
        print(f"  {name:4s}  test ROC-AUC "
              f"{roc_auc_score(y_test, out[name]):.4f}")
    out["label"] = y_test
    return "movie_test", out


def vcbench_scores(data: Path) -> tuple[str, pd.DataFrame]:
    """3-fold out-of-fold over the public 4,500, seed 42."""
    df = rt.load_records(data)
    X, _, _ = rt.build_features(df, None, ["anonymised_prose"], [],
                                EMBED_MODEL, 400)
    y = df["success"].to_numpy()

    out = pd.DataFrame(index=pd.Index(df["founder_uuid"], name="id"))
    folds = list(StratifiedKFold(3, shuffle=True, random_state=SEED)
                 .split(np.zeros(len(y)), y))
    for name, factory in all_models(SEED).items():
        oof = np.zeros(len(y))
        for tr, te in folds:
            model = factory()
            model.fit(X[tr], y[tr])
            oof[te] = model.predict_proba(X[te])[:, 1]
        out[name] = oof
        print(f"  {name:4s}  OOF ROC-AUC {roc_auc_score(y, oof):.4f}")
    out["label"] = y
    return "vcbench_public", out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=["movie", "vcbench"], required=True)
    ap.add_argument("--data", type=Path, default=None,
                    help="vcbench only: the public records CSV ($VCBENCH_DATA)")
    args = ap.parse_args()

    if args.dataset == "movie":
        prefix, scores = movie_scores()
    else:
        if args.data is None:
            raise SystemExit("--data is required for --dataset vcbench")
        prefix, scores = vcbench_scores(args.data)

    # Keep the row order of the shipped ensemble CSV, and sanity-check that
    # the labels line up with it row for row.
    shipped = pd.read_csv(
        PRECOMPUTED / f"{prefix}_traditional_scores.csv").set_index("id")
    scores = scores.reindex(shipped.index)
    assert not scores.isna().to_numpy().any(), "id mismatch vs shipped CSV"
    assert (scores["label"] == shipped["label"]).all(), "label mismatch"

    out = PRECOMPUTED / f"{prefix}_traditional_permodel_scores.csv"
    scores.reset_index().to_csv(out, index=False)
    print(f"wrote {out.relative_to(EXAMPLE_ROOT)}")


if __name__ == "__main__":
    main()
