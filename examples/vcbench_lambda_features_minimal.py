"""Minimal example: human vs. LLM-generated features on VCBench.

Compares three feature-engineering approaches on VCBench founder profiles:

- ``human``    — 7 hand-crafted binary features (no API key needed)
- ``llm``      — LLM-generated Python lambda features
- ``combined`` — both stacked together (default)

Usage::

    python examples/vcbench_lambda_features_minimal.py \
        --input_csv /path/to/vcbench_final_public.csv --feature_mode human

    python examples/vcbench_lambda_features_minimal.py \
        --input_csv /path/to/vcbench_final_public.csv --n_features 8

Cost: < $0.01 with gpt-4.1-nano (default).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from think_reason_learn.core.llms import OpenAIChoice
from think_reason_learn.datasets import (
    VCBENCH_HELPERS,
    VCBENCH_SCHEMA,
    load_vcbench,
    parse_company_size,
    parse_duration,
    parse_qs,
)
from think_reason_learn.features import FeatureEvaluator, FeatureGenerator

# ---------------------------------------------------------------------------
# Human-engineered baseline features
# ---------------------------------------------------------------------------
# Seven binary features a human analyst would try first — no LLM involved,
# just domain intuition.  Interns: read, understand, and extend these!

_SENIOR_ROLES = {
    "ceo",
    "cto",
    "coo",
    "cfo",
    "founder",
    "co-founder",
    "cofounder",
    "president",
    "vp",
    "vice president",
}


def _extract_human_features(record: dict[str, Any]) -> dict[str, int]:
    """Extract 7 hand-crafted binary features from one founder record."""
    educations = record.get("educations", [])
    jobs = record.get("jobs", [])

    top_univ = int(any(parse_qs(e.get("qs_ranking", "")) <= 50 for e in educations))
    has_phd = int(
        any(
            "phd" in e.get("degree", "").lower()
            or "doctor" in e.get("degree", "").lower()
            for e in educations
        )
    )
    prior_exit = int(
        len(record.get("ipos", [])) + len(record.get("acquisitions", [])) > 0
    )
    senior = int(
        any(any(kw in j.get("role", "").lower() for kw in _SENIOR_ROLES) for j in jobs)
    )
    large_co = int(
        any(parse_company_size(j.get("company_size", "")) >= 1000 for j in jobs)
    )
    startup = int(
        any(0 < parse_company_size(j.get("company_size", "")) <= 50 for j in jobs)
    )
    total_yrs = sum(parse_duration(j.get("duration", "")) for j in jobs)
    long_exp = int(total_yrs > 5.0)

    return {
        "top_university": top_univ,
        "has_phd": has_phd,
        "prior_exit": prior_exit,
        "senior_leadership": senior,
        "large_company_exp": large_co,
        "startup_exp": startup,
        "long_experience": long_exp,
    }


def _human_feature_df(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Build binary feature matrix from human-engineered features."""
    return pd.DataFrame([_extract_human_features(r) for r in records])


HUMAN_FEATURE_NAMES = list(_extract_human_features({}).keys())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """Build CLI argument parser."""
    p = argparse.ArgumentParser(
        description="VCBench: human vs. LLM features.",
    )
    p.add_argument("--input_csv", required=True, help="VCBench CSV path.")
    p.add_argument("--label_column", default="success")
    p.add_argument("--max_rows", type=int, default=0, help="0 = all data (default).")
    p.add_argument(
        "--n_features",
        type=int,
        default=8,
        help="LLM features to generate (default: 8).",
    )
    p.add_argument("--test_size", type=float, default=0.40)
    p.add_argument("--model", default="gpt-4.1-nano")
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument(
        "--feature_mode",
        choices=["human", "llm", "combined"],
        default="combined",
        help="human | llm | combined (default).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _report(
    label: str,
    feature_names: list[str],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    random_state: int,
) -> dict[str, Any]:
    """Fit logistic regression, tune threshold, print metrics."""
    clf = LogisticRegression(max_iter=1000, random_state=random_state)
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:, 1]

    # F0.5-optimal threshold on training set (handles class imbalance).
    y_prob_tr = clf.predict_proba(X_train)[:, 1]
    best_t, best_f = 0.5, 0.0
    for t in np.arange(0.05, 0.95, 0.01):
        f = fbeta_score(
            y_train,
            (y_prob_tr >= t).astype(int),
            beta=0.5,
            zero_division=0.0,  # type: ignore[arg-type]
        )
        if f > best_f:
            best_f, best_t = f, float(t)

    y_pred = (y_prob >= best_t).astype(int)
    roc = roc_auc_score(y_test, y_prob)
    pr = average_precision_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0.0)  # type: ignore[arg-type]
    rec = recall_score(y_test, y_pred, zero_division=0.0)  # type: ignore[arg-type]
    f05 = fbeta_score(y_test, y_pred, beta=0.5, zero_division=0.0)  # type: ignore[arg-type]

    # Coefficients
    coefs = sorted(
        zip(feature_names, clf.coef_[0]), key=lambda x: abs(x[1]), reverse=True
    )
    print(f"\n  [{label}] — {len(feature_names)} features, threshold={best_t:.2f}")
    for name, c in coefs:
        print(f"    {'+' if c >= 0 else '-'}{abs(c):.3f}  {name}")

    # Metrics
    print(
        f"\n  ROC-AUC={roc:.3f}  PR-AUC={pr:.3f}  "
        f"Prec={prec:.3f}  Rec={rec:.3f}  F0.5={f05:.3f}  Acc={acc:.3f}"
    )

    return {
        "roc_auc": roc,
        "pr_auc": pr,
        "precision": prec,
        "recall": rec,
        "f0.5": f05,
        "accuracy": acc,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    """Run the feature comparison example."""
    args = _parse_args()
    mode = args.feature_mode
    rs = args.random_state

    # 1. Load data
    print(f"\n{'=' * 60}")
    print(f"  VCBench Feature Example  [mode={mode}]")
    print(f"{'=' * 60}\n")

    records, labels = load_vcbench(
        args.input_csv,
        args.label_column,
        args.max_rows,
        rs,
    )

    # 2. Train / test split (BEFORE LLM generation to prevent leakage)
    idx = np.arange(len(records))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=args.test_size,
        stratify=labels,
        random_state=rs,
    )
    train_recs = [records[i] for i in train_idx]
    test_recs = [records[i] for i in test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    n_pos_test = int(y_test.sum())
    print(f"  {len(records)} rows -> {len(train_idx)} train / {len(test_idx)} test")
    print(f"  Train: {int(y_train.sum())} pos, {len(y_train) - int(y_train.sum())} neg")
    print(f"  Test:  {n_pos_test} pos, {len(y_test) - n_pos_test} neg")
    if n_pos_test < 10:
        print(
            f"  WARNING: only {n_pos_test} test positives — "
            "try --max_rows 0 or --test_size 0.5"
        )

    # 3. Human features (always computed — free and instant)
    h_train = _human_feature_df(train_recs)
    h_test = _human_feature_df(test_recs)

    if mode in ("human", "combined"):
        print("\n  Human features:", ", ".join(HUMAN_FEATURE_NAMES))

    # 4. LLM features (training data only)
    llm_names: list[str] = []
    l_train = pd.DataFrame()
    l_test = pd.DataFrame()

    if mode in ("llm", "combined"):
        print(f"\n  Generating {args.n_features} LLM features " f"with {args.model}...")

        generator = FeatureGenerator(
            schema=VCBENCH_SCHEMA,
            helpers=VCBENCH_HELPERS,
            llm_priority=[OpenAIChoice(model=args.model)],
            temperature=0.7,
        )
        rules = await generator.generate(
            samples=train_recs,
            labels=y_train.tolist(),
            n_rules=args.n_features,
            n_samples=min(60, len(train_recs)),
        )
        if not rules:
            sys.exit("ERROR: No rules generated. Check your API key.")

        evaluator = FeatureEvaluator(rules=rules, helpers=VCBENCH_HELPERS)
        errors = evaluator.compilation_errors
        if errors:
            for name, err in errors.items():
                print(f"  WARNING: {name} failed: {err}")

        l_train = evaluator.evaluate_df(train_recs)
        l_test = evaluator.evaluate_df(test_recs)
        llm_names = [r.name for r in rules]

        ok = len(rules) - len(errors)
        print(f"  Generated {len(rules)} rules ({ok} compiled OK)")
        for i, r in enumerate(rules, 1):
            print(f"    {i}. {r.name}: {r.description}")

    # 5. Evaluate
    summary: dict[str, dict[str, float]] = {}

    if mode == "human":
        summary["human"] = _report(
            "Human Only",
            HUMAN_FEATURE_NAMES,
            h_train.values.astype(float),
            h_test.values.astype(float),
            y_train,
            y_test,
            rs,
        )

    elif mode == "llm":
        summary["llm"] = _report(
            "LLM Only",
            llm_names,
            l_train.values.astype(float),
            l_test.values.astype(float),
            y_train,
            y_test,
            rs,
        )

    else:  # combined — compare all three
        summary["human"] = _report(
            "Human Only",
            HUMAN_FEATURE_NAMES,
            h_train.values.astype(float),
            h_test.values.astype(float),
            y_train,
            y_test,
            rs,
        )
        summary["llm"] = _report(
            "LLM Only",
            llm_names,
            l_train.values.astype(float),
            l_test.values.astype(float),
            y_train,
            y_test,
            rs,
        )

        # Deduplicate LLM features that overlap with human features
        h_set = set(HUMAN_FEATURE_NAMES)
        keep_llm: list[str] = []
        dropped: list[str] = []
        for col in l_train.columns:
            if col in h_set:
                dropped.append(col)
            elif any(
                np.array_equal(l_train[col].values, h_train[hc].values)  # type: ignore[arg-type]
                for hc in h_train.columns
            ):
                dropped.append(col)
            else:
                keep_llm.append(col)

        if dropped:
            print(
                f"\n  Deduplicated {len(dropped)} LLM feature(s): " + ", ".join(dropped)
            )

        comb_names = HUMAN_FEATURE_NAMES + keep_llm
        comb_train = pd.concat(
            [h_train, l_train[keep_llm]],
            axis=1,
        ).values.astype(float)
        comb_test = pd.concat(
            [h_test, l_test[keep_llm]],
            axis=1,
        ).values.astype(float)

        summary["combined"] = _report(
            "Combined",
            comb_names,
            comb_train,
            comb_test,
            y_train,
            y_test,
            rs,
        )

    # 6. Comparison table
    if len(summary) > 1:
        print(f"\n{'=' * 60}")
        print("  Comparison")
        print(f"{'=' * 60}")
        cols = ["ROC-AUC", "PR-AUC", "Prec", "Recall", "F0.5", "Acc"]
        keys = ["roc_auc", "pr_auc", "precision", "recall", "f0.5", "accuracy"]
        print(f"  {'Mode':<12}" + "".join(f" {c:>8}" for c in cols))
        print(f"  {'-' * 66}")
        for name, m in summary.items():
            vals = "".join(f" {m[k]:>8.3f}" for k in keys)
            print(f"  {name:<12}{vals}")

    print(
        "\nDone! Try --feature_mode human/llm/combined, "
        "or change --n_features / --model."
    )


if __name__ == "__main__":
    asyncio.run(main())
