r"""Minimal example: LLM-reasoned features on VCBench.

Unlike lambda features (where the LLM writes code that runs locally),
**reasoned features** ask the LLM to read each founder's profile and
make a subjective judgment — e.g., "Does this founder have strong
domain expertise?"  Each founder requires an LLM call, so this is
slower and more expensive than lambda features.

Usage::

    # Default: 100 founders, 3 questions (~300 LLM calls, < $0.02)
    python examples/vcbench_reasoned_features.py \\
        --input_csv /path/to/vcbench_final_public.csv

    # Quick test with fewer founders
    python examples/vcbench_reasoned_features.py \\
        --input_csv /path/to/vcbench_final_public.csv --max_rows 30

Requires OPENAI_API_KEY in environment or .env file.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel
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

from think_reason_learn.core.llms import LLMChoice, OpenAIChoice, llm
from think_reason_learn.datasets import load_vcbench

# ---------------------------------------------------------------------------
# Reasoned questions
# ---------------------------------------------------------------------------
# These are subjective yes/no questions that require the LLM to read the
# founder's full profile and make a holistic judgment.  Unlike structured
# features (e.g., "has_phd"), these cannot be extracted by parsing fields —
# the LLM must *reason* about the founder's background.

QUESTIONS: list[dict[str, str]] = [
    {
        "name": "domain_expertise",
        "question": (
            "Based on this founder's education and work history, do they "
            "demonstrate strong domain expertise that is directly relevant "
            "to their startup's industry?"
        ),
    },
    {
        "name": "serial_entrepreneur",
        "question": (
            "Does this founder's career trajectory suggest they are a "
            "serial entrepreneur — i.e., they have founded or co-founded "
            "multiple ventures?"
        ),
    },
    {
        "name": "strong_network",
        "question": (
            "Based on this founder's experience at large companies, senior "
            "roles, and diverse industries, would you expect them to have "
            "a strong professional network?"
        ),
    },
]


# ---------------------------------------------------------------------------
# LLM answering
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a VC analyst evaluating startup founders. Given an "
    "anonymised founder summary, answer the question with YES or NO. "
    "Base your answer only on what is stated in the summary."
)


class YesNo(BaseModel):
    """Structured output for yes/no answers."""

    answer: Literal["YES", "NO"]


async def _ask_one(
    question: str,
    prose: str,
    llm_priority: list[LLMChoice],
    semaphore: asyncio.Semaphore,
) -> int:
    """Ask one question about one founder. Returns 1 (YES) or 0 (NO)."""
    query = f"**Question:** {question}\n\n" f"**Founder summary:**\n{prose}"
    async with semaphore:
        response = await llm.respond(
            query=query,
            llm_priority=llm_priority,
            response_format=YesNo,
            instructions=SYSTEM_PROMPT,
            temperature=0.0,
        )
    if response.response is not None:
        return 1 if response.response.answer == "YES" else 0
    return 0


async def _answer_question(
    question_name: str,
    question_text: str,
    prose_list: list[str],
    llm_priority: list[LLMChoice],
    concurrency: int,
) -> list[int]:
    """Answer one question for all founders concurrently.

    Returns a list of 0/1 answers, one per founder.
    """
    semaphore = asyncio.Semaphore(concurrency)
    answers = [0] * len(prose_list)
    completed = 0
    total = len(prose_list)
    t0 = time.time()

    async def worker(idx: int, prose: str) -> None:
        nonlocal completed
        answers[idx] = await _ask_one(
            question_text,
            prose,
            llm_priority,
            semaphore,
        )
        completed += 1
        if completed % 20 == 0 or completed == total:
            elapsed = time.time() - t0
            rate = completed / elapsed if elapsed > 0 else 0
            print(f"    {question_name}: {completed}/{total} " f"({rate:.1f}/s)")

    async with asyncio.TaskGroup() as tg:
        for i, prose in enumerate(prose_list):
            tg.create_task(worker(i, prose))

    return answers


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """Build CLI argument parser."""
    p = argparse.ArgumentParser(
        description="VCBench: LLM-reasoned features example.",
    )
    p.add_argument("--input_csv", required=True, help="VCBench CSV path.")
    p.add_argument("--label_column", default="success")
    p.add_argument(
        "--max_rows", type=int, default=100, help="Founders to evaluate (default: 100)."
    )
    p.add_argument("--test_size", type=float, default=0.40)
    p.add_argument("--model", default="gpt-4.1-nano")
    p.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Max concurrent LLM calls (default: 10).",
    )
    p.add_argument("--random_state", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    """Run the reasoned features example."""
    args = _parse_args()
    rs = args.random_state
    llm_priority: list[LLMChoice] = [OpenAIChoice(model=args.model)]
    n_questions = len(QUESTIONS)

    print(f"\n{'=' * 60}")
    print("  VCBench Reasoned Features Example")
    print(f"{'=' * 60}\n")

    # 1. Load data
    records, labels = load_vcbench(
        args.input_csv,
        args.label_column,
        args.max_rows,
        rs,
    )

    # Check that prose is available.
    prose_list = [r.get("anonymised_prose", "") for r in records]
    if not any(prose_list):
        print("ERROR: No 'anonymised_prose' column found in CSV.")
        print("This script requires founder text descriptions.")
        return

    # 2. Train/test split
    idx = np.arange(len(records))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=args.test_size,
        stratify=labels,
        random_state=rs,
    )
    y_train, y_test = labels[train_idx], labels[test_idx]

    n_pos_test = int(y_test.sum())
    print(
        f"  {len(records)} founders -> "
        f"{len(train_idx)} train / {len(test_idx)} test"
    )
    print(f"  Test positives: {n_pos_test}")
    if n_pos_test < 5:
        print("  WARNING: very few test positives — metrics may be noisy")

    # 3. Answer each question for ALL founders (train + test)
    # Note: unlike lambda features, we answer for all founders because
    # the questions are fixed (not data-dependent), so there's no leakage.
    total_calls = n_questions * len(records)
    print(
        f"\n  Asking {n_questions} questions x {len(records)} founders "
        f"= {total_calls} LLM calls"
    )
    print(f"  Model: {args.model}, concurrency: {args.concurrency}")
    if total_calls > 500:
        print(
            f"  NOTE: {total_calls} calls may take several minutes and"
            " cost $0.50+ depending on model."
        )
        print("  Consider --max_rows 100 for a quick test first.")
    print()

    feature_matrix: dict[str, list[int]] = {}
    t_start = time.time()

    for q in QUESTIONS:
        print(f"  Q: {q['name']} — \"{q['question'][:60]}...\"")
        answers = await _answer_question(
            q["name"],
            q["question"],
            prose_list,
            llm_priority,
            args.concurrency,
        )
        feature_matrix[q["name"]] = answers
        yes_count = sum(answers)
        print(f"    -> {yes_count} YES, {len(answers) - yes_count} NO\n")

    elapsed = time.time() - t_start
    print(
        f"  Total answering time: {elapsed:.1f}s "
        f"({n_questions * len(records) / elapsed:.1f} calls/s)"
    )

    # 4. Build feature matrix and split
    df_features = pd.DataFrame(feature_matrix)
    X_train = df_features.iloc[train_idx].values.astype(float)
    X_test = df_features.iloc[test_idx].values.astype(float)
    feature_names = list(feature_matrix.keys())

    # 5. Train and evaluate
    clf = LogisticRegression(max_iter=1000, random_state=rs)
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:, 1]

    # F0.5-optimal threshold on training set
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

    # Coefficients
    coefs = sorted(
        zip(feature_names, clf.coef_[0]), key=lambda x: abs(x[1]), reverse=True
    )
    print(f"\n{'=' * 60}")
    print(f"  Feature Coefficients (threshold={best_t:.2f})")
    print(f"{'=' * 60}")
    for name, c in coefs:
        print(f"    {'+' if c >= 0 else '-'}{abs(c):.3f}  {name}")

    # Metrics
    roc = roc_auc_score(y_test, y_prob)
    pr = average_precision_score(y_test, y_prob)
    prec = precision_score(y_test, y_pred, zero_division=0.0)  # type: ignore[arg-type]
    rec = recall_score(y_test, y_pred, zero_division=0.0)  # type: ignore[arg-type]
    f05 = fbeta_score(y_test, y_pred, beta=0.5, zero_division=0.0)  # type: ignore[arg-type]
    acc = accuracy_score(y_test, y_pred)

    print(f"\n{'=' * 60}")
    print("  Test Metrics")
    print(f"{'=' * 60}")
    print(
        f"  ROC-AUC={roc:.3f}  PR-AUC={pr:.3f}  "
        f"Prec={prec:.3f}  Rec={rec:.3f}  F0.5={f05:.3f}  Acc={acc:.3f}"
    )

    # Cost note
    print(
        f"\n  LLM calls: {total_calls} | "
        f"Time: {elapsed:.0f}s | "
        f"Model: {args.model}"
    )
    print(
        "  Tip: increase --max_rows for more reliable metrics, "
        "or --concurrency for speed."
    )
    print()


if __name__ == "__main__":
    asyncio.run(main())
