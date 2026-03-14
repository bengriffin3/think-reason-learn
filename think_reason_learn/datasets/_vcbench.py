"""VCBench dataset loading and helper functions.

Canonical implementations of the VCBench-specific parsing helpers, data
loading, and schema definition.  Used by both ``examples/`` scripts and
``experiments/`` pipelines.
"""

from __future__ import annotations

import ast
import json
import sys
from typing import Any

import numpy as np
import pandas as pd

from think_reason_learn.features import DataSchema, HelperFunction, Rule

# ---------------------------------------------------------------------------
# VCBench helper functions
# ---------------------------------------------------------------------------
# These small helpers parse VCBench-specific string encodings.  They are
# available inside LLM-generated lambda expressions at evaluation time.


def parse_qs(qs_str: str) -> float:
    """Convert QS university ranking string to a float.

    Examples: "1" -> 1.0, "200+" -> 250.0, "" -> 999.0 (unranked).
    """
    if not qs_str or qs_str == "":
        return 999.0
    if qs_str == "200+":
        return 250.0
    try:
        return float(qs_str)
    except (ValueError, TypeError):
        return 999.0


def parse_duration(dur_str: str) -> float:
    """Convert job duration bucket to approximate years."""
    mapping = {
        "<2": 1.0,
        "2-3": 2.5,
        "4-5": 4.5,
        "6-9": 7.5,
        ">9": 12.0,
    }
    return mapping.get(dur_str, 0.0)


def parse_company_size(size_str: str) -> int:
    """Convert company size label to approximate employee count."""
    if not size_str:
        return 0
    s = size_str.lower().strip()
    mapping = {
        "self-employed": 1,
        "1-10 employees": 6,
        "11-50 employees": 30,
        "51-200 employees": 125,
        "201-500 employees": 350,
        "501-1000 employees": 750,
        "1001-5000 employees": 3000,
        "5001-10000 employees": 7500,
        "10001+ employees": 15000,
    }
    return mapping.get(s, 0)


# ---------------------------------------------------------------------------
# Registered helpers (for FeatureGenerator / FeatureEvaluator)
# ---------------------------------------------------------------------------

VCBENCH_HELPERS: list[HelperFunction] = [
    HelperFunction(
        name="parse_qs",
        func=parse_qs,
        signature="parse_qs(qs_str: str) -> float",
        docstring='QS ranking string to float. "1"->1.0, "200+"->250.0, ""->999.0',
    ),
    HelperFunction(
        name="parse_duration",
        func=parse_duration,
        signature="parse_duration(dur_str: str) -> float",
        docstring="Job duration bucket to years. '<2'->1.0, '2-3'->2.5, etc.",
    ),
    HelperFunction(
        name="parse_company_size",
        func=parse_company_size,
        signature="parse_company_size(size_str: str) -> int",
        docstring="Company size label to employee count.",
    ),
]

# ---------------------------------------------------------------------------
# Data schema for the LLM prompt
# ---------------------------------------------------------------------------

VCBENCH_SCHEMA = DataSchema(
    description=(
        "A startup founder profile with education, work history, and exit events."
    ),
    schema_text="""\
{
    "industry": str,                          # startup industry
    "educations": [                           # list of degrees
        {"degree": str, "field": str, "qs_ranking": str}
    ],
    "jobs": [                                 # prior work experience
        {"role": str, "company_size": str, "industry": str, "duration": str}
    ],
    "ipos": [dict],                           # IPO events (may be empty)
    "acquisitions": [dict],                   # acquisition events (may be empty)
}""",
    param_name="founder",
    example_rules=[
        Rule(
            name="has_phd",
            description="Founder holds a PhD or doctorate",
            expression=(
                "lambda founder: any('phd' in e.get('degree', '').lower() "
                "or 'doctor' in e.get('degree', '').lower() "
                "for e in founder.get('educations', []))"
            ),
        ),
        Rule(
            name="prior_ipo",
            description="Founder was involved in a prior IPO",
            expression="lambda founder: len(founder.get('ipos', [])) > 0",
        ),
    ],
)

# ---------------------------------------------------------------------------
# Required columns
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = {
    "industry",
    "educations_json",
    "jobs_json",
    "success",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _safe_json_parse(value: Any) -> list[dict]:
    """Parse a JSON-encoded string column into a list of dicts."""
    if pd.isna(value) or value == "":
        return []
    try:
        result = json.loads(value)
        return result if isinstance(result, list) else [result]
    except (json.JSONDecodeError, TypeError):
        try:
            result = ast.literal_eval(str(value))
            return result if isinstance(result, list) else [result]
        except (ValueError, SyntaxError):
            return []


def load_vcbench(
    csv_path: str,
    label_column: str = "success",
    max_rows: int = 0,
    random_state: int = 42,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Load VCBench CSV and convert rows to structured founder dicts.

    Args:
        csv_path: Path to the VCBench training CSV.
        label_column: Binary label column name.
        max_rows: Maximum rows to load (0 = all data).
        random_state: Random seed for stratified subsampling.

    Returns:
        Tuple of (records, labels) where each record is a dict suitable
        for FeatureGenerator / FeatureEvaluator.
    """
    df = pd.read_csv(csv_path)

    # Validate required columns.
    missing = REQUIRED_COLUMNS - set(df.columns)
    if label_column != "success":
        missing.discard("success")
        if label_column not in df.columns:
            missing.add(label_column)
    if missing:
        sys.exit(
            "ERROR: Missing required columns in CSV:"
            f" {sorted(missing)}\n"
            f"Available columns: {sorted(df.columns)}"
        )

    df = df.dropna(subset=[label_column]).copy()
    df[label_column] = df[label_column].astype(int)

    # Stratified subsample so the rare positive class (~9%) is represented.
    if max_rows and len(df) > max_rows:
        from sklearn.model_selection import train_test_split

        df, _ = train_test_split(  # type: ignore[assignment]
            df,
            train_size=max_rows,
            stratify=df[label_column],
            random_state=random_state,
        )
        df = df.reset_index(drop=True)  # type: ignore[union-attr]

    # Convert each row into a structured dict.
    records: list[dict[str, Any]] = []
    has_prose = "anonymised_prose" in df.columns
    for _, row in df.iterrows():
        rec: dict[str, Any] = {
            "industry": row.get("industry", "") or "",
            "educations": _safe_json_parse(row.get("educations_json", "")),
            "jobs": _safe_json_parse(row.get("jobs_json", "")),
            "ipos": _safe_json_parse(row.get("ipos", "")),
            "acquisitions": _safe_json_parse(row.get("acquisitions", "")),
        }
        if has_prose:
            rec["anonymised_prose"] = row.get("anonymised_prose", "") or ""
        records.append(rec)

    labels: np.ndarray = df[label_column].values  # type: ignore[assignment]
    return records, labels
