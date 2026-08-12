"""Draw the example's hero figure: wall-clock cost vs held-out ROC-AUC.

One point per reasoning method plus the traditional ensemble, the reasoning
ensemble and the combined ensemble, on both datasets. Reads
``precomputed/*_scores.csv`` and ``precomputed/timings.json`` and nothing else —
no LLM, no network, no raw dataset — so anyone with a checkout can regenerate
``figures/time_vs_score.{png,svg}`` exactly.

The two panels are not the same kind of number and say so in their titles:

* **Movie** x-values are the measured ``projected_full_run_s`` from
  ``timings.json`` (idle-box calibration, 2026-08-12).
* **VCBench** was never timed. Its x-values multiply the Movie-measured
  ``s_per_call`` by the call count a full VCBench public run makes
  (``_runner_common.estimate_calls`` at each runner's defaults) — projections,
  labelled as such.

Ensemble x-values are the sum of their members' runtimes: the reasoning
ensemble needs all four method runs, the combined ensemble those plus the
traditional baseline. The traditional baseline makes no LLM calls and finishes
in minutes on either dataset; both panels draw it at ~4 minutes.

Every y-value is recomputed here from the score CSVs and asserted against the
reference run at 4 dp, so the figure cannot silently drift from the README
table or the notebooks.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

import _runner_common as rc

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
PRECOMPUTED = EXAMPLE_ROOT / "precomputed"
FIGURES = EXAMPLE_ROOT / "figures"

# Notebook palette (01/04): colour is emphasis, not identity — identity is
# carried by the direct labels, the combined ensemble by the dark blue.
PEER, HIGHLIGHT = "#86b6ef", "#2a78d6"
GRID, AXIS, TICK, VALUE, TITLE = "#e1e0d9", "#c3c2b7", "#898781", "#52514e", "#0b0b0b"

METHODS = {"PI": "pi", "RRF": "rrf", "GPTree": "gptree", "RRM": "rrm"}
TRADITIONAL_S = 250.0  # embedding + sklearn, no LLM; minutes on either dataset

# The reference run's ROC-AUCs (the README table). Any recomputed value that
# does not match at 4 dp aborts the figure rather than drawing it.
REFERENCE_ROC = {
    "vcbench_public": {"PI": 0.6763, "RRF": 0.6604, "GPTree": 0.6161, "RRM": 0.6665,
                       "Reasoning ens": 0.7188, "Traditional ens": 0.7382, "Combined": 0.7610},
    "movie_test": {"PI": 0.6346, "RRF": 0.6406, "GPTree": 0.5728, "RRM": 0.5787,
                   "Reasoning ens": 0.6515, "Traditional ens": 0.6354, "Combined": 0.6704},
}

# Call counts for a full VCBench public run (fit and score all 4,500) at the
# runners' VCBench defaults: PI 10 policies, RRF's 16-question shortlist,
# GPTree depth 3, RRM 3-vote ensemble.
VCBENCH_CALL_PARAMS = {"pi": {"n_policies": 10}, "rrf": {"n_questions": 16},
                       "gptree": {"max_depth": 3}, "rrm": {"ensemble_size": 3}}
VCBENCH_N = 4_500


def load(prefix: str, series: str) -> pd.DataFrame:
    return pd.read_csv(PRECOMPUTED / f"{prefix}_{series}_scores.csv").set_index("id")


def roc_points(prefix: str) -> dict[str, float]:
    """Recompute every series' ROC-AUC from the shipped CSVs, exactly as the notebooks do."""
    traditional = load(prefix, "traditional")
    y = traditional["label"]
    wide = pd.DataFrame(
        {name: load(prefix, key)["score"].reindex(y.index) for name, key in METHODS.items()}
    )
    reasoning = cast(pd.Series, wide.rank(pct=True).mean(axis=1))
    trad = cast(pd.Series, traditional["score"].reindex(y.index))
    series: dict[str, pd.Series] = {name: cast(pd.Series, wide[name]) for name in METHODS}
    series["Reasoning ens"] = reasoning
    series["Traditional ens"] = trad
    series["Combined"] = trad.rank(pct=True) + reasoning.rank(pct=True)
    rocs = {name: round(float(roc_auc_score(y, s)), 4) for name, s in series.items()}
    for name, expected in REFERENCE_ROC[prefix].items():
        assert rocs[name] == expected, (prefix, name, rocs[name], expected)
    return rocs


def method_hours(timings: dict) -> tuple[dict[str, float], dict[str, float]]:
    """(Movie measured, VCBench projected) full-run hours per method."""
    movie = {m: timings["methods"][key]["projected_full_run_s"] / 3600
             for m, key in METHODS.items()}
    vcbench = {}
    for m, key in METHODS.items():
        calls = rc.estimate_calls(key, VCBENCH_N, VCBENCH_N, **VCBENCH_CALL_PARAMS[key])
        vcbench[m] = calls * timings["methods"][key]["s_per_call"] / 3600
    return movie, vcbench


def panel_points(rocs: dict[str, float], hours: dict[str, float]) -> dict[str, tuple[float, float]]:
    reasoning_h = sum(hours.values())
    trad_h = TRADITIONAL_S / 3600
    points = {m: (hours[m], rocs[m]) for m in METHODS}
    points["Traditional ens"] = (trad_h, rocs["Traditional ens"])
    points["Reasoning ens"] = (reasoning_h, rocs["Reasoning ens"])
    points["Combined"] = (reasoning_h + trad_h, rocs["Combined"])
    return points


# Per-point label placement: (dx, dy) in axis-fraction offsets, anchor.
LABEL_OFFSETS = {
    "Movie test": {
        "Traditional ens": (0.0, 0.05, "left"),
        "GPTree": (0.0, -0.055, "center"),
        "RRM": (0.02, 0.03, "left"),
        "PI": (0.0, -0.055, "center"),
        "RRF": (0.0, 0.035, "center"),
        "Reasoning ens": (-0.025, -0.005, "right"),
        "Combined": (0.0, 0.05, "center"),
    },
    "VCBench public": {
        "Traditional ens": (0.015, 0.035, "left"),
        "GPTree": (0.0, -0.055, "center"),
        "RRM": (0.02, -0.05, "left"),
        "PI": (0.0, 0.035, "center"),
        "RRF": (0.0, 0.035, "center"),
        "Reasoning ens": (-0.02, 0.03, "right"),
        "Combined": (-0.02, 0.02, "right"),
    },
}


def draw_panel(ax: Axes, name: str, points: dict[str, tuple[float, float]],
               title: str, xlabel: str) -> None:
    for label, (x, y) in points.items():
        emphasised = label == "Combined"
        ax.scatter(x, y, s=130 if emphasised else 80,
                   color=HIGHLIGHT if emphasised else PEER, zorder=3)
        dx, dy, ha = LABEL_OFFSETS[name][label]
        xspan = ax.get_xlim()[1] - ax.get_xlim()[0]
        yspan = ax.get_ylim()[1] - ax.get_ylim()[0]
        ax.annotate(f"{label}\n{y:.4f}", (x, y), (x + dx * xspan, y + dy * yspan),
                    ha=ha, va="center", fontsize=8.5, color=VALUE, linespacing=1.25,
                    fontweight="bold" if emphasised else "normal")
    ax.axhline(0.5, color=TICK, linewidth=1, linestyle="--")
    ax.text(ax.get_xlim()[1] * 0.99, 0.503, "chance", ha="right", fontsize=8, color=TICK)
    ax.set_title(title, fontsize=11, color=TITLE, pad=10)
    ax.set_xlabel(xlabel, fontsize=9.5, color=VALUE)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS)
    ax.tick_params(colors=TICK, labelsize=9)


def main() -> None:
    plt.rcParams["svg.hashsalt"] = "trl-example"  # deterministic SVG output
    timings = json.loads((PRECOMPUTED / "timings.json").read_text())
    movie_h, vcbench_h = method_hours(timings)
    panels = {
        "Movie test": (panel_points(roc_points("movie_test"), movie_h),
                       "Movie test (n=727) — measured timings",
                       "wall-clock hours, measured"),
        "VCBench public": (panel_points(roc_points("vcbench_public"), vcbench_h),
                           "VCBench public (n=4,500) — projected timings",
                           "wall-clock hours, projected from measured s/call"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), sharey=True)
    for ax, (name, (points, title, xlabel)) in zip(axes, panels.items()):
        xmax = max(x for x, _ in points.values())
        ax.set_xlim(-0.04 * xmax, 1.12 * xmax)
        ax.set_ylim(0.47, 0.80)
        draw_panel(ax, name, points, title, xlabel)
    axes[0].set_ylabel("held-out ROC-AUC", fontsize=9.5, color=VALUE)

    fig.suptitle("What an extra hour of local compute buys", fontsize=13.5,
                 color=TITLE, y=0.99)
    caption = (
        f"qwen2.5-coder:14b via Ollama on {timings['machine']} · calibrated "
        "2026-08-12 on an idle box (planning-grade estimates, YMMV) · ensemble "
        "x = sum of member runtimes · the traditional baseline makes no LLM "
        "calls and is drawn at its ~4-minute runtime · VCBench hours were never "
        "measured: they are Movie-measured s/call × VCBench call counts"
    )
    fig.text(0.5, 0.012, caption, ha="center", fontsize=7.5, color=TICK, wrap=True)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))

    FIGURES.mkdir(exist_ok=True)
    for suffix in ("png", "svg"):
        out = FIGURES / f"time_vs_score.{suffix}"
        # Date: None keeps the SVG byte-identical between runs.
        fig.savefig(out, dpi=200, metadata=None if suffix == "png" else {"Date": None})
        print(f"wrote {out.relative_to(EXAMPLE_ROOT)}")


if __name__ == "__main__":
    main()
