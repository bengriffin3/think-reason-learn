"""Shared plumbing for the four reasoning-method runners.

`run_{pi,rrf,gptree,rrm}.py` each own their method's prompts and
hyperparameters; everything that is the same across all four lives here:
dataset loading, the train/held-out split convention, the answer-vector →
score combiner, output file shapes, timing estimates and the LLM factory call.

Datasets are addressed the same way as in `run_traditional.py` — either a
`--dataset` preset or explicit `--data / --text-field / --label-field /
--id-field / --split-field`. After loading, every runner sees the same four
columns: ``id``, ``text``, ``label`` (0/1) and ``split``.

Nothing here makes an LLM call. `build_llm()` returns the example's factory
object (`src.llm.get_local_llm`) wrapped in a call counter so black-box fit
stages still report progress.
"""
from __future__ import annotations
import argparse, json, logging, os, sys, time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

# Run against the `think_reason_learn` in this checkout, not whatever version
# happens to be installed. The example ships inside the library repo, and its
# artifacts were produced by the code sitting next to it; a stale install
# silently changes method behaviour and constructor arguments.
REPO_ROOT = EXAMPLE_ROOT.parents[1]
if (REPO_ROOT / "think_reason_learn" / "__init__.py").exists():
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MODEL = "qwen2.5-coder:14b"
OLLAMA_BASE_URL = "http://localhost:11434/v1"

# --smoke sizing. Small enough that all four methods finish in well under an
# hour on a local 14b model, big enough that every stage really runs.
SMOKE_FIT, SMOKE_LIMIT = 12, 5
TIMINGS_PATH = EXAMPLE_ROOT / "precomputed" / "timings.json"

# Used only when precomputed/timings.json has nothing for the method. Measured
# on an M2 Mac against qwen2.5-coder:14b Q4_K_M in August 2026: a short
# structured answer takes about four seconds. Run `calibrate_timings.py` to
# replace this guess with numbers from your own box.
FALLBACK_S_PER_CALL = 4.0

logger = logging.getLogger("runner")


# ---------------------------------------------------------------------------
# Dataset presets
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetSpec:
    """How to find a dataset's rows and what to call its two classes."""
    name: str
    text_field: str
    label_field: str
    id_field: str
    split_field: str | None
    fit_split: str            # split the methods are fitted on
    max_chars: int            # 0 = no truncation
    positive: str             # class word for the positive label
    negative: str
    subject: str              # how prompts refer to one row ("founder summary")
    data_env: str = ""
    default_data: str = ""


DATASETS: dict[str, DatasetSpec] = {
    "vcbench": DatasetSpec(
        name="vcbench",
        text_field="anonymised_prose",
        label_field="success",
        id_field="founder_uuid",
        split_field=None,          # one split: the public 4,500
        fit_split="public",
        max_chars=0,
        positive="successful",
        negative="failed",
        subject="founder summary",
        data_env="VCBENCH_DATA",
        default_data="~/.trl-data/vcbench/vcbench_final_public.csv",
    ),
    "movie": DatasetSpec(
        name="movie",
        text_field="summary",
        label_field="nominated",
        id_field="imdb_id",
        split_field="split",       # synthesised from `year` on the HF fetch
        fit_split="train",
        max_chars=4000,
        positive="nominated",
        negative="not nominated",
        subject="plot summary",
    ),
}

MOVIE_HF_DATASET = "Francis2003/Movie-O-Label"
MOVIE_SPLIT_YEAR = 2013


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def add_common_args(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Dataset selection, run sizing and output arguments shared by all four."""
    d = ap.add_argument_group("dataset")
    d.add_argument("--dataset", choices=sorted(DATASETS), default=None,
                   help="preset that fills in the field names below")
    d.add_argument("--data", type=Path, default=None,
                   help="records file (csv/jsonl); VCBench: $VCBENCH_DATA. "
                        "Movie: omit to fetch from HuggingFace")
    d.add_argument("--text-field", default=None)
    d.add_argument("--label-field", default=None)
    d.add_argument("--id-field", default=None)
    d.add_argument("--split-field", default=None,
                   help="column holding train/test membership; omit for a single split")
    d.add_argument("--train-value", default="train")
    d.add_argument("--test-value", default="test")
    d.add_argument("--max-chars", type=int, default=None,
                   help="truncate each row's text (0 = no truncation)")

    r = ap.add_argument_group("run")
    r.add_argument("--model", default=DEFAULT_MODEL)
    r.add_argument("--concurrency", type=int, default=1,
                   help="Ollama serialises by default; 1 avoids async churn")
    r.add_argument("--fit-size", type=int, default=None,
                   help="stratified subsample of the fit split; 0 = all of it")
    r.add_argument("--limit", type=int, default=0,
                   help="cap rows scored per split (stratified); 0 = all")
    r.add_argument("--predict-split", choices=["both", "held-out"], default="both",
                   help="score every row, or only the held-out split")
    r.add_argument("--smoke", action="store_true",
                   help=f"tiny validation run on its own cache: fit {SMOKE_FIT}, "
                        f"score {SMOKE_LIMIT} rows per split")
    r.add_argument("--out-dir", type=Path, default=None,
                   help="default: <example>/results/<method>_<dataset>[_smoke]")
    r.add_argument("--cache-path", type=Path, default=None,
                   help="default: <out-dir>/llm_cache.jsonl")
    r.add_argument("--seed", type=int, default=42)
    return ap


def resolve_spec(args: argparse.Namespace) -> DatasetSpec:
    """Merge `--dataset` defaults with any explicit field overrides."""
    if args.dataset:
        spec = DATASETS[args.dataset]
    else:
        missing = [n for n in ("text_field", "label_field", "id_field")
                   if getattr(args, n) is None]
        if missing or args.data is None:
            raise SystemExit(
                "Pass --dataset {vcbench,movie}, or --data plus "
                "--text-field/--label-field/--id-field.")
        spec = DatasetSpec(
            name=args.data.stem, text_field="", label_field="", id_field="",
            split_field=None, fit_split=args.train_value, max_chars=0,
            positive="YES", negative="NO", subject="sample")

    over = {}
    for attr in ("text_field", "label_field", "id_field", "split_field", "max_chars"):
        value = getattr(args, attr)
        if value is not None:
            over[attr] = value
    if args.split_field is not None:
        over["fit_split"] = args.train_value
    return spec if not over else DatasetSpec(**{**spec.__dict__, **over})


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _read_any(path: Path) -> pd.DataFrame:
    if str(path).endswith(".jsonl"):
        return pd.DataFrame([json.loads(l) for l in path.read_text().splitlines() if l.strip()])
    return pd.read_csv(path)


def _load_movie_from_hf(max_chars: int) -> pd.DataFrame:
    """The Movie temporal split, rebuilt from HuggingFace.

    Same construction as the run that produced `precomputed/movie_*`: all HF
    splits concatenated, rows with a missing year/summary/label dropped,
    deduplicated on `imdb_id` (2,200 rows → 2,188 films — twelve are listed
    twice), and split on year 2013.
    """
    from datasets import load_dataset

    raw = load_dataset(MOVIE_HF_DATASET)
    df = pd.concat([raw[s].to_pandas() for s in raw], ignore_index=True)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df[df["nominated"].notna() & df["summary"].notna() & df["year"].notna()]
    df = df.drop_duplicates(subset="imdb_id", keep="first").reset_index(drop=True)
    text = df["summary"].astype(str)
    return pd.DataFrame({
        "id": df["imdb_id"].astype(str),
        "text": text.str.slice(0, max_chars) if max_chars else text,
        "label": df["nominated"].astype(int),
        "split": np.where(df["year"] < MOVIE_SPLIT_YEAR, "train", "test"),
    })


def load_frame(args: argparse.Namespace, spec: DatasetSpec) -> pd.DataFrame:
    """Return the dataset as `id, text, label, split`, sorted by id.

    Sorting is deliberate: the fit subsample is drawn from this order, so a
    stable order is what makes two runs on two machines comparable.
    """
    if spec.name == "movie" and args.data is None:
        df = _load_movie_from_hf(spec.max_chars)
    else:
        path = args.data
        if path is None:
            raw = os.environ.get(spec.data_env, "") or spec.default_data
            path = Path(raw).expanduser()
            if not raw or not path.exists():
                raise SystemExit(
                    f"{spec.name} data not found at {path}. Set ${spec.data_env} to the "
                    f"CSV, or pass --data. VCBench is not publicly downloadable — "
                    f"request access at https://vcbench.com.")
        raw_df = _read_any(Path(path).expanduser())
        for field in (spec.text_field, spec.label_field, spec.id_field):
            if field not in raw_df.columns:
                raise SystemExit(
                    f"Column {field!r} not in {path}. Columns: {list(raw_df.columns)}")
        text = raw_df[spec.text_field].fillna("").astype(str)
        df = pd.DataFrame({
            "id": raw_df[spec.id_field].astype(str),
            "text": text.str.slice(0, spec.max_chars) if spec.max_chars else text,
            "label": raw_df[spec.label_field].astype(int),
            "split": (raw_df[spec.split_field].astype(str) if spec.split_field
                      else spec.fit_split),
        })

    df = df.drop_duplicates(subset="id", keep="first")
    return df.sort_values("id", kind="stable").reset_index(drop=True)


def stratified(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """Label-stratified subsample of `n` rows, shuffled."""
    if n <= 0 or n >= len(df):
        return df.sample(frac=1, random_state=seed).reset_index(drop=True)
    rate = float(df["label"].mean())
    n_pos = max(1, int(round(n * rate)))
    pos = df[df["label"] == 1].sample(n=min(n_pos, int(df["label"].sum())), random_state=seed)
    neg = df[df["label"] == 0].sample(
        n=min(n - len(pos), int((df["label"] == 0).sum())), random_state=seed)
    return pd.concat([pos, neg]).sample(frac=1, random_state=seed).reset_index(drop=True)


def split_frames(
    df: pd.DataFrame, args: argparse.Namespace, spec: DatasetSpec,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Return (rows to fit on, {split name: rows to score})."""
    fit_pool = df[df["split"] == spec.fit_split]
    if fit_pool.empty:
        raise SystemExit(
            f"No rows with split=={spec.fit_split!r}. Present: "
            f"{sorted(df['split'].unique())}")
    fit_df = stratified(fit_pool, args.fit_size or 0, args.seed)

    wanted = sorted(df["split"].unique())
    if args.predict_split == "held-out":
        wanted = [s for s in wanted if s != spec.fit_split] or [spec.fit_split]
    scored = {}
    for name in wanted:
        rows = df[df["split"] == name]
        scored[name] = (stratified(rows, args.limit, args.seed + 1) if args.limit
                        else rows.reset_index(drop=True))
    return fit_df, scored


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def combine_answers(
    answers: pd.DataFrame, fit_df: pd.DataFrame, scored: dict[str, pd.DataFrame],
    folds: int = 3, seed: int = 42,
) -> dict[str, np.ndarray]:
    """Collapse an answer-vector matrix into one probability per row.

    PI's policies and RRF's questions both produce a 0/1 vector per row. The
    example combines them with logistic regression rather than a mean vote, so
    a member that fires on everything cannot outweigh one that discriminates.

    Rows the methods were fitted on get out-of-fold scores — no row is ever
    scored by a combiner that saw its own label — and every other row is scored
    by a combiner refit on all of the fit split. Same operator as the shipped
    `precomputed/` files.

    Args:
        answers: 0/1 answers, indexed by row id, one column per member.
        fit_df: the rows the method was fitted on (`id`, `label`).
        scored: rows to score, per split.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict

    def matrix(ids: pd.Series) -> np.ndarray:
        return answers.reindex(ids).fillna(0).to_numpy(dtype=int)

    fit_matrix = matrix(fit_df["id"])
    y_fit = fit_df["label"].to_numpy()

    counts = np.bincount(y_fit, minlength=2)
    usable_folds = int(min(folds, counts.min()))
    oof_by_id: dict[str, float] = {}
    if usable_folds >= 2:
        oof = cross_val_predict(
            LogisticRegression(class_weight="balanced", max_iter=1000),
            fit_matrix, y_fit, cv=usable_folds, method="predict_proba")[:, 1]
        oof_by_id = dict(zip(fit_df["id"], oof))
    else:
        logger.warning(
            "Only %d row(s) of the minority class in the fit split — too few for "
            "out-of-fold scoring, so fitted rows are scored in-sample. Expected on "
            "a --smoke run, a problem on a real one.", counts.min())

    model = LogisticRegression(class_weight="balanced", max_iter=1000).fit(fit_matrix, y_fit)
    out = {}
    for split, frame in scored.items():
        scores = model.predict_proba(matrix(frame["id"]))[:, 1]
        out[split] = np.array([oof_by_id.get(row_id, value)
                               for row_id, value in zip(frame["id"], scores)])
    return out


def score_series(y_true: Sequence[int], y_score: Sequence[float]) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    keep = ~np.isnan(y_score)
    if keep.sum() == 0 or len(set(y_true[keep].tolist())) < 2:
        return {}
    return {
        "roc_auc": round(float(roc_auc_score(y_true[keep], y_score[keep])), 4),
        "pr_auc": round(float(average_precision_score(y_true[keep], y_score[keep])), 4),
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_scores(
    out_dir: Path, prefix: str, split: str, method: str,
    frame: pd.DataFrame, with_label: bool,
) -> Path:
    """Write `id,score[,label]` — the shape the shipped `precomputed/` files use.

    Held-out splits ship without labels, so a student's own run drops in
    alongside `precomputed/` for a direct comparison.
    """
    cols = ["id", "score"] + (["label"] if with_label else [])
    path = out_dir / f"{prefix}_{split}_{method}_scores.csv"
    frame[cols].to_csv(path, index=False)
    return path


def write_metrics(out_dir: Path, payload: dict) -> Path:
    path = out_dir / "metrics.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

class CallCounter:
    """Count and pace-report `respond` calls on a wrapped LLM.

    PI's and RRM's fit stages are single awaits with no progress of their own,
    so without this a multi-hour run prints nothing between "fitting" and
    "done". Everything other than `respond` falls through to the wrapped LLM.
    """

    def __init__(self, llm: Any, log_every: int = 25, expected: int | None = None) -> None:
        self._llm = llm
        self._log_every = log_every
        self.expected = expected
        self.calls = 0
        self.started = time.time()

    async def respond(self, *args: Any, **kwargs: Any) -> Any:
        response = await self._llm.respond(*args, **kwargs)
        self.calls += 1
        if self._log_every and self.calls % self._log_every == 0:
            per_call = (time.time() - self.started) / self.calls
            if self.expected:
                left = max(self.expected - self.calls, 0)
                logger.info("  %d/%d calls  %.2fs/call  ETA %s",
                            self.calls, self.expected, per_call, human_time(per_call * left))
            else:
                logger.info("  %d calls  %.2fs/call", self.calls, per_call)
        return response

    @property
    def s_per_call(self) -> float:
        return (time.time() - self.started) / self.calls if self.calls else 0.0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._llm, name)


def env_guard() -> str:
    """Refuse to run against anything that is not a local endpoint.

    These runners make tens of thousands of calls. Pointed at a paid API by a
    stray environment variable that is an expensive accident, so it is a hard
    error rather than a warning.
    """
    base = os.environ.get("OPENAI_BASE_URL", OLLAMA_BASE_URL)
    if "localhost" not in base and "127.0.0.1" not in base:
        raise SystemExit(
            f"OPENAI_BASE_URL={base!r} is not a local endpoint. These scripts are "
            f"written for a local Ollama server ({OLLAMA_BASE_URL}) and refuse to "
            f"make thousands of calls against a paid API.")
    return base


def build_llm(cache_path: Path | None, *, cache_sampled: bool = False,
              expected_calls: int | None = None) -> CallCounter:
    """The example's LLM, cached on disk, wrapped in the progress counter."""
    from src.llm import get_local_llm

    llm = get_local_llm(cache_path, cache_sampled=cache_sampled)
    return CallCounter(llm, expected=expected_calls)


def choice(model: str) -> Any:
    """One entry for a TRL `llm_priority` list, pointed at the local model."""
    from think_reason_learn.core.llms import OpenAIChoice

    return OpenAIChoice(model=model)


# ---------------------------------------------------------------------------
# Timing + presentation
# ---------------------------------------------------------------------------

def human_time(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def estimate_calls(method: str, n_fit: int, n_rows: int, **params: int) -> int:
    """Roughly how many LLM calls a run will make.

    `n_fit` is the rows fitted on, `n_rows` the distinct rows touched (fit set
    plus everything scored). One place for these formulas so the banner's
    estimate and `calibrate_timings.py`'s projection cannot drift apart.

    * PI      one generation call per 10-row context batch, then one call per
              (row, policy).
    * RRF     one call per (row, question), plus one to write the shortlist.
    * GPTree  the fit answers candidate questions for the rows at each node it
              opens, which is not linear in anything; `n_fit * max_depth` is a
              floor. Scoring is one call per level walked.
    * RRM     reason + extract per fitted row, one compile, then `ensemble_size`
              votes per row — including the fitted rows, which get voted on
              during calibration.
    """
    if method == "pi":
        return n_fit // 10 + n_rows * params.get("n_policies", 10)
    if method == "rrf":
        return n_rows * params.get("n_questions", 14) + params.get("generate", 0)
    if method == "gptree":
        depth = params.get("max_depth", 3)
        return n_fit * depth + 40 + n_rows * depth
    if method == "rrm":
        votes = params.get("ensemble_size", 3)
        return n_fit * 2 + 1 + (n_fit + n_rows) * votes
    raise ValueError(f"Unknown method {method!r}")


def s_per_call(method: str, dataset: str | None = None) -> tuple[float, str]:
    """Seconds per LLM call on this machine: measured if we have it, else a constant."""
    if TIMINGS_PATH.exists():
        try:
            timings = json.loads(TIMINGS_PATH.read_text())
            entry = timings.get("methods", {}).get(method) or {}
            value = entry.get("s_per_call")
            if value:
                return float(value), f"measured, {TIMINGS_PATH.name}"
        except Exception:  # a malformed timings file must not stop a run
            logger.warning("Could not read %s; using the fallback rate", TIMINGS_PATH)
    return FALLBACK_S_PER_CALL, "estimate — run calibrate_timings.py for your box"


def banner(method: str, spec: DatasetSpec, args: argparse.Namespace,
           fit_df: pd.DataFrame, scored: dict[str, pd.DataFrame],
           estimated_calls: int, out_dir: Path, cache_path: Path) -> None:
    rate, source = s_per_call(method, spec.name)
    total = " · ".join(f"{k} {len(v):,}" for k, v in scored.items())
    print(f"""
{'=' * 72}
 {method.upper()} · {spec.name} · {args.model}
{'=' * 72}
 fit on      : {len(fit_df):,} rows ({spec.fit_split}, base rate {fit_df['label'].mean():.3f})
 scoring     : {total}
 est. calls  : ~{estimated_calls:,}
 est. time   : ~{human_time(estimated_calls * rate)}  ({rate:.2f}s/call — {source})
 out-dir     : {out_dir}
 cache       : {cache_path}
 restartable : yes — rerun the identical command and finished calls replay
               from the cache for free. Kill it whenever you like.
{'=' * 72}
""", flush=True)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO, stream=sys.stdout,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s", force=True)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def prepare_run(method: str, args: argparse.Namespace) -> tuple[
        DatasetSpec, pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], Path, Path]:
    """Everything every runner does before it touches the model."""
    setup_logging()
    env_guard()
    if args.smoke:
        args.fit_size = args.fit_size or SMOKE_FIT
        args.limit = args.limit or SMOKE_LIMIT
    spec = resolve_spec(args)
    df = load_frame(args, spec)
    fit_df, scored = split_frames(df, args, spec)

    out_dir = args.out_dir or (
        EXAMPLE_ROOT / "results" / f"{method}_{spec.name}{'_smoke' if args.smoke else ''}")
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.cache_path or (out_dir / "llm_cache.jsonl")
    return spec, df, fit_df, scored, out_dir, cache_path


def base_metrics(method: str, spec: DatasetSpec, args: argparse.Namespace,
                 fit_df: pd.DataFrame, started: str, t0: float,
                 counter: CallCounter) -> dict:
    return {
        "method": method,
        "dataset": spec.name,
        "model": args.model,
        "n_fit": int(len(fit_df)),
        "llm_calls": counter.calls,
        "s_per_call": round(counter.s_per_call, 3),
        "wall_clock_s": int(time.time() - t0),
        "started": started,
        "ended": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "splits": {},
    }


def finish(out_dir: Path, metrics: dict, written: list[Path], t0: float) -> dict:
    write_metrics(out_dir, metrics)
    print("\nWrote:")
    for path in written + [out_dir / "metrics.json"]:
        print(f"  {path}")
    for split, values in metrics["splits"].items():
        if values.get("roc_auc") is not None:
            print(f"  {split:8s} ROC-AUC {values['roc_auc']:.4f}  PR-AUC {values['pr_auc']:.4f}"
                  f"  (n={values['n']:,})")
    print(f"Total wall-clock: {human_time(time.time() - t0)}  "
          f"({metrics['llm_calls']:,} LLM calls)")
    return metrics
