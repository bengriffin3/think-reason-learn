"""Traditional-ML baseline for the reasoning-ML example.

Runs a 5-model sklearn suite (LR / HistGB / RF / ExtraTrees / GaussianNB) on
[len(text) + all-MiniLM-L6-v2 embedding + TF-IDF(top-400) + one-hot(structured)]
with proper train-fit / test-transform discipline. Reports per-model + rank-avg
ensemble ROC-AUC + PR-AUC.

Dataset-agnostic: pass any records file (CSV or JSONL) with a text field and a
label column. Three invocation modes:

  # A. Split by a column value (Movie-style records.jsonl with a "split" field)
  python run_traditional.py \\
    --input path/to/records.jsonl --split-field split \\
    --text-fields description --label-col label \\
    --out-dir output/

  # B. Separate train + test files (VCBench-style CSVs)
  python run_traditional.py \\
    --train path/to/train.csv --test path/to/test.csv \\
    --text-fields anonymised_prose --label-col success \\
    --struct-fields industry \\
    --out-dir output/

  # C. No test set → k-fold OOF on train (what students without private access do)
  python run_traditional.py \\
    --train path/to/public.csv --kfold 3 \\
    --text-fields anonymised_prose --label-col success \\
    --out-dir output/

Outputs (under --out-dir):
  metrics.json                — per-model + ensemble AUC + PR-AUC, timings, feature dims
  predictions.csv             — per-row scores from every model + ensemble
"""
from __future__ import annotations
import argparse
import json
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Imports deliberately after the warnings filter, so sklearn's import-time
# warnings are suppressed too.
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import (  # noqa: E402
    ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier,
)
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.naive_bayes import GaussianNB  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--input", type=Path, help="single records file (csv/jsonl); pair with --split-field")
    g.add_argument("--train", type=Path, help="training records file (csv/jsonl)")
    ap.add_argument("--test", type=Path, default=None, help="test records file (with --train)")
    ap.add_argument("--split-field", type=str, default=None, help="column to split on (with --input)")
    ap.add_argument("--train-value", type=str, default="train")
    ap.add_argument("--test-value", type=str, default="test")
    ap.add_argument("--label-col", type=str, required=True)
    ap.add_argument("--text-fields", type=str, required=True, help="comma-separated text column names")
    ap.add_argument("--struct-fields", type=str, default="", help="comma-separated structured columns")
    ap.add_argument("--id-col", type=str, default=None, help="row id column (default: first)")
    ap.add_argument("--kfold", type=int, default=0, help="if >0 and no test set, run k-fold OOF on train")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--embed-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--tfidf-max-features", type=int, default=400)
    ap.add_argument("--out-dir", type=Path, required=True)
    return ap.parse_args()


def load_records(path: Path) -> pd.DataFrame:
    if str(path).endswith(".jsonl"):
        return pd.DataFrame([json.loads(line) for line in path.read_text().splitlines() if line.strip()])
    return pd.read_csv(path)


def concat_text(df: pd.DataFrame, fields: list[str]) -> pd.Series:
    text = df[fields[0]].fillna("").astype(str)
    for f in fields[1:]:
        text = text.str.cat(df[f].fillna("").astype(str), sep=" . ")
    return text


def build_features(
    df_train: pd.DataFrame, df_test: pd.DataFrame | None, text_fields: list[str],
    struct_fields: list[str], embed_model_name: str, tfidf_max_features: int,
) -> tuple[np.ndarray, np.ndarray | None, dict]:
    """Return (X_train, X_test|None, feature_info). Fits transformers on train only."""
    from sentence_transformers import SentenceTransformer
    emb_model = SentenceTransformer(embed_model_name)

    text_train = concat_text(df_train, text_fields)
    text_test = concat_text(df_test, text_fields) if df_test is not None else None

    len_train = text_train.str.len().to_numpy().reshape(-1, 1)
    len_test = text_test.str.len().to_numpy().reshape(-1, 1) if text_test is not None else None

    emb_train = emb_model.encode(text_train.tolist(), batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    emb_test = emb_model.encode(text_test.tolist(), batch_size=128, normalize_embeddings=True, show_progress_bar=False) if text_test is not None else None

    tfidf = TfidfVectorizer(max_features=tfidf_max_features, stop_words="english")
    tf_train = tfidf.fit_transform(text_train).toarray()
    tf_test = tfidf.transform(text_test).toarray() if text_test is not None else None

    struct_train_arr, struct_test_arr = None, None
    if struct_fields:
        train_parts, test_parts = [], []
        for f in struct_fields:
            col_train = df_train[f]
            col_test = df_test[f] if df_test is not None else None
            if col_train.dtype == object:
                combined = pd.concat([col_train, col_test]) if col_test is not None else col_train
                dummies = pd.get_dummies(combined.astype(str), prefix=f).astype(float)
                train_parts.append(dummies.iloc[:len(col_train)].reset_index(drop=True))
                if col_test is not None:
                    test_parts.append(dummies.iloc[len(col_train):].reset_index(drop=True))
            else:
                train_parts.append(pd.DataFrame({f: pd.to_numeric(col_train, errors="coerce")}))
                if col_test is not None:
                    test_parts.append(pd.DataFrame({f: pd.to_numeric(col_test, errors="coerce")}))
        struct_train_arr = pd.concat(train_parts, axis=1).to_numpy(float)
        if test_parts:
            struct_test_arr = pd.concat(test_parts, axis=1).to_numpy(float)

    train_stack = [len_train, emb_train, tf_train]
    test_stack = [len_test, emb_test, tf_test] if df_test is not None else None
    if struct_train_arr is not None:
        train_stack.append(struct_train_arr)
        if test_stack is not None:
            test_stack.append(struct_test_arr)

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(np.concatenate(train_stack, axis=1))
    X_test = imputer.transform(np.concatenate(test_stack, axis=1)) if test_stack is not None else None

    info = {
        "n_features_total": int(X_train.shape[1]),
        "n_features_text_len": 1,
        "n_features_embedding": int(emb_train.shape[1]),
        "n_features_tfidf": int(tf_train.shape[1]),
        "n_features_structured": int(struct_train_arr.shape[1]) if struct_train_arr is not None else 0,
    }
    return X_train, X_test, info


def make_models(seed: int) -> dict:
    return {
        "lr": lambda: make_pipeline(
            StandardScaler(with_mean=False),
            LogisticRegression(max_iter=1500, class_weight="balanced", random_state=seed),
        ),
        "hgb": lambda: HistGradientBoostingClassifier(class_weight="balanced", random_state=seed),
        "rf":  lambda: RandomForestClassifier(250, class_weight="balanced", random_state=seed, n_jobs=-1),
        "et":  lambda: ExtraTreesClassifier(250, class_weight="balanced", random_state=seed, n_jobs=-1),
        "gnb": lambda: make_pipeline(StandardScaler(with_mean=False), GaussianNB()),
    }


def rank01(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank(pct=True).to_numpy()


def score(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    return {
        "roc_auc": round(float(roc_auc_score(y_true, y_score)), 4),
        "pr_auc":  round(float(average_precision_score(y_true, y_score)), 4),
    }


def run_train_test(X_train, y_train, X_test, y_test, seed: int) -> tuple[dict, pd.DataFrame]:
    per_model_metrics, per_model_scores = {}, {}
    for name, factory in make_models(seed).items():
        t0 = time.time()
        model = factory()
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        per_model_scores[name] = probs
        per_model_metrics[name] = {**score(y_test, probs), "fit_predict_s": round(time.time() - t0, 2)}
        print(f"  {name:4s}  ROC-AUC {per_model_metrics[name]['roc_auc']:.4f}  "
              f"PR-AUC {per_model_metrics[name]['pr_auc']:.4f}  ({per_model_metrics[name]['fit_predict_s']:.1f}s)")

    ens = np.mean([rank01(v) for v in per_model_scores.values()], axis=0)
    ens_metrics = score(y_test, ens)
    print(f"  ENS   ROC-AUC {ens_metrics['roc_auc']:.4f}  PR-AUC {ens_metrics['pr_auc']:.4f}  (rank-avg of {len(per_model_scores)})")

    best_name = max(per_model_metrics, key=lambda k: per_model_metrics[k]["roc_auc"])
    best_metrics = {**per_model_metrics[best_name], "model": best_name}

    preds_df = pd.DataFrame(per_model_scores)
    preds_df["ensemble"] = ens
    return {
        "per_model": per_model_metrics,
        "best_single": best_metrics,
        "ensemble": ens_metrics,
    }, preds_df


def run_kfold(X: np.ndarray, y: np.ndarray, k: int, seed: int) -> tuple[dict, pd.DataFrame]:
    per_model_scores = {name: np.zeros(len(y)) for name in make_models(seed)}
    for fold, (tr, te) in enumerate(StratifiedKFold(k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)):
        print(f"  fold {fold + 1}/{k}: {tr.size} train / {te.size} test")
        for name, factory in make_models(seed).items():
            model = factory()
            model.fit(X[tr], y[tr])
            per_model_scores[name][te] = model.predict_proba(X[te])[:, 1]

    per_model_metrics, ens_input = {}, []
    for name, probs in per_model_scores.items():
        per_model_metrics[name] = score(y, probs)
        ens_input.append(rank01(probs))
        print(f"  {name:4s}  OOF ROC-AUC {per_model_metrics[name]['roc_auc']:.4f}  PR-AUC {per_model_metrics[name]['pr_auc']:.4f}")
    ens = np.mean(ens_input, axis=0)
    ens_metrics = score(y, ens)
    print(f"  ENS   OOF ROC-AUC {ens_metrics['roc_auc']:.4f}  PR-AUC {ens_metrics['pr_auc']:.4f}")

    best_name = max(per_model_metrics, key=lambda k_: per_model_metrics[k_]["roc_auc"])
    best_metrics = {**per_model_metrics[best_name], "model": best_name}

    preds_df = pd.DataFrame(per_model_scores)
    preds_df["ensemble"] = ens
    return {
        "per_model": per_model_metrics,
        "best_single": best_metrics,
        "ensemble": ens_metrics,
        "protocol": f"{k}-fold OOF",
    }, preds_df


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    text_fields = [f.strip() for f in args.text_fields.split(",") if f.strip()]
    struct_fields = [f.strip() for f in args.struct_fields.split(",") if f.strip()]

    t0 = time.time()

    if args.input:
        df = load_records(args.input)
        if args.split_field:
            df_train = df[df[args.split_field] == args.train_value].reset_index(drop=True)
            df_test = df[df[args.split_field] == args.test_value].reset_index(drop=True)
            protocol = f"split by {args.split_field}"
        else:
            df_train, df_test = df, None
            protocol = "single input"
    else:
        df_train = load_records(args.train)
        df_test = load_records(args.test) if args.test else None
        protocol = "train + test files" if df_test is not None else "train only"

    print(f"Loaded: train={len(df_train)}  test={0 if df_test is None else len(df_test)}  "
          f"(protocol: {protocol})")
    print(f"Text fields: {text_fields}")
    if struct_fields:
        print(f"Struct fields: {struct_fields}")

    print("Building features (embedding may take a minute)…")
    X_train, X_test, feat_info = build_features(
        df_train, df_test, text_fields, struct_fields, args.embed_model, args.tfidf_max_features,
    )
    print(f"Feature dim: {X_train.shape[1]} (text-len 1 + emb {feat_info['n_features_embedding']} + "
          f"tfidf {feat_info['n_features_tfidf']} + struct {feat_info['n_features_structured']})")

    y_train = df_train[args.label_col].astype(int).to_numpy()
    y_test = df_test[args.label_col].astype(int).to_numpy() if df_test is not None else None

    if X_test is not None:
        print("\n=== Train → Test evaluation ===")
        results, preds_df = run_train_test(X_train, y_train, X_test, y_test, args.seed)
        results["protocol"] = protocol
        results["train_base_rate"] = round(float(y_train.mean()), 4)
        results["test_base_rate"] = round(float(y_test.mean()), 4)
        id_series = df_test[args.id_col] if args.id_col else df_test.iloc[:, 0]
        preds_df.insert(0, id_series.name, id_series.values)
        preds_df["y_true"] = y_test
    elif args.kfold > 0:
        print(f"\n=== {args.kfold}-fold OOF evaluation ===")
        results, preds_df = run_kfold(X_train, y_train, args.kfold, args.seed)
        results["base_rate"] = round(float(y_train.mean()), 4)
        id_series = df_train[args.id_col] if args.id_col else df_train.iloc[:, 0]
        preds_df.insert(0, id_series.name, id_series.values)
        preds_df["y_true"] = y_train
    else:
        raise SystemExit("Provide either --test or --kfold N. Nothing to evaluate against.")

    results["n_train"] = len(df_train)
    results["n_test"] = 0 if df_test is None else len(df_test)
    results["features"] = feat_info
    results["wall_clock_s"] = round(time.time() - t0, 1)
    results["seed"] = args.seed

    (args.out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    preds_df.to_csv(args.out_dir / "predictions.csv", index=False)

    print("\nWrote:")
    print(f"  {args.out_dir / 'metrics.json'}")
    print(f"  {args.out_dir / 'predictions.csv'}")
    print(f"Total wall-clock: {results['wall_clock_s']:.1f}s")


if __name__ == "__main__":
    main()
