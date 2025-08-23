import os
import json
import hashlib
import joblib
import yaml
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_recall_curve, precision_score, recall_score
)

def read_transformed_csv(data_repo: Path) -> pd.DataFrame:
    csv = data_repo / "transformed-data" / "transformed.csv"
    if not csv.exists():
        raise FileNotFoundError(f"transformed.csv not found at: {csv}")
    df = pd.read_csv(csv)
    if "asof_date" not in df.columns:
        raise ValueError("Column 'asof_date' missing in transformed.csv")
    if "churned" not in df.columns:
        raise ValueError("Column 'churned' missing in transformed.csv (required target)")
    df["asof_date"] = pd.to_datetime(df["asof_date"])
    return df

def kolkata_date_str() -> str:
    return datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d")

def hash_file(p: Path, algo="sha256") -> str:
    h = hashlib.new(algo)
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return f"{algo}:{h.hexdigest()}"

def time_holdout_split(df: pd.DataFrame, holdout_ratio: float = 0.2):
    df = df.sort_values("asof_date").reset_index(drop=True)
    split_idx = int(len(df) * (1.0 - holdout_ratio))
    if split_idx <= 0 or split_idx >= len(df):
        raise ValueError("Not enough rows for a time-based split.")
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

def fit_select_model(trainval: pd.DataFrame, feature_cols: list[str], target_col="churned"):
    # Prepare ordered data
    Xy = trainval[feature_cols + [target_col, "asof_date"]].sort_values("asof_date")
    X = Xy[feature_cols].reset_index(drop=True)
    y = Xy[target_col].astype(int).reset_index(drop=True)

    # Pipelines
    num = feature_cols
    pre_lr = ColumnTransformer([("num", StandardScaler(), num)], remainder="drop")
    lr = Pipeline([("pre", pre_lr),
                   ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=None))])

    rf = Pipeline([("clf", RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=42
    ))])

    candidates = {"logreg": lr, "rf": rf}
    tss = TimeSeriesSplit(n_splits=3)

    scores = {}
    for name, pipe in candidates.items():
        pr_aucs = []; rocs = []; f1s = []
        for tr_idx, va_idx in tss.split(X):
            pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            p = pipe.predict_proba(X.iloc[va_idx])[:, 1]
            pr_aucs.append(average_precision_score(y.iloc[va_idx], p))
            rocs.append(roc_auc_score(y.iloc[va_idx], p))
            prec, rec, thr = precision_recall_curve(y.iloc[va_idx], p)
            f1_arr = 2 * prec * rec / (prec + rec + 1e-9)
            thr_star = thr[np.nanargmax(f1_arr[:-1])] if len(thr) else 0.5
            f1s.append(f1_score(y.iloc[va_idx], (p >= thr_star).astype(int)))
        scores[name] = {"pr_auc": float(np.mean(pr_aucs)),
                        "roc_auc": float(np.mean(rocs)),
                        "f1": float(np.mean(f1s))}

    # Select by PR-AUC
    best_name = max(scores.items(), key=lambda kv: kv[1]["pr_auc"])[0]
    best_pipe = candidates[best_name].fit(X, y)

    # Choose threshold on full trainval to maximize F1
    p_all = best_pipe.predict_proba(X)[:, 1]
    prec, rec, thr = precision_recall_curve(y, p_all)
    f1_arr = 2 * prec * rec / (prec + rec + 1e-9)
    thr_star = thr[np.nanargmax(f1_arr[:-1])] if len(thr) else 0.5
    return best_name, best_pipe, float(thr_star), scores

def evaluate(pipe, threshold: float, df: pd.DataFrame, feature_cols: list[str], target_col="churned"):
    X = df[feature_cols]
    y = df[target_col].astype(int)
    p = pipe.predict_proba(X)[:, 1]
    yhat = (p >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "accuracy": float((yhat == y).mean()),
        "precision": float(precision_score(y, yhat, zero_division=0)),
        "recall": float(recall_score(y, yhat, zero_division=0)),
        "f1": float(f1_score(y, yhat, zero_division=0)),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-repo", required=True, help="Path to local checkout of 2025-dmml-data")
    ap.add_argument("--out-subdir", default="models/churn", help="Subdir under data repo to write artifacts to")
    ap.add_argument("--version", default="v0.1.0", help="Model version prefix")
    args = ap.parse_args()

    data_repo = Path(args.data_repo).resolve()
    df = read_transformed_csv(data_repo)

    target = "churned"
    drop_cols = ["customer_id", "asof_date", target]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    # Time-based split (80/20)
    trainval, test = time_holdout_split(df, holdout_ratio=0.20)

    # Train & select
    best_name, pipe, threshold, cv_scores = fit_select_model(trainval, feature_cols, target)

    # Evaluate on test (latest slice)
    test_metrics = evaluate(pipe, threshold, test, feature_cols, target)

    # Versioned artifact dir: vX.Y.Z__YYYY-MM-DD__<shortsha>
    shortsha = os.environ.get("GITHUB_SHA", "local")[:8]
    stamp = kolkata_date_str()  # Asia/Kolkata date
    out_dir = data_repo / args.out_subdir / f"{args.version}__{stamp}__{shortsha}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Persist model bundle
    model_bundle = {
        "pipeline": pipe,
        "threshold": threshold,
        "feature_cols": feature_cols,
        "model_name": best_name,
    }
    joblib.dump(model_bundle, out_dir / "model.pkl")

    # Save metadata
    params = {"model": best_name, "threshold": threshold, "cv_scores": cv_scores}
    (out_dir / "params.yaml").write_text(yaml.dump(params, sort_keys=False))

    (out_dir / "feature_signature.json").write_text(json.dumps(
        {"features": [{"name": c, "dtype": str(df[c].dtype)} for c in feature_cols]},
        indent=2
    ))

    source_csv = data_repo / "transformed-data" / "transformed.csv"
    (out_dir / "TRAINING_SOURCE.txt").write_text(str(source_csv.relative_to(data_repo)))
    (out_dir / "data_hash.txt").write_text(hash_file(source_csv) if source_csv.exists() else "n/a")

    # Save metrics
    (out_dir / "metrics.json").write_text(json.dumps({"test": test_metrics}, indent=2))

    # Also store individual performance files as requested
    perf_dir = out_dir / "performance"
    perf_dir.mkdir(exist_ok=True)
    for k in ("accuracy", "precision", "recall", "f1"):
        (perf_dir / f"{k}.txt").write_text(f"{test_metrics[k]:.6f}\n")

    # Update INDEX.csv (append-only)
    index = data_repo / args.out_subdir / "INDEX.csv"
    row = (f'{args.version},{stamp},{shortsha},'
           f'{test_metrics.get("pr_auc", float("nan")):.6f},'
           f'{test_metrics.get("roc_auc", float("nan")):.6f},'
           f'{test_metrics.get("f1", float("nan")):.6f},'
           f'{out_dir.name}\n')
    header = "version,date,git_sha,pr_auc,roc_auc,f1,artifact_dir\n"
    if not index.exists():
        index.write_text(header + row)
    else:
        with open(index, "a") as f:
            f.write(row)

    # Console summary
    print("Selected model:", best_name)
    print("Threshold:", threshold)
    print("Test metrics:", test_metrics)
    print("Artifacts written to:", out_dir)

if __name__ == "__main__":
    main()
