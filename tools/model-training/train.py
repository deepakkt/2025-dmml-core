import os
import json
import hashlib
import joblib
import yaml
import argparse
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

# --------- Utilities ---------

def read_transformed_csv(data_repo: Path) -> pd.DataFrame:
    csv = data_repo / "transformed-data" / "transformed.csv"
    if not csv.exists():
        raise FileNotFoundError(f"transformed.csv not found at: {csv}")
    df = pd.read_csv(csv)

    if "asof_date" not in df.columns:
        raise ValueError("Column 'asof_date' missing in transformed.csv")
    if "churned" not in df.columns:
        raise ValueError("Column 'churned' missing in transformed.csv (required target)")

    # Parse date and coerce obvious booleans to int (pipeline can still handle them)
    df["asof_date"] = pd.to_datetime(df["asof_date"], errors="coerce")

    # Replace inf/-inf with NaN (to be imputed)
    df = df.replace([np.inf, -np.inf], np.nan)

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


def _has_two_classes(y: pd.Series) -> bool:
    return np.unique(y).size >= 2


def _safe_metric(fn, *args, **kwargs):
    """Run a sklearn metric and return NaN on ValueError (e.g., single-class y)."""
    try:
        return float(fn(*args, **kwargs))
    except Exception:
        return float("nan")


# --------- Model building ---------

def build_pipelines(feature_df: pd.DataFrame):
    """
    Build two candidate pipelines:
      1) LogisticRegression with numeric scaling and categorical OHE
      2) RandomForest with imputation (no scaling, OHE for categoricals)
    """

    # Selectors
    num_selector = selector(dtype_include=np.number)
    cat_selector = selector(dtype_include=["object", "category"])

    # Numeric + Categorical transformers
    num_imputer = SimpleImputer(strategy="median")
    num_scaler = StandardScaler()

    cat_imputer = SimpleImputer(strategy="most_frequent")
    cat_ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    # Pipelines for LR: impute -> scale (num), impute -> OHE (cat)
    pre_lr = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("imputer", num_imputer), ("scaler", num_scaler)]), num_selector),
            ("cat", Pipeline(steps=[("imputer", cat_imputer), ("ohe", cat_ohe)]), cat_selector),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    lr = Pipeline(
        steps=[
            ("pre", pre_lr),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")),
        ]
    )

    # Pipelines for RF: impute (num), impute+OHE (cat)
    pre_rf = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("imputer", num_imputer)]), num_selector),
            ("cat", Pipeline(steps=[("imputer", cat_imputer), ("ohe", cat_ohe)]), cat_selector),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    rf = Pipeline(
        steps=[
            ("pre", pre_rf),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                n_jobs=-1,
                class_weight="balanced_subsample",
                random_state=42,
            )),
        ]
    )

    return {"logreg": lr, "rf": rf}


def fit_select_model(trainval: pd.DataFrame, feature_cols: list[str], target_col="churned"):
    # Prepare ordered data
    Xy = trainval[feature_cols + [target_col, "asof_date"]].sort_values("asof_date")
    X = Xy[feature_cols].reset_index(drop=True)
    y = Xy[target_col].astype(int).reset_index(drop=True)

    candidates = build_pipelines(X)

    # TimeSeries CV; skip folds that don't have both classes in train split
    tss = TimeSeriesSplit(n_splits=3)

    scores = {}
    for name, pipe in candidates.items():
        pr_aucs = []
        rocs = []
        f1s = []
        valid_folds = 0

        for tr_idx, va_idx in tss.split(X):
            y_tr = y.iloc[tr_idx]
            y_va = y.iloc[va_idx]

            # Skip if the training fold is single-class (can't train a classifier meaningfully)
            if not _has_two_classes(y_tr):
                continue

            # Fit
            pipe.fit(X.iloc[tr_idx], y_tr)

            # Predict proba on validation
            p = pipe.predict_proba(X.iloc[va_idx])[:, 1]

            # Metrics (robust to single-class validation)
            pr_aucs.append(_safe_metric(average_precision_score, y_va, p))
            rocs.append(_safe_metric(roc_auc_score, y_va, p))

            # Threshold by F1 on the validation fold (only if both classes present in val)
            if _has_two_classes(y_va):
                prec, rec, thr = precision_recall_curve(y_va, p)
                if len(thr) > 0:
                    f1_arr = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
                    thr_star = thr[int(np.nanargmax(f1_arr))]
                    f1s.append(_safe_metric(f1_score, y_va, (p >= thr_star).astype(int)))
                else:
                    f1s.append(float("nan"))
            else:
                f1s.append(float("nan"))

            valid_folds += 1

        # If no valid folds (extreme edge case), do a single fit on all trainval and compute pseudo-scores
        if valid_folds == 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if _has_two_classes(y):
                    pipe.fit(X, y)
                    p_all = pipe.predict_proba(X)[:, 1]
                    pr_aucs.append(_safe_metric(average_precision_score, y, p_all))
                    rocs.append(_safe_metric(roc_auc_score, y, p_all))
                    # F1 threshold on full trainval
                    prec, rec, thr = precision_recall_curve(y, p_all)
                    if len(thr) > 0:
                        f1_arr = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
                        thr_star = thr[int(np.nanargmax(f1_arr))]
                        f1s.append(_safe_metric(f1_score, y, (p_all >= thr_star).astype(int)))
                    else:
                        f1s.append(float("nan"))
                else:
                    # Degenerate case: bail with zeros
                    pr_aucs.append(float("nan"))
                    rocs.append(float("nan"))
                    f1s.append(float("nan"))

        # Aggregate (nanmean to tolerate undefined folds)
        scores[name] = {
            "pr_auc": float(np.nanmean(pr_aucs)),
            "roc_auc": float(np.nanmean(rocs)),
            "f1": float(np.nanmean(f1s)),
        }

    # Select by PR-AUC (fallback to roc_auc, then f1 if needed)
    def _score_key(k):
        s = scores[k]
        return (
            (s["pr_auc"] if not np.isnan(s["pr_auc"]) else -1.0),
            (s["roc_auc"] if not np.isnan(s["roc_auc"]) else -1.0),
            (s["f1"] if not np.isnan(s["f1"]) else -1.0),
        )

    best_name = max(scores.keys(), key=_score_key)
    best_pipe = candidates[best_name].fit(X, y) if _has_two_classes(y) else candidates[best_name].fit(X, y)

    # Choose threshold on full trainval (maximize F1); if undefined, fallback to 0.5
    p_all = best_pipe.predict_proba(X)[:, 1]
    prec, rec, thr = precision_recall_curve(y, p_all)
    if len(thr) > 0 and _has_two_classes(y):
        f1_arr = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
        thr_star = float(thr[int(np.nanargmax(f1_arr))])
    else:
        thr_star = 0.5

    return best_name, best_pipe, thr_star, scores


def evaluate(pipe, threshold: float, df: pd.DataFrame, feature_cols: list[str], target_col="churned"):
    X = df[feature_cols]
    y = df[target_col].astype(int)

    p = pipe.predict_proba(X)[:, 1]
    yhat = (p >= threshold).astype(int)

    return {
        "roc_auc": _safe_metric(roc_auc_score, y, p),
        "pr_auc": _safe_metric(average_precision_score, y, p),
        "accuracy": float((yhat == y).mean()),
        "precision": _safe_metric(precision_score, y, yhat, zero_division=0),
        "recall": _safe_metric(recall_score, y, yhat, zero_division=0),
        "f1": _safe_metric(f1_score, y, yhat, zero_division=0),
    }


# --------- Main ---------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-repo", required=True, help="Path to local checkout of 2025-dmml-data")
    ap.add_argument("--out-subdir", default="models/churn", help="Subdir under data repo to write artifacts to")
    ap.add_argument("--version", default="v0.1.0", help="Model version prefix")
    ap.add_argument("--holdout", type=float, default=0.20, help="Time-based holdout ratio for test set")
    args = ap.parse_args()

    data_repo = Path(args.data_repo).resolve()
    df = read_transformed_csv(data_repo)

    target = "churned"
    # Feature space = all columns except identifiers, target and date
    drop_cols = {"customer_id", "asof_date", target}
    feature_cols = [c for c in df.columns if c not in drop_cols]

    if len(feature_cols) == 0:
        raise ValueError("No feature columns found after dropping identifiers/date/target.")

    # Time-based split (e.g., 80/20)
    trainval, test = time_holdout_split(df, holdout_ratio=args.holdout)

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
        "feature_cols": feature_cols,  # original column list before transformers
        "model_name": best_name,
    }
    joblib.dump(model_bundle, out_dir / "model.pkl")

    # Save metadata
    params = {"model": best_name, "threshold": float(threshold), "cv_scores": cv_scores}
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
        val = test_metrics.get(k, float("nan"))
        (perf_dir / f"{k}.txt").write_text(f"{val:.6f}\n" if not np.isnan(val) else "nan\n")

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
    # Reduce noisy warnings from sklearn when folds are degenerate
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
