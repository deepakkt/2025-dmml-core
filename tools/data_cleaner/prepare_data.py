#!/usr/bin/env python3
import argparse
import json
import os
from glob import glob
from typing import List

import pandas as pd


# ------------------------
# Helpers
# ------------------------

def _safe_to_datetime(s):
    """Parse to pandas datetime; NaT on failure."""
    return pd.to_datetime(s, errors="coerce")

def _extract_customer_id(df: pd.DataFrame) -> pd.Series:
    """
    Unify customer_id. Supports:
      - 'customer_id.$oid' (flattened via json_normalize)
      - raw 'customer_id' possibly as dict {"$oid": "..."} or as a string
    """
    if "customer_id.$oid" in df.columns:
        return df["customer_id.$oid"].astype(str)
    if "customer_id" in df.columns:
        def _cid(x):
            if isinstance(x, dict):
                return str(x.get("$oid") or x.get("oid") or x.get("id") or "")
            return "" if pd.isna(x) else str(x)
        return df["customer_id"].apply(_cid)
    return pd.Series([""] * len(df))

def _coalesce(df: pd.DataFrame, names: List[str], default=None) -> pd.Series:
    """Return first present column; else constant default Series."""
    for n in names:
        if n in df.columns:
            return df[n]
    return pd.Series([default] * len(df))

def _label_map(series: pd.Series, ordered_classes: List[str]) -> pd.Series:
    """Deterministic label encode; unseen -> 'Unknown' if available, else -1."""
    mapping = {c: i for i, c in enumerate(ordered_classes)}
    unk = mapping.get("Unknown", -1)
    return series.map(lambda v: mapping.get(v, unk)).astype(int)

def _minmax_inplace(df: pd.DataFrame, cols: List[str]) -> None:
    """Min-max normalize to [0,1]; constant/empty -> 0.0."""
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0
            continue
        s = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        mn, mx = s.min(), s.max()
        if pd.isna(mn) or pd.isna(mx) or mx == mn:
            df[c] = 0.0
        else:
            df[c] = (s - mn) / (mx - mn)

def _fmt_ymd(dt):
    """Format datetime to %Y-%m-%d; empty string if NaT/None."""
    if pd.isna(dt):
        return ""
    try:
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return ""


# ------------------------
# IO
# ------------------------

def _find_engagement_paths(data_repo: str) -> List[str]:
    """Find all raw-data/*/engagement/data.json files."""
    pat = os.path.join(data_repo, "raw-data", "*", "engagement", "data.json")
    return sorted(glob(pat))

def _load_all_json(paths: List[str]) -> pd.DataFrame:
    """Load list-of-dict JSONs and add ingest_ts from folder name."""
    dfs = []
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                df = pd.json_normalize(data)
            else:
                df = pd.json_normalize([data])
            parts = os.path.normpath(p).split(os.sep)
            ts = parts[-3] if len(parts) >= 3 else ""
            df["ingest_ts"] = ts
            dfs.append(df)
        except Exception as e:
            print(f"[WARN] Skipping {p}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


# ------------------------
# Cleaning (ENGAGEMENT ONLY)
# ------------------------

def clean_engagement_only(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the engagement cleaning rules:

    4a.i  customer_id.$oid => use as is, skip row if blank
    4a.ii asof_date => skip row if blank
    4a.iii signup_date => skip row if blank
    4a.iv subscription_plan => label encode
    4a.v  monthly_spend => normalize (use zero if missing)
    4a.vi support_tickets_last_90d => normalize (use zero if missing)
    4a.vii avg_session_length_minutes => normalize (use zero if missing)
    4a.viii email_opens_last_30d => normalize (use zero if missing)
    4a.ix auto_renew_enabled => use False if missing, then one-hot encode
    4a.x  last_login_date => translate to "%Y-%m-%d". blanks allowed
    4a.xi churned => label encode
    """

    if df_raw.empty:
        return df_raw.copy()

    df = df_raw.copy()

    # --- Key fields
    df["customer_id"] = _extract_customer_id(df)
    df["asof_date"]   = _coalesce(df, ["asof_date"])
    df["signup_date"] = _coalesce(df, ["signup_date"])
    df["last_login_date"] = _coalesce(df, ["last_login_date"])

    # Parse dates
    df["asof_date"] = _safe_to_datetime(df["asof_date"])
    df["signup_date"] = _safe_to_datetime(df["signup_date"])
    df["last_login_date"] = _safe_to_datetime(df["last_login_date"])

    # Mandatory row filters
    before = len(df)
    df = df[(df["customer_id"].str.len() > 0)]
    df = df[~df["asof_date"].isna()]
    df = df[~df["signup_date"].isna()]
    after = len(df)
    print(f"Engagement: filtered mandatory keys {before} -> {after}")

    # Categorical: subscription_plan -> label encode
    sub = _coalesce(df, ["subscription_plan"]).astype(str).str.strip()
    sub = sub.where(sub.notna() & (sub != ""), "Unknown")
    df["subscription_plan_le"] = _label_map(sub, ["Basic", "Pro", "Enterprise", "Unknown"])

    # churned -> label encode (boolean-like to 0/1)
    churn_raw = _coalesce(df, ["churned"])
    df["churned"] = churn_raw.map(lambda v: 1 if str(v).lower() in {"1","true","t","yes","y"} else 0).astype(int)

    # auto_renew_enabled -> missing -> False -> one-hot
    auto_raw = _coalesce(df, ["auto_renew_enabled"]).map(lambda v: str(v).lower() in {"1","true","t","yes","y"})
    auto_raw = auto_raw.fillna(False)
    dummies = pd.get_dummies(auto_raw, prefix="auto_renew_enabled")
    for col in ["auto_renew_enabled_False", "auto_renew_enabled_True"]:
        if col not in dummies.columns:
            dummies[col] = 0
    df = pd.concat([df, dummies], axis=1)

    # Numerics: fill 0 then normalize in place
    num_cols = [
        "monthly_spend",
        "support_tickets_last_90d",
        "avg_session_length_minutes",
        "email_opens_last_30d",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        else:
            df[c] = 0.0

    _minmax_inplace(df, num_cols)

    # last_login_date -> "%Y-%m-%d" (blanks allowed)
    df["last_login_date"] = df["last_login_date"].map(_fmt_ymd)

    # also standardize asof_date/signup_date to "%Y-%m-%d" for consistency
    df["asof_date"] = df["asof_date"].map(_fmt_ymd)
    df["signup_date"] = df["signup_date"].map(_fmt_ymd)

    # Keep only required/output columns
    out_cols = [
        "customer_id",
        "asof_date",
        "signup_date",
        "last_login_date",
        "subscription_plan_le",
        "monthly_spend",
        "support_tickets_last_90d",
        "avg_session_length_minutes",
        "email_opens_last_30d",
        "auto_renew_enabled_False",
        "auto_renew_enabled_True",
        "churned",
        "ingest_ts",  # keep for traceability; not used in modeling
    ]

    for c in out_cols:
        if c not in df.columns:
            # default types: strings for ids/dates, 0 for numerics/bools
            df[c] = "" if c in {"customer_id","asof_date","signup_date","last_login_date","ingest_ts"} else 0

    return df[out_cols]


def latest_per_key(df: pd.DataFrame, key_cols: List[str]) -> pd.DataFrame:
    """Within each (customer_id, asof_date), keep the row with the latest ingest_ts (lexical and datetime-aware)."""
    if df.empty:
        return df
    ts_parsed = pd.to_datetime(df["ingest_ts"], errors="coerce")
    df = df.assign(_ts=ts_parsed, _lex=df["ingest_ts"].astype(str))
    # sort so that tail(1) keeps the latest (by parsed ts then lexical)
    df = df.sort_values(by=["_ts", "_lex"]).groupby(key_cols, as_index=False).tail(1)
    return df.drop(columns=["_ts", "_lex"])


# ------------------------
# Main
# ------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-repo", required=True, help="Path to local clone of 2025-dmml-data")
    args = ap.parse_args()

    data_repo = args.data_repo

    eng_paths = _find_engagement_paths(data_repo)
    print(f"Found {len(eng_paths)} engagement files.")
    eng_raw = _load_all_json(eng_paths)
    print(f"Loaded engagement rows: {len(eng_raw):,}")

    cleaned = clean_engagement_only(eng_raw)
    print(f"Cleaned engagement rows: {len(cleaned):,}")

    # Deduplicate to latest per (customer_id, asof_date)
    cleaned_latest = latest_per_key(cleaned, ["customer_id", "asof_date"])
    print(f"After latest-per-key: {len(cleaned_latest):,}")

    # Sort for determinism
    cleaned_latest = cleaned_latest.sort_values(by=["asof_date", "customer_id"]).reset_index(drop=True)

    # Write output
    out_dir = os.path.join(data_repo, "prepared-data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "prepared.csv")

    cleaned_latest.to_csv(out_path, index=False)
    print(f"Wrote {len(cleaned_latest):,} rows to {out_path}")

if __name__ == "__main__":
    main()
