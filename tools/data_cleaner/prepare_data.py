#!/usr/bin/env python3
import argparse
import json
import os
import re
from glob import glob
from datetime import datetime

import pandas as pd
from dateutil import parser as dtparser

# ---------- Helpers

def _safe_to_datetime(s):
    """Robustly parse timestamps/dates into pandas datetime; return NaT on failure."""
    return pd.to_datetime(s, errors="coerce", infer_datetime_format=True)

def _extract_customer_id(df):
    """
    Unify customer_id. Supports:
      - nested {"$oid": "..."} as 'customer_id.$oid' (json_normalize)
      - plain string 'customer_id'
    """
    if "customer_id.$oid" in df.columns:
        cid = df["customer_id.$oid"].astype(str)
    elif "customer_id" in df.columns:
        # Could be dicts or strings
        def _cid(x):
            if isinstance(x, dict):
                return x.get("$oid") or x.get("oid") or x.get("id") or ""
            return "" if pd.isna(x) else str(x)
        cid = df["customer_id"].apply(_cid)
    else:
        cid = pd.Series([""] * len(df))
    return cid

def _coalesce_cols(df, candidates, default=None):
    """Return the first existing column among candidates; else a Series[default]."""
    for c in candidates:
        if c in df.columns:
            return df[c]
    return pd.Series([default] * len(df))

def _label_map(series, ordered_classes):
    """Deterministic label encoding based on provided ordered class list."""
    mapping = {cls: i for i, cls in enumerate(ordered_classes)}
    # unseen -> append 'Unknown' if present in order; else -1
    if "Unknown" in mapping:
        return series.map(lambda v: mapping.get(v, mapping["Unknown"])).astype(int)
    return series.map(lambda v: mapping.get(v, -1)).astype(int)

def _minmax_inplace(df, cols):
    """Apply min-max normalization in place (0..1). If constant col -> set to 0.0."""
    for c in cols:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        mn, mx = s.min(), s.max()
        if pd.isna(mn) or pd.isna(mx) or mx == mn:
            df[c] = 0.0
        else:
            df[c] = (s - mn) / (mx - mn)

def _clean_contract_type(ct_raw):
    if pd.isna(ct_raw):
        return None
    s = str(ct_raw).strip()
    s = re.sub(r"annul|anual", "Annual", s, flags=re.IGNORECASE)  # common typos
    s = s.replace("Annul", "Annual")
    s = s.title()
    if s not in {"Monthly", "Quarterly", "Annual"}:
        return None
    return s

def _to_ymd_str(dt):
    if pd.isna(dt):
        return ""
    try:
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return ""

def _parse_ingest_ts(ts_name):
    """Best-effort to parse folder name to comparable sortable key."""
    try:
        return pd.to_datetime(ts_name, errors="coerce")
    except Exception:
        return pd.NaT

# ---------- Loaders

def _load_all_json_rows(paths):
    """Read list-of-dict JSON from files into a single DataFrame via json_normalize."""
    dfs = []
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                df = pd.json_normalize(data)
            else:
                # If single object, wrap
                df = pd.json_normalize([data])
            # derive ingest_ts from raw-data/<ts>/<kind>/data.json
            parts = os.path.normpath(p).split(os.sep)
            # .../raw-data/<timestamp>/<kind>/data.json
            try:
                ts = parts[-3]
            except Exception:
                ts = ""
            df["ingest_ts"] = ts
            dfs.append(df)
        except Exception as e:
            print(f"[WARN] Skipping {p}: {e}")
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)

def _find_paths(data_repo, kind_dir):
    """
    kind_dir in {"engagement", "satisfaction", "satisfication"}.
    Returns all 'raw-data/*/<kind_dir>/data.json'.
    """
    pat = os.path.join(data_repo, "raw-data", "*", kind_dir, "data.json")
    return sorted(glob(pat))

# ---------- Cleaning per dataset

def clean_engagement(df):
    if df.empty:
        return df

    df = df.copy()

    # Keys
    df["customer_id"] = _extract_customer_id(df)
    df["asof_date"]   = _coalesce_cols(df, ["asof_date"])
    df["signup_date"] = _coalesce_cols(df, ["signup_date"])
    df["last_login_date"] = _coalesce_cols(df, ["last_login_date"])

    # Parse dates
    df["asof_date"] = _safe_to_datetime(df["asof_date"])
    df["signup_date"] = _safe_to_datetime(df["signup_date"])
    df["last_login_date"] = _safe_to_datetime(df["last_login_date"])

    # Mandatory row filters
    df = df[(df["customer_id"].str.len() > 0)]
    df = df[~df["asof_date"].isna()]
    df = df[~df["signup_date"].isna()]

    # Categorical encodes
    sub = _coalesce_cols(df, ["subscription_plan"]).astype(str).str.strip()
    sub = sub.where(sub.notna() & (sub != "") , "Unknown")
    df["subscription_plan_le"] = _label_map(sub, ["Basic", "Pro", "Enterprise", "Unknown"])

    churn_raw = _coalesce_cols(df, ["churned"])
    # Normalize truthy/falsy
    df["churned_engagement"] = churn_raw.map(lambda v: 1 if str(v).lower() in {"1","true","t","yes","y"} else 0).astype(int)

    # Booleans → one-hot
    auto_raw = _coalesce_cols(df, ["auto_renew_enabled"]).map(lambda v: str(v).lower() in {"1","true","t","yes","y"})
    auto_raw = auto_raw.fillna(False)
    dummies = pd.get_dummies(auto_raw, prefix="auto_renew_enabled")
    # Ensure both columns exist for stable schema
    for col in ["auto_renew_enabled_False", "auto_renew_enabled_True"]:
        if col not in dummies.columns:
            dummies[col] = 0
    df = pd.concat([df, dummies], axis=1)

    # Numeric fill then normalize later
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

    # Date translations
    df["last_login_date_std"] = df["last_login_date"].map(_to_ymd_str)
    df["days_since_last_login"] = (df["asof_date"] - df["last_login_date"]).dt.days.fillna(-1).astype(int)

    # Keep only columns needed for merge & model
    keep = [
        "customer_id", "asof_date", "signup_date",
        "subscription_plan_le",
        "monthly_spend", "support_tickets_last_90d", "avg_session_length_minutes", "email_opens_last_30d",
        "auto_renew_enabled_False", "auto_renew_enabled_True",
        "last_login_date_std", "days_since_last_login",
        "churned_engagement", "ingest_ts",
    ]
    return df[keep]

def clean_satisfaction(df):
    if df.empty:
        return df

    df = df.copy()

    # Keys
    df["customer_id"] = _extract_customer_id(df)
    df["asof_date"]   = _coalesce_cols(df, ["asof_date"])
    df["asof_date"] = _safe_to_datetime(df["asof_date"])

    # Drop rows missing required keys
    df = df[(df["customer_id"].str.len() > 0)]
    df = df[~df["asof_date"].isna()]

    # Region: drop blanks then label encode
    region = _coalesce_cols(df, ["region"]).astype(str).str.strip()
    df = df[region != ""]
    region = region.loc[df.index]
    df["region_le"] = _label_map(region, ["Rural", "Semi-Urban", "Urban"])

    # Contract type: fix typos, drop blanks, label encode
    ct = _coalesce_cols(df, ["contract_type"]).map(_clean_contract_type)
    df = df[ct.notna()]
    ct = ct.loc[df.index]
    df["contract_type_le"] = _label_map(ct, ["Annual", "Monthly", "Quarterly"])

    # Numerics (fill 0 → normalize later)
    num_cols = [
        "age", "avg_monthly_bill", "payment_delay_days",
        "customer_support_calls_last_6m", "net_promoter_score", "discounts_received_last_6m",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        else:
            df[c] = 0.0

    # Churn
    churn_raw = _coalesce_cols(df, ["churned"])
    df["churned_satisfaction"] = churn_raw.map(lambda v: 1 if str(v).lower() in {"1","true","t","yes","y"} else 0).astype(int)

    keep = [
        "customer_id", "asof_date",
        "age", "avg_monthly_bill", "payment_delay_days",
        "customer_support_calls_last_6m", "net_promoter_score", "discounts_received_last_6m",
        "region_le", "contract_type_le",
        "churned_satisfaction", "ingest_ts",
    ]
    return df[keep]

def latest_per_key(df, key_cols):
    """Within each (customer_id, asof_date) group, keep row with max ingest_ts."""
    if df.empty:
        return df
    # make sortable ts
    ts = df["ingest_ts"].apply(_parse_ingest_ts)
    # fallback: lexical order if parse fails
    ts_rank = ts.fillna(pd.NaT)
    df = df.assign(_sort_ts=ts_rank, _lex=df["ingest_ts"].astype(str))
    df = df.sort_values(by=["_sort_ts", "_lex"]).groupby(key_cols, as_index=False).tail(1)
    return df.drop(columns=["_sort_ts", "_lex"])

def consolidate_churn(row):
    e = row.get("churned_engagement", None)
    s = row.get("churned_satisfaction", None)
    if pd.isna(e) and pd.isna(s):
        return 0
    if pd.isna(e):
        return int(s)
    if pd.isna(s):
        return int(e)
    # If both present and disagree, prefer engagement
    return int(e)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-repo", required=True, help="Path to local clone of 2025-dmml-data")
    args = ap.parse_args()

    data_repo = args.data_repo
    # Find raw JSONs
    eng_paths = _find_paths(data_repo, "engagement")
    sat_paths = _find_paths(data_repo, "satisfaction") + _find_paths(data_repo, "satisfication")

    print(f"Found {len(eng_paths)} engagement files, {len(sat_paths)} satisfaction files.")

    eng_raw = _load_all_json_rows(eng_paths)
    sat_raw = _load_all_json_rows(sat_paths)

    eng = clean_engagement(eng_raw)
    sat = clean_satisfaction(sat_raw)

    # Deduplicate within each dataset by latest ingest
    key_cols = ["customer_id", "asof_date"]
    eng_latest = latest_per_key(eng, key_cols)
    sat_latest = latest_per_key(sat, key_cols)

    # Normalize numeric columns per instructions
    eng_norm_cols = ["monthly_spend", "support_tickets_last_90d", "avg_session_length_minutes", "email_opens_last_30d", "days_since_last_login"]
    _minmax_inplace(eng_latest, eng_norm_cols)

    sat_norm_cols = ["age", "avg_monthly_bill", "payment_delay_days", "customer_support_calls_last_6m", "net_promoter_score", "discounts_received_last_6m"]
    _minmax_inplace(sat_latest, sat_norm_cols)

    # Merge (inner) on keys across latest snapshots
    merged = pd.merge(
        eng_latest, sat_latest,
        on=key_cols, how="inner", suffixes=("_eng", "_sat")
    )

    # Produce consolidated target
    merged["churned"] = merged.apply(consolidate_churn, axis=1).astype(int)

    # Output schema
    out_cols = [
        "customer_id", "asof_date", "signup_date",
        "last_login_date_std",
        "days_since_last_login",
        "subscription_plan_le",
        "auto_renew_enabled_False", "auto_renew_enabled_True",
        "monthly_spend",
        "support_tickets_last_90d",
        "avg_session_length_minutes",
        "email_opens_last_30d",
        "age",
        "region_le",
        "contract_type_le",
        "avg_monthly_bill",
        "payment_delay_days",
        "customer_support_calls_last_6m",
        "net_promoter_score",
        "discounts_received_last_6m",
        "churned",                 # consolidated target
        "churned_engagement",      # traceability
        "churned_satisfaction",    # traceability
    ]

    # Coerce dates to desired text format
    merged["asof_date"] = merged["asof_date"].map(_to_ymd_str)
    merged["signup_date"] = merged["signup_date"].map(_to_ymd_str)

    # Ensure all columns exist
    for c in out_cols:
        if c not in merged.columns:
            merged[c] = "" if c in {"customer_id","asof_date","signup_date","last_login_date_std"} else 0

    out = merged[out_cols].copy()

    # Write
    out_dir = os.path.join(data_repo, "prepared-data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "prepared.csv")
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out):,} rows to {out_path}")

if __name__ == "__main__":
    main()
