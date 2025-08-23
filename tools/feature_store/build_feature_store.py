#!/usr/bin/env python3
"""
Build & materialize a Feast feature store from a SQLite source.

- Source of truth: 2025-dmml-data/transformed-data/transformed.sqlite
  table: main.features_churn_v1 (unique key on [customer_id, asof_date])

- Flow:
  1) Read from SQLite (no parquet dependency in data repo)
  2) Validate schema & values (guardrails)
  3) Export an ephemeral file snapshot to feature_repo/_sqlite_export/features_churn_v1.parquet
     (Feast's local provider expects file-based batch sources for offline)
  4) feast apply + (optional) materialize to Redis
  5) Generate docs/FEATURE_CATALOG.md and docs/feature_catalog.csv

Note:
This keeps your offline store file-based (per assignment), while allowing you to
author the transformed dataset in SQLite. The SQLite → file snapshot is ephemeral
and regenerated each run.
"""
import argparse
import os
from pathlib import Path
from datetime import datetime, timedelta, timezone

import sqlite3
import pandas as pd

# Feast
from feast import FeatureStore, FeatureView, FileSource, Entity, Field
from feast.types import Int64, Float32
from feast import FeatureService

CATALOG_MD = Path("docs/FEATURE_CATALOG.md")
CATALOG_CSV = Path("docs/feature_catalog.csv")

# -----------------------------
# Feature metadata (authoritative)
# -----------------------------
FEATURE_META = {
    # name: {desc, dtype, version, offline, online}
    "plan_0": {"desc": "One-hot plan = Basic", "dtype": Int64, "version": "v1", "offline": True, "online": True},
    "plan_1": {"desc": "One-hot plan = Pro", "dtype": Int64, "version": "v1", "offline": True, "online": True},
    "plan_2": {"desc": "One-hot plan = Enterprise", "dtype": Int64, "version": "v1", "offline": True, "online": True},
    "tenure_days": {"desc": "Days since signup", "dtype": Int64, "version": "v1", "offline": True, "online": True},
    "days_since_last_login": {"desc": "Recency of login in days", "dtype": Int64, "version": "v1", "offline": True, "online": True},
    "last_login_missing": {"desc": "1 if never logged in / unknown", "dtype": Int64, "version": "v1", "offline": True, "online": True},
    "inactive_30d": {"desc": "No activity in last 30 days", "dtype": Int64, "version": "v1", "offline": True, "online": True},
    "inactive_90d": {"desc": "No activity in last 90 days", "dtype": Int64, "version": "v1", "offline": True, "online": True},
    "new_user_30d": {"desc": "Signed up in last 30 days", "dtype": Int64, "version": "v1", "offline": True, "online": True},
    "tickets_per_30d": {"desc": "Support tickets per 30d", "dtype": Float32, "version": "v1", "offline": True, "online": True},
    "session_hours": {"desc": "Total session hours per 30d", "dtype": Float32, "version": "v1", "offline": True, "online": True},
    "email_open_rate_30d": {"desc": "Email open rate (0-1) over 30d", "dtype": Float32, "version": "v1", "offline": True, "online": True},
    "asof_month_sin": {"desc": "Seasonality (month sin)", "dtype": Float32, "version": "v1", "offline": True, "online": True},
    "asof_month_cos": {"desc": "Seasonality (month cos)", "dtype": Float32, "version": "v1", "offline": True, "online": True},
    "auto_renew_enabled": {"desc": "Auto-renew is enabled", "dtype": Int64, "version": "v1", "offline": True, "online": True},
    "auto_renew_off": {"desc": "Auto-renew is disabled", "dtype": Int64, "version": "v1", "offline": True, "online": True},
    "auto_renew_off_and_inactive_30d": {"desc": "Disabled + inactive 30d", "dtype": Int64, "version": "v1", "offline": True, "online": True},
    "tenure_x_auto_renew_off": {"desc": "Tenure * Auto-renew off", "dtype": Float32, "version": "v1", "offline": True, "online": True},
    "inactivity_x_email": {"desc": "Inactive 30d * email open rate", "dtype": Float32, "version": "v1", "offline": True, "online": True},
    "tickets_x_recency": {"desc": "Tickets * (1 / (1+days since login))", "dtype": Float32, "version": "v1", "offline": True, "online": True},
    "delta_email_opens_30d": {"desc": "Δ email open rate (m/m)", "dtype": Float32, "version": "v1", "offline": True, "online": True},
    "delta_session_hours": {"desc": "Δ session hours (m/m)", "dtype": Float32, "version": "v1", "offline": True, "online": True},
    "delta_tickets_30d": {"desc": "Δ tickets per 30d (m/m)", "dtype": Float32, "version": "v1", "offline": True, "online": True},
    "rollmean_email_3": {"desc": "3-pt rolling mean email open rate", "dtype": Float32, "version": "v1", "offline": True, "online": True},
    "rollmean_session_3": {"desc": "3-pt rolling mean session hours", "dtype": Float32, "version": "v1", "offline": True, "online": True},
    "rollmean_tickets_3": {"desc": "3-pt rolling mean tickets", "dtype": Float32, "version": "v1", "offline": True, "online": True},
    "auto_renew_enabled_missing": {"desc": "Missingness indicator for auto_renew_enabled", "dtype": Int64, "version": "v1", "offline": True, "online": True},
    "monthly_spend": {"desc": "Monthly spend in account currency", "dtype": Float32, "version": "v1", "offline": True, "online": True},
    # Label (offline only)
    "churned": {"desc": "Binary label: 1 if churned", "dtype": Int64, "version": "v1", "offline": True, "online": False},
}

ONLINE_FEATURES = [k for k, v in FEATURE_META.items() if v["online"]]
LABELS = [k for k, v in FEATURE_META.items() if (v["offline"] and not v["online"])]

# -----------------------------
# Helpers
# -----------------------------
def load_from_sqlite(sqlite_path: Path, table: str) -> pd.DataFrame:
    if not sqlite_path.exists():
        raise FileNotFoundError(f"SQLite DB not found: {sqlite_path}")
    with sqlite3.connect(sqlite_path) as con:
        df = pd.read_sql_query(f"SELECT * FROM {table}", con)
    return df

def coerce_and_validate(df: pd.DataFrame) -> pd.DataFrame:
    errs = []
    required = ["customer_id", "asof_date"] + list(FEATURE_META.keys())
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise SystemExit("VALIDATION FAILED:\n- Missing columns: " + ", ".join(miss))

    # Coerce types
    df["customer_id"] = df["customer_id"].astype(str)
    df["asof_date"] = pd.to_datetime(df["asof_date"], errors="coerce", utc=True)
    if df["asof_date"].isna().any():
        errs.append("asof_date not parseable to datetime (UTC)")

    # Int-like columns to pandas nullable Int64
    int_cols = [c for c, m in FEATURE_META.items() if m["dtype"] == Int64]
    for c in int_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    # Float-like columns
    float_cols = [c for c, m in FEATURE_META.items() if m["dtype"] == Float32]
    for c in float_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

    # Key nulls
    if df["customer_id"].isna().any():
        errs.append("customer_id has nulls")
    if df["asof_date"].isna().any():
        errs.append("asof_date has nulls")

    # Plans in {0,1} and exclusive
    for c in ["plan_0", "plan_1", "plan_2"]:
        if not set(df[c].dropna().unique()).issubset({0, 1}):
            errs.append(f"{c} must be 0/1")
    if ((df["plan_0"].fillna(0) + df["plan_1"].fillna(0) + df["plan_2"].fillna(0)) != 1).any():
        errs.append("plan_0 + plan_1 + plan_2 must equal 1")

    # Flag columns in {0,1}
    flag_cols = [
        c for c, m in FEATURE_META.items()
        if m["dtype"] == Int64 and c not in ["tenure_days", "days_since_last_login", "churned"]
    ]
    for c in flag_cols:
        if not set(df[c].dropna().unique()).issubset({0, 1}):
            errs.append(f"{c} must be 0/1")

    # Non-negatives
    for c in ["tickets_per_30d", "session_hours", "monthly_spend", "tenure_days", "days_since_last_login"]:
        if (df[c].dropna() < 0).any():
            errs.append(f"{c} must be >= 0")

    # Ranges
    def between(series, lo, hi, name):
        bad = series.dropna()[(series < lo) | (series > hi)]
        if len(bad) > 0:
            errs.append(f"{name} out of range [{lo},{hi}]")

    between(df["email_open_rate_30d"], 0.0, 1.0, "email_open_rate_30d")
    between(df["asof_month_sin"], -1.0, 1.0, "asof_month_sin")
    between(df["asof_month_cos"], -1.0, 1.0, "asof_month_cos")

    # Auto-renew exclusivity
    if ((df["auto_renew_enabled"].fillna(0) + df["auto_renew_off"].fillna(0)) > 1).any():
        errs.append("auto_renew_enabled + auto_renew_off must not exceed 1")

    # Label domain
    if "churned" in df.columns and not set(df["churned"].dropna().unique()).issubset({0, 1}):
        errs.append("churned must be 0/1")

    if errs:
        raise SystemExit("VALIDATION FAILED:\n- " + "\n- ".join(errs))
    return df

def export_ephemeral_snapshot(df: pd.DataFrame, repo_path: Path) -> Path:
    out_dir = repo_path / "_sqlite_export"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "features_churn_v1.parquet"
    # Write a single snapshot file that Feast can read as a FileSource
    df.to_parquet(out_path, index=False)

    return out_path

def ensure_repo_yaml(repo_path: Path, redis_uri: str):
    repo_path.mkdir(parents=True, exist_ok=True)
    (repo_path / "feature_store.yaml").write_text(
f"""project: dm4ml_churn
registry: {repo_path.as_posix()}/registry.db
provider: local
offline_store:
  type: file
online_store:
  type: redis
  connection_string: "{redis_uri}"
entity_key_serialization_version: 2
"""
    )

def build_objects(snapshot_path: Path):
    customer = Entity(name="customer_id", join_keys=["customer_id"])

    file_src = FileSource(
        name="sqlite_snapshot_source",
        path=snapshot_path.as_posix(),  # <- ephemeral snapshot
        timestamp_field="asof_date",
    )

    fields_online = [Field(name=k, dtype=v["dtype"]) for k, v in FEATURE_META.items() if v["online"]]
    fields_labels = [Field(name=k, dtype=v["dtype"]) for k, v in FEATURE_META.items() if (v["offline"] and not v["online"])]

    features_v1 = FeatureView(
        name="customer_features_v1",
        entities=[customer],
        ttl=timedelta(days=365),
        schema=fields_online,
        online=True,
        source=file_src,
        tags={"version": "v1", "owner": "dm4ml"},
    )

    labels_v1 = FeatureView(
        name="customer_labels_v1",
        entities=[customer],
        ttl=timedelta(days=365),
        schema=fields_labels,
        online=False,  # labels NOT online
        source=file_src,
        tags={"version": "v1", "owner": "dm4ml"},
    )

    online_fs = FeatureService(name="online_inference_v1", features=[features_v1])
    train_fs = FeatureService(name="training_service_v1", features=[features_v1, labels_v1])

    return [customer, file_src, features_v1, labels_v1, online_fs, train_fs]

def generate_catalog(df_sample: pd.DataFrame):
    CATALOG_MD.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, meta in FEATURE_META.items():
        rows.append({
            "feature_name": name,
            "version": meta["version"],
            "dtype": meta["dtype"].__name__,
            "description": meta["desc"],
            "source": "SQLite: main.features_churn_v1",
            "offline_available": "Y" if meta["offline"] else "N",
            "online_available": "Y" if meta["online"] else "N",
            "example": None if name not in df_sample.columns else (df_sample[name].dropna().iloc[0] if not df_sample[name].dropna().empty else ""),
        })
    pd.DataFrame(rows).to_csv(CATALOG_CSV, index=False)

    md = [
        "# Feature Catalog",
        "",
        "| Feature | Version | Type | Offline | Online | Description |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        md.append(f"| `{r['feature_name']}` | {r['version']} | {r['dtype']} | {r['offline_available']} | {r['online_available']} | {r['description']} |")
    CATALOG_MD.write_text("\n".join(md))

# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-repo", required=True, help="Path to 2025-dmml-data clone")
    ap.add_argument("--repo-path", required=True, help="Path to write Feast repo (feature_store.yaml, registry, snapshot)")
    ap.add_argument("--redis", default="redis://localhost:6379/0", help="Redis connection string")
    ap.add_argument("--sqlite-db", default=None, help="Override path to transformed.sqlite")
    ap.add_argument("--sqlite-table", default="main.features_churn_v1", help="Table name")
    ap.add_argument("--apply", action="store_true", help="feast apply")
    ap.add_argument("--materialize", action="store_true", help="feast materialize_incremental(now)")
    ap.add_argument("--generate-catalog", action="store_true", help="Write docs/FEATURE_CATALOG.md and CSV")
    args = ap.parse_args()

    data_repo = Path(args.data_repo).resolve()
    sqlite_path = Path(args.sqlite_db) if args.sqlite_db else (data_repo / "transformed-data" / "transformed.sqlite")
    repo_path = Path(args.repo_path).resolve()

    # 1) Load from SQLite
    df = load_from_sqlite(sqlite_path, args.sqlite_table)

    # 2) Guardrails
    df = coerce_and_validate(df)

    # 3) Feast repo config
    ensure_repo_yaml(repo_path, args.redis)

    # 4) Export ephemeral snapshot for Feast batch source
    snapshot_path = export_ephemeral_snapshot(df, repo_path)

    # 5) Define Feast objects against the snapshot
    objects = build_objects(snapshot_path)

    # 6) Apply & materialize
    store = FeatureStore(repo_path=str(repo_path))
    if args.apply:
        store.apply(objects)

    if args.materialize:
        store.materialize_incremental(end_date=datetime.now(timezone.utc))

    # 7) Catalog
    if args.generate_catalog:
        generate_catalog(df_sample=df.head(50))

    print("Done.")

if __name__ == "__main__":
    main()
