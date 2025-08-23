#!/usr/bin/env python3
"""
Build & materialize a Feast feature store from a SQLite source (soft-fail validation).

- Source of truth: 2025-dmml-data/transformed-data/transformed.sqlite
  table: main.features_churn_v1 (unique key on [customer_id, asof_date])

Flow:
  1) Read from SQLite
  2) Validate & coerce (SOFT by default): log invalid rows to stdout, drop them, continue
  3) Export an ephemeral parquet snapshot under feature_repo/_sqlite_export/
  4) feast apply + (optional) materialize to Redis
  5) Generate docs/FEATURE_CATALOG.md and docs/feature_catalog.csv

Use --strict-validation to revert to hard-fail behavior.
"""
import argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone
import sqlite3
import pandas as pd

# Feast
from feast import FeatureStore, FeatureView, FileSource, Entity, Field, FeatureService
from feast.types import Int64, Float32

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
# Load
# -----------------------------
def load_from_sqlite(sqlite_path: Path, table: str) -> pd.DataFrame:
    if not sqlite_path.exists():
        raise FileNotFoundError(f"SQLite DB not found: {sqlite_path}")
    with sqlite3.connect(sqlite_path) as con:
        return pd.read_sql_query(f"SELECT * FROM {table}", con)

# -----------------------------
# Validation (SOFT by default)
# -----------------------------
def clean_and_report(df: pd.DataFrame, *, strict: bool = False, max_examples: int = 8) -> pd.DataFrame:
    """
    Coerce types, log violations to stdout, and drop offending rows.
    If strict=True, raise on any violations instead of dropping.
    """
    required = ["customer_id", "asof_date"] + list(FEATURE_META.keys())
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise SystemExit("VALIDATION FAILED (schema): missing columns -> " + ", ".join(missing_cols))

    # Coercions
    df["customer_id"] = df["customer_id"].astype(str)
    df["asof_date"] = pd.to_datetime(df["asof_date"], errors="coerce", utc=True)

    int_cols = [c for c, m in FEATURE_META.items() if m["dtype"] == Int64]
    for c in int_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    float_cols = [c for c, m in FEATURE_META.items() if m["dtype"] == Float32]
    for c in float_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

    # Start with all valid; progressively mask out violations
    valid_mask = pd.Series(True, index=df.index)
    issues = []

    def flag(name: str, bad_mask: pd.Series, detail: str):
        nonlocal valid_mask, issues
        bad_idx = bad_mask[bad_mask].index
        if len(bad_idx) == 0:
            return
        issues.append((name, len(bad_idx), detail, df.loc[bad_idx, ["customer_id", "asof_date"]].head(max_examples)))
        if strict:
            pass  # collect and raise later
        else:
            valid_mask.loc[bad_idx] = False  # drop these rows

    # Key/timestamp presence
    flag("customer_id null", df["customer_id"].isna(), "customer_id must be non-null")
    flag("asof_date null/invalid", df["asof_date"].isna(), "asof_date must be valid datetime")

    # Plans
    for c in ["plan_0", "plan_1", "plan_2"]:
        flag(f"{c} not in {{0,1}}", ~df[c].isin([0, 1]), f"{c} must be 0/1 integer")
    plan_sum = (df["plan_0"].fillna(0) + df["plan_1"].fillna(0) + df["plan_2"].fillna(0))
    flag("plan exclusivity", plan_sum != 1, "plan_0 + plan_1 + plan_2 must equal 1")

    # Other 0/1 flags (exclude integer measures that aren't flags)
    flag_cols = [
        c for c, m in FEATURE_META.items()
        if m["dtype"] == Int64 and c not in ["tenure_days", "days_since_last_login", "churned"]
        and c not in ["plan_0", "plan_1", "plan_2"]
    ]
    for c in flag_cols:
        flag(f"{c} not in {{0,1}}", ~df[c].isin([0, 1]), f"{c} must be 0/1 integer")

    # Non-negatives
    for c in ["tickets_per_30d", "session_hours", "monthly_spend", "tenure_days", "days_since_last_login"]:
        flag(f"{c} negative", df[c].dropna() < 0, f"{c} must be >= 0")

    # Ranges
    flag("email_open_rate_30d range", ~df["email_open_rate_30d"].between(0.0, 1.0, inclusive="both"),
         "email_open_rate_30d must be in [0,1]")
    flag("asof_month_sin range", ~df["asof_month_sin"].between(-1.0, 1.0, inclusive="both"),
         "asof_month_sin must be in [-1,1]")
    flag("asof_month_cos range", ~df["asof_month_cos"].between(-1.0, 1.0, inclusive="both"),
         "asof_month_cos must be in [-1,1]")

    # Auto-renew exclusivity
    flag("auto_renew exclusivity",
         (df["auto_renew_enabled"].fillna(0) + df["auto_renew_off"].fillna(0)) > 1,
         "auto_renew_enabled + auto_renew_off must not exceed 1")

    # Label domain (offline only)
    if "churned" in df.columns:
        flag("churned not in {0,1}", ~df["churned"].isin([0, 1]), "churned must be 0/1")

    # Report
    if issues:
        print("\n=== DATA QUALITY REPORT (soft validation) ===")
        total = len(df)
        for name, count, detail, sample in issues:
            print(f"- {name}: {count} row(s) -> {detail}")
            if len(sample) > 0:
                print(sample.to_string(index=False))
        dropped = (valid_mask == False).sum()
        kept = total - dropped
        print(f"Summary: total={total}, dropped={dropped}, kept={kept}")
        print("============================================\n")

    # Hard fail if requested
    if strict and issues:
        raise SystemExit("VALIDATION FAILED (strict mode). See report above.")

    # Drop offending rows in soft mode
    df_clean = df[valid_mask].copy()

    if df_clean.empty:
        raise SystemExit("VALIDATION FAILED: no valid rows remain after dropping invalid records.")

    return df_clean

# -----------------------------
# Feast plumbing
# -----------------------------
def export_ephemeral_snapshot(df: pd.DataFrame, repo_path: Path) -> Path:
    out_dir = repo_path / "_sqlite_export"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "features_churn_v1.parquet"
    df.to_parquet(out_path, index=False, engine="pyarrow")
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
        path=snapshot_path.as_posix(),
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
    train_fs  = FeatureService(name="training_service_v1", features=[features_v1, labels_v1])
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
    ap.add_argument("--strict-validation", action="store_true", help="Fail the run on any validation issue")
    args = ap.parse_args()

    data_repo = Path(args.data_repo).resolve()
    sqlite_path = Path(args.sqlite_db) if args.sqlite_db else (data_repo / "transformed-data" / "transformed.sqlite")
    repo_path = Path(args.repo_path).resolve()

    # 1) Load from SQLite
    df = load_from_sqlite(sqlite_path, args.sqlite_table)

    # 2) Soft validation (or strict if flag set)
    df = clean_and_report(df, strict=args.strict_validation, max_examples=8)

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
