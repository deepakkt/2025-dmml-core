#!/usr/bin/env python3
import argparse, json, sqlite3, math, sys, logging, re
from pathlib import Path
import numpy as np
import pandas as pd
from dateutil import tz

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------- args ----------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)          # JSON spec path
    ap.add_argument("--input", required=True)         # prepared.csv
    ap.add_argument("--out-csv", required=True)       # transformed.csv
    ap.add_argument("--out-sqlite", required=True)    # transformed.sqlite
    ap.add_argument("--table-name", default=None)
    ap.add_argument("--profile", default=None, help="scaling profile: tree_baseline or linear_baseline")
    return ap.parse_args()

# ---------- tiny expression runtime ----------
def datediff_day(a, b):
    return (a - b).dt.days

def month_of(d):
    return d.dt.month

def isnull(s):
    return s.isna()

ENV_FUNCS = {
    "DATEDIFF_DAY": datediff_day,
    "MONTH": month_of,
    "SIN": np.sin,
    "COS": np.cos,
    "PI": (lambda: math.pi),
    "ISNULL": isnull,
}

def _normalize_expr(expr: str) -> str:
    """Make SQL-ish tokens Pythonic for eval()."""
    e = expr
    # boolean and NULL normalization (case-insensitive)
    e = re.sub(r"\btrue\b", "True", e, flags=re.I)
    e = re.sub(r"\bfalse\b", "False", e, flags=re.I)
    e = re.sub(r"\bNULL\b", "None", e, flags=re.I)
    # logical ops
    # use spaces to avoid clobbering inside identifiers
    e = re.sub(r"\bAND\b", "&", e, flags=re.I)
    e = re.sub(r"\bOR\b",  "|", e, flags=re.I)
    return e

def _case_when(df: pd.DataFrame, expr: str, env: dict):
    """
    Parse a very small CASE WHEN grammar:
      CASE WHEN <cond1> THEN <v1> WHEN <cond2> THEN <v2> ELSE <vE> END
    """
    raw = expr.strip()
    # slice out contents between CASE and END (case-insensitive)
    head = raw[:4].upper()
    if head != "CASE":
        raise ValueError("CASE expression expected")
    # normalize just enough to find END
    body = raw.strip()[len("CASE "):]
    if body.upper().endswith(" END"):
        body = body[: -len(" END")]
    # iterate WHEN ... THEN ...
    conds, choices = [], []
    rest = body
    default = None
    while rest.strip().upper().startswith("WHEN"):
        # drop leading WHEN
        rest = rest.strip()[len("WHEN "):]
        # split on THEN
        if " THEN " not in rest:
            raise ValueError("Malformed CASE: missing THEN")
        cond, after = rest.split(" THEN ", 1)
        # more WHENs?
        up_after = after.upper()
        if " WHEN " in up_after:
            # split at first ' WHEN ' (preserving subsequent WHENs)
            idx = up_after.index(" WHEN ")
            val, rest = after[:idx], after[idx+1:]  # rest starts with 'WHEN '
            rest = "W" + rest                       # restore leading 'WHEN '
        else:
            # optional ELSE
            if " ELSE " in up_after:
                idx = up_after.index(" ELSE ")
                val, default = after[:idx], after[idx+6:]
            else:
                val, default = after, "None"
            rest = ""

        conds.append(pd_eval(df, cond.strip(), env))
        choices.append(pd_eval(df, val.strip(), env))

    if default is None:
        default = "None"
    default_val = pd_eval(df, default.strip(), env)
    return np.select(conds, choices, default=default_val)

def pd_eval(df: pd.DataFrame, expr: str, env: dict):
    expr = expr.strip()
    # CASE WHEN ...
    if expr[:4].upper() == "CASE":
        return _case_when(df, expr, env)
    # normalize
    expr_py = _normalize_expr(expr)
    # evaluate with column Series available by bare name
    return eval(expr_py, {**ENV_FUNCS}, {**env})

# ---------- main pipeline ----------
def run():
    args = parse_args()
    spec = json.loads(Path(args.spec).read_text())
    table_name = args.table_name or spec["export"]["table_name"]
    profile = args.profile or None

    df = pd.read_csv(args.input)

    # function to refresh eval environment with current columns
    def refresh_env():
        return {c: df[c] for c in df.columns}

    # --- preprocess
    for stage in spec["steps"]:
        if stage["stage"] != "preprocess":
            continue
        for step in stage["steps"]:
            op = step["op"]
            if op == "parse_date":
                col = step["col"]
                errors = step.get("errors", "raise")
                df[col] = pd.to_datetime(df[col], errors=("coerce" if errors == "keep_null" else "raise"))
            elif op == "derive" and step["name"] == "auto_renew_enabled":
                env = refresh_env()
                s = pd.Series(pd_eval(df, step["expr"], env))
                # keep float until impute; cast to int post-impute
                df["auto_renew_enabled"] = s.astype("Float64")
            elif op == "flag_missing":
                on = step["on"]
                name = step["name"]
                df[name] = df[on].isna().astype(int)
            elif op == "impute":
                on = step["on"]; strategy = step["strategy"]
                if strategy == "mode":
                    mode = df[on].mode(dropna=True)
                    fill = mode.iloc[0] if len(mode) else 0
                    df[on] = df[on].fillna(fill).astype(int)
                else:
                    raise NotImplementedError("impute: " + strategy)
            elif op == "one_hot_from_int":
                src = step["source"]; pref = step["prefix"]; vals = step["values"]
                for v in vals:
                    df[f"{pref}{v}"] = (df[src] == v).astype(int)
                if step.get("drop_source", False):
                    df.drop(columns=[src], inplace=True)

    # helper to apply derive/rename steps
    def apply_steps(steps):
        for st in steps:
            op = st["op"]
            if op == "derive":
                name = st["name"]; expr = st["expr"]
                env = refresh_env()
                val = pd_eval(df, expr, env)
                t = st.get("type")
                s = pd.Series(val)
                if t == "int":
                    # keep nullable Int64 if NaNs are present
                    df[name] = s.astype("Int64") if s.isna().any() else s.astype(int)
                elif t == "float":
                    df[name] = s.astype(float)
                else:
                    df[name] = s
            elif op == "rename":
                df.rename(columns={st["from"]: st["to"]}, inplace=True)
            else:
                raise NotImplementedError(op)

    # --- row level features
    for stage in spec["steps"]:
        if stage["stage"] == "row_level_features":
            apply_steps(stage["steps"])

    # --- interactions
    for stage in spec["steps"]:
        if stage["stage"] == "interactions":
            apply_steps(stage["steps"])

    # --- optional trend features (lags, deltas, roll means)
    has_multi = df.groupby("customer_id")["asof_date"].transform("count").gt(1).any()
    for stage in spec["steps"]:
        if stage["stage"] != "trend_features_optional":
            continue
        if stage.get("when") == "HAS_MULTIPLE_SNAPSHOTS_PER_CUSTOMER" and not has_multi:
            continue
        group_col = stage["group_by"]; order_by = stage["order_by"]
        df = df.sort_values([group_col, order_by])
        for st in stage["steps"]:
            op = st["op"]
            if op == "lag":
                name = st["name"]; src = st["source"]; k = int(st["k"])
                df[name] = df.groupby(group_col, sort=True)[src].shift(k)
            elif op == "derive":
                name = st["name"]; expr = st["expr"]
                env = refresh_env()
                df[name] = pd_eval(df, expr, env)
            elif op == "rolling_mean":
                name = st["name"]; src = st["source"]
                window = int(st["window"]); minp = int(st.get("min_periods", window))
                df[name] = (df.groupby(group_col, sort=True)[src]
                              .rolling(window, min_periods=minp)
                              .mean()
                              .reset_index(level=0, drop=True))
            else:
                raise NotImplementedError(op)

    # --- drop-after
    for col in spec.get("drop_after", []):
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # --- scaling/clipping
    def clip_p99_cols(cols):
        for c in cols:
            if c in df.columns:
                low = df[c].quantile(0.01)
                high = df[c].quantile(0.99)
                df[c] = df[c].clip(lower=low, upper=high)

    def zscore_cols(cols, skip_if_in_01=False):
        for c in cols:
            if c not in df.columns:
                continue
            if skip_if_in_01 and df[c].min() >= 0 and df[c].max() <= 1:
                continue
            mu = df[c].mean()
            sd = df[c].std(ddof=0) or 1.0
            df[c] = (df[c] - mu) / sd

    def robust_log1p_then_clip(cols, exclude_value=None):
        for c in cols:
            if c not in df.columns:
                continue
            mask = df[c] != exclude_value if exclude_value is not None else pd.Series(True, index=df.index)
            x = df.loc[mask, c]
            x = np.log1p(x.clip(lower=0))
            p99 = np.nanpercentile(x, 99)
            df.loc[mask, c] = np.clip(x, None, p99)

    for stage in spec["steps"]:
        if stage["stage"] == "scaling_and_clipping" and profile:
            prof = stage.get("profiles", {}).get(profile)
            if prof:
                for opdef in prof:
                    op = opdef["op"]
                    if op == "clip_p99":
                        clip_p99_cols(opdef["cols"])
                    elif op == "zscore":
                        zscore_cols(opdef["cols"], opdef.get("skip_if_in_01", False))
                    elif op == "robust_log1p_then_clip_p99":
                        robust_log1p_then_clip(opdef["cols"], opdef.get("exclude_values_equals"))
                    else:
                        raise NotImplementedError(op)

    # --- final selection
    keys = spec["export"]["keys"]
    label = spec["export"]["label"]
    feature_list = list(spec["final_feature_list"])

    # Create any missing OPTIONAL features (e.g., trend features when no multi-snapshots) as NaN
    for c in feature_list:
        if c not in df.columns:
            df[c] = np.nan

    # keys/label must exist
    for must in keys + [label]:
        if must not in df.columns:
            raise RuntimeError(f"Missing required column '{must}' before export")

    final_cols = keys + feature_list + [label]
    out = df[final_cols].copy()

    # Normalize date string for CSV portability; keep as DATE in SQLite
    if "asof_date" in out.columns:
        out["asof_date"] = pd.to_datetime(out["asof_date"]).dt.strftime("%Y-%m-%d")

    # write CSV
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    # write SQLite
    Path(args.out_sqlite).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(args.out_sqlite)
    out_sql = out.copy()
    out_sql["asof_date"] = pd.to_datetime(out_sql["asof_date"])
    out_sql.to_sql(table_name, conn, if_exists="replace", index=False)
    # unique index on keys
    try:
        cols = ",".join(keys)
        conn.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS idx_{table_name}_keys ON {table_name} ({cols});")
    except Exception as e:
        logging.warning("Index creation failed: %s", e)
    conn.commit(); conn.close()

    logging.info("Wrote %s and %s", args.out_csv, args.out_sqlite)

if __name__ == "__main__":
    run()
