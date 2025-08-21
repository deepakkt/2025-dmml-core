#!/usr/bin/env python3
"""
Data Validation Reporter
- Scans <data-repo>/raw-data/**/data.json
- If sibling data_cleansing_report.csv exists, skip
- Else validate JSON + row-level rules and write CSV next to data.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Iterable, List, Tuple
import pandas as pd

ALLOWED_SUBSCRIPTION = {"Basic", "Pro", "Enterprise"}
ALLOWED_CONTRACT = {"Monthly", "Quarterly", "Annual"}

CSV_NAME = "data_cleansing_report.csv"
CSV_COLS = ["row_number", "column_name", "error"]  # 1-based row_number

# --- helpers -----------------------------------------------------------------

def is_blank(x) -> bool:
    if x is None:
        return True
    if isinstance(x, str) and x.strip() == "":
        return True
    return False

def find_key(d: Dict[str, Any], candidates: Iterable[str]) -> Tuple[str, Any]:
    """Return (key, value) for the first candidate present in dict d (case-insensitive)."""
    lower_map = {k.lower(): k for k in d.keys()}
    for c in candidates:
        if c.lower() in lower_map:
            k = lower_map[c.lower()]
            return k, d[k]
    return "", None

def ensure_list(obj: Any) -> List[Any]:
    if isinstance(obj, list):
        return obj
    raise ValueError("Top-level JSON is not an array of objects")

def write_report(rows: List[Dict[str, Any]], out_csv: Path) -> None:
    df = pd.DataFrame(rows, columns=CSV_COLS)
    # Always write a CSV (even if empty) so future runs will skip processed folders
    df.to_csv(out_csv, index=False)

def determine_kind_from_path(p: Path) -> str:
    parts = [part.lower() for part in p.parts]
    if "engagement" in parts:
        return "engagement"
    if "satisfaction" in parts:
        return "satisfaction"
    return "unknown"

# --- validation rules ---------------------------------------------------------

def validate_engagement(row: Dict[str, Any], idx1: int) -> List[Dict[str, Any]]:
    errs = []

    # customer id cannot be blank
    cust_key, cust_val = find_key(row, ["customer_id", "customer id", "customerId", "cust_id", "id"])
    if is_blank(cust_val):
        errs.append({"row_number": idx1, "column_name": cust_key or "customer_id", "error": "customer id cannot be blank"})

    # subscription plan cannot be blank and must be one of allowed
    plan_key, plan_val = find_key(row, ["subscription_plan", "subscription", "plan", "tier"])
    if is_blank(plan_val):
        errs.append({"row_number": idx1, "column_name": plan_key or "subscription_plan", "error": "subscription plan cannot be blank"})
    else:
        try_val = str(plan_val).strip()
        if try_val not in ALLOWED_SUBSCRIPTION:
            errs.append({"row_number": idx1, "column_name": plan_key or "subscription_plan", "error": f"subscription plan must be one of {sorted(ALLOWED_SUBSCRIPTION)}"})

    # avg_session_length_minutes cannot be blank; must be zero if no logins
    avg_key, avg_val = find_key(row, ["avg_session_length_minutes", "avg_session_minutes", "avg_session_length", "avg_session"])
    if is_blank(avg_val):
        errs.append({"row_number": idx1, "column_name": avg_key or "avg_session_length_minutes", "error": "avg_session_length_minutes cannot be blank"})
    else:
        # numeric check
        try:
            avg_num = float(avg_val)
        except Exception:
            errs.append({"row_number": idx1, "column_name": avg_key or "avg_session_length_minutes", "error": "avg_session_length_minutes must be numeric"})
            avg_num = None

        # infer "has not logged in"
        login_count_key, login_count_val = find_key(row, ["login_count", "logins", "session_count", "total_sessions"])
        last_login_key, last_login_val = find_key(row, ["last_login_at", "last_login", "last_seen_at"])

        no_login = False
        if login_count_key:
            try:
                no_login = (int(login_count_val) == 0)
            except Exception:
                pass
        elif not last_login_val:  # missing/blank last login
            no_login = True

        if no_login and avg_num is not None and avg_num != 0:
            errs.append({"row_number": idx1, "column_name": avg_key or "avg_session_length_minutes", "error": "avg_session_length_minutes must be 0 when customer has not logged in"})

    return errs

def validate_satisfaction(row: Dict[str, Any], idx1: int) -> List[Dict[str, Any]]:
    errs = []

    # customer id cannot be blank
    cust_key, cust_val = find_key(row, ["customer_id", "customer id", "customerId", "cust_id", "id"])
    if is_blank(cust_val):
        errs.append({"row_number": idx1, "column_name": cust_key or "customer_id", "error": "customer id cannot be blank"})

    # contract_type cannot be blank; must be one of allowed
    c_key, c_val = find_key(row, ["contract_type", "contract", "billing_cycle", "term"])
    if is_blank(c_val):
        errs.append({"row_number": idx1, "column_name": c_key or "contract_type", "error": "contract_type cannot be blank"})
    else:
        try_val = str(c_val).strip()
        if try_val not in ALLOWED_CONTRACT:
            errs.append({"row_number": idx1, "column_name": c_key or "contract_type", "error": f"contract_type must be one of {sorted(ALLOWED_CONTRACT)}"})

    return errs

# --- main --------------------------------------------------------------------

def process_data_json(data_json: Path) -> int:
    """
    Returns the number of error rows written to CSV (0 OK, >=0 written).
    """
    out_csv = data_json.with_name(CSV_NAME)
    if out_csv.exists():
        print(f"[SKIP] {out_csv} already exists")
        return 0

    # Validate JSON
    rows: List[Dict[str, Any]] = []
    try:
        with data_json.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        items = ensure_list(payload)
    except Exception as e:
        rows.append({"row_number": 0, "column_name": "__file__", "error": f"Invalid JSON: {e}"})
        write_report(rows, out_csv)
        print(f"[WRITE] {out_csv} (invalid JSON)")
        return len(rows)

    kind = determine_kind_from_path(data_json)
    validators = {
        "engagement": validate_engagement,
        "satisfaction": validate_satisfaction,
    }
    validate_fn = validators.get(kind)

    # Row-level validations
    for i, rec in enumerate(items, start=1):
        if not isinstance(rec, dict):
            rows.append({"row_number": i, "column_name": "__row__", "error": "row is not an object"})
            continue
        if validate_fn:
            rows.extend(validate_fn(rec, i))
        else:
            # Unknown dataset type; still produce a CSV to mark processed
            rows.append({"row_number": i, "column_name": "__dataset__", "error": f"unrecognized dataset folder: {kind}"})

    write_report(rows, out_csv)
    print(f"[WRITE] {out_csv} ({len(rows)} issue rows)")
    return len(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-repo", required=True, help="Path to checked-out 2025-dmml-data repository")
    args = ap.parse_args()

    repo = Path(args.data_repo)
    targets = sorted(repo.glob("raw-data/**/data.json"))

    if not targets:
        print(f"No data.json files found under {repo/'raw-data'}")
        return

    total_files = 0
    total_issues = 0
    for data_json in targets:
        total_files += 1
        total_issues += process_data_json(data_json)

    print(f"Processed files: {total_files}, total issue rows: {total_issues}")

if __name__ == "__main__":
    main()
