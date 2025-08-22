          #!/usr/bin/env python3
          import argparse, json, sqlite3, math, sys, logging
          from pathlib import Path
          import numpy as np
          import pandas as pd
          from dateutil import tz

          logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

          # ---------- helpers ----------
          def parse_args():
              ap = argparse.ArgumentParser()
              ap.add_argument("--spec", required=True)
              ap.add_argument("--input", required=True)   # prepared.csv
              ap.add_argument("--out-csv", required=True)
              ap.add_argument("--out-sqlite", required=True)
              ap.add_argument("--table-name", required=False, default=None)
              ap.add_argument("--profile", required=False, default=None, help="scaling profile: tree_baseline or linear_baseline")
              return ap.parse_args()

          def datediff_day(a, b):
              return (a - b).dt.days

          def month_of(d):
              return d.dt.month

          def ensure_int(x):
              return x.astype("Int64") if x.isna().any() else x.astype(int)

          def isnull(s):
              return s.isna()

          def _case_when(df, expr, env):
              # very small CASE WHEN parser: CASE WHEN <cond1> THEN <v1> WHEN <cond2> THEN <v2> ELSE <vE> END
              body = expr.strip()[len("CASE "):].rsplit(" END", 1)[0]
              parts = []
              rest = body
              choices = []
              conds = []
              default = None
              while rest.strip().startswith("WHEN"):
                  rest = rest.strip()[len("WHEN "):]
                  cond, after = rest.split(" THEN ", 1)
                  if " WHEN " in after:
                      val, rest = after.split(" WHEN ", 1)
                      rest = "WHEN " + rest
                  else:
                      # last THEN ... [ELSE ...]
                      if " ELSE " in after:
                          val, default = after.split(" ELSE ", 1)
                      else:
                          val, default = after, "None"
                      rest = ""
                  conds.append(pd_eval(df, cond.strip(), env))
                  choices.append(pd_eval(df, val.strip(), env))
              if default is None:
                  default = "None"
              default_val = pd_eval(df, default.strip(), env)
              import numpy as np
              return np.select(conds, choices, default=default_val)

          def pd_eval(df, expr, env):
              expr = expr.strip()
              if expr.upper().startswith("CASE "):
                  return _case_when(df, expr, env)
              # normalize operators and literals
              expr_py = (expr.replace(" AND ", " & ").replace(" OR ", " | ")
                              .replace("== true", "== True").replace("== false", "== False"))
              # map column names into env by bare identifiers; we allow df[col] by providing env[col]
              return eval(expr_py, {**ENV_FUNCS}, {**env})

          ENV_FUNCS = {
              "DATEDIFF_DAY": datediff_day,
              "MONTH": month_of,
              "SIN": np.sin,
              "COS": np.cos,
              "PI": (lambda: math.pi),
              "ISNULL": isnull
          }

          def run():
              args = parse_args()
              spec = json.loads(Path(args.spec).read_text())
              table_name = args.table_name or spec["export"]["table_name"]
              profile = args.profile or None

              df = pd.read_csv(args.input)
              # ensure date parsing per spec
              # build env mapping (columns become variables)
              def refresh_env():
                  return {c: df[c] for c in df.columns}

              # --- preprocess
              for stage in spec["steps"]:
                  if stage["stage"] == "preprocess":
                      for step in stage["steps"]:
                          op = step["op"]
                          if op == "parse_date":
                              col = step["col"]
                              errors = step.get("errors", "raise")
                              # keep empty/None as NaT if keep_null requested
                              df[col] = pd.to_datetime(df[col], errors=("coerce" if errors=="keep_null" else "raise"))
                          elif op == "derive" and step["name"] == "auto_renew_enabled":
                              env = refresh_env()
                              df["auto_renew_enabled"] = pd.Series(pd_eval(df, step["expr"], env)).astype("Float64")
                              # keep as int later after impute
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

              # --- row level features, interactions, trends, scaling
              def apply_steps(steps):
                  for st in steps:
                      op = st["op"]
                      if op == "derive":
                          name = st["name"]; expr = st["expr"]
                          env = refresh_env()
                          val = pd_eval(df, expr, env)
                          if st.get("type") == "int":
                              df[name] = pd.Series(val).astype(int)
                          elif st.get("type") == "float":
                              df[name] = pd.Series(val).astype(float)
                          else:
                              df[name] = val
                      elif op == "rename":
                          df.rename(columns={st["from"]: st["to"]}, inplace=True)
                      else:
                          raise NotImplementedError(op)

              # row_level_features
              for stage in spec["steps"]:
                  if stage["stage"] == "row_level_features":
                      apply_steps(stage["steps"])

              # interactions
              for stage in spec["steps"]:
                  if stage["stage"] == "interactions":
                      apply_steps(stage["steps"])

              # trend_features_optional
              multi = df.groupby("customer_id")["asof_date"].transform("count").gt(1).any()
              for stage in spec["steps"]:
                  if stage["stage"] == "trend_features_optional":
                      if stage.get("when") == "HAS_MULTIPLE_SNAPSHOTS_PER_CUSTOMER" and not multi:
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
                              df[name] = (df
                                          .groupby(group_col, sort=True)[src]
                                          .rolling(window, min_periods=minp)
                                          .mean()
                                          .reset_index(level=0, drop=True))
                          else:
                              raise NotImplementedError(op)

              # drop-after
              for col in spec.get("drop_after", []):
                  if col in df.columns:
                      df.drop(columns=[col], inplace=True)

              # scaling_and_clipping profiles
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
                      mask = True
                      if exclude_value is not None:
                          mask = df[c] != exclude_value
                      x = df.loc[mask, c]
                      x = np.log1p(x.clip(lower=0))
                      p99 = np.nanpercentile(x, 99)
                      x = np.clip(x, None, p99)
                      df.loc[mask, c] = x

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

              # final selection
              keys = spec["export"]["keys"]
              label = spec["export"]["label"]
              final_cols = keys + spec["final_feature_list"] + [label]
              missing = [c for c in final_cols if c not in df.columns]
              if missing:
                  raise RuntimeError(f"Missing expected columns in final output: {missing}")
              out = df[final_cols].copy()

              # ensure asof_date string for portability; keep also as DATE in sqlite
              if "asof_date" in out.columns:
                  out["asof_date"] = pd.to_datetime(out["asof_date"]).dt.strftime("%Y-%m-%d")

              # write CSV
              Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
              out.to_csv(args.out_csv, index=False)

              # write SQLite
              Path(args.out_sqlite).parent.mkdir(parents=True, exist_ok=True)
              conn = sqlite3.connect(args.out_sqlite)
              # We’ll store as a table with name from spec/export
              # Cast dates back to DATE via a temp frame
              out_sql = out.copy()
              out_sql["asof_date"] = pd.to_datetime(out_sql["asof_date"])
              out_sql.to_sql(table_name, conn, if_exists="replace", index=False)
              # create a unique index on keys for fast point lookups
              try:
                  cols = ",".join(spec["export"]["keys"])
                  conn.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS idx_{table_name}_keys ON {table_name} ({cols});")
              except Exception as e:
                  logging.warning("Index creation failed: %s", e)
              conn.commit(); conn.close()
              logging.info("Wrote %s and %s", args.out_csv, args.out_sqlite)

          if __name__ == "__main__":
              run()