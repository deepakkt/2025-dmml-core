# Feature Catalog

| Feature | Version | Type | Offline | Online | Description |
|---|---|---|---|---|---|
| `plan_0` | v1 | INT64 | Y | Y | One-hot plan = Basic |
| `plan_1` | v1 | INT64 | Y | Y | One-hot plan = Pro |
| `plan_2` | v1 | INT64 | Y | Y | One-hot plan = Enterprise |
| `tenure_days` | v1 | INT64 | Y | Y | Days since signup |
| `days_since_last_login` | v1 | INT64 | Y | Y | Recency of login in days |
| `last_login_missing` | v1 | INT64 | Y | Y | 1 if never logged in / unknown |
| `inactive_30d` | v1 | INT64 | Y | Y | No activity in last 30 days |
| `inactive_90d` | v1 | INT64 | Y | Y | No activity in last 90 days |
| `new_user_30d` | v1 | INT64 | Y | Y | Signed up in last 30 days |
| `tickets_per_30d` | v1 | FLOAT | Y | Y | Support tickets per 30d |
| `session_hours` | v1 | FLOAT | Y | Y | Total session hours per 30d |
| `email_open_rate_30d` | v1 | FLOAT | Y | Y | Email open rate (0-1) over 30d |
| `asof_month_sin` | v1 | FLOAT | Y | Y | Seasonality (month sin) |
| `asof_month_cos` | v1 | FLOAT | Y | Y | Seasonality (month cos) |
| `auto_renew_enabled` | v1 | INT64 | Y | Y | Auto-renew is enabled |
| `auto_renew_off` | v1 | INT64 | Y | Y | Auto-renew is disabled |
| `auto_renew_off_and_inactive_30d` | v1 | INT64 | Y | Y | Disabled + inactive 30d |
| `tenure_x_auto_renew_off` | v1 | FLOAT | Y | Y | Tenure * Auto-renew off |
| `inactivity_x_email` | v1 | FLOAT | Y | Y | Inactive 30d * email open rate |
| `tickets_x_recency` | v1 | FLOAT | Y | Y | Tickets * (1 / (1+days since login)) |
| `delta_email_opens_30d` | v1 | FLOAT | Y | Y | Δ email open rate (m/m) |
| `delta_session_hours` | v1 | FLOAT | Y | Y | Δ session hours (m/m) |
| `delta_tickets_30d` | v1 | FLOAT | Y | Y | Δ tickets per 30d (m/m) |
| `rollmean_email_3` | v1 | FLOAT | Y | Y | 3-pt rolling mean email open rate |
| `rollmean_session_3` | v1 | FLOAT | Y | Y | 3-pt rolling mean session hours |
| `rollmean_tickets_3` | v1 | FLOAT | Y | Y | 3-pt rolling mean tickets |
| `auto_renew_enabled_missing` | v1 | INT64 | Y | Y | Missingness indicator for auto_renew_enabled |
| `monthly_spend` | v1 | FLOAT | Y | Y | Monthly spend in account currency |
| `churned` | v1 | INT64 | Y | N | Binary label: 1 if churned |