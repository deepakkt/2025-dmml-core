create table main.features_churn_v1
(
    customer_id                     TEXT,
    asof_date                       TIMESTAMP,
    plan_0                          INTEGER,
    plan_1                          INTEGER,
    plan_2                          INTEGER,
    tenure_days                     INTEGER,
    days_since_last_login           INTEGER,
    last_login_missing              INTEGER,
    inactive_30d                    INTEGER,
    inactive_90d                    INTEGER,
    new_user_30d                    INTEGER,
    tickets_per_30d                 REAL,
    session_hours                   REAL,
    email_open_rate_30d             REAL,
    asof_month_sin                  REAL,
    asof_month_cos                  REAL,
    auto_renew_enabled              INTEGER,
    auto_renew_off                  INTEGER,
    auto_renew_off_and_inactive_30d INTEGER,
    tenure_x_auto_renew_off         REAL,
    inactivity_x_email              REAL,
    tickets_x_recency               REAL,
    delta_email_opens_30d           REAL,
    delta_session_hours             REAL,
    delta_tickets_30d               REAL,
    rollmean_email_3                REAL,
    rollmean_session_3              REAL,
    rollmean_tickets_3              REAL,
    auto_renew_enabled_missing      INTEGER,
    monthly_spend                   REAL,
    churned                         INTEGER
);

create unique index main.idx_features_churn_v1_keys
    on main.features_churn_v1 (customer_id, asof_date);

