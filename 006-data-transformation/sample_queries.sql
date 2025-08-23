select avg(features_churn_v1.monthly_spend) from main.features_churn_v1 where
features_churn_v1.churned = 1;

select avg(features_churn_v1.tickets_per_30d) from main.features_churn_v1
where tenure_days > 30;

select features_churn_v1.customer_id, count(*)
from features_churn_v1 group by 1 having
count(*) > 1;