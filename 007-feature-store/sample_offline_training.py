#!/usr/bin/env python3
from feast import FeatureStore
import pandas as pd
from pathlib import Path

# Build an entity dataframe from the latest partition
DATA_REPO = Path("data-repo")
BASE = DATA_REPO / "transformed-data"
latest = sorted([p for p in BASE.glob("dt=*") if p.is_dir()])[-1]
df = pd.read_parquet(list(latest.glob("*.parquet")), columns=["customer_id", "asof_date"])

store = FeatureStore(repo_path="feature_repo")
training_df = store.get_historical_features(
    entity_df=df.rename(columns={"asof_date": "event_timestamp"}),
    features=["training_service_v1:*"],  # includes features + offline-only labels
).to_df()

print(training_df.head())
print("Columns:", list(training_df.columns))
