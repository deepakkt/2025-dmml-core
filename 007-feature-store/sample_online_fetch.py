#!/usr/bin/env python3
from feast import FeatureStore

store = FeatureStore(repo_path="feature_repo")

entities = [{"customer_id": "68a209a39fc554f31784b8e8"}]  # any existing ID
resp = store.get_online_features(
    features=["online_inference_v1:*"],
    entity_rows=entities,
).to_dict()

# NOTE: 'churned' will NOT appear here by design.
print({k: v for k, v in resp.items()})
