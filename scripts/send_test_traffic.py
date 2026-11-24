"""Send REAL rows from the sample dataset to a deployed model, with ground-truth feedback.

This is a test-traffic generator for exercising monitoring. Use --drift to deliberately shift
feature values so you can watch drift detection fire. Nothing here is shown as model metrics:
the dashboard computes everything from the predictions + feedback stored by the API.
"""
import argparse
import os

import requests

from mlflowops.sample_data import generate

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--n", type=int, default=100)
p.add_argument("--drift", action="store_true", help="shift monthly_charges/contract to simulate drift")
p.add_argument("--seed", type=int, default=7)
a = p.parse_args()

url = os.getenv("API_URL", "http://localhost:8000")
hdr = {"X-API-Key": os.getenv("API_KEY", "")}
df = generate(a.n, seed=a.seed)
feats = ["tenure_months", "monthly_charges", "total_charges", "support_calls", "contract", "payment_method"]
x = df[feats].copy()
if a.drift:
    x["monthly_charges"] += 45
    x["contract"] = "Month-to-month"
x = x.astype(object).where(x.notna(), None)
r = requests.post(f"{url}/predict", json={"model_name": a.model, "records": x.to_dict("records")},
                  headers=hdr, timeout=60)
r.raise_for_status()
preds = r.json()["predictions"]
for pr, truth in zip(preds, df["churn"].astype(str)):
    requests.post(f"{url}/feedback", json={"prediction_id": pr["id"], "actual": truth},
                  headers=hdr, timeout=30).raise_for_status()
print(f"sent {len(preds)} predictions with feedback (drift={a.drift})")
