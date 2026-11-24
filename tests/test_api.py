"""API tests. Skipped automatically if fastapi/httpx are not installed."""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("multipart")
from fastapi.testclient import TestClient  # noqa: E402

from mlflowops.api import app  # noqa: E402
from mlflowops.sample_data import generate  # noqa: E402

client = TestClient(app)
FEATS = ["tenure_months", "monthly_charges", "total_charges", "support_calls", "contract", "payment_method"]


def test_health_and_auth(monkeypatch):
    assert client.get("/health").status_code == 200
    monkeypatch.setenv("API_KEY", "secret")
    assert client.get("/models").status_code == 401
    assert client.get("/models", headers={"X-API-Key": "secret"}).status_code == 200


def test_upload_validation():
    r = client.post("/datasets", files={"file": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 415
    r = client.post("/datasets", files={"file": ("x.csv", b"", "text/csv")})
    assert r.status_code == 400


def test_end_to_end_over_http():
    df = generate(600)
    r = client.post("/datasets", files={"file": ("c.csv", df.to_csv(index=False).encode(), "text/csv")})
    assert r.status_code == 201
    ds_id = r.json()["id"]
    r = client.post("/train", json={"dataset_id": ds_id, "target": "churn", "model_name": "m1",
                                    "algorithm": "random_forest", "params": {"n_estimators": 30}})
    assert r.status_code == 201 and 0 < r.json()["metrics"]["accuracy"] <= 1
    assert client.post("/predict", json={"model_name": "m1", "records": [{}]}).status_code == 404
    assert client.post("/models/m1/versions/1/deploy").json()["status"] == "success"
    r = client.post("/predict", json={"model_name": "m1", "records": df[FEATS].head(3).to_dict("records")})
    assert r.status_code == 200 and len(r.json()["predictions"]) == 3
    assert client.post("/predict", json={"model_name": "m1", "records": [{"x": 1}]}).status_code == 422
    assert client.get("/models/m1/monitoring").status_code == 200
    assert client.get("/deployments").json()[0]["status"] == "success"
