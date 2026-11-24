"""End-to-end lifecycle through the service layer (real training, real SQLite, real artifacts)."""
import pytest

from mlflowops import services
from mlflowops.errors import ServiceError
from mlflowops.sample_data import generate

FEATS = ["tenure_months", "monthly_charges", "total_charges", "support_calls", "contract", "payment_method"]


@pytest.fixture
def dataset():
    df = generate(800)
    return services.upload_dataset("churn.csv", df.to_csv(index=False).encode(), "churn"), df


def train(ds, name="churn-model"):
    return services.train_model(ds["id"], "churn", "random_forest", name, {"n_estimators": 40})


def test_full_lifecycle(dataset):
    ds, df = dataset
    v1 = train(ds)
    assert v1["version"] == 1 and v1["stage"] == "staging"
    assert v1["mlflow_status"].startswith("disabled")  # honest about tracking state
    with pytest.raises(services.NotFound):
        services.predict("churn-model", [{}])  # nothing deployed yet

    dep = services.deploy_version("churn-model", 1)
    assert dep["status"] == "success"
    assert services.get_production("churn-model")["version"] == 1

    rows = df[FEATS].head(5).to_dict("records")
    res = services.predict("churn-model", rows)
    assert len(res["predictions"]) == 5 and all(p["label"] in ("0", "1") for p in res["predictions"])

    with pytest.raises(ServiceError):
        services.predict("churn-model", [{"tenure_months": 1}])  # missing features
    bad = dict(rows[0], monthly_charges="abc")
    with pytest.raises(ServiceError):
        services.predict("churn-model", [bad])

    v2 = train(ds)
    assert v2["version"] == 2
    services.deploy_version("churn-model", 2)
    stages = {v["version"]: v["stage"] for v in services.list_versions("churn-model")}
    assert stages == {1: "archived", 2: "production"}
    assert [d["status"] for d in services.list_deployments()] == ["success", "success"]


def test_failed_deploy_is_recorded_as_failed(dataset):
    ds, _ = dataset
    v = train(ds)
    import os
    os.remove(v["artifact_path"])
    with pytest.raises(ServiceError):
        services.deploy_version("churn-model", 1)
    dep = services.list_deployments()[0]
    assert dep["status"] == "failed"
    assert services.get_version("churn-model", 1)["stage"] == "staging"  # not promoted


def test_monitoring_drift_feedback_and_retrain(dataset):
    ds, df = dataset
    train(ds)
    services.deploy_version("churn-model", 1)
    rep = services.monitoring_report("churn-model")
    assert rep["drift"]["status"] == "insufficient_data" and rep["performance"]["status"] == "insufficient_data"
    assert services.retrain("churn-model")["triggered"] is False

    drifted = df[FEATS].head(120).copy()
    drifted["monthly_charges"] += 60
    drifted["contract"] = "Month-to-month"
    res = services.predict("churn-model", drifted.to_dict("records"))
    truth = df["churn"].head(120).astype(str).tolist()
    for p, t in zip(res["predictions"], truth):
        services.submit_feedback(p["id"], t)
    with pytest.raises(ServiceError):
        services.submit_feedback(res["predictions"][0]["id"], "maybe")

    rep = services.monitoring_report("churn-model")
    assert rep["drift"]["drifted"] is True
    assert rep["performance"]["labeled"] == 120 and "accuracy" in rep["performance"]

    out = services.retrain("churn-model", auto_deploy=True)
    assert out["triggered"] and "data_drift" in out["reasons"]
    assert out["new_version"] == 2 and out["feedback_rows_used"] == 120
    assert out["promoted"] == (out["challenger_f1"] >= out["champion_f1"])
    prod = services.get_production("churn-model")["version"]
    assert prod == (2 if out["promoted"] else 1)


def test_invalid_names_and_missing_entities(dataset):
    ds, _ = dataset
    with pytest.raises(ServiceError):
        services.train_model(ds["id"], "churn", "random_forest", "bad name!")
    with pytest.raises(services.NotFound):
        services.get_dataset(999)
    with pytest.raises(services.NotFound):
        services.deploy_version("ghost", 1)
