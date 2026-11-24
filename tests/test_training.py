import pytest
from sklearn.metrics import accuracy_score

from mlflowops import training
from mlflowops.errors import ServiceError
from mlflowops.sample_data import generate


@pytest.mark.parametrize("algo", list(training.ALGORITHMS))
def test_train_each_algorithm_real_metrics(algo):
    df = generate(600)
    r = training.train(df, "churn", algo, {"n_estimators": 30} if algo != "logistic_regression" else {})
    m = r["metrics"]
    assert 0.5 < m["accuracy"] <= 1 and 0 <= m["f1"] <= 1
    assert "customer_id" in r["excluded"]  # id-like column excluded
    # Recompute independently to prove metrics are not fabricated
    _, X_te, _, y_te = training.holdout(df, "churn", [f["name"] for f in r["features"]], 0.2, 42)
    assert accuracy_score(y_te, r["pipeline"].predict(X_te)) == pytest.approx(m["accuracy"])


def test_bad_params_and_algo():
    df = generate(200)
    with pytest.raises(ServiceError):
        training.train(df, "churn", "svm")
    with pytest.raises(ServiceError):
        training.train(df, "churn", "random_forest", {"evil": 1})
    with pytest.raises(ServiceError):
        training.train(df, "churn", "random_forest", {"n_estimators": 10**6})
    with pytest.raises(ServiceError):
        training.train(df, "churn", "random_forest", test_size=0.9)
