import pytest


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOWOPS_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("MLFLOW_ENABLED", "false")
    monkeypatch.setenv("API_KEY", "")
    monkeypatch.setenv("MIN_DRIFT_SAMPLES", "30")
    monkeypatch.setenv("MIN_FEEDBACK_SAMPLES", "20")
    from mlflowops import services
    services._cache.clear()
