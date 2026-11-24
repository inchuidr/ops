"""MLflow experiment tracking. Never pretends: returns the true status of every call."""
from __future__ import annotations

from .config import get_settings


def log_training_run(model_name: str, algorithm: str, params: dict, metrics: dict,
                     pipeline, tags: dict) -> dict:
    s = get_settings()
    if not s.mlflow_enabled:
        return {"run_id": None, "status": "disabled", "message": "MLFLOW_ENABLED=false"}
    try:
        import mlflow
        import mlflow.sklearn
        from mlflow.tracking import MlflowClient
    except ImportError:
        return {"run_id": None, "status": "unavailable", "message": "mlflow is not installed"}
    try:
        mlflow.set_tracking_uri(s.mlflow_uri)
        client = MlflowClient()
        exp = client.get_experiment_by_name(s.experiment)
        if exp is None:
            loc = None if s.mlflow_uri.startswith("http") else (s.home / "mlartifacts").resolve().as_uri()
            exp_id = client.create_experiment(s.experiment, artifact_location=loc)
        else:
            exp_id = exp.experiment_id
        with mlflow.start_run(experiment_id=exp_id, run_name=f"{model_name}-{algorithm}") as run:
            mlflow.log_params({k: str(v) for k, v in params.items()})
            mlflow.log_metrics({k: v for k, v in metrics.items()
                                if isinstance(v, (int, float)) and not isinstance(v, bool)})
            mlflow.set_tags({"model_name": model_name, "algorithm": algorithm,
                             **{k: str(v) for k, v in tags.items()}})
            mlflow.sklearn.log_model(
    pipeline,
    "model",
    serialization_format="skops",
    skops_trusted_types=[
        "numpy.dtype",
        "sklearn.tree._tree.Tree"
    ]
)
            run_id = run.info.run_id
    except Exception as exc:
        return {"run_id": None, "status": "error", "message": f"MLflow logging failed: {exc}"}
    try:
        mv = mlflow.register_model(f"runs:/{run_id}/model", model_name)
        return {"run_id": run_id, "status": "logged", "message": f"registered as MLflow v{mv.version}"}
    except Exception as exc:
        return {"run_id": run_id, "status": "logged", "message": f"run logged; registry failed: {exc}"}
