"""All configuration comes from environment variables (see .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    home: Path
    api_key: str
    mlflow_enabled: bool
    mlflow_uri: str
    experiment: str
    max_upload_mb: int
    psi_threshold: float
    perf_drop_threshold: float
    min_samples: int
    min_feedback: int

    @property
    def db_path(self) -> Path:
        return self.home / "mlflowops.db"

    @property
    def datasets_dir(self) -> Path:
        return self.home / "datasets"

    @property
    def models_dir(self) -> Path:
        return self.home / "models"


def get_settings() -> Settings:
    """Read settings fresh each call so tests can change the environment."""
    home = Path(os.getenv("MLFLOWOPS_HOME", ".mlflowops")).resolve()
    return Settings(
        home=home,
        api_key=os.getenv("API_KEY", ""),
        mlflow_enabled=_bool(os.getenv("MLFLOW_ENABLED", "true")),
        mlflow_uri=os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{home / 'mlflow.db'}"),
        experiment=os.getenv("MLFLOW_EXPERIMENT", "mlflowops"),
        max_upload_mb=int(os.getenv("MAX_UPLOAD_MB", "25")),
        psi_threshold=float(os.getenv("DRIFT_PSI_THRESHOLD", "0.2")),
        perf_drop_threshold=float(os.getenv("PERF_DROP_THRESHOLD", "0.05")),
        min_samples=int(os.getenv("MIN_DRIFT_SAMPLES", "30")),
        min_feedback=int(os.getenv("MIN_FEEDBACK_SAMPLES", "20")),
    )
