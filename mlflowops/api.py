"""FastAPI backend. Thin layer over services.py."""
from __future__ import annotations

import hmac

from fastapi import APIRouter, Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from . import __version__, services
from .config import get_settings
from .errors import ServiceError


def require_key(x_api_key: str | None = Header(default=None)) -> None:
    key = get_settings().api_key
    if key and not hmac.compare_digest(x_api_key or "", key):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


app = FastAPI(title="MLFlowOps API", version=__version__)
router = APIRouter(dependencies=[Depends(require_key)])


@app.exception_handler(ServiceError)
async def service_error_handler(_: Request, exc: ServiceError):
    return JSONResponse(status_code=exc.status, content={"detail": exc.message})


def _public(ds: dict) -> dict:
    """Never expose server filesystem paths to clients."""
    return {k: v for k, v in ds.items() if k != "path"}


class TrainRequest(BaseModel):
    dataset_id: int
    target: str = Field(min_length=1)
    model_name: str
    algorithm: str = "random_forest"
    params: dict = Field(default_factory=dict)
    test_size: float = 0.2
    notes: str = ""


class PredictRequest(BaseModel):
    model_name: str
    records: list[dict] = Field(min_length=1, max_length=1000)


class FeedbackRequest(BaseModel):
    prediction_id: int
    actual: str


class RetrainRequest(BaseModel):
    force: bool = False
    auto_deploy: bool = True


@app.get("/health")
def health():
    return {"status": "ok", "version": __version__}


@router.get("/summary")
def summary():
    return services.summary()


@router.post("/datasets", status_code=201)
def upload_dataset(file: UploadFile = File(...), name: str | None = Form(default=None)):
    limit = get_settings().max_upload_mb * 1024 * 1024
    content = file.file.read(limit + 1)
    return _public(services.upload_dataset(file.filename or "", content, name))


@router.get("/datasets")
def datasets():
    return services.list_datasets()


@router.get("/datasets/{ds_id}")
def dataset(ds_id: int):
    return _public(services.get_dataset(ds_id))


@router.post("/train", status_code=201)
def train(req: TrainRequest):
    return services.train_model(req.dataset_id, req.target, req.algorithm, req.model_name,
                                req.params, req.test_size, req.notes)


@router.get("/models")
def models():
    return services.list_models()


@router.get("/models/{name}/versions")
def versions(name: str):
    return services.list_versions(name, light=False)


@router.post("/models/{name}/versions/{version}/deploy")
def deploy(name: str, version: int):
    return services.deploy_version(name, version)


@router.post("/predict")
def predict(req: PredictRequest):
    return services.predict(req.model_name, req.records)


@router.post("/feedback")
def feedback(req: FeedbackRequest):
    return services.submit_feedback(req.prediction_id, req.actual)


@router.get("/models/{name}/predictions")
def predictions(name: str, limit: int = 100):
    return services.list_predictions(name, min(max(limit, 1), 1000))


@router.get("/models/{name}/monitoring")
def monitoring(name: str):
    return services.monitoring_report(name)


@router.post("/models/{name}/retrain")
def retrain(name: str, req: RetrainRequest = RetrainRequest()):
    return services.retrain(name, req.force, req.auto_deploy)


@router.get("/deployments")
def deployments(limit: int = 200):
    return services.list_deployments(limit)


@router.get("/logs")
def logs(limit: int = 300):
    return services.list_logs(limit)


app.include_router(router)
