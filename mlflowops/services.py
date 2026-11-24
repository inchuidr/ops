"""Business logic: datasets, training, registry, deployment, prediction, monitoring, retraining."""
from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import data as data_mod
from . import monitoring, tracking, training
from .config import get_settings
from .db import connect, log_event, now, query
from .errors import NotFound, ServiceError

NAME_RE = re.compile(r"^[A-Za-z0-9_-]{2,50}$")
_cache: dict[tuple[str, int], object] = {}
_JSON_COLS = {"params_json": "params", "metrics_json": "metrics", "reference_json": "reference",
              "features_json": "features", "classes_json": "classes", "excluded_json": "excluded"}


# ---------------------------------------------------------------- datasets
def upload_dataset(filename: str, content: bytes, name: str | None = None) -> dict:
    s = get_settings()
    df = data_mod.parse_csv(content, filename, s.max_upload_mb)
    prof = data_mod.profile(df)
    s.datasets_dir.mkdir(parents=True, exist_ok=True)
    path = s.datasets_dir / f"{uuid.uuid4().hex}.csv"
    df.to_csv(path, index=False)
    label = (name or Path(filename).stem)[:80]
    with connect() as c:
        cur = c.execute("INSERT INTO datasets(name,path,rows,cols,profile_json,created_at) VALUES(?,?,?,?,?,?)",
                        (label, str(path), prof["rows"], prof["cols"], json.dumps(prof), now()))
        ds_id = cur.lastrowid
    log_event("INFO", "dataset.upload", f"dataset {ds_id} '{label}' ({prof['rows']} rows)")
    return get_dataset(ds_id)


def list_datasets() -> list[dict]:
    rows = query("SELECT id,name,rows,cols,created_at FROM datasets ORDER BY id DESC")
    return rows


def get_dataset(ds_id: int) -> dict:
    rows = query("SELECT * FROM datasets WHERE id=?", (ds_id,))
    if not rows:
        raise NotFound(f"Dataset {ds_id} not found.")
    r = rows[0]
    r["profile"] = json.loads(r.pop("profile_json"))
    return r


def _load_dataset_df(ds: dict) -> pd.DataFrame:
    try:
        return pd.read_csv(ds["path"])
    except Exception as exc:
        raise ServiceError(f"Dataset file unreadable: {exc}", 500) from exc


# ---------------------------------------------------------------- registry
def _hydrate(row: dict) -> dict:
    for col, key in _JSON_COLS.items():
        row[key] = json.loads(row.pop(col)) if row.get(col) else None
    return row


def list_versions(name: str | None = None, light: bool = True) -> list[dict]:
    rows = query("SELECT * FROM model_versions" + (" WHERE name=?" if name else "")
                 + " ORDER BY name, version DESC", (name,) if name else ())
    out = [_hydrate(r) for r in rows]
    if light:
        for r in out:
            r.pop("reference", None)
    return out


def get_version(name: str, version: int) -> dict:
    rows = query("SELECT * FROM model_versions WHERE name=? AND version=?", (name, version))
    if not rows:
        raise NotFound(f"Model {name} v{version} not found.")
    return _hydrate(rows[0])


def get_production(name: str) -> dict:
    rows = query("SELECT * FROM model_versions WHERE name=? AND stage='production'", (name,))
    if not rows:
        raise NotFound(f"No deployed (production) version for model '{name}'.")
    return _hydrate(rows[0])


def list_models() -> list[dict]:
    out = []
    for name in sorted({v["name"] for v in list_versions()}):
        vs = list_versions(name)
        prod = next((v for v in vs if v["stage"] == "production"), None)
        out.append({"name": name, "versions": len(vs), "latest_version": vs[0]["version"],
                    "production_version": prod["version"] if prod else None,
                    "production_metrics": prod["metrics"] if prod else None})
    return out


def train_model(dataset_id: int, target: str, algorithm: str, model_name: str,
                params: dict | None = None, test_size: float = 0.2, notes: str = "",
                df: pd.DataFrame | None = None, features: list[str] | None = None,
                parent_version: int | None = None) -> dict:
    if not NAME_RE.match(model_name or ""):
        raise ServiceError("model_name must be 2-50 chars: letters, digits, '_' or '-'.", 422)
    s = get_settings()
    ds = get_dataset(dataset_id)
    frame = df if df is not None else _load_dataset_df(ds)
    result = training.train(frame, target, algorithm, params, test_size, features=features)

    existing = query("SELECT MAX(version) AS v FROM model_versions WHERE name=?", (model_name,))[0]["v"]
    version = (existing or 0) + 1
    art_dir = s.models_dir / model_name / f"v{version}"
    art_dir.mkdir(parents=True, exist_ok=True)
    art = art_dir / "model.joblib"
    joblib.dump(result["pipeline"], art)

    run = tracking.log_training_run(model_name, algorithm, result["params"], result["metrics"],
                                    result["pipeline"], {"dataset_id": dataset_id, "version": version})
    with connect() as c:
        c.execute(
            "INSERT INTO model_versions(name,version,dataset_id,target,algorithm,params_json,metrics_json,"
            "reference_json,features_json,classes_json,excluded_json,artifact_path,mlflow_run_id,"
            "mlflow_status,stage,parent_version,notes,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (model_name, version, dataset_id, target, algorithm, json.dumps(result["params"]),
             json.dumps(result["metrics"]), json.dumps(result["reference"]), json.dumps(result["features"]),
             json.dumps(result["classes"]), json.dumps(result["excluded"]), str(art), run["run_id"],
             f"{run['status']}: {run['message']}", "staging", parent_version, notes[:500], now()))
    log_event("INFO", "model.train", f"{model_name} v{version} ({algorithm}) "
              f"acc={result['metrics']['accuracy']:.4f}; mlflow={run['status']}")
    return get_version(model_name, version)


# ---------------------------------------------------------------- deployment
def _load_model(v: dict):
    key = (v["name"], v["version"])
    if key not in _cache:
        _cache[key] = joblib.load(v["artifact_path"])
    return _cache[key]


def _reference_row(v: dict) -> pd.DataFrame:
    row = {}
    for f in v["features"]:
        ref = v["reference"][f["name"]]
        row[f["name"]] = ref["mean"] if ref["kind"] == "numeric" else max(ref["proportions"], key=ref["proportions"].get)
    return pd.DataFrame([row])


def deploy_version(name: str, version: int) -> dict:
    """Mark as production ONLY after the artifact loads and a smoke prediction succeeds."""
    v = get_version(name, version)
    try:
        _cache.pop((name, version), None)
        model = _load_model(v)
        pred = model.predict(_reference_row(v))
        if str(pred[0]) not in v["classes"]:
            raise RuntimeError("smoke prediction returned an unknown class")
    except Exception as exc:
        msg = f"Smoke test failed: {exc}"
        with connect() as c:
            c.execute("INSERT INTO deployments(name,version,status,message,created_at) VALUES(?,?,?,?,?)",
                      (name, version, "failed", msg, now()))
        log_event("ERROR", "model.deploy", f"{name} v{version} deployment FAILED: {exc}")
        raise ServiceError(f"Deployment failed. {msg}", 500) from exc
    with connect() as c:
        c.execute("UPDATE model_versions SET stage='archived' WHERE name=? AND stage='production'", (name,))
        c.execute("UPDATE model_versions SET stage='production' WHERE name=? AND version=?", (name, version))
        cur = c.execute("INSERT INTO deployments(name,version,status,message,created_at) VALUES(?,?,?,?,?)",
                        (name, version, "success", "Artifact loaded and smoke prediction passed", now()))
        dep_id = cur.lastrowid
    log_event("INFO", "model.deploy", f"{name} v{version} is now serving")
    return query("SELECT * FROM deployments WHERE id=?", (dep_id,))[0]


def list_deployments(limit: int = 200) -> list[dict]:
    return query("SELECT * FROM deployments ORDER BY id DESC LIMIT ?", (limit,))


def list_logs(limit: int = 300) -> list[dict]:
    return query("SELECT * FROM logs ORDER BY id DESC LIMIT ?", (limit,))


# ---------------------------------------------------------------- prediction
def _prepare(records: list[dict], v: dict) -> pd.DataFrame:
    if not records:
        raise ServiceError("'records' must contain at least one item.", 422)
    if len(records) > 1000:
        raise ServiceError("At most 1000 records per request.", 422)
    df = pd.DataFrame(records)
    names = [f["name"] for f in v["features"]]
    missing = [n for n in names if n not in df.columns]
    if missing:
        raise ServiceError(f"Missing required feature(s): {missing}", 422)
    df = df[names].copy()
    for f in v["features"]:
        c = f["name"]
        if f["kind"] == "numeric":
            try:
                df[c] = pd.to_numeric(df[c])
            except (ValueError, TypeError):
                raise ServiceError(f"Feature '{c}' must be numeric.", 422)
        else:
            df[c] = df[c].astype(object).where(df[c].notna(), np.nan)
    return df


def predict(name: str, records: list[dict], log: bool = True) -> dict:
    v = get_production(name)
    df = _prepare(records, v)
    try:
        model = _load_model(v)
        labels = model.predict(df)
        probs = model.predict_proba(df).max(axis=1)
    except Exception as exc:
        log_event("ERROR", "predict", f"{name} v{v['version']}: {exc}")
        raise ServiceError(f"Prediction failed: {exc}", 500) from exc
    ids = []
    if log:
        with connect() as c:
            for rec, lab, pr in zip(df.to_dict("records"), labels, probs):
                clean = {k: (None if isinstance(x, float) and np.isnan(x) else x) for k, x in rec.items()}
                cur = c.execute("INSERT INTO predictions(name,version,input_json,prediction,probability,created_at)"
                                " VALUES(?,?,?,?,?,?)",
                                (name, v["version"], json.dumps(clean, default=str), str(lab), float(pr), now()))
                ids.append(cur.lastrowid)
    return {"model": name, "version": v["version"],
            "predictions": [{"id": ids[i] if ids else None, "label": str(labels[i]),
                             "probability": round(float(probs[i]), 4)} for i in range(len(labels))]}


def submit_feedback(prediction_id: int, actual: str) -> dict:
    rows = query("SELECT * FROM predictions WHERE id=?", (prediction_id,))
    if not rows:
        raise NotFound(f"Prediction {prediction_id} not found.")
    v = get_version(rows[0]["name"], rows[0]["version"])
    if str(actual) not in v["classes"]:
        raise ServiceError(f"'actual' must be one of {v['classes']}.", 422)
    with connect() as c:
        c.execute("UPDATE predictions SET actual=? WHERE id=?", (str(actual), prediction_id))
    return {"id": prediction_id, "actual": str(actual)}


def list_predictions(name: str, limit: int = 100) -> list[dict]:
    rows = query("SELECT * FROM predictions WHERE name=? ORDER BY id DESC LIMIT ?", (name, limit))
    for r in rows:
        r["input"] = json.loads(r.pop("input_json"))
    return rows


# ---------------------------------------------------------------- monitoring
def monitoring_report(name: str) -> dict:
    s = get_settings()
    v = get_production(name)
    rows = query("SELECT * FROM predictions WHERE name=? AND version=? ORDER BY id DESC LIMIT 1000",
                 (name, v["version"]))
    recent = pd.DataFrame([json.loads(r["input_json"]) for r in rows]) if rows else pd.DataFrame()
    drift = monitoring.drift_report(v["reference"], recent, s.psi_threshold, s.min_samples)

    labeled = [r for r in rows if r["actual"] is not None]
    baseline = v["metrics"]["accuracy"]
    if len(labeled) < s.min_feedback:
        perf = {"status": "insufficient_data", "labeled": len(labeled), "required": s.min_feedback,
                "baseline_accuracy": baseline, "degraded": False}
    else:
        m = training.compute_metrics([r["actual"] for r in labeled], [r["prediction"] for r in labeled])
        degraded = m["accuracy"] < baseline - s.perf_drop_threshold
        perf = {"status": "degraded" if degraded else "healthy", "labeled": len(labeled),
                "accuracy": m["accuracy"], "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
                "baseline_accuracy": baseline, "degraded": degraded,
                "drop_threshold": s.perf_drop_threshold}
    daily: dict[str, int] = {}
    for r in rows:
        daily[r["created_at"][:10]] = daily.get(r["created_at"][:10], 0) + 1
    return {"model": name, "version": v["version"], "predictions_analyzed": len(rows),
            "drift": drift, "performance": perf,
            "daily_volume": [{"date": d, "count": n} for d, n in sorted(daily.items())]}


# ---------------------------------------------------------------- retraining
def retrain(name: str, force: bool = False, auto_deploy: bool = True) -> dict:
    s = get_settings()
    report = monitoring_report(name)
    reasons = (["performance_degraded"] if report["performance"]["degraded"] else []) + \
              (["data_drift"] if report["drift"]["drifted"] else [])
    if force:
        reasons.append("forced")
    if not reasons:
        log_event("INFO", "model.retrain", f"{name}: no retrain needed")
        return {"triggered": False, "reasons": [], "message": "No degradation or drift detected."}

    champ = get_production(name)
    ds = get_dataset(champ["dataset_id"])
    base = _load_dataset_df(ds)
    feats = [f["name"] for f in champ["features"]]
    fb = query("SELECT input_json, actual FROM predictions WHERE name=? AND actual IS NOT NULL", (name,))
    frames = [base[feats + [champ["target"]]]]
    if fb:
        extra = pd.DataFrame([json.loads(r["input_json"]) for r in fb])[feats]
        extra[champ["target"]] = [r["actual"] for r in fb]
        frames.append(extra)
    combined = pd.concat(frames, ignore_index=True)
    new = train_model(champ["dataset_id"], champ["target"], champ["algorithm"], name,
                      params={k: v for k, v in champ["params"].items() if k in training.ALGORITHMS[champ["algorithm"]][1]},
                      notes=f"retrain ({', '.join(reasons)}); +{len(fb)} labeled feedback rows",
                      df=combined, features=feats, parent_version=champ["version"])

    # Compare champion vs challenger on the SAME holdout. (Champion may have seen part of it,
    # which is optimistic for the champion, so this comparison is conservative.)
    _, X_te, _, y_te = training.holdout(combined, champ["target"], feats, 0.2, 42)
    champ_f1 = training.compute_metrics(y_te, _load_model(champ).predict(X_te))["f1"]
    chall_f1 = new["metrics"]["f1"]
    promote = auto_deploy and chall_f1 >= champ_f1
    deployed = None
    if promote:
        deployed = deploy_version(name, new["version"])
    log_event("INFO", "model.retrain", f"{name}: v{new['version']} f1={chall_f1:.4f} vs "
              f"champion v{champ['version']} f1={champ_f1:.4f}; promoted={bool(deployed)}")
    return {"triggered": True, "reasons": reasons, "new_version": new["version"],
            "challenger_f1": chall_f1, "champion_f1": champ_f1, "promoted": bool(deployed),
            "feedback_rows_used": len(fb),
            "message": "Challenger promoted to production." if deployed else
            "Challenger registered but NOT promoted (auto_deploy off or not better than champion)."}


# ---------------------------------------------------------------- summary
def summary() -> dict:
    q = lambda sql: query(sql)[0]["n"]  # noqa: E731
    return {
        "datasets": q("SELECT COUNT(*) n FROM datasets"),
        "model_versions": q("SELECT COUNT(*) n FROM model_versions"),
        "production_models": q("SELECT COUNT(*) n FROM model_versions WHERE stage='production'"),
        "predictions": q("SELECT COUNT(*) n FROM predictions"),
        "deployments_success": q("SELECT COUNT(*) n FROM deployments WHERE status='success'"),
        "deployments_failed": q("SELECT COUNT(*) n FROM deployments WHERE status='failed'"),
        "models": list_models(),
    }
