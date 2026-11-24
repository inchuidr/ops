# MLFlowOps

A compact MLOps platform: dataset upload → training (scikit-learn) → MLflow tracking → versioned registry →
verified deployment behind a FastAPI API → drift/performance monitoring → retraining workflow. Streamlit dashboard on top.

```
Streamlit dashboard ──HTTP──▶ FastAPI (mlflowops/api.py) ──▶ services.py ──▶ SQLite + joblib artifacts
                                                               │
                                                               └──▶ MLflow (params, metrics, model, registry)
```

| Module | File | Role |
|---|---|---|
| Data | `data.py` | CSV validation (type, size, rows, headers, target checks) and profiling |
| Training | `training.py` | Pipelines (impute/scale/one-hot), 3 algorithms, hold-out metrics |
| Tracking | `tracking.py` | MLflow run + registry; reports its real status (`logged`/`disabled`/`unavailable`/`error`) |
| Registry/Deploy | `services.py` | Versions, stages (`staging`→`production`→`archived`), deployment history |
| Monitoring | `monitoring.py` | PSI drift per feature, live accuracy/precision/recall/F1 from ground-truth feedback |
| Retraining | `services.retrain` | Trigger on degradation/drift, retrain on data + feedback, champion/challenger compare |

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env            # set API_KEY
set -a; source .env; set +a
uvicorn mlflowops.api:app --port 8000                     # terminal 1
streamlit run dashboard/app.py                            # terminal 2 -> http://localhost:8501
mlflow server --backend-store-uri "$MLFLOW_TRACKING_URI" --port 5000   # optional MLflow UI (terminal 3)
```
Then in the dashboard: **Datasets** → upload `data/sample_churn.csv` → **Train** (target `churn`) →
**Models & Registry** → Deploy → **Predict**.

Exercise monitoring with real rows from the sample generator (drift mode deliberately shifts inputs):
```bash
python scripts/send_test_traffic.py --model churn-model --n 100            # healthy traffic + feedback
python scripts/send_test_traffic.py --model churn-model --n 100 --drift    # shifted traffic -> drift alert
```
Open **Monitoring**, then **Run retraining workflow**.

## Docker
```bash
cp .env.example .env && docker compose up --build
# dashboard :8501 · API docs :8000/docs · MLflow UI :5000
```

## Deploy to a server
1. Build/push the image (CI pushes to `ghcr.io/<owner>/mlflowops` on `main`).
2. On the host: copy `docker-compose.yml` + `.env` (strong `API_KEY`), run `docker compose up -d`.
3. Put a TLS reverse proxy (Caddy/nginx) in front of ports 8000/8501; don't expose 5000 publicly (MLflow has no auth).
4. Back up the `mlflowops-data` volume (SQLite DB, datasets, model artifacts).

## API
Auth: `X-API-Key` header (when `API_KEY` is set). Interactive docs at `/docs`.
`POST /datasets` · `POST /train` · `GET /models` · `POST /models/{name}/versions/{v}/deploy` · `POST /predict` ·
`POST /feedback` · `GET /models/{name}/monitoring` · `POST /models/{name}/retrain` · `GET /deployments` · `GET /logs`

## Honesty guarantees (by design)
- Metrics are computed on a held-out split with scikit-learn; tests recompute them independently.
- A deployment is recorded `success` only after the artifact loads **and** a smoke prediction passes; otherwise it is recorded `failed`, the version is not promoted, and the API returns an error.
- MLflow status is stored per version; if MLflow is off/missing/failing, that is shown, not hidden.
- Live performance is `insufficient_data` until enough labeled feedback exists; drift likewise until enough predictions.
- "Deployed" means the version is in the `production` stage and served by this API process (`/predict`). It is not an auto-provisioned cloud endpoint.

## Tests / CI
`pytest -q` (unit + end-to-end lifecycle + API tests). GitHub Actions runs ruff, pytest, builds the image, boots it and checks `/health`.

## Known limitations
Classification only; single-node SQLite (swap `db.connect()` for PostgreSQL to scale out); drift uses PSI only;
the bundled dataset is synthetic; the champion-vs-challenger comparison reuses the same hold-out split, which can favor the champion slightly; models are trusted artifacts (joblib) — don't load files from untrusted sources.
