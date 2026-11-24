"""MLFlowOps dashboard (Streamlit). Talks to the FastAPI backend only; shows nothing it did not receive."""
import json
import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.getenv("API_KEY", "")
MLFLOW_UI = os.getenv("MLFLOW_UI_URL", "http://localhost:5000")

st.set_page_config(page_title="MLFlowOps", page_icon="🧪", layout="wide")
st.markdown("""
<style>
 .block-container{padding-top:1.5rem;max-width:1250px}
 [data-testid=stSidebar]{border-right:1px solid rgba(128,128,128,.25)}
 [data-testid=stMetric]{background:rgba(128,128,128,.08);border:1px solid rgba(128,128,128,.2);
   padding:14px 16px;border-radius:12px}
 .pill{display:inline-block;padding:2px 10px;border-radius:999px;font-size:.78rem;font-weight:600}
 .ok{background:#12b76a22;color:#12b76a}.bad{background:#f0443822;color:#f04438}
 .warn{background:#f7900922;color:#f79009}.neutral{background:#88888822}
</style>""", unsafe_allow_html=True)


def call(method: str, path: str, **kw):
    """Return (data, error). Errors are shown to the user verbatim, never swallowed."""
    try:
        r = requests.request(method, f"{API_URL}{path}", headers={"X-API-Key": API_KEY}, timeout=300, **kw)
    except requests.RequestException as exc:
        return None, f"Cannot reach API at {API_URL}: {exc}"
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        return None, f"{r.status_code}: {detail}"
    return r.json(), None


def need(data_err):
    data, err = data_err
    if err:
        st.error(err)
        st.stop()
    return data


def pill(text, kind="neutral"):
    return f'<span class="pill {kind}">{text}</span>'


# ---------------------------------------------------------------- sidebar
st.sidebar.title("🧪 MLFlowOps")
page = st.sidebar.radio("Navigation", ["Dashboard", "Datasets", "Train", "Models & Registry",
                                       "Predict", "Monitoring", "Deployments", "Logs"], label_visibility="collapsed")
health, herr = call("GET", "/health")
st.sidebar.markdown(pill("API online", "ok") if health else pill("API offline", "bad"), unsafe_allow_html=True)
st.sidebar.caption(f"Backend: {API_URL}")
st.sidebar.link_button("Open MLflow UI", MLFLOW_UI)
if herr:
    st.error(herr)
    st.stop()


def model_names():
    return [m["name"] for m in need(call("GET", "/models"))]


def deployed_names():
    return [m["name"] for m in need(call("GET", "/models")) if m["production_version"]]


# ---------------------------------------------------------------- pages
if page == "Dashboard":
    st.title("Dashboard")
    s = need(call("GET", "/summary"))
    c = st.columns(5)
    c[0].metric("Datasets", s["datasets"])
    c[1].metric("Model versions", s["model_versions"])
    c[2].metric("Deployed models", s["production_models"])
    c[3].metric("Predictions served", s["predictions"])
    c[4].metric("Deploys ok / failed", f"{s['deployments_success']} / {s['deployments_failed']}")
    st.subheader("Models")
    if not s["models"]:
        st.info("No models yet. Upload a dataset, then train a model.")
    for m in s["models"]:
        with st.container(border=True):
            top = st.columns([3, 1, 1])
            top[0].markdown(f"### {m['name']}")
            top[1].metric("Versions", m["versions"])
            if m["production_version"]:
                top[2].markdown(pill(f"v{m['production_version']} deployed", "ok"), unsafe_allow_html=True)
                pm = m["production_metrics"]
                cols = st.columns(4)
                for col, k in zip(cols, ["accuracy", "precision", "recall", "f1"]):
                    col.metric(k.capitalize(), f"{pm[k]:.3f}", help="Hold-out test set, weighted average")
            else:
                top[2].markdown(pill("not deployed", "neutral"), unsafe_allow_html=True)
    dep = need(call("GET", "/deployments", params={"limit": 8}))
    if dep:
        st.subheader("Recent deployments")
        st.dataframe(pd.DataFrame(dep)[["created_at", "name", "version", "status", "message"]],
                     hide_index=True, use_container_width=True)

elif page == "Datasets":
    st.title("Datasets")
    up = st.file_uploader("Upload a CSV dataset", type=["csv"])
    name = st.text_input("Dataset name (optional)")
    if up and st.button("Upload", type="primary"):
        d, err = call("POST", "/datasets", files={"file": (up.name, up.getvalue(), "text/csv")},
                      data={"name": name} if name else None)
        st.error(err) if err else st.success(f"Uploaded dataset #{d['id']} ({d['rows']} rows)")
    ds = need(call("GET", "/datasets"))
    if ds:
        st.dataframe(pd.DataFrame(ds), hide_index=True, use_container_width=True)
        pick = st.selectbox("Inspect dataset", [d["id"] for d in ds], format_func=lambda i: f"#{i}")
        prof = need(call("GET", f"/datasets/{pick}"))["profile"]
        c = st.columns(4)
        c[0].metric("Rows", prof["rows"]); c[1].metric("Columns", prof["cols"])
        c[2].metric("Missing cells", prof["missing_total"]); c[3].metric("Duplicate rows", prof["duplicate_rows"])
        st.dataframe(pd.DataFrame(prof["columns"]), hide_index=True, use_container_width=True)
        st.caption("Preprocessing is applied inside the model pipeline: median imputation + scaling for "
                   "numeric, most-frequent imputation + one-hot for categorical; ID-like columns are excluded.")
    else:
        st.info("No datasets yet. A sample is in data/sample_churn.csv (synthetic).")

elif page == "Train":
    st.title("Train a model")
    ds = need(call("GET", "/datasets"))
    if not ds:
        st.info("Upload a dataset first.")
        st.stop()
    with st.form("train"):
        ds_id = st.selectbox("Dataset", [d["id"] for d in ds],
                             format_func=lambda i: next(f"#{d['id']} · {d['name']}" for d in ds if d["id"] == i))
        cols = [c["name"] for c in need(call("GET", f"/datasets/{ds_id}"))["profile"]["columns"]]
        target = st.selectbox("Target column", cols, index=len(cols) - 1)
        mname = st.text_input("Model name", "churn-model")
        algo = st.selectbox("Algorithm", ["random_forest", "gradient_boosting", "logistic_regression"])
        test_size = st.slider("Test split", 0.1, 0.5, 0.2, 0.05)
        n_est = st.number_input("n_estimators (tree models)", 10, 1000, 200)
        go = st.form_submit_button("Train", type="primary")
    if go:
        params = {"n_estimators": int(n_est)} if algo != "logistic_regression" else {}
        with st.spinner("Training…"):
            v, err = call("POST", "/train", json={"dataset_id": ds_id, "target": target, "model_name": mname,
                                                  "algorithm": algo, "params": params, "test_size": test_size})
        if err:
            st.error(err)
        else:
            st.success(f"Trained {v['name']} v{v['version']}")
            m = v["metrics"]
            c = st.columns(4)
            for col, k in zip(c, ["accuracy", "precision", "recall", "f1"]):
                col.metric(k.capitalize(), f"{m[k]:.3f}")
            cm = pd.DataFrame(m["confusion_matrix"], index=[f"actual {l}" for l in m["labels"]],
                              columns=[f"pred {l}" for l in m["labels"]])
            st.write("Confusion matrix (hold-out set)")
            st.dataframe(cm)
            st.caption(f"MLflow: {v['mlflow_status']}")

elif page == "Models & Registry":
    st.title("Models & Registry")
    names = model_names()
    if not names:
        st.info("No models yet.")
        st.stop()
    name = st.selectbox("Model", names)
    vs = need(call("GET", f"/models/{name}/versions"))
    df = pd.DataFrame([{"version": v["version"], "stage": v["stage"], "algorithm": v["algorithm"],
                        **{k: round(v["metrics"][k], 4) for k in ["accuracy", "precision", "recall", "f1"]},
                        "mlflow": v["mlflow_status"], "created": v["created_at"], "notes": v["notes"]} for v in vs])
    st.dataframe(df, hide_index=True, use_container_width=True)
    st.bar_chart(df.set_index("version")[["accuracy", "precision", "recall", "f1"]])
    ver = st.selectbox("Version to deploy", [v["version"] for v in vs])
    if st.button("Deploy this version", type="primary"):
        d, err = call("POST", f"/models/{name}/versions/{ver}/deploy")
        if err:
            st.error(err) 
        else: 
            st.success(f"Deployed {name} v{ver}: {d['message']}")

elif page == "Predict":
    st.title("Prediction playground")
    names = deployed_names()
    if not names:
        st.info("Deploy a model first.")
        st.stop()
    name = st.selectbox("Deployed model", names)
    v = next(x for x in need(call("GET", f"/models/{name}/versions")) if x["stage"] == "production")
    rec = {}
    cols = st.columns(3)
    for i, f in enumerate(v["features"]):
        ref = v["reference"][f["name"]]
        with cols[i % 3]:
            if f["kind"] == "numeric":
                rec[f["name"]] = st.number_input(f["name"], value=round(ref["mean"], 2))
            else:
                rec[f["name"]] = st.selectbox(f["name"], sorted(ref["proportions"]))
    if st.button("Predict", type="primary"):
        r, err = call("POST", "/predict", json={"model_name": name, "records": [rec]})
        if err:
            st.error(err)
        else:
            p = r["predictions"][0]
            st.success(f"Prediction: **{p['label']}** (confidence {p['probability']:.1%}) · "
                       f"prediction id {p['id']} · model v{r['version']}")
    st.divider()
    st.caption("API usage")
    payload = json.dumps({"model_name": name, "records": [rec]})
    st.code(f"curl -X POST {API_URL}/predict -H 'X-API-Key: $API_KEY' "
            f"-H 'Content-Type: application/json' -d '{payload}'", language="bash")

elif page == "Monitoring":
    st.title("Monitoring")
    names = deployed_names()
    if not names:
        st.info("Deploy a model to start monitoring.")
        st.stop()
    name = st.selectbox("Deployed model", names)
    r = need(call("GET", f"/models/{name}/monitoring"))
    st.caption(f"Version {r['version']} · {r['predictions_analyzed']} recent predictions analyzed")
    perf, drift = r["performance"], r["drift"]
    c = st.columns(3)
    kind = {"healthy": "ok", "degraded": "bad", "insufficient_data": "neutral"}[perf["status"]]
    c[0].markdown("**Performance** " + pill(perf["status"].replace("_", " "), kind), unsafe_allow_html=True)
    dk = {"stable": "ok", "drift": "bad", "insufficient_data": "neutral"}[drift["status"]]
    c[1].markdown("**Data drift** " + pill(drift["status"].replace("_", " "), dk), unsafe_allow_html=True)
    if perf["status"] == "insufficient_data":
        st.info(f"Performance needs {perf['required']} labeled predictions (have {perf['labeled']}). "
                "Submit ground truth via POST /feedback.")
    else:
        m = st.columns(4)
        m[0].metric("Live accuracy", f"{perf['accuracy']:.3f}", f"{perf['accuracy'] - perf['baseline_accuracy']:+.3f} vs baseline")
        m[1].metric("Precision", f"{perf['precision']:.3f}")
        m[2].metric("Recall", f"{perf['recall']:.3f}")
        m[3].metric("F1", f"{perf['f1']:.3f}")
    if drift["status"] == "insufficient_data":
        st.info(f"Drift needs {drift['required']} predictions (have {drift['samples']}).")
    else:
        st.subheader("Feature drift (PSI)")
        st.caption(f"≥0.1 moderate, ≥{drift['threshold']} drift")
        st.bar_chart(pd.DataFrame(drift["features"]).set_index("feature")["psi"])
    if r["daily_volume"]:
        st.subheader("Prediction volume")
        st.line_chart(pd.DataFrame(r["daily_volume"]).set_index("date"))
    st.divider()
    st.subheader("Retraining")
    force = st.checkbox("Force retrain even if no degradation is detected")
    if st.button("Run retraining workflow", type="primary"):
        with st.spinner("Retraining…"):
            out, err = call("POST", f"/models/{name}/retrain", json={"force": force, "auto_deploy": True})
        if err:
            st.error(err)
        elif not out["triggered"]:
            st.info(out["message"])
        else:
            (st.success if out["promoted"] else st.warning)(
                f"{out['message']} Reasons: {', '.join(out['reasons'])}. New version v{out['new_version']}: "
                f"challenger F1 {out['challenger_f1']:.3f} vs champion {out['champion_f1']:.3f}.")

elif page == "Deployments":
    st.title("Deployment history")
    dep = need(call("GET", "/deployments"))
    if dep:
        st.dataframe(pd.DataFrame(dep), hide_index=True, use_container_width=True)
    else:
        st.info("No deployments yet.")


elif page == "Logs":
    st.title("Event log")

    logs = need(call("GET", "/logs"))

    if logs:
        logs_df = pd.DataFrame(logs)

        _ = st.dataframe(
            logs_df,
            hide_index=True,
            use_container_width=True
        )
    else:
        _ = st.info("No events yet.")
        