"""Data drift (Population Stability Index) and live performance metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-4


def psi(expected, actual) -> float:
    e = np.clip(np.asarray(expected, dtype=float), EPS, None)
    a = np.clip(np.asarray(actual, dtype=float), EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def feature_psi(ref: dict, values: pd.Series) -> float | None:
    values = values.dropna()
    if values.empty:
        return None
    if ref["kind"] == "numeric":
        inner = np.asarray(ref["inner_edges"])
        vals = pd.to_numeric(values, errors="coerce").dropna().to_numpy()
        if len(vals) == 0:
            return None
        counts = np.bincount(np.searchsorted(inner, vals, side="right"), minlength=len(inner) + 1)
        return psi(ref["proportions"], counts / counts.sum())
    props = values.astype(str).value_counts(normalize=True).to_dict()
    cats = sorted(set(ref["proportions"]) | set(props))
    return psi([ref["proportions"].get(c, 0.0) for c in cats], [props.get(c, 0.0) for c in cats])


def drift_report(reference: dict, recent: pd.DataFrame, threshold: float, min_samples: int) -> dict:
    if len(recent) < min_samples:
        return {"status": "insufficient_data", "samples": int(len(recent)), "required": min_samples,
                "drifted": False, "features": []}
    feats = []
    for col, ref in reference.items():
        if col not in recent.columns:
            continue
        score = feature_psi(ref, recent[col])
        if score is None:
            continue
        level = "drift" if score >= threshold else ("moderate" if score >= 0.1 else "stable")
        feats.append({"feature": col, "psi": round(score, 4), "level": level})
    drifted = [f["feature"] for f in feats if f["level"] == "drift"]
    return {"status": "drift" if drifted else "stable", "samples": int(len(recent)),
            "required": min_samples, "drifted": bool(drifted), "drifted_features": drifted,
            "threshold": threshold, "features": feats}
