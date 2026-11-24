"""Real scikit-learn training + evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import check_target
from .errors import ServiceError

# algorithm -> (class, {param: (type, min, max)}, fixed defaults)
ALGORITHMS = {
    "logistic_regression": (LogisticRegression, {"C": (float, 1e-4, 1e3), "max_iter": (int, 50, 5000)},
                            {"max_iter": 1000}),
    "random_forest": (RandomForestClassifier, {"n_estimators": (int, 10, 1000), "max_depth": (int, 1, 50),
                                               "min_samples_leaf": (int, 1, 100)},
                      {"n_estimators": 200}),
    "gradient_boosting": (GradientBoostingClassifier, {"n_estimators": (int, 10, 1000),
                                                       "learning_rate": (float, 1e-3, 1.0),
                                                       "max_depth": (int, 1, 10)}, {}),
}


def _validate_params(algorithm: str, params: dict | None) -> dict:
    if algorithm not in ALGORITHMS:
        raise ServiceError(f"Unknown algorithm '{algorithm}'. Choose from {sorted(ALGORITHMS)}.", 422)
    allowed = ALGORITHMS[algorithm][1]
    clean = {}
    for key, value in (params or {}).items():
        if key not in allowed:
            raise ServiceError(f"Parameter '{key}' not allowed for {algorithm}. Allowed: {sorted(allowed)}.", 422)
        typ, lo, hi = allowed[key]
        try:
            value = typ(value)
        except (TypeError, ValueError):
            raise ServiceError(f"Parameter '{key}' must be {typ.__name__}.", 422)
        if not lo <= value <= hi:
            raise ServiceError(f"Parameter '{key}' must be between {lo} and {hi}.", 422)
        clean[key] = value
    return clean


def select_features(df: pd.DataFrame, target: str) -> tuple[list[str], list[str]]:
    """Return (feature columns, excluded id-like columns)."""
    feats, excluded = [], []
    for c in df.columns:
        if c == target:
            continue
        s = df[c]
        if not is_numeric_dtype(s) and s.nunique(dropna=True) > 50 and s.nunique() / len(s) > 0.9:
            excluded.append(c)
        else:
            feats.append(c)
    if not feats:
        raise ServiceError("No usable feature columns remain after excluding the target.", 422)
    return feats, excluded


def build_reference(X: pd.DataFrame) -> dict:
    """Training-time feature distributions used as the drift baseline."""
    ref = {}
    for c in X.columns:
        s = X[c].dropna()
        if is_numeric_dtype(X[c]):
            inner = np.unique(np.quantile(s, np.linspace(0, 1, 11))[1:-1]) if len(s) else np.array([])
            counts = np.bincount(np.searchsorted(inner, s.to_numpy(), side="right"), minlength=len(inner) + 1)
            ref[c] = {"kind": "numeric", "inner_edges": [float(e) for e in inner],
                      "proportions": (counts / max(counts.sum(), 1)).tolist(),
                      "mean": float(s.mean()) if len(s) else 0.0}
        else:
            props = s.astype(str).value_counts(normalize=True)
            ref[c] = {"kind": "categorical", "proportions": {k: float(v) for k, v in props.items()}}
    return ref


def compute_metrics(y_true, y_pred) -> dict:
    labels = sorted(set(map(str, y_true)) | set(map(str, y_pred)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "average": "weighted",
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def holdout(df: pd.DataFrame, target: str, features: list[str], test_size: float, seed: int):
    df = df.dropna(subset=[target])
    X, y = df[features], df[target].astype(str)
    try:
        return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    except ValueError as exc:
        raise ServiceError(f"Cannot split data: {exc}", 422) from exc


def train(df: pd.DataFrame, target: str, algorithm: str, params: dict | None = None,
          test_size: float = 0.2, seed: int = 42, features: list[str] | None = None) -> dict:
    if not 0.1 <= test_size <= 0.5:
        raise ServiceError("test_size must be between 0.1 and 0.5.", 422)
    check_target(df, target)
    clean = _validate_params(algorithm, params)
    excluded: list[str] = []
    if features is None:
        features, excluded = select_features(df, target)
    X_tr, X_te, y_tr, y_te = holdout(df, target, features, test_size, seed)

    num = [c for c in features if is_numeric_dtype(df[c])]
    cat = [c for c in features if c not in num]
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])
    cls, _, defaults = ALGORITHMS[algorithm]
    kwargs = {**defaults, **clean}
    if algorithm != "logistic_regression":
        kwargs["random_state"] = seed
    pipe = Pipeline([("pre", pre), ("clf", cls(**kwargs))])
    try:
        pipe.fit(X_tr, y_tr)
    except Exception as exc:
        raise ServiceError(f"Training failed: {exc}", 422) from exc

    metrics = compute_metrics(y_te, pipe.predict(X_te))
    metrics.update({"train_rows": int(len(X_tr)), "test_rows": int(len(X_te))})
    return {
        "pipeline": pipe, "metrics": metrics, "params": kwargs,
        "reference": build_reference(X_tr),
        "features": [{"name": c, "kind": "numeric" if c in num else "categorical"} for c in features],
        "classes": [str(c) for c in pipe.classes_], "excluded": excluded,
    }
