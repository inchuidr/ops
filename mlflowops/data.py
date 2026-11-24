"""Dataset parsing, validation and profiling."""
from __future__ import annotations

import io

import pandas as pd
from pandas.api.types import is_numeric_dtype

from .errors import ServiceError

MIN_ROWS = 30


def parse_csv(content: bytes, filename: str, max_mb: int) -> pd.DataFrame:
    if not filename.lower().endswith(".csv"):
        raise ServiceError("Only .csv files are supported.", 415)
    if not content:
        raise ServiceError("The uploaded file is empty.")
    if len(content) > max_mb * 1024 * 1024:
        raise ServiceError(f"File exceeds the {max_mb} MB upload limit.", 413)
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:  # pandas raises many types
        raise ServiceError(f"Could not parse CSV: {exc}") from exc
    df.columns = [str(c).strip() for c in df.columns]
    if df.shape[1] < 2:
        raise ServiceError("Dataset needs at least 2 columns (features + target).")
    if len(df) < MIN_ROWS:
        raise ServiceError(f"Dataset needs at least {MIN_ROWS} rows (got {len(df)}).")
    if df.columns.duplicated().any():
        raise ServiceError("Duplicate column names are not allowed.")
    if any(c == "" or c.startswith("Unnamed") for c in df.columns):
        raise ServiceError("All columns must have a header name.")
    return df


def profile(df: pd.DataFrame) -> dict:
    cols = []
    for c in df.columns:
        s = df[c]
        cols.append({
            "name": c,
            "kind": "numeric" if is_numeric_dtype(s) else "categorical",
            "missing": int(s.isna().sum()),
            "unique": int(s.nunique(dropna=True)),
        })
    return {"rows": int(len(df)), "cols": int(df.shape[1]),
            "missing_total": int(df.isna().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()), "columns": cols}


def check_target(df: pd.DataFrame, target: str) -> None:
    if target not in df.columns:
        raise ServiceError(f"Target column '{target}' not found in dataset.", 422)
    y = df[target].dropna()
    n_classes = y.nunique()
    if n_classes < 2:
        raise ServiceError("Target must contain at least 2 classes.", 422)
    if n_classes > 20:
        raise ServiceError(f"Target has {n_classes} distinct values; this platform supports "
                           "classification with up to 20 classes.", 422)
    if y.astype(str).value_counts().min() < 2:
        raise ServiceError("Every target class needs at least 2 rows.", 422)
