"""SQLite persistence (stdlib only). Swap for PostgreSQL by replacing connect()."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from .config import get_settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets(
  id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, path TEXT NOT NULL,
  rows INTEGER, cols INTEGER, profile_json TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS model_versions(
  id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, version INTEGER NOT NULL,
  dataset_id INTEGER, target TEXT, algorithm TEXT, params_json TEXT, metrics_json TEXT,
  reference_json TEXT, features_json TEXT, classes_json TEXT, excluded_json TEXT,
  artifact_path TEXT, mlflow_run_id TEXT, mlflow_status TEXT, stage TEXT DEFAULT 'staging',
  parent_version INTEGER, notes TEXT, created_at TEXT, UNIQUE(name, version));
CREATE TABLE IF NOT EXISTS deployments(
  id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, version INTEGER, status TEXT,
  message TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS predictions(
  id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, version INTEGER, input_json TEXT,
  prediction TEXT, probability REAL, actual TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS logs(
  id INTEGER PRIMARY KEY AUTOINCREMENT, level TEXT, event TEXT, message TEXT, created_at TEXT);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect():
    s = get_settings()
    s.home.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(s.db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def query(sql: str, params: tuple = ()) -> list[dict]:
    with connect() as conn:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]


def log_event(level: str, event: str, message: str) -> None:
    with connect() as conn:
        conn.execute("INSERT INTO logs(level,event,message,created_at) VALUES(?,?,?,?)",
                     (level, event, message, now()))
