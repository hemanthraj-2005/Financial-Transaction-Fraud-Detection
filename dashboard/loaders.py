"""Data access for the dashboard. No Streamlit imports here, so it can be tested with pytest."""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd

from src.db import analytics, queries
from src.utils.errors import DatasetNotFoundError
from src.utils.io_utils import table_paths

REGISTRY = {
    # src/db/queries.py
    "summary": queries.summary,
    "run_info": queries.run_info,
    "risk_band_counts": queries.risk_band_counts,
    "top_transactions": queries.top_transactions,
    "top_accounts": queries.top_accounts,
    "top_communities": queries.top_communities,
    "community_members": queries.community_members,
    "account_detail": queries.account_detail,
    "account_transactions": queries.account_transactions,
    # src/db/analytics.py
    "fraud_by_type": analytics.fraud_by_type,
    "daily_counts": analytics.daily_counts,
    "hourly_counts": analytics.hourly_counts,
    "risk_band_by_type": analytics.risk_band_by_type,
    "amount_distribution": analytics.amount_distribution,
    "risk_histogram": analytics.risk_histogram,
    "pr_curve": analytics.pr_curve,
    "confusion_counts": analytics.confusion_counts,
    "reason_code_counts": analytics.reason_code_counts,
    "search_accounts": analytics.search_accounts,
    "accounts_by_ids": analytics.accounts_by_ids,
    "counterparties": analytics.counterparties,
    "neighborhood": analytics.neighborhood,
}


def open_readonly(db_path: Path | str) -> sqlite3.Connection:
    """Open the database read-only (never creates a missing file)."""
    path = Path(db_path)
    if not path.exists():
        raise DatasetNotFoundError(f"Database not found: {path}. Run: python -m src.models.scoring_pipeline")
    return sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)


def run_query(db_path: Path | str, name: str, *args, **kwargs):
    if name not in REGISTRY:
        raise ValueError(f"Unknown query '{name}'.")
    with closing(open_readonly(db_path)) as conn:
        return REGISTRY[name](conn, *args, **kwargs)


def db_status(db_path: Path | str) -> tuple[bool, str]:
    path = Path(db_path)
    if not path.exists():
        return False, f"Database not found at {path}. Build it first:"
    try:
        with closing(open_readonly(path)) as conn:
            n = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    except sqlite3.Error as exc:
        return False, f"The database exists but is incomplete or unreadable ({exc}). Re-run the scoring pipeline:"
    if n == 0:
        return False, "The database has no scored transactions. Re-run the pipelines:"
    return True, f"{n:,} scored transactions"


def read_json(path: Path | str) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def read_csv(path: Path | str) -> pd.DataFrame | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return None


def load_optional_table(stem: Path | str) -> pd.DataFrame | None:
    """Load <stem>.parquet or <stem>.csv, or None if neither exists."""
    parquet_path, csv_path = table_paths(stem)
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None