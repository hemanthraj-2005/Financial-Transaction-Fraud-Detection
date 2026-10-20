"""SQLite storage for scored results.

The dashboard (Phase 9) reads this database and never recomputes anything heavy.

Tables
  transactions : every scored transaction (model score, 0-100 risk score, band, reasons)
  accounts     : one row per account: risk aggregated from its transactions + graph/community context
  communities  : Phase 6 community statistics and suspicion scores
  run_info     : key/value metadata (model, threshold, risk mapping, data source)

Columns starting with label_ use the fraud label: display / evaluation only.
Graph and community columns on accounts come from the full-data Phase 5/6 runs: display only.
"""
from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

TRANSACTION_COLUMNS = [
    "tx_id", "step", "day", "hour_of_day", "type", "amount", "nameOrig", "nameDest", "split",
    "model_score", "risk_score", "risk_band", "predicted", "reason_codes", "reasons", "label_isFraud",
]
ACCOUNT_DB_COLUMNS = [
    "account", "is_merchant", "n_transactions", "n_sent", "n_received", "total_sent", "total_received",
    "max_risk", "mean_risk", "risk_score", "risk_band", "n_alerts", "n_high_risk_tx", "riskiest_tx_id",
    "reason_codes", "reasons", "total_degree", "pagerank", "betweenness", "pass_through_ratio",
    "community_id", "community_suspicion_score", "community_flagged", "label_fraud_tx",
]
COMMUNITY_DB_COLUMNS = [
    "community_id", "n_accounts", "n_merchants", "merchant_share", "internal_edges", "internal_tx_count",
    "internal_amount", "external_out_amount", "external_in_amount", "internal_flow_share", "internal_density",
    "repeat_ratio", "mean_pass_through_ratio", "share_in_small_cycle", "top_account_by_pagerank",
    "label_fraud_tx", "label_fraud_accounts", "label_fraud_rate_pct", "eligible", "suspicion_score", "flagged",
]
TABLE_COLUMNS = {
    "transactions": TRANSACTION_COLUMNS,
    "accounts": ACCOUNT_DB_COLUMNS,
    "communities": COMMUNITY_DB_COLUMNS,
    "run_info": ["key", "value"],
}

_CREATE = {
    "transactions": """
        CREATE TABLE transactions (
            tx_id INTEGER PRIMARY KEY,
            step INTEGER NOT NULL,
            day INTEGER,
            hour_of_day INTEGER,
            type TEXT NOT NULL,
            amount REAL NOT NULL,
            nameOrig TEXT NOT NULL,
            nameDest TEXT NOT NULL,
            split TEXT NOT NULL,
            model_score REAL NOT NULL,
            risk_score REAL NOT NULL,
            risk_band TEXT NOT NULL,
            predicted INTEGER NOT NULL,
            reason_codes TEXT,
            reasons TEXT,
            label_isFraud INTEGER NOT NULL
        )""",
    "accounts": """
        CREATE TABLE accounts (
            account TEXT PRIMARY KEY,
            is_merchant INTEGER NOT NULL,
            n_transactions INTEGER NOT NULL,
            n_sent INTEGER NOT NULL,
            n_received INTEGER NOT NULL,
            total_sent REAL NOT NULL,
            total_received REAL NOT NULL,
            max_risk REAL NOT NULL,
            mean_risk REAL NOT NULL,
            risk_score REAL NOT NULL,
            risk_band TEXT NOT NULL,
            n_alerts INTEGER NOT NULL,
            n_high_risk_tx INTEGER NOT NULL,
            riskiest_tx_id INTEGER,
            reason_codes TEXT,
            reasons TEXT,
            total_degree INTEGER,
            pagerank REAL,
            betweenness REAL,
            pass_through_ratio REAL,
            community_id INTEGER,
            community_suspicion_score REAL,
            community_flagged INTEGER,
            label_fraud_tx INTEGER NOT NULL
        )""",
    "communities": """
        CREATE TABLE communities (
            community_id INTEGER PRIMARY KEY,
            n_accounts INTEGER,
            n_merchants INTEGER,
            merchant_share REAL,
            internal_edges INTEGER,
            internal_tx_count INTEGER,
            internal_amount REAL,
            external_out_amount REAL,
            external_in_amount REAL,
            internal_flow_share REAL,
            internal_density REAL,
            repeat_ratio REAL,
            mean_pass_through_ratio REAL,
            share_in_small_cycle REAL,
            top_account_by_pagerank TEXT,
            label_fraud_tx INTEGER,
            label_fraud_accounts INTEGER,
            label_fraud_rate_pct REAL,
            eligible INTEGER,
            suspicion_score REAL,
            flagged INTEGER
        )""",
    "run_info": "CREATE TABLE run_info (key TEXT PRIMARY KEY, value TEXT)",
}
_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_tx_orig ON transactions(nameOrig)",
    "CREATE INDEX IF NOT EXISTS idx_tx_dest ON transactions(nameDest)",
    "CREATE INDEX IF NOT EXISTS idx_tx_risk ON transactions(risk_score)",
    "CREATE INDEX IF NOT EXISTS idx_tx_split ON transactions(split)",
    "CREATE INDEX IF NOT EXISTS idx_acc_risk ON accounts(risk_score)",
    "CREATE INDEX IF NOT EXISTS idx_acc_comm ON accounts(community_id)",
    "CREATE INDEX IF NOT EXISTS idx_comm_score ON communities(suspicion_score)",
]

from config import DB_PATH

def connect(path: Path | str = DB_PATH) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(path))


def create_schema(conn: sqlite3.Connection) -> None:
    """(Re)create all tables. Safe to call repeatedly: existing data is dropped."""
    for table in TABLE_COLUMNS:
        conn.execute(f"DROP TABLE IF EXISTS {table}")
    for statement in _CREATE.values():
        conn.execute(statement)
    conn.commit()


def create_indexes(conn: sqlite3.Connection) -> None:
    for statement in _INDEXES:
        conn.execute(statement)
    conn.commit()


def _py(value):
    """Convert numpy / pandas scalars to plain Python values (NaN -> NULL)."""
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def insert_dataframe(conn: sqlite3.Connection, table: str, df: pd.DataFrame, chunk_size: int = 50_000) -> int:
    """Insert a DataFrame whose columns match the table exactly (same names, same order)."""
    if table not in TABLE_COLUMNS:
        raise ValueError(f"Unknown table '{table}'. Choose from {list(TABLE_COLUMNS)}.")
    expected = TABLE_COLUMNS[table]
    if list(df.columns) != expected:
        raise ValueError(f"Columns for '{table}' must be exactly {expected}; got {list(df.columns)}")
    placeholders = ", ".join("?" * len(expected))
    sql = f"INSERT INTO {table} ({', '.join(expected)}) VALUES ({placeholders})"
    total = 0
    for start in range(0, len(df), chunk_size):
        part = df.iloc[start : start + chunk_size]
        records = [tuple(_py(v) for v in row) for row in part.itertuples(index=False, name=None)]
        conn.executemany(sql, records)
        total += len(records)
    conn.commit()
    return total


def write_run_info(conn: sqlite3.Connection, info: dict) -> None:
    conn.execute("DELETE FROM run_info")
    conn.executemany(
        "INSERT INTO run_info (key, value) VALUES (?, ?)",
        [(key, json.dumps(value, default=str)) for key, value in info.items()],
    )
    conn.commit()