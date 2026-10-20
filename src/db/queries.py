"""Read-only queries used by the dashboard (all parameterised)."""
from __future__ import annotations

import json
import sqlite3

import numpy as np
import pandas as pd

from src.utils.errors import InvalidAccountError

BANDS = ["Low", "Medium", "High", "Critical"]


def _frame(conn: sqlite3.Connection, sql: str, params=()) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn, params=list(params))


def _positive(n: int, name: str = "n") -> int:
    if int(n) < 1:
        raise ValueError(f"{name} must be at least 1.")
    return int(n)


def _plain(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and value != value:
        return None
    return value


def run_info(conn: sqlite3.Connection) -> dict:
    rows = conn.execute("SELECT key, value FROM run_info").fetchall()
    return {key: json.loads(value) for key, value in rows}


def risk_band_counts(conn: sqlite3.Connection, split: str | None = None) -> dict:
    where, params = ("WHERE split = ?", [split]) if split else ("", [])
    frame = _frame(conn, f"SELECT risk_band, COUNT(*) AS n FROM transactions {where} GROUP BY risk_band", params)
    counts = dict.fromkeys(BANDS, 0)
    counts.update({row.risk_band: int(row.n) for row in frame.itertuples()})
    return counts


def summary(conn: sqlite3.Connection) -> dict:
    def scalar(sql):
        return conn.execute(sql).fetchone()[0] or 0

    return {
        "n_transactions": int(scalar("SELECT COUNT(*) FROM transactions")),
        "n_accounts": int(scalar("SELECT COUNT(*) FROM accounts")),
        "n_communities": int(scalar("SELECT COUNT(*) FROM communities")),
        "n_alerts": int(scalar("SELECT SUM(predicted) FROM transactions")),
        "bands_all": risk_band_counts(conn),
        "bands_test": risk_band_counts(conn, "test"),
    }


def top_transactions(conn, n: int = 50, split: str | None = None, min_risk: float | None = None) -> pd.DataFrame:
    conditions, params = [], []
    if split is not None:
        conditions.append("split = ?")
        params.append(split)
    if min_risk is not None:
        conditions.append("risk_score >= ?")
        params.append(float(min_risk))
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    params.append(_positive(n))
    return _frame(conn, f"SELECT * FROM transactions {where} ORDER BY risk_score DESC, tx_id ASC LIMIT ?", params)


def top_accounts(conn, n: int = 50, min_risk: float | None = None, include_merchants: bool = True) -> pd.DataFrame:
    conditions, params = [], []
    if min_risk is not None:
        conditions.append("risk_score >= ?")
        params.append(float(min_risk))
    if not include_merchants:
        conditions.append("is_merchant = 0")
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    params.append(_positive(n))
    return _frame(conn, f"SELECT * FROM accounts {where} ORDER BY risk_score DESC, account ASC LIMIT ?", params)


def top_communities(conn, n: int = 20, flagged_only: bool = False) -> pd.DataFrame:
    extra = "AND flagged = 1" if flagged_only else ""
    return _frame(
        conn,
        f"SELECT * FROM communities WHERE eligible = 1 {extra} ORDER BY suspicion_score DESC, community_id ASC LIMIT ?",
        [_positive(n)],
    )


def community_members(conn, community_id: int, limit: int = 200) -> pd.DataFrame:
    return _frame(
        conn,
        "SELECT * FROM accounts WHERE community_id = ? ORDER BY risk_score DESC, account ASC LIMIT ?",
        [int(community_id), _positive(limit, "limit")],
    )


def account_detail(conn, account: str) -> dict:
    frame = _frame(conn, "SELECT * FROM accounts WHERE account = ?", [account])
    if frame.empty:
        raise InvalidAccountError(f"Account '{account}' was not found in the database.")
    return {key: _plain(value) for key, value in frame.iloc[0].to_dict().items()}


def account_transactions(conn, account: str, limit: int = 200) -> pd.DataFrame:
    """All transactions touching the account, oldest first, with its role in each."""
    return _frame(
        conn,
        "SELECT *, CASE WHEN nameOrig = ? THEN 'sent' ELSE 'received' END AS role "
        "FROM transactions WHERE nameOrig = ? OR nameDest = ? ORDER BY step ASC, tx_id ASC LIMIT ?",
        [account, account, account, _positive(limit, "limit")],
    )