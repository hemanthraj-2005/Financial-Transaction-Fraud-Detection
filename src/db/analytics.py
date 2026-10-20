"""Analytic read-only queries for the dashboard. All are parameterised and return DataFrames."""
from __future__ import annotations

import sqlite3
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from src.models.explain import REASON_RULES

SPLITS = ["fit", "val", "test"]
_REASON_TEXT = {rule.code: rule.message for rule in REASON_RULES}
_ACCOUNT_COLUMNS = ["account", "risk_score", "risk_band", "n_transactions", "is_merchant", "community_id"]


def _frame(conn: sqlite3.Connection, sql: str, params=()) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn, params=list(params))


def _check_split(split) -> None:
    if split is not None and split not in SPLITS:
        raise ValueError(f"split must be None or one of {SPLITS}")


def _where(split) -> tuple[str, list]:
    _check_split(split)
    return ("WHERE split = ?", [split]) if split else ("", [])


# ------------------------------------------------------------------ counts by type / time
def fraud_by_type(conn, split=None) -> pd.DataFrame:
    where, params = _where(split)
    frame = _frame(
        conn,
        f"""SELECT type, COUNT(*) AS n_transactions, SUM(label_isFraud) AS n_fraud, SUM(predicted) AS n_alerts,
                   AVG(risk_score) AS avg_risk, SUM(amount) AS total_amount
            FROM transactions {where} GROUP BY type ORDER BY n_transactions DESC, type ASC""",
        params,
    )
    frame[["n_fraud", "n_alerts"]] = frame[["n_fraud", "n_alerts"]].fillna(0)
    frame["fraud_rate_pct"] = 100.0 * frame["n_fraud"].astype(float) / frame["n_transactions"].astype(float)
    return frame


def _count_by(conn, column: str, split) -> pd.DataFrame:
    where, params = _where(split)
    frame = _frame(
        conn,
        f"""SELECT {column}, COUNT(*) AS n_transactions, SUM(label_isFraud) AS n_fraud, SUM(predicted) AS n_alerts
            FROM transactions {where} GROUP BY {column} ORDER BY {column} ASC""",
        params,
    )
    frame = frame.dropna(subset=[column])
    frame[["n_fraud", "n_alerts"]] = frame[["n_fraud", "n_alerts"]].fillna(0)
    return frame.astype({column: "int64"}).reset_index(drop=True)


def daily_counts(conn, split=None) -> pd.DataFrame:
    return _count_by(conn, "day", split)


def hourly_counts(conn, split=None) -> pd.DataFrame:
    return _count_by(conn, "hour_of_day", split)


def risk_band_by_type(conn, split=None) -> pd.DataFrame:
    where, params = _where(split)
    return _frame(
        conn, f"SELECT type, risk_band, COUNT(*) AS n FROM transactions {where} GROUP BY type, risk_band", params
    )


# ------------------------------------------------------------------ distributions and curves
def amount_distribution(conn, split=None, bins: int = 40) -> pd.DataFrame:
    """Share of each class (%) per log10(amount + 1) bin."""
    where, params = _where(split)
    frame = _frame(conn, f"SELECT amount, label_isFraud FROM transactions {where}", params)
    columns = ["log10_amount", "class_name", "share_pct"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    x = np.log10(frame["amount"].to_numpy(dtype=float) + 1.0)
    labels = frame["label_isFraud"].to_numpy()
    edges = np.linspace(x.min(), x.max() + 1e-9, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    parts = []
    for label, name in ((0, "Legitimate"), (1, "Fraud")):
        values = x[labels == label]
        if len(values) == 0:
            continue
        counts, _ = np.histogram(values, bins=edges)
        parts.append(pd.DataFrame({"log10_amount": centers, "class_name": name, "share_pct": 100.0 * counts / len(values)}))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)


def risk_histogram(conn, split=None, bins: int = 20) -> pd.DataFrame:
    """Share of each class (%) per risk-score bin (0-100)."""
    where, params = _where(split)
    frame = _frame(conn, f"SELECT risk_score, label_isFraud FROM transactions {where}", params)
    columns = ["risk_mid", "class_name", "share_pct"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    edges = np.linspace(0.0, 100.0, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    scores = frame["risk_score"].to_numpy(dtype=float)
    labels = frame["label_isFraud"].to_numpy()
    parts = []
    for label, name in ((0, "Legitimate"), (1, "Fraud")):
        values = scores[labels == label]
        if len(values) == 0:
            continue
        counts, _ = np.histogram(values, bins=edges)
        parts.append(pd.DataFrame({"risk_mid": centers, "class_name": name, "share_pct": 100.0 * counts / len(values)}))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)


def pr_curve(conn, split="test", max_points: int = 1500) -> pd.DataFrame:
    where, params = _where(split)
    frame = _frame(conn, f"SELECT model_score, label_isFraud FROM transactions {where}", params)
    columns = ["precision", "recall", "threshold"]
    y = frame["label_isFraud"].to_numpy(dtype=int)
    if len(y) == 0 or y.sum() == 0:
        return pd.DataFrame(columns=columns)
    precision, recall, thresholds = precision_recall_curve(y, frame["model_score"].to_numpy(dtype=float))
    curve = pd.DataFrame({"precision": precision[:-1], "recall": recall[:-1], "threshold": thresholds})
    if len(curve) > max_points:
        curve = curve.iloc[np.unique(np.linspace(0, len(curve) - 1, max_points).astype(int))]
    return curve.reset_index(drop=True)


def confusion_counts(conn, split=None) -> dict:
    where, params = _where(split)
    frame = _frame(
        conn,
        f"SELECT predicted, label_isFraud, COUNT(*) AS n FROM transactions {where} GROUP BY predicted, label_isFraud",
        params,
    )
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for row in frame.itertuples(index=False):
        key = ("t" if row.predicted == row.label_isFraud else "f") + ("p" if row.predicted == 1 else "n")
        counts[key] += int(row.n)
    return counts


def reason_code_counts(conn, split=None, min_risk=None) -> pd.DataFrame:
    _check_split(split)
    conditions, params = ["reason_codes IS NOT NULL", "reason_codes <> ''"], []
    if split:
        conditions.append("split = ?")
        params.append(split)
    if min_risk is not None:
        conditions.append("risk_score >= ?")
        params.append(float(min_risk))
    frame = _frame(conn, f"SELECT reason_codes FROM transactions WHERE {' AND '.join(conditions)}", params)
    counter = Counter(code for text in frame["reason_codes"] for code in text.split(";") if code)
    out = pd.DataFrame(sorted(counter.items(), key=lambda item: (-item[1], item[0])), columns=["code", "count"])
    out["description"] = out["code"].map(_REASON_TEXT).fillna("")
    return out


# ------------------------------------------------------------------ account lookups
def _like_pattern(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def search_accounts(conn, text: str, limit: int = 20) -> pd.DataFrame:
    text = (text or "").strip()
    if not text:
        return pd.DataFrame(columns=_ACCOUNT_COLUMNS)
    return _frame(
        conn,
        f"""SELECT {', '.join(_ACCOUNT_COLUMNS)} FROM accounts WHERE account LIKE ? ESCAPE '\\'
            ORDER BY risk_score DESC, account ASC LIMIT ?""",
        [_like_pattern(text), max(1, int(limit))],
    )


def accounts_by_ids(conn, ids) -> pd.DataFrame:
    ids = [str(i) for i in ids]
    if not ids:
        return pd.DataFrame(columns=_ACCOUNT_COLUMNS)
    parts = []
    for start in range(0, len(ids), 500):
        chunk = ids[start : start + 500]
        placeholders = ", ".join("?" * len(chunk))
        parts.append(
            _frame(conn, f"SELECT {', '.join(_ACCOUNT_COLUMNS)} FROM accounts WHERE account IN ({placeholders})", chunk)
        )
    return pd.concat(parts, ignore_index=True)


def counterparties(conn, account: str, limit: int = 50) -> pd.DataFrame:
    return _frame(
        conn,
        """SELECT nameDest AS counterparty, 'sent_to' AS direction, COUNT(*) AS n_tx,
                  SUM(amount) AS total_amount, MAX(risk_score) AS max_risk
           FROM transactions WHERE nameOrig = ? GROUP BY nameDest
           UNION ALL
           SELECT nameOrig, 'received_from', COUNT(*), SUM(amount), MAX(risk_score)
           FROM transactions WHERE nameDest = ? GROUP BY nameOrig
           ORDER BY total_amount DESC, counterparty ASC LIMIT ?""",
        [account, account, max(1, int(limit))],
    )


def neighborhood(conn, account: str, max_neighbors: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(edges, nodes) of the account and its largest counterparties (by total amount)."""
    empty_edges = pd.DataFrame(columns=["src", "dst", "n_tx", "total_amount", "max_risk"])
    empty_nodes = pd.DataFrame(columns=["account", "risk_score", "risk_band", "is_center"])
    cp = counterparties(conn, account, limit=1000)
    if cp.empty:
        return empty_edges, empty_nodes
    totals = cp.groupby("counterparty")["total_amount"].sum().sort_values(ascending=False, kind="stable")
    neighbors = [str(a) for a in totals.head(max(1, int(max_neighbors))).index]
    members = [account, *neighbors]
    placeholders = ", ".join("?" * len(members))
    edges = _frame(
        conn,
        f"""SELECT nameOrig AS src, nameDest AS dst, COUNT(*) AS n_tx, SUM(amount) AS total_amount,
                   MAX(risk_score) AS max_risk
            FROM transactions WHERE nameOrig IN ({placeholders}) AND nameDest IN ({placeholders})
            GROUP BY nameOrig, nameDest""",
        members + members,
    )
    nodes = accounts_by_ids(conn, members)[["account", "risk_score", "risk_band"]]
    nodes = nodes.assign(is_center=nodes["account"] == account)
    return edges, nodes