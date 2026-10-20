"""Risk score (0-100), risk bands and account-level aggregation.

Probability -> risk is piecewise linear and ANCHORED at the decision threshold t (tuned on validation):

    p = 0            -> 0
    p = t / 2        -> 30   (top of "Low")
    p = t            -> 61   (bottom of "High": the model flags the transaction)
    p = (1 + t) / 2  -> 81   (bottom of "Critical")
    p = 1            -> 100

Bands (project-defined, calibratable, NOT scientifically validated):
    Low 0-30 | Medium 31-60 | High 61-80 | Critical 81-100
Risk is floored to one decimal so a displayed value never crosses a band boundary.
An account's risk score is the MAXIMUM risk of the transactions that touch it (as sender or receiver).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.db.database import ACCOUNT_DB_COLUMNS
from src.models.train import evaluate
from src.utils.validation import require_columns, require_non_empty

RISK_BANDS = [("Low", 0, 30), ("Medium", 31, 60), ("High", 61, 80), ("Critical", 81, 100)]
BAND_NAMES = [name for name, _, _ in RISK_BANDS]
ALERT_RISK = 61.0
_EPS = 1e-6

ACCOUNT_RISK_COLUMNS = [
    "account", "is_merchant", "n_transactions", "n_sent", "n_received", "total_sent", "total_received",
    "max_risk", "mean_risk", "risk_score", "risk_band", "n_alerts", "n_high_risk_tx", "riskiest_tx_id",
    "label_fraud_tx",
]
_ACCOUNT_NEEDED = ["tx_id", "nameOrig", "nameDest", "amount", "risk_score", "predicted", "isFraud"]


def risk_anchors(threshold: float) -> tuple[list[float], list[float]]:
    """(probabilities, risk values) of the mapping's anchor points."""
    t = float(np.clip(threshold, _EPS, 1.0 - _EPS))
    return [0.0, t / 2, t, (1 + t) / 2, 1.0], [0.0, 30.0, ALERT_RISK, 81.0, 100.0]


def probability_to_risk(p, threshold: float) -> np.ndarray:
    xp, fp = risk_anchors(threshold)
    risk = np.interp(np.clip(np.asarray(p, dtype=float), 0.0, 1.0), xp, fp)
    return np.floor(risk * 10.0) / 10.0


def risk_band(risk) -> np.ndarray:
    r = np.asarray(risk, dtype=float)
    return np.select([r < 31.0, r < 61.0, r < 81.0], ["Low", "Medium", "High"], default="Critical")


def aggregate_accounts(scored: pd.DataFrame) -> pd.DataFrame:
    """One row per account from scored transactions (needs tx_id, nameOrig, nameDest, amount, risk_score,
    predicted, isFraud). ``label_fraud_tx`` uses the label: display / evaluation only."""
    require_non_empty(scored, "scored transactions")
    require_columns(scored, _ACCOUNT_NEEDED, "scored transactions")
    base = ["tx_id", "amount", "risk_score", "predicted", "isFraud"]
    sent = scored[["nameOrig", *base]].rename(columns={"nameOrig": "account"})
    sent["is_sent"] = 1
    received = scored[["nameDest", *base]].rename(columns={"nameDest": "account"})
    received["is_sent"] = 0
    long = pd.concat([sent, received], ignore_index=True)
    long["amount_sent"] = long["amount"] * long["is_sent"]
    long["amount_received"] = long["amount"] * (1 - long["is_sent"])
    long["high"] = (long["risk_score"] >= ALERT_RISK).astype("int64")

    acc = long.groupby("account").agg(
        n_transactions=("tx_id", "size"),
        n_sent=("is_sent", "sum"),
        total_sent=("amount_sent", "sum"),
        total_received=("amount_received", "sum"),
        max_risk=("risk_score", "max"),
        mean_risk=("risk_score", "mean"),
        n_alerts=("predicted", "sum"),
        n_high_risk_tx=("high", "sum"),
        label_fraud_tx=("isFraud", "sum"),
    ).reset_index()
    acc["n_received"] = acc["n_transactions"] - acc["n_sent"]
    acc["mean_risk"] = acc["mean_risk"].round(2)
    acc["risk_score"] = acc["max_risk"]
    acc["risk_band"] = risk_band(acc["risk_score"].to_numpy())
    acc["is_merchant"] = acc["account"].str.startswith("M").astype("int64")

    riskiest = (
        long.sort_values(["account", "risk_score"], ascending=[True, False], kind="stable")
        .drop_duplicates("account")[["account", "tx_id"]]
        .rename(columns={"tx_id": "riskiest_tx_id"})
    )
    acc = acc.merge(riskiest, on="account", how="left", validate="one_to_one")
    int_columns = ["n_transactions", "n_sent", "n_received", "n_alerts", "n_high_risk_tx", "label_fraud_tx"]
    acc[int_columns] = acc[int_columns].astype("int64")
    return acc[ACCOUNT_RISK_COLUMNS].sort_values("account").reset_index(drop=True)


def build_account_table(
    scored: pd.DataFrame,
    metrics: pd.DataFrame | None = None,
    membership: pd.DataFrame | None = None,
    community_stats: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Account risk + reasons of its riskiest transaction + optional graph/community context (display only).

    ``scored`` must also have reason_codes and reasons columns."""
    accounts = aggregate_accounts(scored)
    by_tx = scored.set_index("tx_id")[["reason_codes", "reasons"]]
    for column in ("reason_codes", "reasons"):
        accounts[column] = by_tx[column].reindex(accounts["riskiest_tx_id"]).to_numpy()
    if metrics is not None:
        accounts = accounts.merge(
            metrics[["account", "total_degree", "pagerank", "betweenness", "pass_through_ratio"]],
            on="account", how="left", validate="one_to_one",
        )
    if membership is not None:
        accounts = accounts.merge(
            membership[["account", "community_id"]], on="account", how="left", validate="one_to_one"
        )
        if community_stats is not None:
            stats = community_stats[["community_id", "suspicion_score", "flagged"]].rename(
                columns={"suspicion_score": "community_suspicion_score", "flagged": "community_flagged"}
            )
            accounts = accounts.merge(stats, on="community_id", how="left", validate="many_to_one")
    for column in ACCOUNT_DB_COLUMNS:
        if column not in accounts.columns:
            accounts[column] = np.nan
    return accounts[ACCOUNT_DB_COLUMNS]


def account_level_check(scored: pd.DataFrame, split: str = "test") -> dict:
    """Honest account-level evaluation on one window. Uses the label: reporting only.

    'Fraud account' = the account touches at least one fraud transaction in that window
    (victims as well as mules), so this is not a mule-detection benchmark."""
    sub = scored[scored["split"] == split]
    result = {
        "split": split,
        "definition": "account touches at least one fraud transaction in this window",
        "n_accounts": 0, "n_fraud_accounts": 0,
        "precision_at_high": None, "recall_at_high": None, "pr_auc": None, "roc_auc": None,
    }
    if sub.empty:
        return result
    acc = aggregate_accounts(sub)
    y = (acc["label_fraud_tx"] > 0).astype(int).to_numpy()
    m = evaluate(y, acc["risk_score"].to_numpy(), ALERT_RISK)
    result.update(
        n_accounts=int(len(acc)), n_fraud_accounts=int(y.sum()),
        precision_at_high=m["precision"], recall_at_high=m["recall"], pr_auc=m["pr_auc"], roc_auc=m["roc_auc"],
    )
    return result