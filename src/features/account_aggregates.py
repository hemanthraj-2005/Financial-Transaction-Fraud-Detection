"""Account-level aggregates (Pandas engine): one row per account.

IMPORTANT: aggregates are computed from whatever rows you pass in.
 * For dashboards, EDA and the graph: pass all (clean) transactions.
 * For ML features: pass only the TRAINING window, otherwise the aggregates
   contain information from the future (leakage). Phase 7 does this.
Columns starting with ``label_`` use the fraud label and must never be fed to
a model unless computed from training rows only.
"""
from __future__ import annotations

import pandas as pd

from src.preprocessing.schema import ACCOUNT_COLUMNS
from src.utils.validation import require_columns, require_non_empty

_NEEDED = ["nameOrig", "nameDest", "amount", "step", "isFraud"]


def build_account_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    require_non_empty(df)
    require_columns(df, _NEEDED)

    sent = df.groupby("nameOrig", sort=False).agg(
        out_tx_count=("amount", "size"),
        out_amount_sum=("amount", "sum"),
        out_unique_counterparties=("nameDest", "nunique"),
        out_first_step=("step", "min"),
        out_last_step=("step", "max"),
        label_fraud_sent=("isFraud", "sum"),
    )
    received = df.groupby("nameDest", sort=False).agg(
        in_tx_count=("amount", "size"),
        in_amount_sum=("amount", "sum"),
        in_unique_counterparties=("nameOrig", "nunique"),
        in_first_step=("step", "min"),
        in_last_step=("step", "max"),
        label_fraud_received=("isFraud", "sum"),
    )
    acc = sent.join(received, how="outer")
    acc.index.name = "account"

    acc["first_step"] = acc[["out_first_step", "in_first_step"]].min(axis=1)
    acc["last_step"] = acc[["out_last_step", "in_last_step"]].max(axis=1)
    count_cols = [
        "out_tx_count", "out_amount_sum", "out_unique_counterparties", "label_fraud_sent",
        "in_tx_count", "in_amount_sum", "in_unique_counterparties", "label_fraud_received",
    ]
    acc[count_cols] = acc[count_cols].fillna(0)

    # exact number of distinct neighbours, ignoring direction
    edges = df[["nameOrig", "nameDest"]].drop_duplicates()
    both = pd.concat(
        [
            edges.rename(columns={"nameOrig": "account", "nameDest": "counterparty"}),
            edges.rename(columns={"nameDest": "account", "nameOrig": "counterparty"}),
        ]
    ).drop_duplicates()
    acc["n_counterparties"] = both.groupby("account").size().reindex(acc.index).fillna(0)

    acc["total_tx_count"] = acc["out_tx_count"] + acc["in_tx_count"]
    acc["total_amount"] = acc["out_amount_sum"] + acc["in_amount_sum"]
    acc["out_amount_avg"] = (acc["out_amount_sum"] / acc["out_tx_count"].where(acc["out_tx_count"] > 0)).fillna(0.0)
    acc["in_amount_avg"] = (acc["in_amount_sum"] / acc["in_tx_count"].where(acc["in_tx_count"] > 0)).fillna(0.0)
    acc["avg_tx_amount"] = (acc["total_amount"] / acc["total_tx_count"].where(acc["total_tx_count"] > 0)).fillna(0.0)

    acc = acc.reset_index()
    acc["is_merchant"] = acc["account"].str.startswith("M").astype("int8")
    int_cols = [
        "out_tx_count", "out_unique_counterparties", "in_tx_count", "in_unique_counterparties",
        "total_tx_count", "n_counterparties", "first_step", "last_step",
        "label_fraud_sent", "label_fraud_received",
    ]
    acc[int_cols] = acc[int_cols].astype("int64")
    return acc[ACCOUNT_COLUMNS].sort_values("account").reset_index(drop=True)