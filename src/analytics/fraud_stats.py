"""Group-by fraud statistics (Pandas engine). The Spark engine produces the same columns."""
from __future__ import annotations

import pandas as pd

from src.preprocessing.schema import STATS_COLUMNS
from src.utils.validation import require_columns, require_non_empty


def _stats(df: pd.DataFrame, key: str) -> pd.DataFrame:
    grouped = df.assign(fraud_amount=df["amount"] * df["isFraud"]).groupby(key).agg(
        n_transactions=("amount", "size"),
        n_fraud=("isFraud", "sum"),
        total_amount=("amount", "sum"),
        fraud_amount=("fraud_amount", "sum"),
    )
    grouped["fraud_rate_pct"] = (100.0 * grouped["n_fraud"] / grouped["n_transactions"]).round(4)
    grouped = grouped.reset_index().sort_values(key).reset_index(drop=True)
    return grouped[[key, *STATS_COLUMNS]]


def fraud_stats(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (stats by transaction type, stats by day). Needs the ``day`` feature."""
    require_non_empty(df)
    require_columns(df, ["type", "day", "amount", "isFraud"])
    return _stats(df, "type"), _stats(df, "day")