"""Data cleaning (Pandas engine).

Rules (identical to the Spark engine in spark_pipeline.py):
  null_required        any required column is missing/unparseable
  negative_amount      amount < 0
  invalid_step         step < 1
  unknown_type         type not in VALID_TRANSACTION_TYPES
  invalid_label        isFraud not in {0, 1}
  invalid_flag_column  isFlaggedFraud not in {0, 1}
  negative_balance     any balance column < 0
  self_transfer        nameOrig == nameDest (not a meaningful graph edge)
  blank_account        empty account id

Rows violating ANY rule are removed. Violation counts are reported per rule and
may overlap (one row can break several rules). Exact duplicate rows are then
removed (the first occurrence is kept).

Zero-amount rows are KEPT (they are valid records); they are only counted.
Each row gets a stable ``tx_id`` (its position in the input file) so that
history features have a deterministic order and rows can be traced later.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import BALANCE_COLUMNS, NUMERIC_COLUMNS, REQUIRED_COLUMNS, VALID_TRANSACTION_TYPES
from src.utils.logging_utils import get_logger
from src.utils.validation import require_columns, require_non_empty

logger = get_logger("cleaning")

RULE_NAMES = [
    "null_required",
    "negative_amount",
    "invalid_step",
    "unknown_type",
    "invalid_label",
    "invalid_flag_column",
    "negative_balance",
    "self_transfer",
    "blank_account",
]


def validity_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Boolean DataFrame: one column per rule, True where the row VIOLATES it."""
    flags = pd.DataFrame(index=df.index)
    flags["null_required"] = df[REQUIRED_COLUMNS].isna().any(axis=1)
    flags["negative_amount"] = df["amount"] < 0
    flags["invalid_step"] = df["step"] < 1
    flags["unknown_type"] = ~df["type"].isin(VALID_TRANSACTION_TYPES)
    flags["invalid_label"] = ~df["isFraud"].isin([0, 1])
    flags["invalid_flag_column"] = ~df["isFlaggedFraud"].isin([0, 1])
    negative_balance = pd.Series(False, index=df.index)
    for column in BALANCE_COLUMNS:
        negative_balance |= df[column] < 0
    flags["negative_balance"] = negative_balance
    flags["self_transfer"] = df["nameOrig"] == df["nameDest"]
    flags["blank_account"] = (
        df["nameOrig"].astype(str).str.strip().eq("")
        | df["nameDest"].astype(str).str.strip().eq("")
    )
    return flags[RULE_NAMES]


def clean_transactions(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return (clean DataFrame, cleaning report). The input is not modified."""
    require_non_empty(df)
    require_columns(df)
    df = df.copy()
    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if "tx_id" not in df.columns:
        df.insert(0, "tx_id", np.arange(len(df), dtype="int64"))

    n_input = len(df)
    flags = validity_flags(df)
    invalid = flags.any(axis=1)
    valid = df.loc[~invalid]

    duplicated = valid.duplicated(subset=REQUIRED_COLUMNS, keep="first")
    clean = (
        valid.loc[~duplicated]
        .copy()
        .astype(
            {
                "tx_id": "int64",
                "step": "int32",
                "isFraud": "int8",
                "isFlaggedFraud": "int8",
                "amount": "float64",
                "oldbalanceOrg": "float64",
                "newbalanceOrig": "float64",
                "oldbalanceDest": "float64",
                "newbalanceDest": "float64",
            }
        )
        .reset_index(drop=True)
    )

    report = {
        "n_input": int(n_input),
        "rule_violations": {name: int(flags[name].sum()) for name in RULE_NAMES},
        "n_invalid_removed": int(invalid.sum()),
        "duplicates_removed": int(duplicated.sum()),
        "n_clean": int(len(clean)),
        "zero_amount_rows": int((df["amount"] == 0).sum()),
    }
    logger.info(
        "Cleaning: %d in -> %d invalid removed, %d duplicates removed -> %d clean",
        n_input,
        report["n_invalid_removed"],
        report["duplicates_removed"],
        report["n_clean"],
    )
    return clean, report