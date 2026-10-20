"""Transaction-level feature engineering (Pandas engine).

Input : the clean DataFrame from cleaning.clean_transactions (has ``tx_id``).
Output: the same rows, sorted by (step, tx_id), with schema.OUTPUT_COLUMNS.

Leakage safety
--------------
* History features use ONLY earlier rows (strictly before the current one in
  (step, tx_id) order) and never use the fraud label.
* Balance-derived features are kept but listed in schema.LEAKAGE_RISK_COLUMNS,
  because PaySim fraud empties the sender's account; Phase 7 ablates them.
* Encoders are fixed mappings (config.TYPE_CODES), so nothing is "fitted" on
  the test period. Scaling is fitted separately on training data only
  (src/preprocessing/scaling.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    BASE_TIMESTAMP,
    NIGHT_END_HOUR,
    REQUIRED_COLUMNS,
    TYPE_CODES,
    VALID_TRANSACTION_TYPES,
)
from src.preprocessing.schema import OUTPUT_COLUMNS
from src.utils.validation import require_columns, require_non_empty


def _add_history(out: pd.DataFrame, key: str, prefix: str, with_mean: bool) -> None:
    """Past-only count / amount-sum / gap-since-last features per account."""
    grouped = out.groupby(key, sort=False)
    prev_count = grouped.cumcount()
    prev_sum = (grouped["amount"].cumsum() - out["amount"]).clip(lower=0.0)
    gap = out["step"] - grouped["step"].shift(1)

    out[f"{prefix}_prev_tx_count"] = prev_count.astype("int32")
    out[f"{prefix}_prev_amount_sum"] = prev_sum
    if with_mean:
        out[f"{prefix}_prev_amount_mean"] = np.where(prev_count > 0, prev_sum / prev_count.clip(lower=1), 0.0)
    out[f"{prefix}_steps_since_last_tx"] = gap.fillna(-1).astype("int32")  # -1 = no earlier transaction


def add_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    require_non_empty(df)
    require_columns(df, [*REQUIRED_COLUMNS, "tx_id"])

    out = df.sort_values(["step", "tx_id"], kind="stable").reset_index(drop=True)
    step0 = out["step"].astype("int64") - 1  # step 1 = hour 0 of day 0

    # ---- time ---------------------------------------------------------
    out["timestamp"] = pd.Timestamp(BASE_TIMESTAMP) + pd.to_timedelta(step0, unit="h")
    out["hour_of_day"] = (step0 % 24).astype("int8")
    out["day"] = (step0 // 24).astype("int16")
    out["is_night"] = (out["hour_of_day"] < NIGHT_END_HOUR).astype("int8")

    # ---- amount / categorical encoding ----------------------------------
    out["log_amount"] = np.log1p(out["amount"])
    out["type_code"] = out["type"].map(TYPE_CODES).astype("int8")
    for tx_type in VALID_TRANSACTION_TYPES:
        out[f"type_{tx_type}"] = (out["type"] == tx_type).astype("int8")
    out["dest_is_merchant"] = out["nameDest"].str.startswith("M").astype("int8")

    # ---- balance-derived (leakage-risk, see module docstring) ----------------
    out["balance_error_orig"] = out["newbalanceOrig"] + out["amount"] - out["oldbalanceOrg"]
    out["balance_error_dest"] = out["oldbalanceDest"] + out["amount"] - out["newbalanceDest"]
    out["orig_balance_emptied"] = ((out["oldbalanceOrg"] > 0) & (out["newbalanceOrig"] == 0)).astype("int8")
    out["amount_to_oldbalance_ratio"] = (out["amount"] / out["oldbalanceOrg"].where(out["oldbalanceOrg"] > 0)).fillna(0.0)

    # ---- past-only account history ----------------------------------------
    _add_history(out, "nameOrig", "orig", with_mean=True)
    _add_history(out, "nameDest", "dest", with_mean=False)

    return out[OUTPUT_COLUMNS]