"""Column groups shared by the Pandas and Spark engines.

Both engines must produce exactly OUTPUT_COLUMNS (same names, same order), so
the rest of the project never needs to know which engine ran.
"""
from __future__ import annotations

from config import BALANCE_COLUMNS, REQUIRED_COLUMNS, VALID_TRANSACTION_TYPES

TYPE_ONEHOT_COLUMNS = [f"type_{t}" for t in VALID_TRANSACTION_TYPES]

TIME_FEATURES = ["hour_of_day", "day", "is_night"]

# Derived from balances. In PaySim, fraudulent transfers empty the sender's
# account, so these (and the raw balance columns) can leak the label. Phase 7
# trains with and without them (ablation) and reports the difference.
BALANCE_DERIVED_FEATURES = [
    "balance_error_orig",
    "balance_error_dest",
    "orig_balance_emptied",
    "amount_to_oldbalance_ratio",
]

# "Past-only" behaviour features: for each transaction they use ONLY earlier
# transactions (ordered by step, then tx_id). They never use the label.
HISTORY_FEATURES = [
    "orig_prev_tx_count",
    "orig_prev_amount_sum",
    "orig_prev_amount_mean",
    "orig_steps_since_last_tx",
    "dest_prev_tx_count",
    "dest_prev_amount_sum",
    "dest_steps_since_last_tx",
]

# Columns that can leak the fraud label; excluded or ablated in Phase 7.
LEAKAGE_RISK_COLUMNS = [*BALANCE_COLUMNS, *BALANCE_DERIVED_FEATURES, "isFlaggedFraud"]

ENGINEERED_COLUMNS = [
    "timestamp",
    *TIME_FEATURES,
    "log_amount",
    "type_code",
    *TYPE_ONEHOT_COLUMNS,
    "dest_is_merchant",
    *BALANCE_DERIVED_FEATURES,
    *HISTORY_FEATURES,
]

OUTPUT_COLUMNS = ["tx_id", *REQUIRED_COLUMNS, *ENGINEERED_COLUMNS]

# One row per account. Columns starting with "label_" are derived from the
# fraud label: use them for display/analysis, never as ML features unless the
# aggregates were computed from the TRAINING window only.
ACCOUNT_COLUMNS = [
    "account",
    "is_merchant",
    "out_tx_count",
    "out_amount_sum",
    "out_amount_avg",
    "out_unique_counterparties",
    "in_tx_count",
    "in_amount_sum",
    "in_amount_avg",
    "in_unique_counterparties",
    "total_tx_count",
    "total_amount",
    "avg_tx_amount",
    "n_counterparties",
    "first_step",
    "last_step",
    "label_fraud_sent",
    "label_fraud_received",
]

STATS_COLUMNS = ["n_transactions", "n_fraud", "fraud_rate_pct", "total_amount", "fraud_amount"]