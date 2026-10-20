"""Feature sets for the models and the ablation.

tx_only          : transaction features + past-only account history (no graph, no balances)
tx_graph         : + graph and community features   <- PRIMARY (honest) feature set
tx_graph_balance : + balance columns (they reveal the label in PaySim; ablation only)

Never features: isFlaggedFraud (rule-based flag), the label itself, step/day (time
identifiers), ids, and anything starting with 'label_'.
"""
from __future__ import annotations

from config import BALANCE_COLUMNS
from src.features.graph_features import GRAPH_FEATURE_COLUMNS
from src.preprocessing.schema import BALANCE_DERIVED_FEATURES, HISTORY_FEATURES, TYPE_ONEHOT_COLUMNS

TX_FEATURES = [
    "amount",
    "log_amount",
    "hour_of_day",
    "is_night",
    "dest_is_merchant",
    *TYPE_ONEHOT_COLUMNS,
    *HISTORY_FEATURES,
]
BALANCE_FEATURES = [*BALANCE_COLUMNS, *BALANCE_DERIVED_FEATURES]
GRAPH_FEATURES = list(GRAPH_FEATURE_COLUMNS)

FEATURE_SETS = {
    "tx_only": list(TX_FEATURES),
    "tx_graph": [*TX_FEATURES, *GRAPH_FEATURES],
    "tx_graph_balance": [*TX_FEATURES, *GRAPH_FEATURES, *BALANCE_FEATURES],
}
PRIMARY_FEATURE_SET = "tx_graph"

NEVER_FEATURES = [
    "isFlaggedFraud", "isFraud", "step", "day", "tx_id", "nameOrig", "nameDest", "timestamp", "block", "split",
]

BASE_COLUMNS = ["tx_id", "step", "nameOrig", "nameDest", "amount", "isFraud"]
# columns read from the Phase 3 transactions table
LOAD_COLUMNS = list(dict.fromkeys([*BASE_COLUMNS, *TX_FEATURES, *BALANCE_FEATURES]))