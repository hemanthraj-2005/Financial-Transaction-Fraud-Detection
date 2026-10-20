"""Graph construction: transactions -> aggregated edge list -> weighted directed graph.

Nodes : accounts.
Edges : one directed edge per (sender, receiver) pair, carrying
          weight   = total amount sent on that pair
          tx_count = number of transactions on that pair
The edge TABLE also has ``label_fraud_tx`` (label-derived, display only). The
graph object itself never contains the label, so every metric computed from it
is label-free.

Leakage: use ``max_step`` to build the graph from the training window only.
"""
from __future__ import annotations

import networkx as nx
import pandas as pd

from src.utils.errors import EmptyDatasetError
from src.utils.logging_utils import get_logger
from src.utils.validation import require_columns, require_non_empty

logger = get_logger("graph")

EDGE_COLUMNS = [
    "src",
    "dst",
    "total_amount",
    "tx_count",
    "avg_amount",
    "first_step",
    "last_step",
    "label_fraud_tx",
]
_TX_NEEDED = ["tx_id", "step", "nameOrig", "nameDest", "amount", "isFraud"]


def select_graph_transactions(
    tx: pd.DataFrame, max_step: int | None = None, max_tx: int | None = None
) -> tuple[pd.DataFrame, dict]:
    """Pick the transactions the graph is built from.

    max_step: keep only steps <= max_step (use the end of the training window for ML).
    max_tx  : keep only the first N transactions in time order (memory guard).
    """
    require_non_empty(tx, "transactions table")
    require_columns(tx, _TX_NEEDED, "transactions table")
    n_input = int(len(tx))

    used = tx
    if max_step is not None:
        used = used[used["step"] <= max_step]
    used = used.sort_values(["step", "tx_id"], kind="stable")

    truncated = False
    if max_tx is not None and len(used) > max_tx:
        used = used.head(max_tx)
        truncated = True
        logger.warning(
            "Graph limited to the first %d transactions (by time) to keep memory in check. "
            "Raise --max-tx if you have the RAM.",
            max_tx,
        )
    if len(used) == 0:
        raise EmptyDatasetError("No transactions left to build the graph (check --max-step).")

    info = {
        "n_input_tx": n_input,
        "n_tx_used": int(len(used)),
        "max_step_filter": max_step,
        "max_tx_limit": max_tx,
        "truncated_by_max_tx": truncated,
        "first_step_used": int(used["step"].min()),
        "last_step_used": int(used["step"].max()),
    }
    return used.reset_index(drop=True), info


def build_edge_list(tx: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transactions into one row per (src, dst) pair."""
    require_non_empty(tx, "transactions")
    require_columns(tx, _TX_NEEDED, "transactions")
    edges = (
        tx.groupby(["nameOrig", "nameDest"], sort=False)
        .agg(
            total_amount=("amount", "sum"),
            tx_count=("amount", "size"),
            first_step=("step", "min"),
            last_step=("step", "max"),
            label_fraud_tx=("isFraud", "sum"),
        )
        .reset_index()
        .rename(columns={"nameOrig": "src", "nameDest": "dst"})
    )
    edges["avg_amount"] = edges["total_amount"] / edges["tx_count"]
    edges = edges.astype(
        {"tx_count": "int64", "first_step": "int64", "last_step": "int64", "label_fraud_tx": "int64"}
    )
    return edges[EDGE_COLUMNS].sort_values(["src", "dst"]).reset_index(drop=True)


def build_digraph(edges: pd.DataFrame) -> nx.DiGraph:
    """Weighted directed graph. Edge attributes: weight (amount), tx_count."""
    require_non_empty(edges, "edge list")
    require_columns(edges, ["src", "dst", "total_amount", "tx_count"], "edge list")
    graph = nx.DiGraph()
    graph.add_edges_from(
        (s, d, {"weight": float(w), "tx_count": int(c)})
        for s, d, w, c in zip(edges["src"], edges["dst"], edges["total_amount"], edges["tx_count"])
    )
    return graph