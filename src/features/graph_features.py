"""Leakage-safe graph and community features for ML (time-block snapshots).

For every time block b the pipeline builds a graph from the transactions in
blocks < b ONLY, computes node metrics and communities on that snapshot, and
attaches them to the transactions of block b. So a transaction never "sees":
  * its own block or any later block (no future information), or
  * its own label (the only label-based feature, the community fraud rate, uses
    labels from earlier blocks and never from steps after ``fit_end``).

Block 0 has no history, so its rows get all-zero features and
``graph_history_available = 0`` (the pipeline drops this warm-up block).

Features per side (s = sender, r = receiver):
  node metrics     : degree, amounts, pass-through ratio, reciprocal partners, PageRank,
                     approximate betweenness, component sizes, small-cycle flag
  <side>_in_graph  : 1 if the account already existed in the snapshot
  community stats  : size, internal flow share, density, pass-through (label-free)
  comm_fraud_rate  : smoothed fraud rate of the community from KNOWN labels (label-based)
Pair features: same_community, pair_prev_tx_count, pair_reverse_tx_count.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import ML_BETWEENNESS_K, ML_COMMUNITY_SMOOTHING, ML_N_BLOCKS
from src.features.time_split import assign_blocks
from src.graph.communities import build_undirected, detect_communities
from src.graph.community_stats import community_statistics
from src.graph.construction import build_digraph, build_edge_list
from src.graph.metrics import compute_node_metrics
from src.utils.errors import FraudProjectError
from src.utils.logging_utils import get_logger
from src.utils.validation import require_columns, require_non_empty

logger = get_logger("graph_features")

GRAPH_INPUT_COLUMNS = ["tx_id", "step", "nameOrig", "nameDest", "amount", "isFraud"]

NODE_FEATURES = [
    "in_degree",
    "out_degree",
    "in_amount",
    "out_amount",
    "pass_through_ratio",
    "reciprocal_partners",
    "pagerank",
    "betweenness",
    "wcc_size",
    "scc_size",
    "in_small_cycle",
]
COMMUNITY_FEATURES = {
    "n_accounts": "comm_size",
    "internal_flow_share": "comm_flow_share",
    "internal_density": "comm_density",
    "mean_pass_through_ratio": "comm_pass_through",
}


def _side_columns(prefix: str) -> list[str]:
    return [
        *[f"{prefix}_{c}" for c in NODE_FEATURES],
        f"{prefix}_in_graph",
        *[f"{prefix}_{name}" for name in COMMUNITY_FEATURES.values()],
        f"{prefix}_comm_fraud_rate",
    ]


GRAPH_FEATURE_COLUMNS = [
    *_side_columns("s"),
    *_side_columns("r"),
    "same_community",
    "pair_prev_tx_count",
    "pair_reverse_tx_count",
    "graph_history_available",
]


def _zero_features(n_rows: int) -> pd.DataFrame:
    return pd.DataFrame(0.0, index=range(n_rows), columns=GRAPH_FEATURE_COLUMNS)


def _community_fraud_rates(known: pd.DataFrame, membership: pd.DataFrame, smoothing: float):
    """Smoothed fraud rate per community from KNOWN-label transactions. Returns (rates, prior)."""
    if known.empty:
        return pd.Series(dtype=float), 0.0
    comm = membership.set_index("account")["community_id"]
    c_src = known["nameOrig"].map(comm).to_numpy()
    c_dst = known["nameDest"].map(comm).to_numpy()
    y = known["isFraud"].to_numpy(dtype=float)
    differ = c_src != c_dst  # a transaction counts once for each community it touches
    table = pd.DataFrame(
        {"c": np.concatenate([c_src, c_dst[differ]]), "f": np.concatenate([y, y[differ]])}
    )
    grouped = table.groupby("c")["f"].agg(n="size", f="sum")
    prior = float(y.mean())
    rates = (grouped["f"] + smoothing * prior) / (grouped["n"] + smoothing)
    return rates, prior


def _block_features(
    rows: pd.DataFrame, history: pd.DataFrame, fit_end: int | None, betweenness_k: int, smoothing: float
) -> pd.DataFrame:
    """Features for the transactions in ``rows``, computed from ``history`` only."""
    out = _zero_features(len(rows))
    if history.empty:
        return out
    edges = build_edge_list(history.assign(isFraud=0))  # the graph itself is label-free
    graph = build_digraph(edges)
    if graph.number_of_nodes() < 2:
        return out

    metrics, _ = compute_node_metrics(graph, betweenness_k=betweenness_k)
    node = metrics.set_index("account")

    membership = cstats = None
    try:
        membership, _ = detect_communities(build_undirected(edges))
        cstats = community_statistics(edges, membership, metrics).set_index("community_id")
    except FraudProjectError as exc:
        logger.warning("Community features skipped for this snapshot: %s", exc)
        membership = None

    rates, prior, comm_of = pd.Series(dtype=float), 0.0, None
    if membership is not None:
        known = history if fit_end is None else history[history["step"] <= fit_end]
        rates, prior = _community_fraud_rates(known, membership, smoothing)
        comm_of = membership.set_index("account")["community_id"]

    cids = {}
    for prefix, key in (("s", "nameOrig"), ("r", "nameDest")):
        accounts = rows[key].to_numpy()
        found = node.reindex(accounts)
        for col in NODE_FEATURES:
            out[f"{prefix}_{col}"] = found[col].fillna(0.0).to_numpy(dtype=float)
        out[f"{prefix}_in_graph"] = found["in_degree"].notna().to_numpy(dtype=float)

        cid = np.full(len(rows), -1, dtype="int64")
        if comm_of is not None:
            cid = comm_of.reindex(accounts).fillna(-1).astype("int64").to_numpy()
            stats = cstats.reindex(cid)
            for src_col, name in COMMUNITY_FEATURES.items():
                out[f"{prefix}_{name}"] = stats[src_col].fillna(0.0).to_numpy(dtype=float)
            out[f"{prefix}_comm_fraud_rate"] = rates.reindex(cid).fillna(prior).to_numpy(dtype=float)
        cids[prefix] = cid

    out["same_community"] = ((cids["s"] == cids["r"]) & (cids["s"] >= 0)).astype(float)

    pairs = rows[["nameOrig", "nameDest"]].reset_index(drop=True)
    forward = edges[["src", "dst", "tx_count"]].rename(
        columns={"src": "nameOrig", "dst": "nameDest", "tx_count": "pair_prev_tx_count"}
    )
    reverse = edges[["src", "dst", "tx_count"]].rename(
        columns={"src": "nameDest", "dst": "nameOrig", "tx_count": "pair_reverse_tx_count"}
    )
    out["pair_prev_tx_count"] = (
        pairs.merge(forward, on=["nameOrig", "nameDest"], how="left")["pair_prev_tx_count"]
        .fillna(0).to_numpy(dtype=float)
    )
    out["pair_reverse_tx_count"] = (
        pairs.merge(reverse, on=["nameOrig", "nameDest"], how="left")["pair_reverse_tx_count"]
        .fillna(0).to_numpy(dtype=float)
    )
    out["graph_history_available"] = 1.0
    return out[GRAPH_FEATURE_COLUMNS]


def compute_graph_features(
    tx: pd.DataFrame,
    n_blocks: int = ML_N_BLOCKS,
    fit_end: int | None = None,
    betweenness_k: int = ML_BETWEENNESS_K,
    smoothing: float = ML_COMMUNITY_SMOOTHING,
) -> pd.DataFrame:
    """Return one row per transaction: tx_id, block and GRAPH_FEATURE_COLUMNS.

    fit_end: labels from steps after this are NEVER used (pass the end of the fit window).
    The output is sorted by (step, tx_id).
    """
    require_non_empty(tx, "transactions")
    require_columns(tx, GRAPH_INPUT_COLUMNS, "transactions")
    tx = tx[GRAPH_INPUT_COLUMNS].sort_values(["step", "tx_id"], kind="stable").reset_index(drop=True)

    min_step = int(tx["step"].min())
    span = int(tx["step"].max()) - min_step + 1
    if span < 2:
        raise FraudProjectError("Graph snapshots need at least 2 distinct time steps.")
    n_blocks = max(2, min(int(n_blocks), span))
    block = assign_blocks(tx["step"], min_step, span, n_blocks)
    block_values = block.to_numpy()

    pieces = []
    for b in range(n_blocks):
        positions = np.flatnonzero(block_values == b)
        if len(positions) == 0:
            continue
        rows = tx.iloc[positions]
        history = tx[block < b]
        logger.info("Snapshot %d/%d: %d earlier transactions -> %d rows", b + 1, n_blocks, len(history), len(rows))
        feats = _block_features(rows, history, fit_end, betweenness_k, smoothing)
        feats.index = positions
        pieces.append(feats)

    out = pd.concat(pieces).sort_index()
    positions = out.index.to_numpy()
    out.insert(0, "tx_id", tx["tx_id"].to_numpy()[positions])
    out.insert(1, "block", block_values[positions])
    return out.reset_index(drop=True)