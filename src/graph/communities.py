"""Community detection on the undirected weighted projection of the transaction graph.

Steps
  1. build_undirected: merge A->B and B->A into one undirected edge whose weight is the sum
     of the per-edge weights (log1p(amount) by default).
  2. detect_communities: Louvain (python-louvain, seeded). If it is unavailable or fails,
     fall back to networkx greedy modularity and record the reason.
  3. Communities are renumbered by size (id 0 = largest), so results are stable to read.

Detection is LABEL-FREE: it never looks at isFraud.
"""
from __future__ import annotations

from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd

from config import COMMUNITY_RESOLUTION, COMMUNITY_WEIGHT_MODE, RANDOM_SEED
from src.utils.errors import EmptyDatasetError
from src.utils.logging_utils import get_logger
from src.utils.validation import require_columns, require_non_empty

logger = get_logger("community")

MEMBERSHIP_COLUMNS = ["account", "community_id", "community_size"]
WEIGHT_MODES = ("log_amount", "tx_count", "amount")


def build_undirected(edges: pd.DataFrame, weight_mode: str = COMMUNITY_WEIGHT_MODE) -> nx.Graph:
    """Undirected weighted graph from the aggregated edge list."""
    if weight_mode not in WEIGHT_MODES:
        raise ValueError(f"weight_mode must be one of {WEIGHT_MODES}")
    require_non_empty(edges, "edge list")
    require_columns(edges, ["src", "dst", "total_amount", "tx_count"], "edge list")

    if weight_mode == "log_amount":
        weight = np.log1p(edges["total_amount"].astype(float))
    elif weight_mode == "tx_count":
        weight = edges["tx_count"].astype(float)
    else:
        weight = edges["total_amount"].astype(float)

    swap = edges["src"] > edges["dst"]  # order each pair so A->B and B->A land on the same key
    pairs = pd.DataFrame(
        {
            "u": edges["src"].where(~swap, edges["dst"]),
            "v": edges["dst"].where(~swap, edges["src"]),
            "w": weight.to_numpy(),
        }
    )
    merged = pairs.groupby(["u", "v"], sort=False)["w"].sum().reset_index()

    graph = nx.Graph()
    graph.add_weighted_edges_from(zip(merged["u"], merged["v"], merged["w"]))
    if graph.size(weight="weight") <= 0:
        raise EmptyDatasetError("All edge weights are zero, so communities cannot be detected.")
    return graph


def detect_communities(
    graph: nx.Graph,
    method: str = "louvain",
    resolution: float = COMMUNITY_RESOLUTION,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, dict]:
    """Return (membership table, info dict). method: "louvain" (with fallback) or "greedy"."""
    if method not in {"louvain", "greedy"}:
        raise ValueError("method must be 'louvain' or 'greedy'")
    if graph.number_of_edges() == 0:
        raise EmptyDatasetError("The graph has no edges, so communities cannot be detected.")

    info = {
        "method_requested": method,
        "fallback_reason": None,
        "resolution": resolution,
        "seed": seed,
    }
    partition = None
    modularity = 0.0

    if method == "louvain":
        try:
            import community as community_louvain  # package name: python-louvain

            partition = community_louvain.best_partition(
                graph, weight="weight", resolution=resolution, random_state=seed
            )
            modularity = community_louvain.modularity(partition, graph, weight="weight")
            info["method_used"] = "louvain"
        except Exception as exc:  # missing package or any incompatibility
            info["fallback_reason"] = f"{type(exc).__name__}: {exc}"
            logger.warning("Louvain unavailable (%s). Falling back to greedy modularity.", info["fallback_reason"])
            partition = None

    if partition is None:
        groups = nx.community.greedy_modularity_communities(graph, weight="weight", resolution=resolution)
        partition = {node: i for i, group in enumerate(groups) for node in group}
        modularity = nx.community.modularity(graph, groups, weight="weight", resolution=resolution)
        info["method_used"] = "greedy_modularity"

    # renumber: largest community = 0 (ties broken by the original label, so it is deterministic)
    sizes = Counter(partition.values())
    order = sorted(sizes, key=lambda c: (-sizes[c], c))
    remap = {old: new for new, old in enumerate(order)}

    membership = pd.DataFrame(
        {"account": list(partition), "community_id": [remap[c] for c in partition.values()]}
    )
    membership["community_id"] = membership["community_id"].astype("int64")
    membership["community_size"] = (
        membership.groupby("community_id")["account"].transform("size").astype("int64")
    )
    membership = membership[MEMBERSHIP_COLUMNS].sort_values("account").reset_index(drop=True)

    info["modularity"] = round(float(modularity), 6)
    info["n_communities"] = int(membership["community_id"].nunique())
    logger.info(
        "%s: %d communities over %d accounts (modularity %.3f)",
        info["method_used"], info["n_communities"], len(membership), info["modularity"],
    )
    return membership, info