"""Node-level graph metrics (all label-free).

degree          : unique counterparties (in = senders, out = receivers)
amounts         : total value received / sent
pass_through    : min(in, out) / max(in, out). Near 1 means money flows through
                  the account (mule-like), near 0 means it mostly only receives or only sends
reciprocal      : accounts that this account both sent to and received from
PageRank        : weighted by amount (importance in the money flow)
betweenness     : APPROXIMATE (k sampled start nodes), computed on the largest
                  weakly connected component (capped by node count), unweighted.
                  Accounts outside that scope get 0.
components      : weakly connected component size; strongly connected component
                  size. A strongly connected group of 2+ accounts means money can
                  travel in a loop (cycle / ring structure).
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from config import BETWEENNESS_K, BETWEENNESS_MAX_NODES, PAGERANK_ALPHA, RANDOM_SEED, SMALL_CYCLE_MAX
from src.utils.errors import EmptyDatasetError
from src.utils.logging_utils import get_logger

logger = get_logger("graph")

NODE_COLUMNS = [
    "account",
    "in_degree",
    "out_degree",
    "total_degree",
    "degree_centrality",
    "in_tx_count",
    "out_tx_count",
    "in_amount",
    "out_amount",
    "pass_through_ratio",
    "reciprocal_partners",
    "pagerank",
    "betweenness",
    "wcc_id",
    "wcc_size",
    "scc_size",
    "in_small_cycle",
]


def _component_maps(components) -> tuple[dict, dict]:
    """Map node -> component id (largest = 0) and node -> component size."""
    comp_id: dict = {}
    comp_size: dict = {}
    for i, comp in enumerate(sorted(components, key=len, reverse=True)):
        size = len(comp)
        for node in comp:
            comp_id[node] = i
            comp_size[node] = size
    return comp_id, comp_size


def betweenness_scope(graph: nx.DiGraph, max_nodes: int = BETWEENNESS_MAX_NODES) -> tuple[nx.DiGraph, str]:
    """Subgraph used for betweenness: the largest weakly connected component,
    reduced to its top-``max_nodes`` accounts by degree if it is too large."""
    largest = max(nx.weakly_connected_components(graph), key=len)
    if len(largest) > max_nodes:
        nodes = sorted(largest, key=lambda n: graph.degree(n), reverse=True)[:max_nodes]
        scope = (
            f"top {max_nodes} nodes by degree of the largest weakly connected component "
            f"({len(largest)} nodes)"
        )
    else:
        nodes = largest
        scope = f"largest weakly connected component ({len(largest)} nodes)"
    return graph.subgraph(nodes).copy(), scope


def compute_node_metrics(
    graph: nx.DiGraph,
    betweenness_k: int | None = BETWEENNESS_K,
    betweenness_max_nodes: int = BETWEENNESS_MAX_NODES,
    alpha: float = PAGERANK_ALPHA,
    small_cycle_max: int = SMALL_CYCLE_MAX,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, dict]:
    """Return (one row per account, info dict about components and betweenness)."""
    n = graph.number_of_nodes()
    if n < 2:
        raise EmptyDatasetError("The graph needs at least 2 accounts to compute metrics.")
    nodes = list(graph.nodes)

    in_degree = dict(graph.in_degree())
    out_degree = dict(graph.out_degree())
    in_amount = dict(graph.in_degree(weight="weight"))
    out_amount = dict(graph.out_degree(weight="weight"))
    in_tx = dict(graph.in_degree(weight="tx_count"))
    out_tx = dict(graph.out_degree(weight="tx_count"))

    logger.info("Computing PageRank on %d nodes / %d edges ...", n, graph.number_of_edges())
    pagerank = nx.pagerank(graph, alpha=alpha, weight="weight", max_iter=200)

    wcc_id, wcc_size = _component_maps(nx.weakly_connected_components(graph))
    _, scc_size = _component_maps(nx.strongly_connected_components(graph))

    sub, scope = betweenness_scope(graph, betweenness_max_nodes)
    exact = betweenness_k is None or betweenness_k >= sub.number_of_nodes()
    k = None if exact else int(betweenness_k)
    logger.info("Betweenness on %s, %s ...", scope, "exact" if exact else f"sampled with k={k}")
    betweenness = nx.betweenness_centrality(sub, k=k, normalized=True, weight=None, seed=seed)

    reciprocal = [len(set(graph.successors(a)) & set(graph.predecessors(a))) for a in nodes]
    in_amt = np.array([in_amount[a] for a in nodes], dtype="float64")
    out_amt = np.array([out_amount[a] for a in nodes], dtype="float64")
    low, high = np.minimum(in_amt, out_amt), np.maximum(in_amt, out_amt)
    pass_through = np.divide(low, high, out=np.zeros_like(high), where=high > 0)

    total_degree = np.array([in_degree[a] + out_degree[a] for a in nodes], dtype="int64")
    sizes = np.array([scc_size[a] for a in nodes], dtype="int64")

    metrics = pd.DataFrame(
        {
            "account": nodes,
            "in_degree": [in_degree[a] for a in nodes],
            "out_degree": [out_degree[a] for a in nodes],
            "total_degree": total_degree,
            "degree_centrality": total_degree / (n - 1),
            "in_tx_count": [in_tx[a] for a in nodes],
            "out_tx_count": [out_tx[a] for a in nodes],
            "in_amount": in_amt,
            "out_amount": out_amt,
            "pass_through_ratio": pass_through,
            "reciprocal_partners": reciprocal,
            "pagerank": [pagerank[a] for a in nodes],
            "betweenness": [betweenness.get(a, 0.0) for a in nodes],
            "wcc_id": [wcc_id[a] for a in nodes],
            "wcc_size": [wcc_size[a] for a in nodes],
            "scc_size": sizes,
            "in_small_cycle": ((sizes >= 2) & (sizes <= small_cycle_max)).astype("int8"),
        }
    )
    metrics = metrics.astype({c: "int64" for c in (
        "in_degree", "out_degree", "in_tx_count", "out_tx_count", "reciprocal_partners",
        "wcc_id", "wcc_size", "scc_size")})
    metrics = metrics[NODE_COLUMNS].sort_values("account").reset_index(drop=True)

    scc_values = sorted(set(scc_size.values()), reverse=True)
    info = {
        "n_nodes": int(n),
        "n_edges": int(graph.number_of_edges()),
        "n_weakly_connected_components": int(max(wcc_id.values()) + 1),
        "largest_weakly_connected_component": int(max(wcc_size.values())),
        "largest_strongly_connected_component": int(scc_values[0]),
        "n_accounts_in_small_cycles": int(metrics["in_small_cycle"].sum()),
        "betweenness": {
            "scope": scope,
            "nodes_in_scope": int(sub.number_of_nodes()),
            "exact": bool(exact),
            "k_sampled_start_nodes": k,
            "note": "Approximate and unweighted. Accounts outside the scope have betweenness 0.",
        },
    }
    return metrics, info


def top_accounts(metrics: pd.DataFrame, column: str, n: int = 10) -> list[dict]:
    top = metrics.nlargest(n, column)[["account", column]]
    return top.to_dict("records")