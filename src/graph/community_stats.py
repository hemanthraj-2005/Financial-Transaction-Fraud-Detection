"""Per-community statistics and the (heuristic) suspicion score.

Label-free columns describe the structure of the community. The three columns that
start with ``label_`` use the fraud label: they are for display and reporting only,
and must never be model inputs unless computed from the TRAINING window only.

Suspicion score: a project-defined heuristic (NOT a validated fraud probability).
Among communities with at least COMMUNITY_MIN_SIZE accounts, each signal is turned into a
0..1 percentile position, the signals are combined with COMMUNITY_SCORE_WEIGHTS and the
result is rescaled to 0..100. Signals with no variation are skipped, so they do not
distort the score. The score is a RELATIVE ranking within one graph.
"""
from __future__ import annotations

import pandas as pd

from config import COMMUNITY_FLAG_SCORE, COMMUNITY_MIN_SIZE, COMMUNITY_SCORE_WEIGHTS
from src.utils.errors import FraudProjectError
from src.utils.validation import require_columns, require_non_empty

COMMUNITY_STATS_COLUMNS = [
    "community_id",
    "n_accounts",
    "n_merchants",
    "merchant_share",
    "internal_edges",
    "internal_tx_count",
    "internal_amount",
    "external_out_amount",
    "external_in_amount",
    "internal_flow_share",
    "internal_density",
    "repeat_ratio",
    "mean_pass_through_ratio",
    "share_in_small_cycle",
    "top_account_by_pagerank",
    "label_fraud_tx",
    "label_fraud_accounts",
    "label_fraud_rate_pct",
]
SCORED_COLUMNS = [*COMMUNITY_STATS_COLUMNS, "eligible", "suspicion_score", "flagged"]

_INT_COLUMNS = ["n_accounts", "n_merchants", "internal_edges", "internal_tx_count",
                "label_fraud_tx", "label_fraud_accounts"]


def community_statistics(
    edges: pd.DataFrame, membership: pd.DataFrame, metrics: pd.DataFrame | None = None
) -> pd.DataFrame:
    """One row per community. ``metrics`` is the Phase 5 node-metrics table (optional)."""
    require_non_empty(edges, "edge list")
    require_columns(edges, ["src", "dst", "total_amount", "tx_count", "label_fraud_tx"], "edge list")
    require_columns(membership, ["account", "community_id"], "membership table")

    comm = membership.set_index("account")["community_id"]
    e = edges[["src", "dst", "total_amount", "tx_count", "label_fraud_tx"]].copy()
    e["c_src"] = e["src"].map(comm)
    e["c_dst"] = e["dst"].map(comm)
    if e[["c_src", "c_dst"]].isna().any().any():
        raise FraudProjectError(
            "Some edge accounts are missing from the membership table. The edge table and the "
            "membership table must come from the same run: re-run the community pipeline."
        )
    e["c_src"] = e["c_src"].astype("int64")
    e["c_dst"] = e["c_dst"].astype("int64")

    same = e["c_src"] == e["c_dst"]
    internal, cross = e[same], e[~same]

    base = membership.groupby("community_id").agg(n_accounts=("account", "size"))
    merchant = membership["account"].str.startswith("M")  # PaySim: merchant ids start with "M"
    base["n_merchants"] = merchant.groupby(membership["community_id"]).sum()

    inner = internal.groupby("c_src").agg(
        internal_edges=("src", "size"),
        internal_tx_count=("tx_count", "sum"),
        internal_amount=("total_amount", "sum"),
        _int_fraud=("label_fraud_tx", "sum"),
    )
    inner.index.name = "community_id"
    out_cross = cross.groupby("c_src").agg(
        external_out_amount=("total_amount", "sum"),
        _out_tx=("tx_count", "sum"),
        _out_fraud=("label_fraud_tx", "sum"),
    )
    out_cross.index.name = "community_id"
    in_cross = cross.groupby("c_dst").agg(
        external_in_amount=("total_amount", "sum"),
        _in_tx=("tx_count", "sum"),
        _in_fraud=("label_fraud_tx", "sum"),
    )
    in_cross.index.name = "community_id"

    stats = base.join(inner).join(out_cross).join(in_cross).fillna(0)

    n = stats["n_accounts"]
    pairs = n * (n - 1)
    touch_amount = stats["internal_amount"] + stats["external_out_amount"] + stats["external_in_amount"]
    touch_tx = stats["internal_tx_count"] + stats["_out_tx"] + stats["_in_tx"]

    stats["merchant_share"] = stats["n_merchants"] / n
    stats["internal_density"] = (stats["internal_edges"] / pairs.where(pairs > 0)).fillna(0.0)
    stats["internal_flow_share"] = (stats["internal_amount"] / touch_amount.where(touch_amount > 0)).fillna(0.0)
    stats["repeat_ratio"] = (
        stats["internal_tx_count"] / stats["internal_edges"].where(stats["internal_edges"] > 0)
    ).fillna(0.0)

    # label-derived (display only): a fraud transaction counts once for each community it touches
    stats["label_fraud_tx"] = stats["_int_fraud"] + stats["_out_fraud"] + stats["_in_fraud"]
    stats["label_fraud_rate_pct"] = (
        100.0 * stats["label_fraud_tx"] / touch_tx.where(touch_tx > 0)
    ).fillna(0.0)
    fraud_edges = e[e["label_fraud_tx"] > 0]
    fraud_accounts = pd.Index(pd.concat([fraud_edges["src"], fraud_edges["dst"]]).unique())
    is_fraud_account = membership["account"].isin(fraud_accounts)
    stats["label_fraud_accounts"] = is_fraud_account.groupby(membership["community_id"]).sum()

    if metrics is not None:
        require_columns(metrics, ["account", "pass_through_ratio", "in_small_cycle", "pagerank"], "node metrics")
        m = membership[["account", "community_id"]].merge(
            metrics[["account", "pass_through_ratio", "in_small_cycle", "pagerank"]], on="account", how="left"
        )
        m[["pass_through_ratio", "in_small_cycle", "pagerank"]] = m[
            ["pass_through_ratio", "in_small_cycle", "pagerank"]
        ].fillna(0.0)
        grouped = m.groupby("community_id")
        stats["mean_pass_through_ratio"] = grouped["pass_through_ratio"].mean()
        stats["share_in_small_cycle"] = grouped["in_small_cycle"].mean()
        top = m.loc[grouped["pagerank"].idxmax()]
        stats["top_account_by_pagerank"] = top.set_index("community_id")["account"]
    else:
        stats["mean_pass_through_ratio"] = 0.0
        stats["share_in_small_cycle"] = 0.0
        stats["top_account_by_pagerank"] = ""

    stats[_INT_COLUMNS] = stats[_INT_COLUMNS].astype("int64")
    stats = stats.reset_index().sort_values("community_id").reset_index(drop=True)
    return stats[COMMUNITY_STATS_COLUMNS]


def score_communities(
    stats: pd.DataFrame,
    min_size: int = COMMUNITY_MIN_SIZE,
    weights: dict | None = None,
    flag_score: float = COMMUNITY_FLAG_SCORE,
) -> pd.DataFrame:
    """Add ``eligible``, ``suspicion_score`` (0-100) and ``flagged`` columns."""
    weights = COMMUNITY_SCORE_WEIGHTS if weights is None else weights
    require_non_empty(stats, "community stats")
    require_columns(stats, ["n_accounts", *weights], "community stats")

    out = stats.copy()
    eligible = out["n_accounts"] >= min_size
    out["eligible"] = eligible.astype("int8")
    score = pd.Series(0.0, index=out.index)

    if int(eligible.sum()) >= 2:
        total_weight = 0.0
        combined = pd.Series(0.0, index=out.index[eligible])
        for column, weight in weights.items():
            values = out.loc[eligible, column].astype(float)
            if values.nunique() <= 1:
                continue  # no variation: this signal cannot separate communities
            ranks = values.rank(method="average")
            ranks = (ranks - ranks.min()) / (ranks.max() - ranks.min())  # exactly 0..1
            combined += weight * ranks
            total_weight += weight
        if total_weight > 0:
            score.loc[combined.index] = 100.0 * combined / total_weight

    out["suspicion_score"] = score.round(2)
    out["flagged"] = (eligible & (out["suspicion_score"] >= flag_score)).astype("int8")
    return out