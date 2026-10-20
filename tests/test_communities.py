"""Phase 6 tests: undirected projection, community detection, statistics, scoring, pipeline."""
import json

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from src.data.synthetic_generator import generate_synthetic_transactions
from src.graph.communities import MEMBERSHIP_COLUMNS, build_undirected, detect_communities
from src.graph.community_pipeline import run_community_detection
from src.graph.community_stats import COMMUNITY_STATS_COLUMNS, community_statistics, score_communities
from src.graph.construction import build_digraph, build_edge_list
from src.graph.metrics import compute_node_metrics
from src.preprocessing.cleaning import clean_transactions
from src.utils.errors import EmptyDatasetError, FraudProjectError
from src.utils.io_utils import load_table


# ------------------------------------------------------------------ helpers
def _edges(rows):
    """rows: (src, dst, total_amount, tx_count, label_fraud_tx)"""
    return pd.DataFrame(
        [dict(src=s, dst=d, total_amount=float(a), tx_count=int(t), label_fraud_tx=int(f))
         for s, d, a, t, f in rows]
    )


def _clique_rows(prefix, n, amount=100.0):
    nodes = [f"{prefix}{i}" for i in range(n)]
    return [(nodes[i], nodes[j], amount, 2, 0) for i in range(n) for j in range(i + 1, n)]


def _two_cliques():
    return _edges(_clique_rows("A", 5) + _clique_rows("B", 4) + [("A0", "B0", 1.0, 1, 0)])


@pytest.fixture(scope="module")
def synth_tx():
    raw = generate_synthetic_transactions(n_normal=2000, n_customers=600, seed=5)
    return clean_transactions(raw)[0]


@pytest.fixture(scope="module")
def synth_edges(synth_tx):
    return build_edge_list(synth_tx)


# ------------------------------------------------------------------ undirected projection
def test_build_undirected_merges_both_directions():
    edges = _edges([("A", "B", 10, 1, 0), ("B", "A", 5, 3, 0)])
    graph = build_undirected(edges, "amount")
    assert graph.number_of_edges() == 1
    assert graph["A"]["B"]["weight"] == 15.0
    assert build_undirected(edges, "tx_count")["A"]["B"]["weight"] == 4.0
    assert np.isclose(build_undirected(edges, "log_amount")["A"]["B"]["weight"], np.log1p(10) + np.log1p(5))


def test_weight_mode_validation():
    with pytest.raises(ValueError):
        build_undirected(_edges([("A", "B", 1, 1, 0)]), "bogus")
    with pytest.raises(EmptyDatasetError):
        build_undirected(_edges([("A", "B", 0, 1, 0)]), "amount")  # total weight is zero


# ------------------------------------------------------------------ detection
@pytest.mark.parametrize(
    "method,expected",
    [("louvain", {"louvain", "greedy_modularity"}), ("greedy", {"greedy_modularity"})],
)
def test_two_cliques_are_separated(method, expected):
    graph = build_undirected(_two_cliques(), "amount")
    membership, info = detect_communities(graph, method=method)
    assert info["method_used"] in expected
    assert list(membership.columns) == MEMBERSHIP_COLUMNS
    assert membership["community_id"].nunique() == 2

    ids = membership.set_index("account")["community_id"]
    assert len({ids[f"A{i}"] for i in range(5)}) == 1
    assert len({ids[f"B{i}"] for i in range(4)}) == 1
    assert ids["A0"] != ids["B0"]
    assert ids["A0"] == 0  # the larger community gets id 0
    assert membership.set_index("account").loc["A0", "community_size"] == 5
    assert info["modularity"] > 0.3


def test_detection_rejects_bad_input():
    with pytest.raises(ValueError):
        detect_communities(nx.Graph(), method="bogus")
    with pytest.raises(EmptyDatasetError):
        detect_communities(nx.Graph())


def test_detection_is_label_free_and_reproducible(synth_edges):
    first, _ = detect_communities(build_undirected(synth_edges))
    flipped, _ = detect_communities(build_undirected(synth_edges.assign(label_fraud_tx=0)))
    again, _ = detect_communities(build_undirected(synth_edges))
    pd.testing.assert_frame_equal(first, flipped)
    pd.testing.assert_frame_equal(first, again)


# ------------------------------------------------------------------ statistics
def _small_case():
    edges = _edges(
        [
            ("A", "B", 100, 2, 1),
            ("B", "A", 50, 1, 0),
            ("C", "D", 30, 1, 0),
            ("B", "C", 10, 1, 1),  # the only edge between the two communities
        ]
    )
    membership = pd.DataFrame(
        {"account": list("ABCD"), "community_id": [0, 0, 1, 1], "community_size": [2, 2, 2, 2]}
    )
    return edges, membership


def test_statistics_exact_values():
    edges, membership = _small_case()
    stats = community_statistics(edges, membership)
    assert list(stats.columns) == COMMUNITY_STATS_COLUMNS
    s = stats.set_index("community_id")

    c0 = s.loc[0]
    assert (c0.n_accounts, c0.n_merchants, c0.internal_edges, c0.internal_tx_count) == (2, 0, 2, 3)
    assert (c0.internal_amount, c0.external_out_amount, c0.external_in_amount) == (150.0, 10.0, 0.0)
    assert np.isclose(c0.internal_density, 1.0)
    assert np.isclose(c0.internal_flow_share, 150 / 160)
    assert np.isclose(c0.repeat_ratio, 1.5)
    assert (c0.label_fraud_tx, c0.label_fraud_accounts) == (2, 2)
    assert np.isclose(c0.label_fraud_rate_pct, 50.0)
    assert c0.mean_pass_through_ratio == 0.0 and c0.top_account_by_pagerank == ""

    c1 = s.loc[1]
    assert (c1.internal_edges, c1.internal_tx_count, c1.internal_amount) == (1, 1, 30.0)
    assert (c1.external_out_amount, c1.external_in_amount) == (0.0, 10.0)
    assert np.isclose(c1.internal_density, 0.5)
    assert np.isclose(c1.internal_flow_share, 0.75)
    assert np.isclose(c1.repeat_ratio, 1.0)
    assert (c1.label_fraud_tx, c1.label_fraud_accounts) == (1, 1)
    assert np.isclose(c1.label_fraud_rate_pct, 50.0)


def test_statistics_with_metrics():
    edges, membership = _small_case()
    metrics = pd.DataFrame(
        {
            "account": list("ABCD"),
            "pass_through_ratio": [1.0, 0.5, 0.2, 0.0],
            "in_small_cycle": [1, 0, 0, 0],
            "pagerank": [0.4, 0.1, 0.3, 0.2],
        }
    )
    s = community_statistics(edges, membership, metrics).set_index("community_id")
    assert np.isclose(s.loc[0, "mean_pass_through_ratio"], 0.75)
    assert np.isclose(s.loc[0, "share_in_small_cycle"], 0.5)
    assert s.loc[0, "top_account_by_pagerank"] == "A"
    assert np.isclose(s.loc[1, "mean_pass_through_ratio"], 0.1)
    assert s.loc[1, "top_account_by_pagerank"] == "C"


def test_statistics_rejects_unknown_accounts():
    edges, membership = _small_case()
    with pytest.raises(FraudProjectError):
        community_statistics(edges, membership[membership["account"] != "D"])


# ------------------------------------------------------------------ scoring
_WEIGHTS = {
    "internal_flow_share": 0.35,
    "internal_density": 0.25,
    "mean_pass_through_ratio": 0.25,
    "share_in_small_cycle": 0.15,
}


def test_score_communities_ranks_and_flags():
    stats = pd.DataFrame(
        {
            "community_id": [0, 1, 2, 3],
            "n_accounts": [5, 5, 5, 2],
            "internal_flow_share": [0.9, 0.5, 0.1, 1.0],
            "internal_density": [0.8, 0.4, 0.1, 1.0],
            "mean_pass_through_ratio": [0.9, 0.5, 0.1, 1.0],
            "share_in_small_cycle": [0.0, 0.0, 0.0, 0.0],  # constant: skipped
        }
    )
    scored = score_communities(stats, min_size=3, weights=_WEIGHTS, flag_score=70.0)
    assert scored["eligible"].tolist() == [1, 1, 1, 0]
    assert np.allclose(scored["suspicion_score"], [100.0, 50.0, 0.0, 0.0])  # size-2 community is not scored
    assert scored["flagged"].tolist() == [1, 0, 0, 0]


def test_score_constant_signals_give_zero():
    stats = pd.DataFrame(
        {
            "community_id": [0, 1, 2],
            "n_accounts": [4, 4, 4],
            "internal_flow_share": [0.5, 0.5, 0.5],
            "internal_density": [0.3, 0.3, 0.3],
            "mean_pass_through_ratio": [0.2, 0.2, 0.2],
            "share_in_small_cycle": [0.0, 0.0, 0.0],
        }
    )
    scored = score_communities(stats, min_size=3, weights=_WEIGHTS, flag_score=70.0)
    assert (scored["suspicion_score"] == 0.0).all()
    assert (scored["flagged"] == 0).all()


# ------------------------------------------------------------------ pipeline
def test_pipeline_end_to_end(tmp_path, synth_edges):
    metrics, _ = compute_node_metrics(build_digraph(synth_edges), betweenness_k=10)
    report = run_community_detection(
        edges=synth_edges, metrics=metrics,
        output_dir=tmp_path / "out", report_path=tmp_path / "community_report.json",
    )
    membership = load_table(tmp_path / "out" / "community_membership")
    stats = load_table(tmp_path / "out" / "community_stats")

    assert len(membership) == len(metrics)
    assert int(stats["n_accounts"].sum()) == len(membership)
    assert report["n_communities"] == len(stats)
    assert {"eligible", "suspicion_score", "flagged"} <= set(stats.columns)

    saved = json.loads((tmp_path / "community_report.json").read_text())
    assert "top_suspicious_communities" in saved and "fraud_signal_check" in saved