"""Phase 5 tests: graph construction, node metrics, pipeline."""
import json

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from src.data.synthetic_generator import generate_synthetic_transactions
from src.graph.construction import EDGE_COLUMNS, build_digraph, build_edge_list, select_graph_transactions
from src.graph.metrics import NODE_COLUMNS, betweenness_scope, compute_node_metrics
from src.graph.pipeline import run_graph_analytics
from src.preprocessing.cleaning import clean_transactions
from src.utils.errors import EmptyDatasetError
from src.utils.io_utils import load_table

# (step, sender, receiver, amount, isFraud)
ROWS = [
    (1, "A", "B", 100, 0),
    (2, "A", "B", 50, 1),
    (3, "B", "C", 30, 0),
    (4, "C", "A", 20, 0),
    (5, "D", "B", 10, 0),
]


def _tx(rows):
    return pd.DataFrame(
        [dict(tx_id=i, step=s, nameOrig=o, nameDest=d, amount=float(a), isFraud=int(f))
         for i, (s, o, d, a, f) in enumerate(rows)]
    )


@pytest.fixture(scope="module")
def synth_tx():
    raw = generate_synthetic_transactions(n_normal=2000, n_customers=600, seed=5)
    return clean_transactions(raw)[0]


def _small_graph():
    return build_digraph(build_edge_list(_tx(ROWS)))


# ------------------------------------------------------------ construction
def test_select_transactions_orders_and_limits():
    tx = _tx([(3, "A", "B", 1, 0), (1, "A", "B", 2, 0), (2, "C", "D", 3, 0)])
    used, info = select_graph_transactions(tx, max_tx=2)
    assert used["step"].tolist() == [1, 2]
    assert info["truncated_by_max_tx"] is True and info["n_tx_used"] == 2


def test_select_transactions_max_step():
    tx = _tx([(1, "A", "B", 1, 0), (2, "A", "B", 2, 0), (3, "C", "D", 3, 0)])
    used, info = select_graph_transactions(tx, max_step=2)
    assert len(used) == 2 and info["last_step_used"] == 2
    with pytest.raises(EmptyDatasetError):
        select_graph_transactions(tx, max_step=0)


def test_edge_list_aggregates_pairs():
    edges = build_edge_list(_tx(ROWS))
    assert list(edges.columns) == EDGE_COLUMNS
    assert len(edges) == 4
    ab = edges.set_index(["src", "dst"]).loc[("A", "B")]
    assert (ab.total_amount, ab.tx_count, ab.avg_amount) == (150.0, 2, 75.0)
    assert (ab.first_step, ab.last_step, ab.label_fraud_tx) == (1, 2, 1)


def test_build_digraph_weights():
    graph = _small_graph()
    assert graph.number_of_nodes() == 4 and graph.number_of_edges() == 4
    assert graph["A"]["B"]["weight"] == 150.0
    assert graph["A"]["B"]["tx_count"] == 2


# ------------------------------------------------------------ metrics
def test_node_metrics_known_graph():
    metrics, info = compute_node_metrics(_small_graph())
    assert list(metrics.columns) == NODE_COLUMNS
    m = metrics.set_index("account")

    b = m.loc["B"]
    assert (b.in_degree, b.out_degree, b.total_degree) == (2, 1, 3)
    assert (b.in_amount, b.out_amount) == (160.0, 30.0)
    assert (b.in_tx_count, b.out_tx_count) == (3, 1)
    assert np.isclose(b.pass_through_ratio, 30.0 / 160.0)
    assert (b.scc_size, b.wcc_size, b.in_small_cycle) == (3, 4, 1)

    d = m.loc["D"]
    assert (d.in_degree, d.scc_size, d.in_small_cycle, d.reciprocal_partners) == (0, 1, 0, 0)
    assert m.loc["A"].in_small_cycle == 1
    assert info["largest_strongly_connected_component"] == 3
    assert info["n_weakly_connected_components"] == 1


def test_pagerank_sums_to_one():
    metrics, _ = compute_node_metrics(_small_graph())
    assert np.isclose(metrics["pagerank"].sum(), 1.0, atol=1e-6)
    assert (metrics["pagerank"] > 0).all()


def test_betweenness_exact_on_path():
    graph = build_digraph(build_edge_list(_tx([(1, "X", "Y", 5, 0), (2, "Y", "Z", 5, 0)])))
    metrics, info = compute_node_metrics(graph)
    m = metrics.set_index("account")
    assert info["betweenness"]["exact"] is True
    assert np.isclose(m.loc["Y", "betweenness"], 0.5)
    assert m.loc["X", "betweenness"] == 0.0 and m.loc["Z", "betweenness"] == 0.0


def test_degree_centrality_matches_networkx():
    graph = _small_graph()
    metrics, _ = compute_node_metrics(graph)
    expected = nx.degree_centrality(graph)
    for account, value in metrics.set_index("account")["degree_centrality"].items():
        assert np.isclose(value, expected[account])


def test_betweenness_scope_caps_nodes():
    chain = nx.path_graph(10, create_using=nx.DiGraph)
    capped, scope = betweenness_scope(chain, max_nodes=4)
    assert capped.number_of_nodes() == 4 and "top 4" in scope
    full, scope_full = betweenness_scope(chain, max_nodes=100)
    assert full.number_of_nodes() == 10 and "top" not in scope_full


def test_metrics_are_label_free(synth_tx):
    flipped = synth_tx.copy()
    flipped["isFraud"] = 1 - flipped["isFraud"]
    m1, _ = compute_node_metrics(build_digraph(build_edge_list(synth_tx)), betweenness_k=15)
    m2, _ = compute_node_metrics(build_digraph(build_edge_list(flipped)), betweenness_k=15)
    pd.testing.assert_frame_equal(m1, m2)


def test_metrics_reject_tiny_graph():
    graph = nx.DiGraph()
    graph.add_node("A")
    with pytest.raises(EmptyDatasetError):
        compute_node_metrics(graph)


# ------------------------------------------------------------ pipeline
def test_pipeline_end_to_end(tmp_path, synth_tx):
    report = run_graph_analytics(tx=synth_tx, betweenness_k=20, output_dir=tmp_path / "out",
                                 report_path=tmp_path / "graph_report.json")
    metrics = load_table(tmp_path / "out" / "graph_metrics")
    edges = load_table(tmp_path / "out" / "graph_edges")
    assert list(metrics.columns) == NODE_COLUMNS
    assert list(edges.columns) == EDGE_COLUMNS
    assert report["n_nodes"] == len(metrics) and report["n_edges"] == len(edges)
    saved = json.loads((tmp_path / "graph_report.json").read_text())
    assert "top_accounts" in saved and "fraud_vs_graph_signals" in saved


def test_pipeline_respects_max_step_and_max_tx(tmp_path, synth_tx):
    by_step = run_graph_analytics(tx=synth_tx, max_step=100, betweenness_k=10,
                                  output_dir=tmp_path / "a", report_path=tmp_path / "a.json")
    assert by_step["selection"]["last_step_used"] <= 100

    capped = run_graph_analytics(tx=synth_tx, max_tx=500, betweenness_k=10,
                                 output_dir=tmp_path / "b", report_path=tmp_path / "b.json")
    assert capped["selection"]["n_tx_used"] == 500
    assert capped["selection"]["truncated_by_max_tx"] is True