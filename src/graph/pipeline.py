"""Graph analytics pipeline.

Reads the Phase 3 transactions table, builds the weighted directed graph,
computes node metrics and saves:
    data/processed/graph_edges.parquet     one row per (src, dst) pair
    data/processed/graph_metrics.parquet   one row per account (label-free metrics)
    reports/graph_report.json              graph statistics, top accounts, sanity check
(.csv instead of .parquet if pyarrow is missing)

Usage:
    python -m src.graph.pipeline
    python -m src.graph.pipeline --max-tx 200000 --k 100
    python -m src.graph.pipeline --max-step 520     # training window only (for ML)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from config import (
    BETWEENNESS_K,
    GRAPH_EDGES_STEM,
    GRAPH_MAX_TX,
    GRAPH_METRICS_STEM,
    GRAPH_REPORT_PATH,
    PROCESSED_DIR,
    TRANSACTIONS_CLEAN_STEM,
    ensure_directories,
)
from src.graph.construction import build_digraph, build_edge_list, select_graph_transactions
from src.graph.metrics import compute_node_metrics, top_accounts
from src.utils.errors import DatasetNotFoundError, FraudProjectError
from src.utils.io_utils import save_table, table_paths, write_json
from src.utils.logging_utils import get_logger

logger = get_logger("graph")

_TX_COLUMNS = ["tx_id", "step", "nameOrig", "nameDest", "amount", "isFraud"]


def _load_transactions(stem: Path) -> pd.DataFrame:
    """Load only the columns the graph needs (saves a lot of RAM on the full dataset)."""
    parquet_path, csv_path = table_paths(stem)
    if parquet_path.exists():
        return pd.read_parquet(parquet_path, columns=_TX_COLUMNS)
    if csv_path.exists():
        return pd.read_csv(csv_path, usecols=_TX_COLUMNS)
    raise DatasetNotFoundError(
        f"Processed table '{stem.name}' not found in {stem.parent}. "
        "Run the preprocessing pipeline first:  python -m src.preprocessing.pipeline"
    )


def fraud_vs_graph_signals(edges: pd.DataFrame, metrics: pd.DataFrame) -> dict:
    """DISPLAY-ONLY sanity check: do accounts touching fraud look different in the graph?

    Uses the fraud label, so it is for understanding and reporting only.
    Never feed it to a model.
    """
    fraud_edges = edges[edges["label_fraud_tx"] > 0]
    fraud_accounts = set(fraud_edges["src"]) | set(fraud_edges["dst"])
    flag = metrics["account"].isin(fraud_accounts)
    result = {
        "note": "Uses the fraud label: for display and reporting only, not a model input.",
        "n_accounts_touching_fraud": int(flag.sum()),
        "comparison": None,
    }
    if flag.sum() == 0 or flag.all():
        return result
    columns = ["total_degree", "pagerank", "pass_through_ratio", "reciprocal_partners", "in_small_cycle"]
    grouped = metrics.groupby(flag.rename("touches_fraud"))[columns].mean()
    result["comparison"] = grouped.rename(index={True: "touches_fraud", False: "no_fraud"}).round(6).to_dict("index")
    return result


def run_graph_analytics(
    max_step: int | None = None,
    max_tx: int | None = GRAPH_MAX_TX,
    betweenness_k: int | None = BETWEENNESS_K,
    tx: pd.DataFrame | None = None,
    output_dir: Path | str = PROCESSED_DIR,
    report_path: Path | str = GRAPH_REPORT_PATH,
) -> dict:
    started = time.perf_counter()
    output_dir = Path(output_dir)
    if tx is None:
        tx = _load_transactions(TRANSACTIONS_CLEAN_STEM)

    used, selection = select_graph_transactions(tx, max_step=max_step, max_tx=max_tx)
    edges = build_edge_list(used)
    graph = build_digraph(edges)
    metrics, info = compute_node_metrics(graph, betweenness_k=betweenness_k)

    edges_path = save_table(edges, output_dir / GRAPH_EDGES_STEM.name)
    metrics_path = save_table(metrics, output_dir / GRAPH_METRICS_STEM.name)

    n, e = info["n_nodes"], info["n_edges"]
    report = {
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "selection": selection,
        "n_nodes": n,
        "n_edges": e,
        "density": e / (n * (n - 1)) if n > 1 else 0.0,
        "components": {k: v for k, v in info.items() if k not in {"n_nodes", "n_edges", "betweenness"}},
        "betweenness": info["betweenness"],
        "top_accounts": {
            "by_pagerank": top_accounts(metrics, "pagerank"),
            "by_betweenness": top_accounts(metrics, "betweenness"),
            "by_total_degree": top_accounts(metrics, "total_degree"),
            "by_pass_through_ratio": top_accounts(metrics, "pass_through_ratio"),
        },
        "fraud_vs_graph_signals": fraud_vs_graph_signals(edges, metrics),
        "outputs": {"edges": str(edges_path), "metrics": str(metrics_path)},
        "notes": [
            "Node metrics are label-free. For ML, rebuild with --max-step set to the end of the training window.",
            "Betweenness is approximate (sampled) and unweighted; PageRank is weighted by amount.",
            "Graph metrics are risk indicators, not proof of fraud.",
        ],
    }
    write_json(report, report_path)
    logger.info("Graph report written to %s", report_path)
    return report


def print_summary(report: dict) -> None:
    sel, comp, btw = report["selection"], report["components"], report["betweenness"]
    print("\n" + "=" * 64)
    print(f"Graph analytics finished in {report['runtime_seconds']} s")
    print("-" * 64)
    print(f"Transactions used  : {sel['n_tx_used']:,} of {sel['n_input_tx']:,} "
          f"(steps {sel['first_step_used']}..{sel['last_step_used']})"
          + ("  [TRUNCATED by --max-tx]" if sel["truncated_by_max_tx"] else ""))
    print(f"Nodes / edges      : {report['n_nodes']:,} / {report['n_edges']:,}")
    print(f"Weak components    : {comp['n_weakly_connected_components']:,} "
          f"(largest {comp['largest_weakly_connected_component']:,} accounts)")
    print(f"Largest strong comp: {comp['largest_strongly_connected_component']:,} accounts | "
          f"accounts in small cycles: {comp['n_accounts_in_small_cycles']:,}")
    print(f"Betweenness        : {btw['scope']}, "
          + ("exact" if btw["exact"] else f"sampled (k={btw['k_sampled_start_nodes']})"))
    print("-" * 64)
    for name, rows in report["top_accounts"].items():
        top = rows[0] if rows else {}
        print(f"Top {name:<22}: {top}")
    comparison = report["fraud_vs_graph_signals"]["comparison"]
    if comparison:
        print("-" * 64)
        print("Mean graph signals (display only, uses the label):")
        for group, values in comparison.items():
            print(f"  {group:<14}: {values}")
    print("-" * 64)
    for name, location in report["outputs"].items():
        print(f"{name:<19}: {location}")
    print("=" * 64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the transaction graph and compute node metrics.")
    parser.add_argument("--max-step", type=int, default=None, help="use only steps <= this value (training window)")
    parser.add_argument("--max-tx", type=int, default=GRAPH_MAX_TX, help="use only the first N transactions by time")
    parser.add_argument("--k", type=int, default=BETWEENNESS_K, help="sampled start nodes for betweenness")
    args = parser.parse_args()
    ensure_directories()
    try:
        report = run_graph_analytics(max_step=args.max_step, max_tx=args.max_tx, betweenness_k=args.k)
    except FraudProjectError as exc:
        print(f"\nERROR: {exc}")
        raise SystemExit(1)
    print_summary(report)


if __name__ == "__main__":
    main()