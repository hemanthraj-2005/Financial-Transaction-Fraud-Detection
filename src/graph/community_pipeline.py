"""Community detection pipeline.

Reads the Phase 5 outputs (graph_edges, graph_metrics), detects communities, computes
per-community statistics and the suspicion score, and saves:
    data/processed/community_membership.parquet   one row per account
    data/processed/community_stats.parquet        one row per community
    reports/community_report.json
(.csv instead of .parquet if pyarrow is missing)

Usage:
    python -m src.graph.community_pipeline
    python -m src.graph.community_pipeline --resolution 1.5 --weight tx_count
    python -m src.graph.community_pipeline --method greedy
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from config import (
    COMMUNITY_MEMBERSHIP_STEM,
    COMMUNITY_REPORT_PATH,
    COMMUNITY_RESOLUTION,
    COMMUNITY_STATS_STEM,
    COMMUNITY_WEIGHT_MODE,
    GRAPH_EDGES_STEM,
    GRAPH_METRICS_STEM,
    PROCESSED_DIR,
    ensure_directories,
)
from src.graph.communities import WEIGHT_MODES, build_undirected, detect_communities
from src.graph.community_stats import community_statistics, score_communities
from src.utils.errors import DatasetNotFoundError, FraudProjectError
from src.utils.io_utils import load_table, save_table, write_json
from src.utils.logging_utils import get_logger

logger = get_logger("community")

_TOP_COLUMNS = [
    "community_id", "n_accounts", "suspicion_score", "internal_flow_share", "internal_density",
    "mean_pass_through_ratio", "label_fraud_tx", "label_fraud_rate_pct",
]


def fraud_signal_check(scored: pd.DataFrame) -> dict:
    """DISPLAY-ONLY: does the label-free score line up with the real fraud rate?

    Uses the fraud label, so it is for understanding and reporting only.
    """
    eligible = scored[scored["eligible"] == 1]
    result = {
        "note": "Uses the fraud label: for display and reporting only, not a model input.",
        "n_eligible_communities": int(len(eligible)),
        "spearman_score_vs_fraud_rate": None,
        "mean_fraud_rate_pct_flagged": None,
        "mean_fraud_rate_pct_not_flagged": None,
    }
    if len(eligible) >= 3:
        corr = eligible["suspicion_score"].corr(eligible["label_fraud_rate_pct"], method="spearman")
        result["spearman_score_vs_fraud_rate"] = None if pd.isna(corr) else round(float(corr), 4)
    flagged = eligible[eligible["flagged"] == 1]
    other = eligible[eligible["flagged"] == 0]
    if len(flagged):
        result["mean_fraud_rate_pct_flagged"] = round(float(flagged["label_fraud_rate_pct"].mean()), 4)
    if len(other):
        result["mean_fraud_rate_pct_not_flagged"] = round(float(other["label_fraud_rate_pct"].mean()), 4)
    return result


def run_community_detection(
    method: str = "louvain",
    resolution: float = COMMUNITY_RESOLUTION,
    weight_mode: str = COMMUNITY_WEIGHT_MODE,
    edges: pd.DataFrame | None = None,
    metrics: pd.DataFrame | None = None,
    output_dir: Path | str = PROCESSED_DIR,
    report_path: Path | str = COMMUNITY_REPORT_PATH,
) -> dict:
    started = time.perf_counter()
    output_dir = Path(output_dir)

    loaded_from_disk = edges is None
    if edges is None:
        try:
            edges = load_table(GRAPH_EDGES_STEM)
        except DatasetNotFoundError as exc:
            raise DatasetNotFoundError(
                "Graph edge table not found. Run the graph pipeline first:  python -m src.graph.pipeline"
            ) from exc
    if loaded_from_disk and metrics is None:
        try:
            metrics = load_table(GRAPH_METRICS_STEM)
        except DatasetNotFoundError:
            logger.warning("graph_metrics not found: pass-through and cycle signals will be 0. "
                           "Run python -m src.graph.pipeline for the full score.")

    graph = build_undirected(edges, weight_mode)
    membership, info = detect_communities(graph, method=method, resolution=resolution)
    stats = community_statistics(edges, membership, metrics)
    scored = score_communities(stats)

    membership_path = save_table(membership, output_dir / COMMUNITY_MEMBERSHIP_STEM.name)
    stats_path = save_table(scored, output_dir / COMMUNITY_STATS_STEM.name)

    sizes = scored["n_accounts"]
    eligible = scored[scored["eligible"] == 1]
    report = {
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "detection": {**info, "weight_mode": weight_mode},
        "n_accounts": int(len(membership)),
        "n_communities": int(len(scored)),
        "size_summary": {
            "largest_sizes": [int(s) for s in sizes.nlargest(5)],
            "median_size": float(sizes.median()),
            "n_size_1": int((sizes == 1).sum()),
            "n_size_2": int((sizes == 2).sum()),
            "n_size_3_or_more": int((sizes >= 3).sum()),
        },
        "n_flagged_communities": int(scored["flagged"].sum()),
        "top_suspicious_communities": eligible.nlargest(10, "suspicion_score")[_TOP_COLUMNS].to_dict("records"),
        "fraud_signal_check": fraud_signal_check(scored),
        "outputs": {"membership": str(membership_path), "stats": str(stats_path)},
        "notes": [
            "Detection and the suspicion score are label-free. Columns starting with 'label_' are display-only.",
            "The suspicion score is a project-defined heuristic: a relative ranking, not a fraud probability.",
            "For ML, build the graph with --max-step set to the end of the training window first.",
        ],
    }
    write_json(report, report_path)
    logger.info("Community report written to %s", report_path)
    return report


def print_summary(report: dict) -> None:
    det, size = report["detection"], report["size_summary"]
    print("\n" + "=" * 64)
    print(f"Community detection finished in {report['runtime_seconds']} s")
    print("-" * 64)
    print(f"Method             : {det['method_used']} (requested {det['method_requested']}), "
          f"resolution {det['resolution']}, weight '{det['weight_mode']}'")
    if det["fallback_reason"]:
        print(f"Fallback reason    : {det['fallback_reason']}")
    print(f"Accounts / groups  : {report['n_accounts']:,} / {report['n_communities']:,} communities "
          f"(modularity {det['modularity']})")
    print(f"Largest sizes      : {size['largest_sizes']} | median size {size['median_size']}")
    print(f"Size 1 / 2 / 3+    : {size['n_size_1']} / {size['n_size_2']} / {size['n_size_3_or_more']}")
    print(f"Flagged communities: {report['n_flagged_communities']}")
    top = report["top_suspicious_communities"]
    if top:
        print("-" * 64)
        print("Top suspicious communities (score is a relative heuristic):")
        for row in top[:5]:
            print(f"  #{row['community_id']:<4} size {row['n_accounts']:<5} score {row['suspicion_score']:<6} "
                  f"flow {row['internal_flow_share']:.2f}  density {row['internal_density']:.3f}  "
                  f"fraud rate {row['label_fraud_rate_pct']:.2f}% (label, display only)")
    check = report["fraud_signal_check"]
    print("-" * 64)
    print(f"Spearman(score, fraud rate): {check['spearman_score_vs_fraud_rate']}  "
          f"| fraud rate flagged vs not: {check['mean_fraud_rate_pct_flagged']} / "
          f"{check['mean_fraud_rate_pct_not_flagged']}  (display only, uses the label)")
    print("-" * 64)
    for name, location in report["outputs"].items():
        print(f"{name:<19}: {location}")
    print("=" * 64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect communities and score them.")
    parser.add_argument("--method", choices=["louvain", "greedy"], default="louvain")
    parser.add_argument("--resolution", type=float, default=COMMUNITY_RESOLUTION)
    parser.add_argument("--weight", choices=list(WEIGHT_MODES), default=COMMUNITY_WEIGHT_MODE)
    args = parser.parse_args()
    ensure_directories()
    try:
        report = run_community_detection(args.method, args.resolution, args.weight)
    except FraudProjectError as exc:
        print(f"\nERROR: {exc}")
        raise SystemExit(1)
    print_summary(report)


if __name__ == "__main__":
    main()