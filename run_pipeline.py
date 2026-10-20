"""Run the whole project pipeline end to end.

    python run_pipeline.py                          # everything (PaySim if present, else synthetic dev data)
    python run_pipeline.py --source synthetic       # development data only
    python run_pipeline.py --source paysim --max-tx 150000 --blocks 6
    python run_pipeline.py --from ml                # re-run from the ML step onwards
    python run_pipeline.py --only score             # re-run one step
    python run_pipeline.py --dry-run                # show the plan, run nothing

Steps, in order:
    preprocess   clean data, engineer features, account aggregates         (Phase 3)
    eda          Plotly charts                                             (Phase 4)
    graph        weighted directed graph + node metrics                    (Phase 5)
    communities  community detection + suspicion scores                    (Phase 6)
    ml           time split, leakage-safe features, models, ablation       (Phase 7)
    score        0-100 risk scores, reason codes, account risk, SQLite     (Phase 8)

Then start the dashboard:   streamlit run dashboard/app.py
"""
from __future__ import annotations

import argparse
import time
import traceback
from datetime import datetime, timezone

from config import GRAPH_MAX_TX, ML_MAX_TX, ML_N_BLOCKS, PIPELINE_REPORT_PATH, ensure_directories
from src.utils.errors import FraudProjectError
from src.utils.io_utils import write_json

STEP_NAMES = ["preprocess", "eda", "graph", "communities", "ml", "score"]
MODEL_CHOICES = ["logistic_regression", "random_forest", "xgboost"]


# ------------------------------------------------------------------ step runners (heavy imports are lazy)
def _limits(args) -> tuple[int | None, int | None]:
    """(graph transaction limit, ML transaction limit). 0 means no limit."""
    graph = args.graph_max_tx if args.graph_max_tx is not None else (
        args.max_tx if args.max_tx is not None else GRAPH_MAX_TX)
    ml = args.ml_max_tx if args.ml_max_tx is not None else (
        args.max_tx if args.max_tx is not None else ML_MAX_TX)
    return (graph or None), (ml or None)


def _run_preprocess(args) -> dict:
    from src.preprocessing.pipeline import run_preprocessing

    report = run_preprocessing(source=args.source, nrows=args.nrows, engine=args.engine)
    return {
        "source": report["source"],
        "engine": report["engine"],
        "clean_rows": report["cleaning"]["n_clean"],
        "accounts": report["n_accounts"],
        "fraud_rate_pct": report["class_balance"]["fraud_rate_pct"],
    }


def _run_eda(args) -> dict:
    from src.analytics.eda import run_eda

    summary = run_eda()
    return {"charts": len(summary["charts"]), "fraud_rate_pct": summary["fraud_rate_pct"]}


def _run_graph(args) -> dict:
    from src.graph.pipeline import run_graph_analytics

    graph_limit, _ = _limits(args)
    report = run_graph_analytics(max_tx=graph_limit)
    return {
        "nodes": report["n_nodes"],
        "edges": report["n_edges"],
        "truncated": report["selection"]["truncated_by_max_tx"],
    }


def _run_communities(args) -> dict:
    from src.graph.community_pipeline import run_community_detection

    report = run_community_detection()
    return {
        "method": report["detection"]["method_used"],
        "communities": report["n_communities"],
        "flagged": report["n_flagged_communities"],
    }


def _run_ml(args) -> dict:
    from src.models.pipeline import run_ml_pipeline

    _, ml_limit = _limits(args)
    report = run_ml_pipeline(
        max_tx=ml_limit, n_blocks=args.blocks, model_names=tuple(args.models),
        run_ablation=not args.skip_ablation,
    )
    best = report["selected_model"]
    return {
        "model": best["name"],
        "test_pr_auc": best["test"]["pr_auc"],
        "test_precision": best["test"]["precision"],
        "test_recall": best["test"]["recall"],
    }


def _run_score(args) -> dict:
    from src.models.scoring_pipeline import run_scoring_pipeline

    report = run_scoring_pipeline()
    counts = report["counts"]
    return {
        "transactions": counts["transactions"],
        "accounts": counts["accounts"],
        "alerts": counts["alerts_predicted"],
    }


RUNNERS = {
    "preprocess": _run_preprocess,
    "eda": _run_eda,
    "graph": _run_graph,
    "communities": _run_communities,
    "ml": _run_ml,
    "score": _run_score,
}


# ------------------------------------------------------------------ orchestration
def select_steps(from_step: str | None = None, only: list | None = None, skip: list | None = None) -> list[str]:
    """Steps to run, always in pipeline order."""
    named = [*(only or []), *(skip or []), *([from_step] if from_step else [])]
    unknown = [name for name in named if name not in STEP_NAMES]
    if unknown:
        raise ValueError(f"Unknown step(s) {unknown}. Choose from {STEP_NAMES}.")
    if only and from_step:
        raise ValueError("Use either --only or --from, not both.")
    steps = list(STEP_NAMES)
    if only:
        steps = [s for s in STEP_NAMES if s in only]
    elif from_step:
        steps = STEP_NAMES[STEP_NAMES.index(from_step):]
    if skip:
        steps = [s for s in steps if s not in skip]
    if not steps:
        raise ValueError("No steps left to run.")
    return steps


def run_all(args, runners: dict | None = None, report_path=PIPELINE_REPORT_PATH, log=print) -> dict:
    """Run the selected steps in order. Stops at the first failure and always records what happened."""
    runners = RUNNERS if runners is None else runners
    steps = select_steps(args.from_step, args.only, args.skip)
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    results: list[dict] = []
    status = "planned" if args.dry_run else "ok"
    for index, name in enumerate(steps):
        if args.dry_run:
            results.append({"name": name, "status": "planned"})
            continue
        log(f"\n=== [{index + 1}/{len(steps)}] {name} ===")
        step_started = time.perf_counter()
        try:
            summary = runners[name](args)
            results.append({"name": name, "status": "ok",
                            "seconds": round(time.perf_counter() - step_started, 2), "summary": summary})
        except FraudProjectError as exc:
            results.append({"name": name, "status": "failed",
                            "seconds": round(time.perf_counter() - step_started, 2), "error": str(exc)})
            log(f"\nStep '{name}' failed: {exc}")
            status = "failed"
            break
        except Exception as exc:  # unexpected bug: record it, show the traceback, stop cleanly
            results.append({"name": name, "status": "error",
                            "seconds": round(time.perf_counter() - step_started, 2),
                            "error": f"{type(exc).__name__}: {exc}"})
            log(traceback.format_exc())
            status = "error"
            break

    done = {r["name"] for r in results}
    if not args.dry_run:
        results.extend({"name": name, "status": "not_run"} for name in steps if name not in done)

    report = {
        "status": status,
        "started_at_utc": started_at,
        "total_seconds": round(time.perf_counter() - started, 2),
        "args": vars(args),
        "steps": results,
    }
    if not args.dry_run:
        write_json(report, report_path)
    return report


def _fmt(summary) -> str:
    return ", ".join(f"{key}={value}" for key, value in (summary or {}).items())


def print_summary(report: dict) -> None:
    print("\n" + "=" * 78)
    print(f"Pipeline {report['status']} in {report['total_seconds']} s")
    print("-" * 78)
    for step in report["steps"]:
        seconds = f"{step['seconds']:>7.1f} s" if "seconds" in step else " " * 9
        detail = _fmt(step.get("summary")) or step.get("error", "")
        print(f"{step['name']:<12} {step['status']:<8} {seconds}  {detail}")
    print("=" * 78)
    if report["status"] == "ok":
        print("Next:  streamlit run dashboard/app.py")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the fraud detection pipeline end to end (preprocess, eda, graph, communities, ml, score)."
    )
    parser.add_argument("--source", choices=["auto", "paysim", "synthetic"], default="auto")
    parser.add_argument("--nrows", type=int, default=None, help="preprocess only the first N rows of the raw file")
    parser.add_argument("--engine", choices=["auto", "pandas", "spark"], default="auto")
    parser.add_argument("--max-tx", type=int, default=None, help="limit for BOTH graph and ML steps (0 = no limit)")
    parser.add_argument("--graph-max-tx", type=int, default=None, help="limit for the graph step only")
    parser.add_argument("--ml-max-tx", type=int, default=None, help="limit for the ML step only")
    parser.add_argument("--blocks", type=int, default=ML_N_BLOCKS, help="time blocks for graph snapshots in ML")
    parser.add_argument("--models", nargs="+", choices=MODEL_CHOICES, default=MODEL_CHOICES)
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--from", dest="from_step", choices=STEP_NAMES, default=None, help="start at this step")
    parser.add_argument("--only", nargs="+", choices=STEP_NAMES, default=None, help="run only these steps")
    parser.add_argument("--skip", nargs="+", choices=STEP_NAMES, default=None, help="skip these steps")
    parser.add_argument("--dry-run", action="store_true", help="show the plan and run nothing")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    ensure_directories()
    try:
        report = run_all(args)
    except ValueError as exc:
        print(f"\nERROR: {exc}")
        return 2
    print_summary(report)
    return 0 if report["status"] in {"ok", "planned"} else 1


if __name__ == "__main__":
    raise SystemExit(main())