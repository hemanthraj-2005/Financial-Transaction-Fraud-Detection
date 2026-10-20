"""ML pipeline: time split -> leakage-safe graph features -> train/compare -> ablation -> save.

Reads  data/processed/transactions_clean (Phase 3).
Writes data/processed/ml_features, data/processed/ml_predictions,
       models/best_model.joblib, models/model_metadata.json,
       reports/model_comparison.csv, reports/ml_report.json

Usage:
    python -m src.models.pipeline
    python -m src.models.pipeline --max-tx 200000 --blocks 6
    python -m src.models.pipeline --models logistic_regression xgboost --skip-ablation
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from config import (
    BEST_MODEL_PATH,
    MODEL_METADATA_PATH,
    MODELS_DIR,
    ML_BETWEENNESS_K,
    ML_COMPARISON_PATH,
    ML_FEATURES_STEM,
    ML_FIT_FRACTION,
    ML_MAX_TX,
    ML_MIN_FRAUD_PER_WINDOW,
    ML_N_BLOCKS,
    ML_PREDICTIONS_STEM,
    ML_REPORT_PATH,
    ML_VAL_FRACTION,
    PREPROCESSING_REPORT_PATH,
    PROCESSED_DIR,
    RANDOM_SEED,
    TRANSACTIONS_CLEAN_STEM,
    ensure_directories,
)
from src.features.feature_sets import FEATURE_SETS, LOAD_COLUMNS, PRIMARY_FEATURE_SET
from src.features.graph_features import GRAPH_INPUT_COLUMNS, compute_graph_features
from src.features.time_split import assign_split, split_boundaries
from src.models.train import MODEL_NAMES, fit_and_evaluate, select_best, to_matrix
from src.utils.errors import DatasetNotFoundError, FraudProjectError, MissingDependencyError
from src.utils.io_utils import save_table, table_paths, write_json
from src.utils.logging_utils import get_logger
from src.utils.seed import set_global_seed
from src.utils.validation import require_columns, require_non_empty

logger = get_logger("ml")

LEAKAGE_CONTROLS = [
    "Time-based split by step: fit (first part), validation (threshold tuning + model selection), test (last part, untouched).",
    "Graph/community features for a transaction in block b come from a graph built on blocks < b only (no future data).",
    "The only label-based feature (community fraud rate) uses labels from earlier blocks and from steps <= end of the "
    "fit window, so a row's own label and any validation/test label never reach its features.",
    "Balance columns and balance-derived features are excluded from the primary feature set (ablation only); isFlaggedFraud is never used.",
    "The scaler (Logistic Regression) is fitted on the fit window only; model choice uses validation PR-AUC, never the test window.",
    "Block 0 has no history, so it is dropped as a warm-up.",
]


def _load_transactions() -> pd.DataFrame:
    parquet_path, csv_path = table_paths(TRANSACTIONS_CLEAN_STEM)
    if parquet_path.exists():
        return pd.read_parquet(parquet_path, columns=LOAD_COLUMNS)
    if csv_path.exists():
        return pd.read_csv(csv_path, usecols=LOAD_COLUMNS)
    raise DatasetNotFoundError(
        f"Processed table '{TRANSACTIONS_CLEAN_STEM.name}' not found in {TRANSACTIONS_CLEAN_STEM.parent}. "
        "Run the preprocessing pipeline first:  python -m src.preprocessing.pipeline"
    )


def _data_source() -> str:
    try:
        return str(json.loads(Path(PREPROCESSING_REPORT_PATH).read_text()).get("source", "unknown"))
    except Exception:
        return "unknown"


def _check_windows(data: pd.DataFrame) -> dict:
    counts = (
        data.groupby("split")["isFraud"].agg(n_rows="size", n_fraud="sum")
        .reindex(["fit", "val", "test"]).fillna(0).astype(int)
    )
    for name, row in counts.iterrows():
        if row["n_fraud"] < ML_MIN_FRAUD_PER_WINDOW:
            raise FraudProjectError(
                f"The {name} window has only {row['n_fraud']} fraud case(s) (need at least "
                f"{ML_MIN_FRAUD_PER_WINDOW}). Use more data (--max-tx), fewer --blocks, or different split fractions."
            )
    return {name: {"n_rows": int(r["n_rows"]), "n_fraud": int(r["n_fraud"])} for name, r in counts.iterrows()}


def _comparison_row(feature_set: str, f) -> dict:
    return {
        "feature_set": feature_set,
        "model": f.name,
        "n_features": len(f.columns),
        "threshold": round(f.threshold, 6),
        "val_pr_auc": f.val["pr_auc"],
        "test_precision": f.test["precision"],
        "test_recall": f.test["recall"],
        "test_f1": f.test["f1"],
        "test_roc_auc": f.test["roc_auc"],
        "test_pr_auc": f.test["pr_auc"],
        "tp": f.test["tp"], "fp": f.test["fp"], "fn": f.test["fn"], "tn": f.test["tn"],
        "fit_seconds": round(f.fit_seconds, 2),
    }


def run_ml_pipeline(
    max_tx: int | None = ML_MAX_TX,
    n_blocks: int = ML_N_BLOCKS,
    fit_fraction: float = ML_FIT_FRACTION,
    val_fraction: float = ML_VAL_FRACTION,
    model_names: tuple = tuple(MODEL_NAMES),
    run_ablation: bool = True,
    betweenness_k: int = ML_BETWEENNESS_K,
    tx: pd.DataFrame | None = None,
    output_dir: Path | str = PROCESSED_DIR,
    models_dir: Path | str = MODELS_DIR,
    report_path: Path | str = ML_REPORT_PATH,
    comparison_path: Path | str = ML_COMPARISON_PATH,
    source: str | None = None,
) -> dict:
    started = time.perf_counter()
    set_global_seed()
    output_dir, models_dir = Path(output_dir), Path(models_dir)
    unknown = [m for m in model_names if m not in MODEL_NAMES]
    if unknown:
        raise ValueError(f"Unknown model(s) {unknown}. Choose from {MODEL_NAMES}.")

    if tx is None:
        tx = _load_transactions()
    require_non_empty(tx, "transactions table")
    require_columns(tx, LOAD_COLUMNS, "transactions table")
    tx = tx[LOAD_COLUMNS].sort_values(["step", "tx_id"], kind="stable").reset_index(drop=True)
    n_input = int(len(tx))
    truncated = bool(max_tx) and n_input > max_tx
    if truncated:
        logger.warning("Using the first %d of %d transactions (by time) to keep the laptop run feasible.", max_tx, n_input)
        tx = tx.head(max_tx).reset_index(drop=True)

    bounds = split_boundaries(tx["step"], fit_fraction, val_fraction)
    tx["split"] = assign_split(tx["step"], bounds)

    graph_feats = compute_graph_features(
        tx[GRAPH_INPUT_COLUMNS], n_blocks=n_blocks, fit_end=bounds["fit_end"], betweenness_k=betweenness_k
    )
    n_blocks_used = int(graph_feats["block"].max()) + 1
    data = tx.merge(graph_feats, on="tx_id", how="left", validate="one_to_one")

    warmup = data["block"] < 1
    n_warmup = int(warmup.sum())
    data = data.loc[~warmup].reset_index(drop=True)
    window_counts = _check_windows(data)
    fit, val, test = (data[data["split"] == s] for s in ("fit", "val", "test"))

    primary_cols = FEATURE_SETS[PRIMARY_FEATURE_SET]
    results = []
    for name in model_names:
        try:
            results.append(fit_and_evaluate(name, primary_cols, fit, val, test))
        except MissingDependencyError as exc:
            logger.warning("Skipping %s: %s", name, exc)
    if not results:
        raise FraudProjectError("No model could be trained.")
    best = select_best(results)

    rows = [_comparison_row(PRIMARY_FEATURE_SET, r) for r in results]
    if run_ablation:
        for set_name, cols in FEATURE_SETS.items():
            if set_name != PRIMARY_FEATURE_SET:
                rows.append(_comparison_row(set_name, fit_and_evaluate(best.name, cols, fit, val, test)))
    comparison = pd.DataFrame(rows)
    order = {name: i for i, name in enumerate(FEATURE_SETS)}
    comparison = comparison.sort_values(
        ["feature_set", "model"], key=lambda col: col.map(order) if col.name == "feature_set" else col
    ).reset_index(drop=True)
    Path(comparison_path).parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(comparison_path, index=False)

    # ---- outputs ---------------------------------------------------------------------------------
    id_columns = ["tx_id", "step", "block", "split", "isFraud"]
    all_feature_cols = FEATURE_SETS["tx_graph_balance"]
    features_table = pd.concat([data[id_columns], data[all_feature_cols].astype("float32")], axis=1)
    features_path = save_table(features_table, output_dir / ML_FEATURES_STEM.name)

    scores = best.model.predict_proba(to_matrix(data, best.columns))[:, 1]
    predictions = data[id_columns].copy()
    predictions["score"] = scores
    predictions["predicted"] = (scores >= best.threshold).astype("int8")
    predictions["threshold"] = best.threshold
    predictions_path = save_table(predictions, output_dir / ML_PREDICTIONS_STEM.name)

    models_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_name": best.name,
        "feature_set": PRIMARY_FEATURE_SET,
        "feature_columns": best.columns,
        "threshold": best.threshold,
        "split_bounds": bounds,
        "n_blocks": n_blocks_used,
        "seed": RANDOM_SEED,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "validation_metrics": best.val,
        "test_metrics": best.test,
    }
    model_path = models_dir / BEST_MODEL_PATH.name
    joblib.dump({"model": best.model, **metadata}, model_path)
    write_json(metadata, models_dir / MODEL_METADATA_PATH.name)

    source = source or _data_source()
    report = {
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "source": source,
        "synthetic_data_warning": source == "synthetic",
        "rows": {"input": n_input, "used": int(len(tx)), "truncated_by_max_tx": truncated,
                 "warmup_block_dropped": n_warmup, "modelled": int(len(data))},
        "split": {**bounds, "windows": window_counts, "n_blocks": n_blocks_used},
        "feature_sets": {name: len(cols) for name, cols in FEATURE_SETS.items()},
        "primary_feature_set": PRIMARY_FEATURE_SET,
        "selected_model": {
            "name": best.name, "threshold": best.threshold, "selected_by": "validation PR-AUC",
            "validation": best.val, "test": best.test, "top_features": best.importance,
        },
        "model_comparison": comparison.to_dict("records"),
        "leakage_controls": LEAKAGE_CONTROLS,
        "outputs": {"model": str(model_path), "features": str(features_path),
                    "predictions": str(predictions_path), "comparison_csv": str(comparison_path)},
        "notes": [
            "Metrics come from a time-based test window; they are batch results, not real-time detection.",
            "tx_graph_balance is an ABLATION: balance columns reveal the label in PaySim, so its scores are inflated.",
            "Scores for fit-window rows in ml_predictions are in-sample; use split == 'test' for honest evaluation.",
            "Fraud labels are assumed known right after each block ends; real systems have label delays.",
        ],
    }
    write_json(report, report_path)
    logger.info("ML report written to %s", report_path)
    return report


def print_summary(report: dict) -> None:
    rows, split = report["rows"], report["split"]
    print("\n" + "=" * 78)
    print(f"ML pipeline finished in {report['runtime_seconds']} s  (data source: {report['source']})")
    if report["synthetic_data_warning"]:
        print("*** SYNTHETIC development data - do not report these results ***")
    print("-" * 78)
    print(f"Rows used / modelled : {rows['used']:,} / {rows['modelled']:,} "
          f"(warm-up block dropped: {rows['warmup_block_dropped']:,})"
          + ("  [TRUNCATED by --max-tx]" if rows["truncated_by_max_tx"] else ""))
    print(f"Split (steps)        : fit <= {split['fit_end']} | val <= {split['val_end']} | test <= {split['max_step']}")
    for name, w in split["windows"].items():
        print(f"  {name:<5}: {w['n_rows']:>9,} rows, {w['n_fraud']:>6,} fraud")
    print("-" * 78)
    table = pd.DataFrame(report["model_comparison"])[
        ["feature_set", "model", "val_pr_auc", "test_precision", "test_recall", "test_f1", "test_roc_auc", "test_pr_auc"]
    ]
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    best = report["selected_model"]
    print("-" * 78)
    print(f"Selected model       : {best['name']} (by validation PR-AUC), threshold {best['threshold']:.4f}")
    print(f"Test                 : precision {best['test']['precision']}, recall {best['test']['recall']}, "
          f"F1 {best['test']['f1']}, PR-AUC {best['test']['pr_auc']}")
    print("Top features         : " + ", ".join(f["feature"] for f in best["top_features"][:6]))
    print("-" * 78)
    for name, location in report["outputs"].items():
        print(f"{name:<20}: {location}")
    print("=" * 78)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and compare fraud models with a leakage-safe time split.")
    parser.add_argument("--max-tx", type=int, default=ML_MAX_TX, help="use the first N transactions by time")
    parser.add_argument("--blocks", type=int, default=ML_N_BLOCKS, help="time blocks for graph snapshots")
    parser.add_argument("--fit-fraction", type=float, default=ML_FIT_FRACTION)
    parser.add_argument("--val-fraction", type=float, default=ML_VAL_FRACTION)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=MODEL_NAMES)
    parser.add_argument("--k", type=int, default=ML_BETWEENNESS_K, help="betweenness sample size per snapshot")
    parser.add_argument("--skip-ablation", action="store_true")
    args = parser.parse_args()
    ensure_directories()
    try:
        report = run_ml_pipeline(
            max_tx=args.max_tx, n_blocks=args.blocks, fit_fraction=args.fit_fraction,
            val_fraction=args.val_fraction, model_names=tuple(args.models),
            run_ablation=not args.skip_ablation, betweenness_k=args.k,
        )
    except FraudProjectError as exc:
        print(f"\nERROR: {exc}")
        raise SystemExit(1)
    print_summary(report)


if __name__ == "__main__":
    main()