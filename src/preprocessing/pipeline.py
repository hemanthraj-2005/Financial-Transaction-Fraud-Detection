"""Preprocessing pipeline: clean -> features -> account aggregates -> save + report.

Run from the terminal:

    python -m src.preprocessing.pipeline                       # auto source, auto engine
    python -m src.preprocessing.pipeline --source synthetic    # development data only
    python -m src.preprocessing.pipeline --source paysim --engine spark
    python -m src.preprocessing.pipeline --nrows 500000        # first N rows (time-ordered file)

Engines
  pandas : in-memory, fine up to a few hundred thousand rows
  spark  : PySpark, for the full PaySim file (6.3M rows) and beyond
  auto   : Spark if the input is >= SPARK_AUTO_THRESHOLD_MB and Spark+Java are installed, else Pandas

Outputs (in --output-dir, default data/processed):
  transactions_clean.parquet   cleaned rows + engineered features (schema.OUTPUT_COLUMNS)
  account_aggregates.parquet   one row per account (schema.ACCOUNT_COLUMNS)
  fraud_stats_by_type.csv, fraud_stats_by_day.csv
  reports/preprocessing_report.json  cleaning counts, class balance, column inspection
(.csv instead of .parquet if pyarrow is not installed.)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from config import (
    ACCOUNT_AGG_STEM,
    BASE_TIMESTAMP,
    PREPROCESSING_REPORT_PATH,
    PROCESSED_DIR,
    RAW_DIR,
    SPARK_AUTO_THRESHOLD_MB,
    SYNTHETIC_PATH,
    TRANSACTIONS_CLEAN_STEM,
    ensure_directories,
)
from src.analytics.fraud_stats import fraud_stats
from src.data.loader import DOWNLOAD_HELP, find_paysim_file, load_transactions
from src.data.synthetic_generator import generate_synthetic_transactions, save_synthetic
from src.features.account_aggregates import build_account_aggregates
from src.features.transaction_feature import add_transaction_features
from src.preprocessing import schema
from src.preprocessing.cleaning import clean_transactions
from src.preprocessing.diagnostics import class_balance_report, inspect_columns
from src.preprocessing.spark_pipeline import run_spark_engine, spark_available
from src.utils.errors import DatasetNotFoundError, FraudProjectError
from src.utils.io_utils import save_table, write_json
from src.utils.logging_utils import get_logger

logger = get_logger("pipeline")


def resolve_input(
    source: str = "auto",
    path: Path | str | None = None,
    raw_dir: Path = RAW_DIR,
    synthetic_path: Path = SYNTHETIC_PATH,
) -> tuple[Path, str]:
    """Return (csv path, source label). Falls back to synthetic data ONLY in 'auto' mode."""
    if source not in {"auto", "paysim", "synthetic"}:
        raise ValueError("source must be 'auto', 'paysim' or 'synthetic'")
    if path:
        return Path(path), ("synthetic" if source == "synthetic" else "paysim" if source == "paysim" else "custom")
    if source in {"auto", "paysim"}:
        found = find_paysim_file(raw_dir)
        if found:
            return found, "paysim"
        if source == "paysim":
            raise DatasetNotFoundError(DOWNLOAD_HELP.format(raw_dir=Path(raw_dir).resolve()))
        logger.warning("PaySim not found - using SYNTHETIC development data. Do NOT report results from it.")
    synthetic_path = Path(synthetic_path)
    if not synthetic_path.exists():
        logger.info("Generating synthetic development data at %s", synthetic_path)
        ensure_directories()
        save_synthetic(generate_synthetic_transactions(), path=synthetic_path)
    return synthetic_path, "synthetic"


def resolve_engine(engine: str, input_path: Path | str) -> str:
    if engine not in {"auto", "pandas", "spark"}:
        raise ValueError("engine must be 'auto', 'pandas' or 'spark'")
    if engine != "auto":
        return engine
    size_mb = Path(input_path).stat().st_size / 1e6
    if size_mb >= SPARK_AUTO_THRESHOLD_MB:
        if spark_available():
            logger.info("Input is %.0f MB -> using the Spark engine.", size_mb)
            return "spark"
        logger.warning(
            "Input is %.0f MB but PySpark/Java is not available. Using Pandas, which may be slow and "
            "memory-hungry. Install Java 17 + pyspark, or use --nrows to process a sample.",
            size_mb,
        )
    return "pandas"


def _run_pandas_engine(input_path: Path, label: str, nrows: int | None, tx_stem: Path, acc_stem: Path) -> dict:
    df = load_transactions("synthetic" if label == "synthetic" else "paysim", path=input_path, nrows=nrows)
    inspection = inspect_columns(df)
    clean, cleaning = clean_transactions(df)
    features = add_transaction_features(clean)
    accounts = build_account_aggregates(features)
    by_type, by_day = fraud_stats(features)
    return {
        "cleaning": cleaning,
        "n_accounts": int(len(accounts)),
        "by_type": by_type,
        "by_day": by_day,
        "inspection": inspection,
        "inspection_note": "computed on all loaded rows",
        "transactions_path": save_table(features, tx_stem),
        "accounts_path": save_table(accounts, acc_stem),
    }


def run_preprocessing(
    source: str = "auto",
    path: Path | str | None = None,
    nrows: int | None = None,
    engine: str = "auto",
    output_dir: Path | str = PROCESSED_DIR,
    report_path: Path | str = PREPROCESSING_REPORT_PATH,
) -> dict:
    """Run the full preprocessing pipeline and return the report dictionary."""
    started = time.perf_counter()
    output_dir = Path(output_dir)
    tx_stem = output_dir / TRANSACTIONS_CLEAN_STEM.name
    acc_stem = output_dir / ACCOUNT_AGG_STEM.name

    input_path, label = resolve_input(source, path)
    chosen_engine = resolve_engine(engine, input_path)
    logger.info("Input: %s (%s data) | engine: %s", input_path.name, label, chosen_engine)

    if chosen_engine == "pandas":
        result = _run_pandas_engine(input_path, label, nrows, tx_stem, acc_stem)
    else:
        result = run_spark_engine(input_path, nrows, tx_stem, acc_stem)
        sample = pd.read_csv(input_path, nrows=100_000)
        result["inspection"] = inspect_columns(sample)
        result["inspection_note"] = "computed on the first 100,000 rows only (Spark engine)"

    by_type, by_day = result["by_type"], result["by_day"]
    output_dir.mkdir(parents=True, exist_ok=True)
    by_type.to_csv(output_dir / "fraud_stats_by_type.csv", index=False)
    by_day.to_csv(output_dir / "fraud_stats_by_day.csv", index=False)

    n_fraud = int(by_type["n_fraud"].sum())
    n_total = int(by_type["n_transactions"].sum())
    balance = class_balance_report(n_fraud, n_total - n_fraud)
    if label == "synthetic":
        balance["warning"] = "SYNTHETIC development data - the ratios above are NOT real-world figures."

    report = {
        "engine": chosen_engine,
        "source": label,
        "synthetic_data_warning": label == "synthetic",
        "input_path": str(input_path),
        "nrows_limit": nrows,
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "cleaning": result["cleaning"],
        "n_accounts": result["n_accounts"],
        "class_balance": balance,
        "fraud_rate_by_type": by_type.to_dict("records"),
        "column_inspection_note": result["inspection_note"],
        "column_inspection": result["inspection"].to_dict("records"),
        "outputs": {
            "transactions": str(result["transactions_path"]),
            "accounts": str(result["accounts_path"]),
            "fraud_stats_by_type": str(output_dir / "fraud_stats_by_type.csv"),
            "fraud_stats_by_day": str(output_dir / "fraud_stats_by_day.csv"),
        },
        "feature_groups": {
            "time": schema.TIME_FEATURES,
            "type_encoding": ["type_code", *schema.TYPE_ONEHOT_COLUMNS],
            "balance_derived_LEAKAGE_RISK": schema.BALANCE_DERIVED_FEATURES,
            "history_past_only": schema.HISTORY_FEATURES,
        },
        "notes": [
            f"timestamp = {BASE_TIMESTAMP} + (step-1) hours. The date is an arbitrary anchor; only hour-of-day "
            "and day offset are meaningful.",
            "History features use only earlier rows (ordered by step, then tx_id) and never the label.",
            "Account aggregates cover ALL rows; for ML they must be recomputed from the training window only.",
            "Columns starting with 'label_' are label-derived and must not be used as model inputs.",
        ],
    }
    write_json(report, report_path)
    logger.info("Report written to %s", report_path)
    return report


def print_summary(report: dict) -> None:
    c, b = report["cleaning"], report["class_balance"]
    print("\n" + "=" * 64)
    print(f"Preprocessing finished in {report['runtime_seconds']} s  (engine: {report['engine']}, source: {report['source']})")
    if report["synthetic_data_warning"]:
        print("*** SYNTHETIC development data - do not report results from this run ***")
    print("-" * 64)
    print(f"Rows in            : {c['n_input']:,}")
    print(f"Invalid removed    : {c['n_invalid_removed']:,}   (by rule, may overlap: "
          f"{ {k: v for k, v in c['rule_violations'].items() if v} or 'none' })")
    print(f"Duplicates removed : {c['duplicates_removed']:,}")
    print(f"Clean rows         : {c['n_clean']:,}   | accounts: {report['n_accounts']:,}")
    print(f"Fraud              : {b['n_fraud']:,} of {b['n_total']:,} ({b['fraud_rate_pct']}%) -> {b['severity']}")
    print(f"Suggested scale_pos_weight: {b['suggested_scale_pos_weight']}")
    print("-" * 64)
    for name, location in report["outputs"].items():
        print(f"{name:<20}: {location}")
    print("=" * 64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the preprocessing pipeline.")
    parser.add_argument("--source", choices=["auto", "paysim", "synthetic"], default="auto")
    parser.add_argument("--path", type=Path, default=None, help="explicit input CSV")
    parser.add_argument("--nrows", type=int, default=None, help="process only the first N rows")
    parser.add_argument("--engine", choices=["auto", "pandas", "spark"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--report-path", type=Path, default=PREPROCESSING_REPORT_PATH)
    args = parser.parse_args()
    ensure_directories()
    try:
        report = run_preprocessing(args.source, args.path, args.nrows, args.engine, args.output_dir, args.report_path)
    except FraudProjectError as exc:
        print(f"\nERROR: {exc}")
        raise SystemExit(1)
    print_summary(report)


if __name__ == "__main__":
    main()