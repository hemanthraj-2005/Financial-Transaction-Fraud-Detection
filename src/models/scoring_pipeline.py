"""Risk scoring pipeline: model scores -> 0-100 risk -> reasons -> account risk -> SQLite.

Reads  models/best_model.joblib, data/processed/ml_predictions, ml_features, transactions_clean
       (+ optional graph_metrics, community_membership, community_stats from Phases 5/6)
Writes data/processed/fraud.db and reports/risk_report.json

Usage:
    python -m src.models.scoring_pipeline
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from config import (
    BEST_MODEL_PATH,
    COMMUNITY_MEMBERSHIP_STEM,
    COMMUNITY_STATS_STEM,
    DB_PATH,
    GRAPH_METRICS_STEM,
    ML_FEATURES_STEM,
    ML_PREDICTIONS_STEM,
    MODELS_DIR,
    PREPROCESSING_REPORT_PATH,
    PROCESSED_DIR,
    REASON_FLAG_MAX_PREVALENCE,
    REASON_MAX,
    REASON_MIN_RISK,
    REASON_PERCENTILE,
    RISK_REPORT_PATH,
    TRANSACTIONS_CLEAN_STEM,
    ensure_directories,
)
from src.db.database import (
    COMMUNITY_DB_COLUMNS,
    TRANSACTION_COLUMNS,
    connect,
    create_indexes,
    create_schema,
    insert_dataframe,
    write_run_info,
)
from src.models.explain import REASON_RULES, explain_transactions, fit_baselines, importance_weights
from src.models.risk import (
    BAND_NAMES,
    RISK_BANDS,
    account_level_check,
    build_account_table,
    probability_to_risk,
    risk_anchors,
    risk_band,
)
from src.utils.errors import DatasetNotFoundError, FraudProjectError, MissingModelError
from src.utils.io_utils import table_paths, write_json
from src.utils.logging_utils import get_logger

logger = get_logger("scoring")

ML_HINT = "Run the ML pipeline first:  python -m src.models.pipeline"
CLEAN_HINT = "Run the preprocessing pipeline first:  python -m src.preprocessing.pipeline"


def _optional_table(stem: Path, columns: list[str] | None = None) -> pd.DataFrame | None:
    parquet_path, csv_path = table_paths(stem)
    if parquet_path.exists():
        return pd.read_parquet(parquet_path, columns=columns)
    if csv_path.exists():
        return pd.read_csv(csv_path, usecols=columns)
    return None


def _required_table(stem: Path, columns: list[str], hint: str) -> pd.DataFrame:
    table = _optional_table(stem, columns)
    if table is None:
        raise DatasetNotFoundError(f"Table '{stem.name}' not found in {stem.parent}. {hint}")
    return table


def _data_source() -> str:
    try:
        return str(json.loads(Path(PREPROCESSING_REPORT_PATH).read_text()).get("source", "unknown"))
    except Exception:
        return "unknown"


def _band_counts(bands: pd.Series) -> dict:
    counts = bands.value_counts().reindex(BAND_NAMES, fill_value=0)
    return {name: int(value) for name, value in counts.items()}


def run_scoring_pipeline(
    output_dir: Path | str = PROCESSED_DIR,
    models_dir: Path | str = MODELS_DIR,
    db_path: Path | str = DB_PATH,
    report_path: Path | str = RISK_REPORT_PATH,
    source: str | None = None,
) -> dict:
    started = time.perf_counter()
    output_dir, models_dir = Path(output_dir), Path(models_dir)

    model_path = models_dir / BEST_MODEL_PATH.name
    if not model_path.exists():
        raise MissingModelError(f"No trained model found at {model_path}. {ML_HINT}")
    payload = joblib.load(model_path)
    threshold = float(payload["threshold"])
    columns = list(payload["feature_columns"])

    predictions = _required_table(
        output_dir / ML_PREDICTIONS_STEM.name,
        ["tx_id", "step", "block", "split", "isFraud", "score", "predicted"], ML_HINT,
    )
    clean = _required_table(
        output_dir / TRANSACTIONS_CLEAN_STEM.name,
        ["tx_id", "type", "amount", "nameOrig", "nameDest", "day", "hour_of_day"], CLEAN_HINT,
    )
    scored = predictions.merge(clean, on="tx_id", how="left", validate="one_to_one")
    if scored["nameOrig"].isna().any():
        raise FraudProjectError(
            "Some scored transactions are missing from transactions_clean (the tables are out of sync). "
            f"{ML_HINT}"
        )

    scored["risk_score"] = probability_to_risk(scored["score"].to_numpy(), threshold)
    scored["risk_band"] = risk_band(scored["risk_score"].to_numpy())

    # ---- reason codes (Medium and above only) ----------------------------------------------------------
    features = _required_table(output_dir / ML_FEATURES_STEM.name, ["tx_id", "split", "isFraud", *columns], ML_HINT)
    legit_fit = features[(features["split"] == "fit") & (features["isFraud"] == 0)]
    rules = [rule for rule in REASON_RULES if rule.feature in columns]
    baselines = fit_baselines(legit_fit, rules, REASON_PERCENTILE, REASON_FLAG_MAX_PREVALENCE)
    weights = importance_weights(payload["model"], payload["model_name"], columns)

    scored["reason_codes"] = ""
    scored["reasons"] = ""
    mask = (scored["risk_score"] >= REASON_MIN_RISK).to_numpy()
    if mask.any():
        ids = scored.loc[mask, "tx_id"].to_numpy()
        try:
            selected = features.set_index("tx_id").loc[ids]
        except KeyError as exc:
            raise FraudProjectError(f"ml_features and ml_predictions do not match. {ML_HINT}") from exc
        explained = explain_transactions(selected, baselines, weights, REASON_MAX)
        scored["reason_codes"] = scored["tx_id"].map(explained["reason_codes"]).fillna("")
        scored["reasons"] = scored["tx_id"].map(explained["reasons"]).fillna("")

    # ---- accounts + optional graph / community context (display only) -----------------------------------
    metrics = _optional_table(
        output_dir / GRAPH_METRICS_STEM.name, ["account", "total_degree", "pagerank", "betweenness", "pass_through_ratio"]
    )
    membership = _optional_table(output_dir / COMMUNITY_MEMBERSHIP_STEM.name, ["account", "community_id"])
    community_stats = _optional_table(output_dir / COMMUNITY_STATS_STEM.name)
    if metrics is None or membership is None or community_stats is None:
        logger.warning("Graph/community tables not all found: those account columns will be empty. "
                       "Run python -m src.graph.pipeline and python -m src.graph.community_pipeline.")
    accounts = build_account_table(scored, metrics, membership, community_stats)
    communities = community_stats[COMMUNITY_DB_COLUMNS] if community_stats is not None else None

    # ---- database ------------------------------------------------------------------------------------
    transactions = scored.rename(columns={"score": "model_score", "isFraud": "label_isFraud"})[TRANSACTION_COLUMNS]
    source = source or _data_source()
    xp, fp = risk_anchors(threshold)
    run_info = {
        "model_name": payload["model_name"],
        "feature_set": payload["feature_set"],
        "threshold": threshold,
        "risk_anchors": {"probability": xp, "risk": fp},
        "risk_bands": RISK_BANDS,
        "data_source": source,
        "split_bounds": payload.get("split_bounds"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "notes": "Risk bands are project-defined. fit-window scores are in-sample; use split='test' for honest numbers.",
    }
    with closing(connect(db_path)) as conn:
        create_schema(conn)
        insert_dataframe(conn, "transactions", transactions)
        insert_dataframe(conn, "accounts", accounts)
        if communities is not None:
            insert_dataframe(conn, "communities", communities)
        write_run_info(conn, run_info)
        create_indexes(conn)

    reason_counts = Counter(code for text in scored["reason_codes"] for code in text.split(";") if code)
    report = {
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "source": source,
        "synthetic_data_warning": source == "synthetic",
        "model": {"name": payload["model_name"], "feature_set": payload["feature_set"], "threshold": threshold},
        "risk_mapping": {"anchors": run_info["risk_anchors"], "bands": RISK_BANDS},
        "counts": {
            "transactions": int(len(transactions)),
            "accounts": int(len(accounts)),
            "communities": int(len(communities)) if communities is not None else 0,
            "alerts_predicted": int(scored["predicted"].sum()),
        },
        "band_counts": {
            "all": _band_counts(scored["risk_band"]),
            "test": _band_counts(scored.loc[scored["split"] == "test", "risk_band"]),
        },
        "top_reason_codes": [{"code": c, "count": n} for c, n in reason_counts.most_common(10)],
        "account_level_check": account_level_check(scored, "test"),
        "outputs": {"database": str(db_path), "report": str(report_path)},
        "notes": [
            "Risk bands are project-defined and calibratable, not scientifically validated.",
            "Reason codes explain unusual features of a transaction; they are not proof of fraud.",
            "Scores for fit-window rows are in-sample. Evaluate on split = 'test'.",
            "Graph/community columns on accounts come from full-data Phase 5/6 runs: display only.",
        ],
    }
    write_json(report, report_path)
    logger.info("Risk report written to %s", report_path)
    return report


def print_summary(report: dict) -> None:
    counts, bands = report["counts"], report["band_counts"]
    print("\n" + "=" * 70)
    print(f"Risk scoring finished in {report['runtime_seconds']} s  (data source: {report['source']})")
    if report["synthetic_data_warning"]:
        print("*** SYNTHETIC development data - do not report these results ***")
    print("-" * 70)
    print(f"Model              : {report['model']['name']} ({report['model']['feature_set']}), "
          f"threshold {report['model']['threshold']:.4f}")
    print(f"Transactions       : {counts['transactions']:,} | accounts: {counts['accounts']:,} | "
          f"communities: {counts['communities']:,} | alerts: {counts['alerts_predicted']:,}")
    print(f"Bands (all rows)   : {bands['all']}")
    print(f"Bands (test only)  : {bands['test']}")
    top = ", ".join(f"{r['code']} ({r['count']})" for r in report["top_reason_codes"][:5]) or "none"
    print(f"Top reasons        : {top}")
    check = report["account_level_check"]
    print("-" * 70)
    print(f"Account check (test window): {check['n_accounts']:,} accounts, {check['n_fraud_accounts']:,} touch fraud | "
          f"precision@High {check['precision_at_high']} | recall@High {check['recall_at_high']} | "
          f"PR-AUC {check['pr_auc']}")
    print("-" * 70)
    for name, location in report["outputs"].items():
        print(f"{name:<19}: {location}")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score transactions and accounts and write the SQLite database.")
    parser.add_argument("--db-path", type=Path, default=DB_PATH)
    args = parser.parse_args()
    ensure_directories()
    try:
        report = run_scoring_pipeline(db_path=args.db_path)
    except FraudProjectError as exc:
        print(f"\nERROR: {exc}")
        raise SystemExit(1)
    print_summary(report)


if __name__ == "__main__":
    main()