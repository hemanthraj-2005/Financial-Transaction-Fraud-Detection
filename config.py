"""Central configuration for the Financial Fraud Graph project.

All paths and constants live here. Any directory can be overridden with an
environment variable, so nothing is hard-coded elsewhere in the code base:

    FRAUD_DATA_DIR         -> root folder that holds raw/, processed/, synthetic/
    FRAUD_MODELS_DIR       -> folder for saved models
    FRAUD_REPORTS_DIR      -> folder for charts and reports
    FRAUD_SEED             -> global random seed (default 42)
    FRAUD_SPARK_MASTER     -> Spark master URL (default local[*])
    FRAUD_SPARK_MEMORY     -> Spark driver memory (default 4g)
    FRAUD_SPARK_PARTITIONS -> spark.sql.shuffle.partitions (default 8)
    FRAUD_GRAPH_MAX_TX     -> transactions used for the graph (default 500000)
    FRAUD_ML_MAX_TX        -> transactions used for ML (default 300000)
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
DATA_DIR = _env_path("FRAUD_DATA_DIR", PROJECT_ROOT / "data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
MODELS_DIR = _env_path("FRAUD_MODELS_DIR", PROJECT_ROOT / "models")
REPORTS_DIR = _env_path("FRAUD_REPORTS_DIR", PROJECT_ROOT / "reports")
DB_PATH = PROCESSED_DIR / "fraud.db"

# Processed outputs. A "stem" has no extension: the table is saved as
# <stem>.parquet (preferred) or <stem>.csv (fallback when pyarrow is missing).
TRANSACTIONS_CLEAN_STEM = PROCESSED_DIR / "transactions_clean"
ACCOUNT_AGG_STEM = PROCESSED_DIR / "account_aggregates"
FRAUD_STATS_BY_TYPE_PATH = PROCESSED_DIR / "fraud_stats_by_type.csv"
FRAUD_STATS_BY_DAY_PATH = PROCESSED_DIR / "fraud_stats_by_day.csv"
PREPROCESSING_REPORT_PATH = REPORTS_DIR / "preprocessing_report.json"
EDA_DIR = REPORTS_DIR / "eda"

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
RANDOM_SEED = int(os.environ.get("FRAUD_SEED", "42"))

# --------------------------------------------------------------------------
# Dataset: PaySim (Kaggle "Synthetic Financial Datasets For Fraud Detection")
# --------------------------------------------------------------------------
KAGGLE_DATASET = "ealaxi/paysim1"
KAGGLE_URL = "https://www.kaggle.com/datasets/ealaxi/paysim1"
PAYSIM_FILENAME = "PS_20174392719_1491204439457_log.csv"

# Development / testing data ONLY (never the main project dataset)
SYNTHETIC_FILENAME = "synthetic_transactions.csv"
SYNTHETIC_PATH = SYNTHETIC_DIR / SYNTHETIC_FILENAME
SYNTHETIC_META_PATH = SYNTHETIC_DIR / "synthetic_metadata.json"

# PaySim schema (documented assumption: the pipeline expects these columns)
REQUIRED_COLUMNS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
]
NUMERIC_COLUMNS = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
]
BALANCE_COLUMNS = [
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]
VALID_TRANSACTION_TYPES = ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"]
# Fixed, deterministic label encoding (no fitting needed, identical in Pandas and Spark)
TYPE_CODES = {t: i for i, t in enumerate(sorted(VALID_TRANSACTION_TYPES))}
LABEL_COLUMN = "isFraud"

# --------------------------------------------------------------------------
# Time handling
# --------------------------------------------------------------------------
HOURS_PER_STEP = 1  # PaySim: 1 step = 1 hour, 743 steps (~30 days)
# PaySim has no calendar dates, only "step". We anchor step 1 to an arbitrary
# start so that real timestamps exist. Only hour-of-day / day-offset carry
# meaning; the calendar date itself is NOT real.
BASE_TIMESTAMP = "2024-01-01 00:00:00"
NIGHT_END_HOUR = 6  # hours 0-5 are treated as "night"

# --------------------------------------------------------------------------
# Spark
# --------------------------------------------------------------------------
SPARK_MASTER = os.environ.get("FRAUD_SPARK_MASTER", "local[*]")
SPARK_DRIVER_MEMORY = os.environ.get("FRAUD_SPARK_MEMORY", "4g")
SPARK_SHUFFLE_PARTITIONS = int(os.environ.get("FRAUD_SPARK_PARTITIONS", "8"))
# engine="auto" uses Spark for input files at least this large (if Spark is installed)
SPARK_AUTO_THRESHOLD_MB = 50

# --------------------------------------------------------------------------
# Graph analytics (Phase 5)
# --------------------------------------------------------------------------
GRAPH_EDGES_STEM = PROCESSED_DIR / "graph_edges"
GRAPH_METRICS_STEM = PROCESSED_DIR / "graph_metrics"
GRAPH_REPORT_PATH = REPORTS_DIR / "graph_report.json"
GRAPH_MAX_TX = int(os.environ.get("FRAUD_GRAPH_MAX_TX", "500000"))  # graph uses the first N transactions by time
BETWEENNESS_K = 200            # number of sampled start nodes for approximate betweenness
BETWEENNESS_MAX_NODES = 20000  # cap on the subgraph used for betweenness
PAGERANK_ALPHA = 0.85
SMALL_CYCLE_MAX = 50           # strongly connected groups of 2..50 accounts count as "small cycles"

# --------------------------------------------------------------------------
# Community detection (Phase 6)
# --------------------------------------------------------------------------
COMMUNITY_MEMBERSHIP_STEM = PROCESSED_DIR / "community_membership"
COMMUNITY_STATS_STEM = PROCESSED_DIR / "community_stats"
COMMUNITY_REPORT_PATH = REPORTS_DIR / "community_report.json"
COMMUNITY_RESOLUTION = 1.0            # Louvain resolution (higher = more, smaller communities)
COMMUNITY_WEIGHT_MODE = "log_amount"  # "log_amount" | "tx_count" | "amount"
COMMUNITY_MIN_SIZE = 3                # smaller communities are not scored
COMMUNITY_FLAG_SCORE = 70.0           # eligible communities scoring >= this are flagged
# Label-free signals used for the suspicion score (project-defined, not validated)
COMMUNITY_SCORE_WEIGHTS = {
    "internal_flow_share": 0.35,        # share of value that stays inside the community
    "internal_density": 0.25,           # how fully connected the members are
    "mean_pass_through_ratio": 0.25,    # members forward most of what they receive
    "share_in_small_cycle": 0.15,       # members that sit on money loops
}

# --------------------------------------------------------------------------
# Machine learning (Phase 7)
# --------------------------------------------------------------------------
ML_FEATURES_STEM = PROCESSED_DIR / "ml_features"
ML_PREDICTIONS_STEM = PROCESSED_DIR / "ml_predictions"
ML_REPORT_PATH = REPORTS_DIR / "ml_report.json"
ML_COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
ML_MAX_TX = int(os.environ.get("FRAUD_ML_MAX_TX", "300000"))  # use the first N transactions by time
ML_FIT_FRACTION = 0.56       # first 56% of the step range -> model fitting
ML_VAL_FRACTION = 0.14       # next 14% -> threshold tuning + model selection (the last 30% is the test window)
ML_N_BLOCKS = 8              # time blocks used for graph snapshots (block 0 is a warm-up and is dropped)
ML_BETWEENNESS_K = 50        # sampled start nodes for betweenness inside each snapshot
ML_COMMUNITY_SMOOTHING = 20.0
ML_MIN_FRAUD_PER_WINDOW = 5  # fit / val / test windows each need at least this many fraud cases

# --------------------------------------------------------------------------
# Risk scoring, explanations and database (Phase 8)
# --------------------------------------------------------------------------
RISK_REPORT_PATH = REPORTS_DIR / "risk_report.json"
REASON_PERCENTILE = 0.95           # a "high" reason needs a value above this percentile of legitimate fit-window activity
REASON_FLAG_MAX_PREVALENCE = 0.25  # a yes/no flag is only evidence if at most this share of legitimate transactions have it
REASON_MAX = 3                     # reasons stored per transaction
REASON_MIN_RISK = 31.0             # reasons are generated for Medium-and-above transactions only

# --------------------------------------------------------------------------
# End-to-end runner (Phase 10)
# --------------------------------------------------------------------------
PIPELINE_REPORT_PATH = REPORTS_DIR / "pipeline_run.json"


def ensure_directories() -> None:
    """Create all project data/output directories if they do not exist."""
    for directory in (RAW_DIR, PROCESSED_DIR, SYNTHETIC_DIR, MODELS_DIR, REPORTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)