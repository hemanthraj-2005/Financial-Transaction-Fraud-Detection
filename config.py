"""Central configuration for the Financial Fraud Graph project.

All paths and constants live here. Any directory can be overridden with an
environment variable, so nothing is hard-coded elsewhere in the code base:

    FRAUD_DATA_DIR    -> root folder that holds raw/, processed/, synthetic/
    FRAUD_MODELS_DIR  -> folder for saved models
    FRAUD_REPORTS_DIR -> folder for charts and reports
    FRAUD_SEED        -> global random seed (default 42)
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
DB_PATH = PROCESSED_DIR / "fraud.db"  # used from the database phase onwards

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
LABEL_COLUMN = "isFraud"
HOURS_PER_STEP = 1  # PaySim: 1 step = 1 hour, 743 steps (~30 days)


def ensure_directories() -> None:
    """Create all project data/output directories if they do not exist."""
    for directory in (RAW_DIR, PROCESSED_DIR, SYNTHETIC_DIR, MODELS_DIR, REPORTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)