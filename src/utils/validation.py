"""Validation helpers shared by the whole project."""
from __future__ import annotations

import importlib
from typing import Iterable

import pandas as pd

from config import BALANCE_COLUMNS, LABEL_COLUMN, REQUIRED_COLUMNS, VALID_TRANSACTION_TYPES
from src.utils.errors import (
    EmptyDatasetError,
    InvalidValuesError,
    MissingColumnsError,
    MissingDependencyError,
)


def require_non_empty(df: pd.DataFrame, name: str = "dataset") -> None:
    if df is None or len(df) == 0:
        raise EmptyDatasetError(f"The {name} is empty (0 rows). Check the source file.")


def require_columns(df: pd.DataFrame, columns: Iterable[str] = REQUIRED_COLUMNS, name: str = "dataset") -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise MissingColumnsError(
            f"The {name} is missing required column(s): {missing}. "
            f"Found columns: {list(df.columns)}. This project expects the PaySim schema."
        )


def require_dependency(module: str, pip_name: str | None = None, purpose: str = "") -> None:
    """Raise a helpful error if an optional package is not installed."""
    try:
        importlib.import_module(module)
    except ImportError as exc:
        pip_name = pip_name or module
        extra = f" It is needed for {purpose}." if purpose else ""
        raise MissingDependencyError(
            f"Python package '{module}' is not installed.{extra} Install it with: pip install {pip_name}"
        ) from exc


def transaction_quality_report(df: pd.DataFrame) -> dict:
    """Count data-quality problems in a PaySim-style DataFrame.

    This only *reports*; cleaning happens in the preprocessing phase.
    """
    require_non_empty(df)
    require_columns(df)

    amount = pd.to_numeric(df["amount"], errors="coerce")
    step = pd.to_numeric(df["step"], errors="coerce")
    label = pd.to_numeric(df[LABEL_COLUMN], errors="coerce")

    null_counts = {c: int(n) for c, n in df[REQUIRED_COLUMNS].isna().sum().items() if n > 0}
    negative_balances = {
        c: int((pd.to_numeric(df[c], errors="coerce") < 0).sum()) for c in BALANCE_COLUMNS
    }
    negative_balances = {c: n for c, n in negative_balances.items() if n > 0}

    report = {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "null_counts": null_counts,
        "negative_amounts": int((amount < 0).sum()),
        "zero_amounts": int((amount == 0).sum()),
        "negative_balances": negative_balances,
        "invalid_steps": int((step < 0).sum() + step.isna().sum()),
        "unknown_types": int((~df["type"].isin(VALID_TRANSACTION_TYPES)).sum()),
        "invalid_labels": int((~label.isin([0, 1])).sum()),
        "self_transfers": int((df["nameOrig"] == df["nameDest"]).sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }
    valid_labels = label[label.isin([0, 1])]
    n_fraud = int((valid_labels == 1).sum())
    n_legit = int((valid_labels == 0).sum())
    report["n_fraud"] = n_fraud
    report["n_legitimate"] = n_legit
    report["fraud_rate_pct"] = round(100.0 * n_fraud / max(len(valid_labels), 1), 4)
    report["imbalance_ratio"] = round(n_legit / n_fraud, 1) if n_fraud else None
    return report


_PROBLEM_KEYS = ["negative_amounts", "invalid_steps", "unknown_types", "invalid_labels"]


def raise_if_invalid(report: dict) -> None:
    """Used for strict validation: raise if any hard problem was found."""
    problems = {k: report[k] for k in _PROBLEM_KEYS if report.get(k)}
    if report.get("null_counts"):
        problems["null_counts"] = report["null_counts"]
    if problems:
        raise InvalidValuesError(f"Dataset contains invalid values: {problems}")