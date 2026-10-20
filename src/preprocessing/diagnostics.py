"""Column inspection and class-imbalance detection."""
from __future__ import annotations

import pandas as pd


def inspect_columns(df: pd.DataFrame, n_examples: int = 3) -> pd.DataFrame:
    """One row per column: dtype, missing values, distinct values, example values."""
    rows = []
    total = max(len(df), 1)
    for column in df.columns:
        series = df[column]
        rows.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "null_count": int(series.isna().sum()),
                "null_pct": round(100.0 * series.isna().sum() / total, 4),
                "n_unique": int(series.nunique(dropna=True)),
                "examples": ", ".join(map(str, series.dropna().unique()[:n_examples])),
            }
        )
    return pd.DataFrame(rows)


def class_balance_report(n_fraud: int, n_legit: int) -> dict:
    """Describe class imbalance and what to do about it.

    The severity labels are project-defined conventions, not a standard.
    """
    total = n_fraud + n_legit
    rate = 100.0 * n_fraud / total if total else 0.0
    if n_fraud == 0:
        severity = "no fraud cases (cannot train or evaluate a classifier)"
    elif rate < 1.0:
        severity = "severe imbalance (<1% fraud)"
    elif rate < 10.0:
        severity = "moderate imbalance (1-10% fraud)"
    else:
        severity = "mild imbalance or balanced (>=10% fraud)"

    ratio = round(n_legit / n_fraud, 1) if n_fraud else None
    return {
        "n_total": int(total),
        "n_fraud": int(n_fraud),
        "n_legitimate": int(n_legit),
        "fraud_rate_pct": round(rate, 4),
        "imbalance_ratio_legit_per_fraud": ratio,
        "severity": severity,
        "suggested_scale_pos_weight": ratio,
        "recommendations": [
            "Do not use accuracy: predicting 'legitimate' every time would already score "
            f"{100 - rate:.2f}%.",
            "Evaluate with precision, recall, F1 and especially PR-AUC (plus ROC-AUC).",
            "Handle imbalance with class weights / scale_pos_weight (suggested value above) "
            "rather than oversampling.",
            "If resampling is ever used, apply it to the TRAINING split only, never to the test split.",
            "Split by time so the test period is genuinely in the future.",
        ],
    }