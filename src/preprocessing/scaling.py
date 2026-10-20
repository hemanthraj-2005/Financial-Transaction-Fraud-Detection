"""Leakage-safe feature scaling.

``FeatureScaler`` learns mean and standard deviation from the data passed to
``fit`` (ALWAYS the training split only) and then applies those same numbers
to any other split. Parameters are saved as JSON so the dashboard can scale a
single new transaction exactly the way the training data was scaled.

Tree models (Random Forest, XGBoost) do not need scaling; Logistic Regression
does. Phase 7 uses this for the Logistic Regression baseline.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.utils.errors import FraudProjectError, MissingColumnsError


class FeatureScaler:
    def __init__(self, columns: list[str] | None = None) -> None:
        self.columns = list(columns) if columns else []
        self.mean_: dict[str, float] = {}
        self.std_: dict[str, float] = {}

    @property
    def is_fitted(self) -> bool:
        return bool(self.mean_)

    def fit(self, df: pd.DataFrame, columns: list[str] | None = None) -> "FeatureScaler":
        if columns:
            self.columns = list(columns)
        missing = [c for c in self.columns if c not in df.columns]
        if missing:
            raise MissingColumnsError(f"Cannot fit scaler, missing columns: {missing}")
        if len(df) == 0:
            raise FraudProjectError("Cannot fit a scaler on an empty DataFrame.")
        for column in self.columns:
            self.mean_[column] = float(df[column].mean())
            std = float(df[column].std(ddof=0))
            self.std_[column] = std if std > 0 else 1.0  # constant column: avoid divide-by-zero
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise FraudProjectError("FeatureScaler is not fitted. Call fit() on the training data first.")
        missing = [c for c in self.columns if c not in df.columns]
        if missing:
            raise MissingColumnsError(f"Cannot scale, missing columns: {missing}")
        out = df.copy()
        for column in self.columns:
            out[column] = (out[column].astype("float64") - self.mean_[column]) / self.std_[column]
        return out

    def fit_transform(self, df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
        return self.fit(df, columns).transform(df)

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"columns": self.columns, "mean": self.mean_, "std": self.std_}, indent=2))
        return path

    @classmethod
    def load(cls, path: Path | str) -> "FeatureScaler":
        path = Path(path)
        if not path.exists():
            raise FraudProjectError(f"Scaler file not found: {path}")
        data = json.loads(path.read_text())
        scaler = cls(data["columns"])
        scaler.mean_, scaler.std_ = data["mean"], data["std"]
        return scaler