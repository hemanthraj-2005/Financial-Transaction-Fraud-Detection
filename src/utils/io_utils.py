"""Small I/O helpers: Parquet tables (with CSV fallback) and JSON reports."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.errors import DatasetNotFoundError
from src.utils.logging_utils import get_logger

logger = get_logger("io")


def remove_path(path: Path | str) -> None:
    """Delete a file or a directory (Spark writes Parquet as a directory)."""
    path = Path(path)
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def table_paths(stem: Path | str) -> tuple[Path, Path]:
    stem = Path(stem)
    return stem.parent / f"{stem.name}.parquet", stem.parent / f"{stem.name}.csv"


def save_table(df: pd.DataFrame, stem: Path | str) -> Path:
    """Save as <stem>.parquet; fall back to <stem>.csv if pyarrow is missing.

    Any older version of the table (either format) is removed first, so a
    later load never picks up stale data.
    """
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    parquet_path, csv_path = table_paths(stem)
    remove_path(parquet_path)
    remove_path(csv_path)
    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path
    except ImportError:
        logger.warning(
            "pyarrow is not installed - saving '%s' as CSV instead. Run 'pip install pyarrow' for Parquet.",
            stem.name,
        )
    df.to_csv(csv_path, index=False)
    return csv_path


def load_table(stem: Path | str) -> pd.DataFrame:
    """Load <stem>.parquet (file or Spark directory) or <stem>.csv."""
    parquet_path, csv_path = table_paths(stem)
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    raise DatasetNotFoundError(
        f"Processed table '{Path(stem).name}' not found in {Path(stem).parent}. "
        "Run the preprocessing pipeline first:  python -m src.preprocessing.pipeline"
    )


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, Path)):
        return str(obj)
    return str(obj)


def write_json(obj: dict, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default))
    return path