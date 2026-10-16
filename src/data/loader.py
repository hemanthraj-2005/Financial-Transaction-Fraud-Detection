"""Dataset loading with validation and friendly error messages.

Main dataset : PaySim (real project dataset, downloaded separately).
Fallback     : synthetic development data, used ONLY when PaySim is absent and
               the caller allows it. A loud warning is always logged and the
               DataFrame is tagged: ``df.attrs["source"] == "synthetic"``.

Usage:
    python -m src.data.loader                    # auto: PaySim, else synthetic
    python -m src.data.loader --source paysim    # fail if PaySim is missing
    python -m src.data.loader --source synthetic
    python -m src.data.loader --nrows 100000     # first N rows (file is time-ordered)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config import (
    NUMERIC_COLUMNS,
    PAYSIM_FILENAME,
    RAW_DIR,
    REQUIRED_COLUMNS,
    SYNTHETIC_PATH,
    ensure_directories,
)
from src.data.synthetic_generator import generate_synthetic_transactions, save_synthetic
from src.utils.errors import DatasetNotFoundError, EmptyDatasetError, FraudProjectError
from src.utils.logging_utils import get_logger
from src.utils.validation import (
    raise_if_invalid,
    require_columns,
    require_non_empty,
    transaction_quality_report,
)

logger = get_logger("loader")

DOWNLOAD_HELP = (
    "PaySim dataset not found.\n"
    "  Option A (manual): open https://www.kaggle.com/datasets/ealaxi/paysim1 , download the zip, "
    f"extract it and place '{PAYSIM_FILENAME}' in:\n      {{raw_dir}}\n"
    "  Option B (automatic): pip install kaggle, set up your Kaggle API token, then run:\n"
    "      python -m src.data.download\n"
    "  To try the app without the real data, use development data:\n"
    "      python -m src.data.synthetic_generator\n"
    "      python -m src.data.loader --source synthetic"
)


def find_paysim_file(raw_dir: Path = RAW_DIR) -> Path | None:
    """Return the PaySim CSV in raw_dir (exact name first, else any CSV), or None."""
    raw_dir = Path(raw_dir)
    exact = raw_dir / PAYSIM_FILENAME
    if exact.exists():
        return exact
    candidates = sorted(raw_dir.glob("*.csv")) if raw_dir.exists() else []
    return candidates[0] if candidates else None


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Convert numeric columns; unparseable values become NaN (reported, cleaned in Phase 3)."""
    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def read_csv_checked(path: Path, nrows: int | None = None) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise DatasetNotFoundError(f"File not found: {path}")
    try:
        df = pd.read_csv(path, nrows=nrows)
    except pd.errors.EmptyDataError as exc:
        raise EmptyDatasetError(f"The file is empty: {path}") from exc
    except pd.errors.ParserError as exc:
        raise FraudProjectError(f"Could not parse {path} as CSV: {exc}") from exc
    require_non_empty(df, f"file {path.name}")
    require_columns(df, REQUIRED_COLUMNS, f"file {path.name}")
    return _coerce_numeric(df)


def load_transactions(
    source: str = "auto",
    path: Path | str | None = None,
    nrows: int | None = None,
    raw_dir: Path = RAW_DIR,
    synthetic_path: Path = SYNTHETIC_PATH,
    strict: bool = False,
) -> pd.DataFrame:
    """Load and validate transactions.

    source: "paysim" (real data), "synthetic" (dev/test data) or "auto".
    strict: if True, raise InvalidValuesError when invalid values are present.
    The returned DataFrame has ``attrs["source"]`` and ``attrs["path"]`` set.
    """
    if source not in {"paysim", "synthetic", "auto"}:
        raise ValueError("source must be 'paysim', 'synthetic' or 'auto'")

    chosen: Path | None = Path(path) if path else None
    label = source

    if chosen is None and source in {"paysim", "auto"}:
        chosen = find_paysim_file(raw_dir)
        label = "paysim"
        if chosen is None:
            if source == "paysim":
                raise DatasetNotFoundError(DOWNLOAD_HELP.format(raw_dir=Path(raw_dir).resolve()))
            logger.warning("PaySim not found - falling back to SYNTHETIC development data. "
                           "Do NOT report results from synthetic data.")
            source = "synthetic"

    if chosen is None and source == "synthetic":
        label = "synthetic"
        chosen = Path(synthetic_path)
        if not chosen.exists():
            logger.info("Synthetic file missing - generating development data at %s", chosen)
            ensure_directories()
            save_synthetic(generate_synthetic_transactions(), path=chosen)

    if path and source == "synthetic":
        label = "synthetic"

    df = read_csv_checked(chosen, nrows=nrows)
    report = transaction_quality_report(df)
    if strict:
        raise_if_invalid(report)

    df.attrs["source"] = label
    df.attrs["path"] = str(chosen)
    df.attrs["quality_report"] = report
    logger.info("Loaded %d rows from %s data (%s)", len(df), label, chosen.name)
    return df


def describe_dataset(df: pd.DataFrame) -> str:
    """Human-readable summary (used by the CLI and later by reports)."""
    report = df.attrs.get("quality_report") or transaction_quality_report(df)
    lines = [
        f"Source            : {df.attrs.get('source', 'unknown')}"
        + ("  (SYNTHETIC - development/testing only)" if df.attrs.get("source") == "synthetic" else ""),
        f"File              : {df.attrs.get('path', 'n/a')}",
        f"Rows x columns    : {report['n_rows']:,} x {report['n_columns']}",
        f"Fraud / legit     : {report['n_fraud']:,} / {report['n_legitimate']:,} "
        f"({report['fraud_rate_pct']}% fraud, imbalance 1:{report['imbalance_ratio']})",
        f"Null values       : {report['null_counts'] or 'none'}",
        f"Negative amounts  : {report['negative_amounts']}",
        f"Zero amounts      : {report['zero_amounts']}",
        f"Unknown types     : {report['unknown_types']}",
        f"Self transfers    : {report['self_transfers']}",
        f"Duplicate rows    : {report['duplicate_rows']}",
        f"Types             : {df['type'].value_counts().to_dict()}",
        f"Steps (time)      : {int(df['step'].min())} .. {int(df['step'].max())}",
        f"Unique accounts   : {len(set(df['nameOrig']) | set(df['nameDest'])):,}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Load and inspect the transaction dataset.")
    parser.add_argument("--source", choices=["auto", "paysim", "synthetic"], default="auto")
    parser.add_argument("--path", type=Path, default=None, help="explicit CSV path")
    parser.add_argument("--nrows", type=int, default=None, help="load only the first N rows")
    parser.add_argument("--strict", action="store_true", help="fail on invalid values")
    args = parser.parse_args()
    try:
        df = load_transactions(args.source, args.path, args.nrows, strict=args.strict)
    except FraudProjectError as exc:
        print(f"\nERROR: {exc}")
        raise SystemExit(1)
    print("\n" + describe_dataset(df))
    print("\nFirst rows:\n" + df.head().to_string(index=False))


if __name__ == "__main__":
    main()