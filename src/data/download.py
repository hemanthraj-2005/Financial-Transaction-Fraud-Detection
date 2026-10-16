"""Download PaySim from Kaggle (optional - manual download works too).

Requirements: ``pip install kaggle`` and a Kaggle API token
(https://www.kaggle.com/settings -> API -> Create New Token, which saves
kaggle.json to ~/.kaggle/ on macOS/Linux or C:\\Users\\<you>\\.kaggle\\ on Windows).

Usage:
    python -m src.data.download
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from config import KAGGLE_DATASET, KAGGLE_URL, PAYSIM_FILENAME, RAW_DIR, ensure_directories
from src.data.loader import find_paysim_file
from src.utils.errors import FraudProjectError, MissingDependencyError
from src.utils.logging_utils import get_logger

logger = get_logger("download")


def manual_instructions(raw_dir: Path = RAW_DIR) -> str:
    return (
        "Manual download:\n"
        f"  1. Open {KAGGLE_URL} (free Kaggle account required)\n"
        "  2. Click Download -> the zip contains one CSV\n"
        f"  3. Extract it and put '{PAYSIM_FILENAME}' in {Path(raw_dir).resolve()}"
    )


def download_paysim(raw_dir: Path = RAW_DIR) -> Path:
    ensure_directories()
    existing = find_paysim_file(raw_dir)
    if existing:
        logger.info("PaySim already present: %s", existing)
        return existing

    if shutil.which("kaggle") is None:
        raise MissingDependencyError(
            "The 'kaggle' command was not found. Install it with: pip install kaggle\n" + manual_instructions(raw_dir)
        )

    cmd = ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(raw_dir), "--unzip"]
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise FraudProjectError(
            "Kaggle download failed (is your API token set up?).\n"
            f"{result.stderr.strip() or result.stdout.strip()}\n" + manual_instructions(raw_dir)
        )

    found = find_paysim_file(raw_dir)
    if not found:
        raise FraudProjectError("Download finished but no CSV was found.\n" + manual_instructions(raw_dir))
    logger.info("Downloaded PaySim to %s", found)
    return found


def main() -> None:
    try:
        download_paysim()
    except FraudProjectError as exc:
        print(f"\nERROR: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()