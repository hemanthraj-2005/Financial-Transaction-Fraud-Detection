"""Time-based split and time blocks.

The split is by STEP (time), never random: the model is trained on the past and
evaluated on the future, like a real deployment.

    fit  : first ML_FIT_FRACTION of the step range
    val  : next ML_VAL_FRACTION (threshold tuning + model selection)
    test : the rest (touched once, for the final numbers)

Blocks cut the step range into equal-width pieces. Graph features for a row in
block b are computed from blocks < b only (see graph_features.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.errors import FraudProjectError

MIN_SPAN_STEPS = 10


def split_boundaries(steps: pd.Series, fit_fraction: float, val_fraction: float) -> dict:
    if not (0 < fit_fraction < 1 and 0 < val_fraction < 1 and fit_fraction + val_fraction < 1):
        raise ValueError("fit and val fractions must be in (0, 1) and sum to less than 1 (the rest is the test window)")
    min_step, max_step = int(steps.min()), int(steps.max())
    span = max_step - min_step + 1
    if span < MIN_SPAN_STEPS:
        raise FraudProjectError(
            f"The data covers only {span} time step(s); a time-based split needs at least {MIN_SPAN_STEPS}. "
            "Use more data (raise --max-tx)."
        )
    fit_end = min_step + int(span * fit_fraction) - 1
    val_end = min_step + int(span * (fit_fraction + val_fraction)) - 1
    if not (min_step <= fit_end < val_end < max_step):
        raise FraudProjectError("The time range is too short for the requested split fractions.")
    return {
        "min_step": min_step,
        "fit_end": fit_end,
        "val_end": val_end,
        "max_step": max_step,
        "fit_fraction": fit_fraction,
        "val_fraction": val_fraction,
    }


def assign_split(steps: pd.Series, bounds: dict) -> pd.Series:
    """Label every row 'fit', 'val' or 'test' from its step."""
    values = steps.to_numpy()
    labels = np.select([values <= bounds["fit_end"], values <= bounds["val_end"]], ["fit", "val"], default="test")
    return pd.Series(labels, index=steps.index)


def assign_blocks(steps: pd.Series, min_step: int, span: int, n_blocks: int) -> pd.Series:
    """Block index 0..n_blocks-1 (equal-width step ranges)."""
    values = steps.to_numpy(dtype="int64")
    return pd.Series(((values - int(min_step)) * int(n_blocks)) // int(span), index=steps.index, dtype="int64")