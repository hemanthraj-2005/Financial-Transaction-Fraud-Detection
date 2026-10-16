"""Reproducibility helper."""
from __future__ import annotations

import os
import random

import numpy as np

from config import RANDOM_SEED


def set_global_seed(seed: int = RANDOM_SEED) -> int:
    """Seed Python, NumPy and the hash seed. Returns the seed used."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return seed