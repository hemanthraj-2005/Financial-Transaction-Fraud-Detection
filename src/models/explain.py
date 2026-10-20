"""Reason codes: why was this transaction scored as risky?

Method (deliberately simple and explainable):
  * Baseline = legitimate transactions of the FIT window only (no validation/test labels).
  * "high" rule: fires when a feature is above the 95th percentile of that baseline.
  * "flag" rule: fires when a yes/no feature is 1 AND at most 25% of legitimate transactions have it
    (a flag that is common among legitimate activity is no evidence).
  * Reasons are ranked by (how extreme the value is) x (how important the feature is to the model)
    and the top few are kept.
Reasons describe unusual features of a transaction. They are explanations, not proof of fraud.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import REASON_FLAG_MAX_PREVALENCE, REASON_MAX, REASON_PERCENTILE
from src.models.train import feature_importance
from src.utils.validation import require_non_empty

GRID_POINTS = 1001  # quantile grid per feature (0%, 0.1%, ..., 100%)


@dataclass(frozen=True)
class ReasonRule:
    code: str
    feature: str
    kind: str  # "high" or "flag"
    message: str


REASON_RULES = [
    ReasonRule("HIGH_AMOUNT", "amount", "high", "Unusually large amount"),
    ReasonRule("SENDER_LARGE_HISTORY", "orig_prev_amount_sum", "high", "Sender has already moved an unusually large total"),
    ReasonRule("SENDER_ACTIVE", "orig_prev_tx_count", "high", "Sender is unusually active"),
    ReasonRule("RECEIVER_ACTIVE", "dest_prev_tx_count", "high", "Receiver is unusually active (possible collection account)"),
    ReasonRule("RECEIVER_LARGE_INFLOW", "dest_prev_amount_sum", "high", "Receiver has already collected an unusually large total"),
    ReasonRule("SENDER_PASS_THROUGH", "s_pass_through_ratio", "high", "Sender forwards almost everything it receives (pass-through)"),
    ReasonRule("RECEIVER_PASS_THROUGH", "r_pass_through_ratio", "high", "Receiver forwards almost everything it receives (pass-through)"),
    ReasonRule("SENDER_CENTRAL", "s_pagerank", "high", "Sender is unusually central in the money flow"),
    ReasonRule("RECEIVER_CENTRAL", "r_pagerank", "high", "Receiver is unusually central in the money flow"),
    ReasonRule("SENDER_BRIDGE", "s_betweenness", "high", "Sender bridges many otherwise separate accounts"),
    ReasonRule("RECEIVER_BRIDGE", "r_betweenness", "high", "Receiver bridges many otherwise separate accounts"),
    ReasonRule("SENDER_CIRCULAR", "s_reciprocal_partners", "high", "Sender exchanges money both ways with the same accounts (circular flow)"),
    ReasonRule("RECEIVER_CIRCULAR", "r_reciprocal_partners", "high", "Receiver exchanges money both ways with the same accounts (circular flow)"),
    ReasonRule("RECEIVER_FAN_IN", "r_in_degree", "high", "Receiver gets money from unusually many different accounts (fan-in)"),
    ReasonRule("SENDER_FAN_OUT", "s_out_degree", "high", "Sender pays unusually many different accounts (fan-out)"),
    ReasonRule("REPEAT_PAIR", "pair_prev_tx_count", "high", "Repeated transfers between the same two accounts"),
    ReasonRule("ROUND_TRIP", "pair_reverse_tx_count", "high", "Money has also flowed back between these two accounts (round trip)"),
    ReasonRule("SENDER_RISKY_COMMUNITY", "s_comm_fraud_rate", "high", "Sender's community has an unusually high earlier fraud rate"),
    ReasonRule("RECEIVER_RISKY_COMMUNITY", "r_comm_fraud_rate", "high", "Receiver's community has an unusually high earlier fraud rate"),
    ReasonRule("SENDER_CLOSED_COMMUNITY", "s_comm_flow_share", "high", "Sender's community mostly trades within itself"),
    ReasonRule("RECEIVER_CLOSED_COMMUNITY", "r_comm_flow_share", "high", "Receiver's community mostly trades within itself"),
    ReasonRule("TYPE_TRANSFER", "type_TRANSFER", "flag", "Transaction type is TRANSFER"),
    ReasonRule("TYPE_CASH_OUT", "type_CASH_OUT", "flag", "Transaction type is CASH_OUT"),
    ReasonRule("SENDER_IN_CYCLE", "s_in_small_cycle", "flag", "Sender is part of a small circular-flow group"),
    ReasonRule("RECEIVER_IN_CYCLE", "r_in_small_cycle", "flag", "Receiver is part of a small circular-flow group"),
    ReasonRule("SAME_COMMUNITY", "same_community", "flag", "Both accounts sit in the same tight community"),
    ReasonRule("NIGHT_TIME", "is_night", "flag", "Made during night hours"),
]


@dataclass
class Baseline:
    rule: ReasonRule
    cutoff: float = 0.0
    grid: np.ndarray | None = None
    prevalence: float = 0.0


def importance_weights(model, name: str, columns: list[str]) -> dict:
    """Feature -> importance scaled to 0..1 (1 = the model's most important feature)."""
    items = feature_importance(model, name, columns, top=len(columns))
    top = max((item["importance"] for item in items), default=0.0)
    if top <= 0:
        return {column: 1.0 for column in columns}
    return {item["feature"]: item["importance"] / top for item in items}


def fit_baselines(
    baseline_rows: pd.DataFrame,
    rules=REASON_RULES,
    percentile: float = REASON_PERCENTILE,
    flag_max_prevalence: float = REASON_FLAG_MAX_PREVALENCE,
) -> dict:
    """Learn per-rule baselines from LEGITIMATE FIT-WINDOW rows. Inactive rules are left out."""
    require_non_empty(baseline_rows, "baseline (legitimate fit-window) rows")
    baselines = {}
    for rule in rules:
        if rule.feature not in baseline_rows.columns:
            continue
        values = baseline_rows[rule.feature].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            continue
        if rule.kind == "high":
            grid = np.quantile(values, np.linspace(0.0, 1.0, GRID_POINTS))
            cutoff = float(grid[int(round(percentile * (GRID_POINTS - 1)))])
            baselines[rule.code] = Baseline(rule, cutoff=cutoff, grid=grid)
        elif rule.kind == "flag":
            prevalence = float((values > 0.5).mean())
            if prevalence <= flag_max_prevalence:
                baselines[rule.code] = Baseline(rule, prevalence=prevalence)
        else:
            raise ValueError(f"Unknown rule kind '{rule.kind}' for {rule.code}")
    return baselines


def _describe(baseline: Baseline, share: float) -> str:
    if baseline.rule.kind == "high":
        return f"{baseline.rule.message} (higher than {min(99.9, 100.0 * share):.1f}% of legitimate activity)"
    return f"{baseline.rule.message} (seen in only {100.0 * baseline.prevalence:.1f}% of legitimate transactions)"


def explain_transactions(
    features: pd.DataFrame, baselines: dict, weights: dict, top_n: int = REASON_MAX
) -> pd.DataFrame:
    """Return reason_codes ('A;B;C') and reasons ('text | text | text') per row, same index as ``features``."""
    n = len(features)
    out = pd.DataFrame({"reason_codes": [""] * n, "reasons": [""] * n}, index=features.index)
    items = [b for b in baselines.values() if b.rule.feature in features.columns]
    if n == 0 or not items:
        return out

    strength = np.zeros((n, len(items)))
    shares = np.zeros((n, len(items)))
    weight = np.array([0.25 + 0.75 * float(weights.get(b.rule.feature, 0.0)) for b in items])
    for j, b in enumerate(items):
        values = np.nan_to_num(features[b.rule.feature].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        if b.rule.kind == "high":
            share = np.clip(np.searchsorted(b.grid, values, side="left") / (len(b.grid) - 1), 0.0, 1.0)
            strength[:, j] = np.where(values > b.cutoff, share, 0.0)
            shares[:, j] = share
        else:
            strength[:, j] = np.where(values > 0.5, 1.0 - b.prevalence, 0.0)

    order = np.argsort(-(strength * weight), axis=1, kind="stable")[:, :top_n]
    codes, texts = [], []
    for i in range(n):
        picked = [j for j in order[i] if strength[i, j] > 0]
        codes.append(";".join(items[j].rule.code for j in picked))
        texts.append(" | ".join(_describe(items[j], shares[i, j]) for j in picked))
    out["reason_codes"] = codes
    out["reasons"] = texts
    return out