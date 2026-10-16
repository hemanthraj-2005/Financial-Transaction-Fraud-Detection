"""Synthetic PaySim-format transaction generator.

*** DEVELOPMENT / TESTING DATA ONLY ***
This data is randomly generated. It exists so the application and the test
suite can run before the real PaySim dataset is downloaded. It must NEVER be
used for reported results; all real experiments use PaySim.

The output has exactly the PaySim columns. It contains:
  * normal activity (payments, cash-in/out, transfers, debits)
  * legitimate high-value transfers (so "large amount" alone is not fraud)
  * layering chains          A -> B -> C -> D -> cash-out
  * circular mule rings      A -> B -> C -> A (repeated rounds)
  * fan-in mule accounts     many victims -> one mule -> cash-out
  * repeated structured transfers just below 10,000
  * camouflage: fraud accounts also make a few ordinary legitimate payments

Like PaySim, fraudulent transfers drain the sender's balance to zero. That
signature is what makes balance features leak the label in PaySim, so the
modelling phase treats balance-derived features carefully.

Usage:
    python -m src.data.synthetic_generator --rows 30000 --seed 42
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    RANDOM_SEED,
    REQUIRED_COLUMNS,
    SYNTHETIC_META_PATH,
    SYNTHETIC_PATH,
    ensure_directories,
)
from src.utils.logging_utils import get_logger

logger = get_logger("synthetic")

TX_TYPES = ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"]
TX_PROBS = [0.34, 0.08, 0.35, 0.22, 0.01]
FLAGGED_THRESHOLD = 200_000  # PaySim rule: TRANSFER above 200,000 is "flagged"
SYNTHETIC_NOTICE = "SYNTHETIC DATA - development and testing only. Not real transactions."


class _FraudRows:
    """Small helper that collects fraud/camouflage rows."""

    def __init__(self, max_step: int) -> None:
        self.max_step = max_step
        self.rows: list[dict] = []

    def add(self, step, tx_type, amount, orig, old_o, new_o, dest, old_d, new_d, fraud) -> None:
        self.rows.append(
            {
                "step": int(min(max(step, 1), self.max_step)),
                "type": tx_type,
                "amount": round(float(amount), 2),
                "nameOrig": orig,
                "oldbalanceOrg": round(float(old_o), 2),
                "newbalanceOrig": round(float(new_o), 2),
                "nameDest": dest,
                "oldbalanceDest": round(float(old_d), 2),
                "newbalanceDest": round(float(new_d), 2),
                "isFraud": int(fraud),
                "isFlaggedFraud": 0,
            }
        )

    def drained_transfer(self, step, amount, orig, dest) -> None:
        """Fraud signature: sender balance is emptied, receiver starts at zero."""
        self.add(step, "TRANSFER", amount, orig, amount, 0.0, dest, 0.0, amount, 1)

    def cash_out(self, step, amount, orig, dest) -> None:
        self.add(step, "CASH_OUT", amount, orig, amount, 0.0, dest, 0.0, amount, 1)


def _normal_transactions(rng, n, customers, merchants, max_step, step_probs) -> pd.DataFrame:
    n_customers = len(customers)
    types = rng.choice(TX_TYPES, size=n, p=TX_PROBS)
    steps = rng.choice(np.arange(1, max_step + 1), size=n, p=step_probs)
    amount = rng.lognormal(mean=9.0, sigma=1.1, size=n)

    # Legitimate high-value transfers: large amounts are not automatically fraud
    big = (types == "TRANSFER") & (rng.random(n) < 0.05)
    amount[big] = rng.uniform(250_000, 3_000_000, size=int(big.sum()))
    amount = np.round(amount, 2)

    orig_idx = rng.integers(0, n_customers, size=n)
    dest_idx = (orig_idx + rng.integers(1, n_customers, size=n)) % n_customers  # never a self-loop
    to_merchant = np.isin(types, ["PAYMENT", "DEBIT"])
    dest = np.where(to_merchant, rng.choice(merchants, size=n), customers[dest_idx])

    outgoing = types != "CASH_IN"
    old_orig = np.where(
        outgoing, amount * (1.0 + rng.exponential(3.0, size=n)), rng.lognormal(9.0, 1.5, size=n)
    )
    new_orig = np.where(outgoing, old_orig - amount, old_orig + amount)

    has_balance = (~to_merchant) & (rng.random(n) < 0.7)
    old_dest = np.where(has_balance, rng.lognormal(9.0, 1.5, size=n), 0.0)
    new_dest = np.where(to_merchant, 0.0, old_dest + amount)
    old_dest = np.where(to_merchant, 0.0, old_dest)

    return pd.DataFrame(
        {
            "step": steps,
            "type": types,
            "amount": amount,
            "nameOrig": customers[orig_idx],
            "oldbalanceOrg": np.round(old_orig, 2),
            "newbalanceOrig": np.round(new_orig, 2),
            "nameDest": dest,
            "oldbalanceDest": np.round(old_dest, 2),
            "newbalanceDest": np.round(new_dest, 2),
            "isFraud": 0,
            "isFlaggedFraud": 0,
        }
    )


def generate_synthetic_transactions(
    n_normal: int = 30_000,
    n_customers: int = 3_000,
    n_merchants: int = 300,
    max_step: int = 743,
    n_chains: int = 15,
    n_rings: int = 6,
    n_fan_in: int = 8,
    n_repeat_pairs: int = 10,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Generate a PaySim-format DataFrame with embedded fraud patterns."""
    rng = np.random.default_rng(seed)
    customers = np.array([f"C{i:09d}" for i in range(n_customers)])
    merchants = np.array([f"M{i:09d}" for i in range(n_merchants)])

    # Daytime-heavy activity: lower volume at night (hour of day = step % 24)
    hours = np.arange(1, max_step + 1) % 24
    weights = 1.0 + 0.8 * np.sin((hours - 6) / 24.0 * 2 * np.pi)
    step_probs = weights / weights.sum()

    # Reserve a pool of accounts for fraud patterns (sampled without replacement)
    pool = rng.permutation(n_customers)
    cursor = {"i": 0}

    def take(k: int) -> np.ndarray:
        start, end = cursor["i"], cursor["i"] + k
        if end > n_customers:
            raise ValueError(
                f"Not enough customers ({n_customers}) for the requested fraud patterns. "
                "Increase n_customers or reduce n_chains / n_rings / n_fan_in / n_repeat_pairs."
            )
        cursor["i"] = end
        return customers[pool[start:end]]

    fraud = _FraudRows(max_step)
    fraud_accounts: list[str] = []

    # 1) Layering chains: A -> B -> C -> D, amount shrinks slightly per hop, then cash-out
    for _ in range(n_chains):
        length = int(rng.integers(3, 6))
        accts = take(length)
        step = int(rng.integers(1, max_step - length - 2))
        amount = float(rng.lognormal(12.0, 0.8))
        for i in range(length - 1):
            fraud.drained_transfer(step + i, amount, accts[i], accts[i + 1])
            amount *= float(rng.uniform(0.93, 0.99))
        fraud.cash_out(step + length - 1, amount, accts[-1], rng.choice(customers))
        fraud_accounts.extend(accts.tolist())

    # 2) Circular mule rings with repeated rounds
    for _ in range(n_rings):
        size = int(rng.integers(5, 9))
        ring = take(size)
        rounds = int(rng.integers(3, 6))
        step = int(rng.integers(1, max_step - size * rounds - 2))
        base = float(rng.lognormal(11.5, 0.5))
        for r in range(rounds):
            for i in range(size):
                fraud.drained_transfer(
                    step + r * size + i, base * float(rng.uniform(0.95, 1.0)), ring[i], ring[(i + 1) % size]
                )
        fraud.cash_out(step + rounds * size, base, ring[0], rng.choice(customers))
        fraud_accounts.extend(ring.tolist())

    # 3) Fan-in: many victims send to one mule, mule cashes out the total
    for _ in range(n_fan_in):
        n_victims = int(rng.integers(8, 21))
        mule = take(1)[0]
        victims = take(n_victims)
        step = int(rng.integers(1, max_step - 10))
        total = 0.0
        for victim in victims:
            amount = float(rng.lognormal(10.5, 0.6))
            fraud.drained_transfer(step + int(rng.integers(0, 4)), amount, victim, mule)
            total += amount
        fraud.cash_out(step + 5, total, mule, rng.choice(customers))
        fraud_accounts.extend([mule, *victims.tolist()])

    # 4) Repeated structured transfers kept just below 10,000
    for _ in range(n_repeat_pairs):
        sender, receiver = take(2)
        times = int(rng.integers(4, 9))
        step = int(rng.integers(1, max_step - 60))
        for _t in range(times):
            amount = float(rng.uniform(9_000, 9_900))
            step += int(rng.integers(1, 6))
            fraud.add(step, "TRANSFER", amount, sender, amount * rng.uniform(1.5, 5.0),
                      0.0, receiver, 0.0, amount, 1)
            # keep balances consistent: new = old - amount
            fraud.rows[-1]["newbalanceOrig"] = round(fraud.rows[-1]["oldbalanceOrg"] - amount, 2)
        fraud_accounts.extend([sender, receiver])

    # 5) Camouflage: ~40% of fraud accounts also make ordinary, legitimate payments
    for account in rng.choice(np.array(fraud_accounts), size=int(0.4 * len(fraud_accounts)), replace=False):
        for _ in range(int(rng.integers(1, 4))):
            amount = float(rng.lognormal(7.5, 0.8))
            old = amount * float(rng.uniform(1.5, 6.0))
            fraud.add(int(rng.integers(1, max_step + 1)), "PAYMENT", amount, account, old, old - amount,
                      rng.choice(merchants), 0.0, 0.0, 0)

    normal = _normal_transactions(rng, n_normal, customers, merchants, max_step, step_probs)
    df = pd.concat([normal, pd.DataFrame(fraud.rows)], ignore_index=True)
    df["isFlaggedFraud"] = ((df["type"] == "TRANSFER") & (df["amount"] > FLAGGED_THRESHOLD)).astype(int)
    df = df.sort_values("step", kind="stable").reset_index(drop=True)
    return df[REQUIRED_COLUMNS]


def save_synthetic(df: pd.DataFrame, path: Path = SYNTHETIC_PATH, meta_path: Path = SYNTHETIC_META_PATH,
                   seed: int = RANDOM_SEED) -> Path:
    """Write the CSV plus a metadata file that clearly labels it as synthetic."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    meta = {
        "notice": SYNTHETIC_NOTICE,
        "seed": seed,
        "rows": int(len(df)),
        "fraud_rows": int(df["isFraud"].sum()),
        "fraud_rate_pct": round(100 * float(df["isFraud"].mean()), 3),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    Path(meta_path).write_text(json.dumps(meta, indent=2))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SYNTHETIC dev/test transactions (PaySim format).")
    parser.add_argument("--rows", type=int, default=30_000, help="number of normal transactions")
    parser.add_argument("--customers", type=int, default=3_000)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--output", type=Path, default=SYNTHETIC_PATH)
    args = parser.parse_args()

    ensure_directories()
    df = generate_synthetic_transactions(n_normal=args.rows, n_customers=args.customers, seed=args.seed)
    out = save_synthetic(df, path=args.output, seed=args.seed)
    logger.info("%s", SYNTHETIC_NOTICE)
    logger.info("Wrote %d rows (%d fraud, %.2f%%) to %s", len(df), df["isFraud"].sum(),
                100 * df["isFraud"].mean(), out)


if __name__ == "__main__":
    main()