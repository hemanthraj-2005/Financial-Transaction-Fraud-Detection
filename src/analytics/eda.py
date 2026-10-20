"""Exploratory data analysis (Plotly).

Reads the Phase 3 outputs (transactions_clean + account_aggregates) and writes
interactive HTML charts to reports/eda/ plus an eda_summary.json.

Notes
-----
* Charts that need the fraud label (fraud rates, top fraud accounts) are for
  UNDERSTANDING the data. They are not model inputs.
* Heavy charts use grouped counts or pre-binned histograms, so they stay fast
  even on the full 6.3M-row PaySim file.

Usage:
    python -m src.analytics.eda
    python -m src.analytics.eda --output-dir reports/eda
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import ACCOUNT_AGG_STEM, EDA_DIR, TRANSACTIONS_CLEAN_STEM
from src.utils.errors import FraudProjectError
from src.utils.io_utils import load_table, write_json
from src.utils.logging_utils import get_logger
from src.utils.validation import require_columns, require_non_empty

logger = get_logger("eda")

LEGIT_COLOR = "#2a6fbb"
FRAUD_COLOR = "#d1495b"
NEUTRAL_COLOR = "#6b7280"
TEMPLATE = "plotly_white"
LEAK_TYPES = ["TRANSFER", "CASH_OUT"]  # in PaySim, fraud occurs only in these types

_TX_NEEDED = ["type", "amount", "log_amount", "day", "hour_of_day", "isFraud",
              "isFlaggedFraud", "orig_balance_emptied"]
_ACC_NEEDED = ["account", "total_amount", "label_fraud_sent", "label_fraud_received"]


def _style(fig: go.Figure, title: str, height: int = 460) -> go.Figure:
    fig.update_layout(title=title, template=TEMPLATE, height=height, margin=dict(t=80, b=50, l=60, r=30))
    return fig


def _empty_note(fig: go.Figure, text: str, row: int, col: int) -> None:
    fig.add_annotation(text=text, showarrow=False, row=row, col=col, xref="x domain", yref="y domain",
                       x=0.5, y=0.5, font=dict(color=NEUTRAL_COLOR))


# ---------------------------------------------------------------- summary numbers
def summary_numbers(tx: pd.DataFrame, acc: pd.DataFrame) -> dict:
    n = int(len(tx))
    fraud = tx["isFraud"] == 1
    flagged = tx["isFlaggedFraud"] == 1
    n_fraud = int(fraud.sum())
    risky = tx["type"].isin(LEAK_TYPES)

    def pct(num: float, den: float) -> float | None:
        return round(100.0 * num / den, 2) if den else None

    emptied = tx["orig_balance_emptied"] == 1
    return {
        "n_transactions": n,
        "n_accounts": int(len(acc)),
        "n_fraud": n_fraud,
        "fraud_rate_pct": round(100.0 * n_fraud / n, 4) if n else 0.0,
        "step_range": [int(tx["step"].min()), int(tx["step"].max())] if "step" in tx and n else None,
        "flag_rule_recall_pct": pct(int((fraud & flagged).sum()), n_fraud),
        "flag_rule_precision_pct": pct(int((fraud & flagged).sum()), int(flagged.sum())),
        "leakage_check_pct_balance_emptied": {
            "fraud": pct(int((fraud & risky & emptied).sum()), int((fraud & risky).sum())),
            "legit": pct(int((~fraud & risky & emptied).sum()), int((~fraud & risky).sum())),
            "note": "Computed on TRANSFER and CASH_OUT rows only. A large gap means balance "
                    "features can leak the label (handled in the Phase 7 ablation).",
        },
    }


# ---------------------------------------------------------------- figures
def fig_class_imbalance(tx: pd.DataFrame) -> go.Figure:
    counts = tx["isFraud"].value_counts().reindex([0, 1], fill_value=0)
    total = max(int(counts.sum()), 1)
    fig = go.Figure(
        go.Bar(
            x=["Legitimate", "Fraud"],
            y=counts.values,
            marker_color=[LEGIT_COLOR, FRAUD_COLOR],
            text=[f"{v:,} ({100 * v / total:.3f}%)" for v in counts.values],
            textposition="outside",
        )
    )
    fig.update_yaxes(type="log", title="Transactions (log scale)")
    _style(fig, "Class imbalance: fraud is extremely rare (accuracy would be misleading)")
    return fig


def fig_fraud_by_type(tx: pd.DataFrame) -> go.Figure:
    g = tx.groupby("type").agg(n=("isFraud", "size"), n_fraud=("isFraud", "sum")).reset_index()
    g["rate"] = 100.0 * g["n_fraud"] / g["n"]
    g = g.sort_values("n", ascending=False)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Transactions per type (log scale)", "Fraud rate per type (%)"))
    fig.add_trace(go.Bar(x=g["type"], y=g["n"], marker_color=NEUTRAL_COLOR, name="Transactions",
                         text=[f"{v:,}" for v in g["n"]], textposition="outside"), row=1, col=1)
    fig.add_trace(go.Bar(x=g["type"], y=g["rate"], marker_color=FRAUD_COLOR, name="Fraud rate %",
                         text=[f"{r:.2f}% ({int(f):,})" for r, f in zip(g["rate"], g["n_fraud"])],
                         textposition="outside"), row=1, col=2)
    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_yaxes(title="Fraud rate (%)", row=1, col=2)
    fig.update_layout(showlegend=False)
    return _style(fig, "Fraud by transaction type")


def fig_fraud_over_time(tx: pd.DataFrame) -> go.Figure:
    g = tx.groupby("day").agg(n=("isFraud", "size"), n_fraud=("isFraud", "sum")).reset_index().sort_values("day")
    g["rate"] = 100.0 * g["n_fraud"] / g["n"]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=g["day"], y=g["n_fraud"], name="Fraud transactions", marker_color=FRAUD_COLOR,
                         opacity=0.75), secondary_y=False)
    fig.add_trace(go.Scatter(x=g["day"], y=g["rate"], name="Fraud rate (%)", mode="lines+markers",
                             line=dict(color=LEGIT_COLOR)), secondary_y=True)
    fig.update_xaxes(title="Day (step 1 = day 0, hour 0)")
    fig.update_yaxes(title="Fraud transactions", secondary_y=False)
    fig.update_yaxes(title="Fraud rate (%)", secondary_y=True)
    return _style(fig, "Fraud over time (per day)")


def fig_fraud_by_hour(tx: pd.DataFrame) -> go.Figure:
    g = tx.groupby("hour_of_day").agg(n=("isFraud", "size"), n_fraud=("isFraud", "sum")).reset_index()
    g["rate"] = 100.0 * g["n_fraud"] / g["n"]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Transactions by hour of day", "Fraud rate by hour of day (%)"))
    fig.add_trace(go.Bar(x=g["hour_of_day"], y=g["n"], marker_color=NEUTRAL_COLOR), row=1, col=1)
    fig.add_trace(go.Bar(x=g["hour_of_day"], y=g["rate"], marker_color=FRAUD_COLOR), row=1, col=2)
    fig.update_xaxes(title="Hour of day", dtick=2)
    fig.update_layout(showlegend=False)
    return _style(fig, "Time-of-day pattern")


def fig_amount_distribution(tx: pd.DataFrame, bins: int = 40) -> go.Figure:
    values = tx["log_amount"].to_numpy(dtype=float)
    edges = np.linspace(float(values.min()), float(values.max()) + 1e-9, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    fig = go.Figure()
    for label, name, color in ((0, "Legitimate", LEGIT_COLOR), (1, "Fraud", FRAUD_COLOR)):
        subset = tx.loc[tx["isFraud"] == label, "log_amount"].to_numpy(dtype=float)
        if len(subset) == 0:
            continue
        counts, _ = np.histogram(subset, bins=edges)
        fig.add_trace(go.Bar(x=centers, y=100.0 * counts / len(subset), name=name, marker_color=color, opacity=0.6))
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title="log(1 + amount)")
    fig.update_yaxes(title="% of that class's transactions")
    return _style(fig, "Amount distribution: fraud vs legitimate (each class normalised to 100%)")


def fig_top_accounts(acc: pd.DataFrame, n: int = 15) -> go.Figure:
    acc = acc.assign(fraud_tx_involved=acc["label_fraud_sent"] + acc["label_fraud_received"])
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.18,
                        subplot_titles=(f"Top {n} accounts by total value", f"Top {n} accounts by fraud transactions"))

    top_value = acc.nlargest(n, "total_amount").iloc[::-1]
    fig.add_trace(go.Bar(y=top_value["account"].astype(str), x=top_value["total_amount"], orientation="h",
                         marker_color=NEUTRAL_COLOR), row=1, col=1)

    top_fraud = acc[acc["fraud_tx_involved"] > 0].nlargest(n, "fraud_tx_involved").iloc[::-1]
    if len(top_fraud):
        fig.add_trace(go.Bar(y=top_fraud["account"].astype(str), x=top_fraud["fraud_tx_involved"],
                             orientation="h", marker_color=FRAUD_COLOR), row=1, col=2)
    else:
        _empty_note(fig, "No fraud-labelled accounts", 1, 2)
    fig.update_xaxes(title="Total amount (sent + received)", row=1, col=1)
    fig.update_xaxes(title="Fraud transactions (sent + received)", row=1, col=2)
    fig.update_layout(showlegend=False)
    return _style(fig, "Top accounts (label-based view: for analysis only, never a model input)", height=560)


def fig_balance_leakage(tx: pd.DataFrame) -> go.Figure:
    subset = tx[tx["type"].isin(LEAK_TYPES)]
    g = subset.groupby("isFraud")["orig_balance_emptied"].mean() * 100.0
    labels = {0: "Legitimate", 1: "Fraud"}
    colors = {0: LEGIT_COLOR, 1: FRAUD_COLOR}
    fig = go.Figure(
        go.Bar(x=[labels[i] for i in g.index], y=g.values, marker_color=[colors[i] for i in g.index],
               text=[f"{v:.1f}%" for v in g.values], textposition="outside")
    )
    fig.update_yaxes(title="% of transactions that empty the sender's balance", range=[0, 110])
    return _style(fig, "Leakage check (TRANSFER + CASH_OUT only): balance features reveal the label")


def fig_flag_rule(tx: pd.DataFrame) -> go.Figure:
    fraud = tx["isFraud"] == 1
    flagged = tx["isFlaggedFraud"] == 1
    hit = int((fraud & flagged).sum())
    recall = 100.0 * hit / max(int(fraud.sum()), 1)
    precision = 100.0 * hit / max(int(flagged.sum()), 1)
    fig = go.Figure(
        go.Bar(x=["Fraud caught by the flag (recall)", "Flagged that are fraud (precision)"],
               y=[recall, precision], marker_color=[FRAUD_COLOR, LEGIT_COLOR],
               text=[f"{recall:.2f}%", f"{precision:.2f}%"], textposition="outside")
    )
    fig.update_yaxes(title="%", range=[0, 110])
    return _style(fig, "The built-in isFlaggedFraud rule: why it is excluded from the features")


FIGURES = {
    "01_class_imbalance": fig_class_imbalance,
    "02_fraud_by_type": fig_fraud_by_type,
    "03_fraud_over_time": fig_fraud_over_time,
    "04_fraud_by_hour": fig_fraud_by_hour,
    "05_amount_distribution": fig_amount_distribution,
    "07_balance_leakage": fig_balance_leakage,
    "08_flag_rule": fig_flag_rule,
}


# ---------------------------------------------------------------- runner
def run_eda(
    output_dir: Path | str = EDA_DIR,
    tx: pd.DataFrame | None = None,
    acc: pd.DataFrame | None = None,
) -> dict:
    """Build all charts, save them as HTML and write eda_summary.json."""
    tx = load_table(TRANSACTIONS_CLEAN_STEM) if tx is None else tx
    acc = load_table(ACCOUNT_AGG_STEM) if acc is None else acc
    require_non_empty(tx, "transactions table")
    require_non_empty(acc, "account table")
    require_columns(tx, _TX_NEEDED, "transactions table")
    require_columns(acc, _ACC_NEEDED, "account table")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figures = {name: build(tx) for name, build in FIGURES.items()}
    figures["06_top_accounts"] = fig_top_accounts(acc)

    paths = {}
    for name, fig in sorted(figures.items()):
        path = output_dir / f"{name}.html"
        fig.write_html(str(path), include_plotlyjs="cdn")  # needs internet when you open the file
        paths[name] = str(path)
        logger.info("Saved %s", path.name)

    summary = summary_numbers(tx, acc)
    summary["charts"] = paths
    write_json(summary, output_dir / "eda_summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate EDA charts from the preprocessed data.")
    parser.add_argument("--output-dir", type=Path, default=EDA_DIR)
    args = parser.parse_args()
    try:
        summary = run_eda(args.output_dir)
    except FraudProjectError as exc:
        print(f"\nERROR: {exc}")
        raise SystemExit(1)
    print(f"\nTransactions: {summary['n_transactions']:,} | accounts: {summary['n_accounts']:,} | "
          f"fraud: {summary['n_fraud']:,} ({summary['fraud_rate_pct']}%)")
    print(f"isFlaggedFraud rule: recall {summary['flag_rule_recall_pct']}%, "
          f"precision {summary['flag_rule_precision_pct']}%")
    leak = summary["leakage_check_pct_balance_emptied"]
    print(f"Sender balance emptied (TRANSFER/CASH_OUT): fraud {leak['fraud']}% vs legit {leak['legit']}%")
    print(f"\nCharts written to {args.output_dir}  (open the .html files in a browser)")


if __name__ == "__main__":
    main()