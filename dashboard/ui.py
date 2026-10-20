"""Shared Streamlit helpers: page start/guard, KPI rows, tables, formatting."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard import data

SPLIT_CHOICES = {
    "All data": None,
    "Test window (honest evaluation)": "test",
    "Validation window": "val",
    "Fit window (in-sample)": "fit",
}
BANDS = ["Low", "Medium", "High", "Critical"]

_ACCOUNT_VIEW = [
    "account", "risk_score", "risk_band", "n_transactions", "n_alerts", "total_sent", "total_received",
    "community_id", "reasons",
]


def start(title: str, caption: str | None = None) -> dict:
    """Title + database guard. Returns run_info; stops the page with instructions if the database is missing."""
    st.title(title)
    if caption:
        st.caption(caption)
    ok, message = data.status()
    if not ok:
        st.error(message)
        st.code("python -m src.models.pipeline\npython -m src.models.scoring_pipeline", language="bash")
        st.stop()
    info = data.q("run_info")
    if info.get("data_source") == "synthetic":
        st.warning("SYNTHETIC development data: do not report results from this dashboard.")
    return info


def split_selector(key: str, default_index: int = 0):
    choice = st.sidebar.selectbox("Data window", list(SPLIT_CHOICES), index=default_index, key=key)
    value = SPLIT_CHOICES[choice]
    if value == "fit":
        st.sidebar.caption("Fit-window scores are in-sample: the model was trained on these rows.")
    return value


def kpis(items: list) -> None:
    columns = st.columns(len(items))
    for column, (label, value, help_text) in zip(columns, items):
        column.metric(label, value, help=help_text)


def fmt_int(value) -> str:
    return "n/a" if value is None else f"{int(value):,}"


def fmt_float(value, digits: int = 3) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def fmt_pct(value, digits: int = 2) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}%"


def fmt_money(value) -> str:
    if value is None:
        return "n/a"
    v = float(value)
    for limit, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(v) >= limit:
            return f"{v / limit:.2f}{suffix}"
    return f"{v:,.0f}"


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def accounts_table(df: pd.DataFrame, key: str, selectable: bool = False) -> str | None:
    """Show an accounts table. If selectable, returns the selected account ID (or None)."""
    if df is None or df.empty:
        st.info("No accounts match.")
        return None
    view = df[[c for c in _ACCOUNT_VIEW if c in df.columns]].reset_index(drop=True)
    config = {
        "risk_score": st.column_config.ProgressColumn("Risk", min_value=0, max_value=100, format="%.1f"),
        "reasons": st.column_config.TextColumn("Top reasons (riskiest transaction)", width="large"),
        "total_sent": st.column_config.NumberColumn("Total sent", format="%.0f"),
        "total_received": st.column_config.NumberColumn("Total received", format="%.0f"),
    }
    config = {name: spec for name, spec in config.items() if name in view.columns}
    if selectable:
        event = st.dataframe(view, hide_index=True, column_config=config, on_select="rerun",
                             selection_mode="single-row", key=key)
        rows = event.selection.rows
        return str(view.iloc[rows[0]]["account"]) if rows else None
    st.dataframe(view, hide_index=True, column_config=config, key=key)
    return None