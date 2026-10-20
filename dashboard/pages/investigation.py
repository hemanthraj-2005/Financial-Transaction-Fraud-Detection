"""Investigation: one account in detail."""
import pandas as pd
import streamlit as st

from dashboard import charts, data, ui
from src.utils.errors import InvalidAccountError

ui.start("Investigation", "One account in detail. Risk scores and reasons are indicators, not proof of fraud.")

# an account chosen on another page arrives through "inv_pending"
if "inv_pending" in st.session_state:
    st.session_state["inv_input"] = st.session_state.pop("inv_pending")
if "inv_input" not in st.session_state:
    first = data.q("top_accounts", 1, include_merchants=False)
    st.session_state["inv_input"] = str(first.iloc[0]["account"]) if not first.empty else ""


def _pick():
    choice = st.session_state.get("inv_pick")
    if choice and not choice.startswith("("):
        st.session_state["inv_input"] = choice


picks = data.q("top_accounts", 25, include_merchants=False)
st.sidebar.selectbox(
    "Quick pick (highest risk)", ["(choose an account)", *picks["account"].astype(str).tolist()],
    key="inv_pick", on_change=_pick,
)
account = st.text_input("Account ID", key="inv_input").strip()

if not account:
    st.info("Enter an account ID, or use the quick pick in the sidebar.")
    st.stop()

try:
    detail = data.q("account_detail", account)
except InvalidAccountError:
    st.error(f"Account '{account}' was not found.")
    matches = data.q("search_accounts", account, 10)
    if not matches.empty:
        st.write("Did you mean one of these?")
        ui.accounts_table(matches, key="inv_matches")
    st.stop()

ui.kpis([
    ("Risk score", ui.fmt_float(detail["risk_score"], 1), "Highest risk of any transaction touching this account."),
    ("Band", str(detail["risk_band"]), None),
    ("Transactions", ui.fmt_int(detail["n_transactions"]), None),
    ("Model alerts", ui.fmt_int(detail["n_alerts"]), "Transactions flagged High+."),
    ("Total sent", ui.fmt_money(detail["total_sent"]), None),
    ("Total received", ui.fmt_money(detail["total_received"]), None),
])

st.subheader("Why this account is risky")
reasons = [r for r in str(detail.get("reasons") or "").split(" | ") if r]
if reasons:
    for text in reasons:
        st.markdown(f"- {text}")
else:
    st.info("No reasons were generated: the riskiest transaction is below Medium, or no rule fired strongly.")
st.caption("Reasons come from the account's riskiest transaction and describe unusual features, not proof of wrongdoing.")

signals = [
    ("Total degree", detail.get("total_degree")),
    ("PageRank", detail.get("pagerank")),
    ("Betweenness (approx.)", detail.get("betweenness")),
    ("Pass-through ratio", detail.get("pass_through_ratio")),
    ("Community", detail.get("community_id")),
    ("Community suspicion score", detail.get("community_suspicion_score")),
]
signals = [(name, str(value)) for name, value in signals if value is not None]
if signals:
    with st.expander("Graph and community context (full-data run, display only)"):
        st.table(pd.DataFrame(signals, columns=["Signal", "Value"]))

tx = data.q("account_transactions", account, 500)
tab_network, tab_timeline, tab_counterparties, tab_transactions = st.tabs(
    ["Network", "Timeline", "Counterparties", "Transactions"]
)

with tab_network:
    max_neighbors = st.slider("Counterparties shown", 5, 40, 20, key="inv_neighbors")
    edges, nodes = data.q("neighborhood", account, max_neighbors)
    st.plotly_chart(charts.network_figure(edges, nodes, account), key="inv_network")
    st.caption("Shows the account and its largest counterparties by total amount, plus the transfers among them.")

with tab_timeline:
    st.plotly_chart(charts.timeline_figure(tx, account), key="inv_timeline")

with tab_counterparties:
    counterparties = data.q("counterparties", account, 100)
    if counterparties.empty:
        st.info("No counterparties.")
    else:
        st.dataframe(counterparties, hide_index=True, key="inv_cp_table")

with tab_transactions:
    if tx.empty:
        st.info("No transactions.")
    else:
        view = tx[["tx_id", "step", "role", "type", "amount", "nameOrig", "nameDest", "split", "risk_score",
                   "risk_band", "reasons"]]
        st.dataframe(
            view, hide_index=True, key="inv_tx_table",
            column_config={"risk_score": st.column_config.ProgressColumn("Risk", min_value=0, max_value=100, format="%.1f")},
        )
        st.download_button("Download transactions (CSV)", ui.csv_bytes(view), f"{account}_transactions.csv", "text/csv")
        if len(tx) >= 500:
            st.caption("Showing the oldest 500 transactions of this account.")