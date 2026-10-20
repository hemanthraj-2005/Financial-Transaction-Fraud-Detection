"""Suspicious Accounts: filter, search and open an account in the Investigation view."""
import streamlit as st

from dashboard import data, ui

ui.start("Suspicious Accounts", "An account's risk is the highest risk of any transaction touching it.")

st.sidebar.header("Filters")
min_risk = st.sidebar.slider("Minimum risk score", 0, 100, 61, key="acc_min_risk")
include_merchants = st.sidebar.checkbox("Include merchants", value=False, key="acc_merchants")
n_rows = st.sidebar.selectbox("Rows", [50, 100, 250, 500], index=0, key="acc_rows")

query = st.text_input("Search account ID (partial match, ignores the risk filter)", key="acc_search").strip()
if query:
    table = data.q("search_accounts", query, n_rows)
else:
    table = data.q("top_accounts", n_rows, min_risk=float(min_risk), include_merchants=include_merchants)

st.caption(f"{len(table):,} account(s) shown. Select a row to investigate it.")
selected = ui.accounts_table(table, key="acc_table", selectable=True)

if selected:
    st.write(f"Selected: **{selected}**")
    if st.button("Open in Investigation", key="acc_open"):
        st.session_state["inv_pending"] = selected
        st.switch_page("pages/investigation.py")

if not table.empty:
    st.download_button("Download this table (CSV)", ui.csv_bytes(table), "suspicious_accounts.csv", "text/csv")
st.caption("Risk scores are indicators for human review, not proof of fraud.")