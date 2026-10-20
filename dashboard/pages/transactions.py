"""Transaction Analytics: fraud by type, time and amount, plus the riskiest transactions."""
import streamlit as st

from dashboard import charts, data, ui

ui.start("Transaction Analytics", "Patterns in the scored transactions. Fraud labels are shown for understanding only.")
split = ui.split_selector("tx_split")

st.plotly_chart(charts.type_figure(data.q("fraud_by_type", split)), key="tx_type")

left, right = st.columns(2)
with left:
    st.plotly_chart(charts.daily_figure(data.q("daily_counts", split)), key="tx_daily")
with right:
    st.plotly_chart(charts.hourly_figure(data.q("hourly_counts", split)), key="tx_hourly")

left, right = st.columns(2)
with left:
    st.plotly_chart(charts.amount_figure(data.q("amount_distribution", split)), key="tx_amount")
with right:
    st.plotly_chart(charts.band_by_type_figure(data.q("risk_band_by_type", split)), key="tx_band_type")

st.subheader("Why transactions were flagged")
reasons = data.q("reason_code_counts", split, 31.0)
st.plotly_chart(charts.reasons_figure(reasons), key="tx_reasons")
if not reasons.empty:
    st.dataframe(reasons, hide_index=True, key="tx_reasons_table")

st.subheader("Riskiest transactions")
c1, c2 = st.columns(2)
min_risk = c1.slider("Minimum risk score", 0, 100, 61, key="tx_min_risk")
n_rows = c2.selectbox("Rows", [25, 50, 100, 250], index=1, key="tx_rows")
top = data.q("top_transactions", n_rows, split=split, min_risk=float(min_risk))
if top.empty:
    st.info("No transactions at or above this risk in the selected window.")
else:
    view = top[["tx_id", "step", "type", "amount", "nameOrig", "nameDest", "split", "risk_score", "risk_band",
                "label_isFraud", "reasons"]].rename(columns={"label_isFraud": "labelled_fraud"})
    st.dataframe(
        view, hide_index=True, key="tx_top_table",
        column_config={"risk_score": st.column_config.ProgressColumn("Risk", min_value=0, max_value=100, format="%.1f")},
    )
    st.download_button("Download this table (CSV)", ui.csv_bytes(view), "riskiest_transactions.csv", "text/csv")
st.caption("'labelled_fraud' is the dataset's fraud label, shown for understanding only.")