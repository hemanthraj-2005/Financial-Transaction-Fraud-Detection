"""Overview: headline numbers, risk-band distribution and the riskiest accounts."""
import streamlit as st

from dashboard import charts, data, ui

info = ui.start("Overview", "Batch risk scoring of transactions and accounts. This is not real-time detection.")

summary = data.q("summary")
by_type = data.q("fraud_by_type")
n_tx = summary["n_transactions"]
n_fraud = int(by_type["n_fraud"].sum()) if not by_type.empty else 0

ui.kpis([
    ("Transactions scored", ui.fmt_int(n_tx), None),
    ("Accounts", ui.fmt_int(summary["n_accounts"]), None),
    ("Model alerts (High+)", ui.fmt_int(summary["n_alerts"]), "Transactions the model flags as fraud (risk 61 or above)."),
    ("Labelled fraud", ui.fmt_int(n_fraud), "From the dataset's fraud label. Shown for understanding only."),
    ("Labelled fraud rate", ui.fmt_pct(100 * n_fraud / n_tx if n_tx else 0, 3), None),
])

st.caption(
    f"Model: {info.get('model_name', 'n/a')} on feature set '{info.get('feature_set', 'n/a')}', "
    f"decision threshold {ui.fmt_float(info.get('threshold'), 4)}. Data source: {info.get('data_source', 'unknown')}."
)

left, right = st.columns(2)
with left:
    st.plotly_chart(charts.band_bar(summary["bands_all"], "Risk bands: all scored transactions"), key="ov_bands_all")
with right:
    st.plotly_chart(charts.band_bar(summary["bands_test"], "Risk bands: test window only"), key="ov_bands_test")

st.subheader("Riskiest accounts")
ui.accounts_table(data.q("top_accounts", 10, include_merchants=False), key="ov_accounts")
st.caption("Open 'Suspicious Accounts' to filter and search, or 'Investigation' to look at one account in detail.")

with st.expander("How to read these numbers"):
    st.markdown(
        "- **Risk score (0-100)** comes from the model's probability, anchored so that the model's decision "
        "threshold maps to 61 (the bottom of High).\n"
        "- **Bands** (Low 0-30, Medium 31-60, High 61-80, Critical 81-100) are project-defined, not scientifically validated.\n"
        "- **Scores on the fit window are in-sample**; use the test window for honest evaluation.\n"
        "- **Reasons and graph signals are indicators, not proof of fraud.**"
    )