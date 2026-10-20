"""Streamlit entry point. Run from the project root:   streamlit run dashboard/app.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # so pages can import config, src and dashboard

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Fraud Graph Analytics", layout="wide")

navigation = st.navigation(
    {
        "Analytics": [
            st.Page("pages/overview.py", title="Overview", default=True),
            st.Page("pages/transactions.py", title="Transaction Analytics"),
            st.Page("pages/graph.py", title="Graph Analytics"),
            st.Page("pages/fraud_detection.py", title="Fraud Detection"),
        ],
        "Investigate": [
            st.Page("pages/suspicious_accounts.py", title="Suspicious Accounts"),
            st.Page("pages/suspicious_communities.py", title="Suspicious Communities"),
            st.Page("pages/investigation.py", title="Investigation"),
        ],
    }
)
st.sidebar.caption("Batch analytics on precomputed results, not real-time detection. Risk bands are project-defined.")
navigation.run()