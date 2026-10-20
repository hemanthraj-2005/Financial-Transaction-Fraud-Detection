"""Graph Analytics: size of the transaction graph and which accounts are structurally important."""
import streamlit as st

from config import GRAPH_REPORT_PATH
from dashboard import charts, data, ui

ui.start("Graph Analytics", "Accounts are nodes, transactions are weighted directed edges.")

report = data.read_json(GRAPH_REPORT_PATH)
metrics = data.graph_metrics()
if report is None or metrics is None:
    st.info("Graph tables not found. Build them with:")
    st.code("python -m src.graph.pipeline", language="bash")
    st.stop()

selection, components = report["selection"], report["components"]
ui.kpis([
    ("Accounts (nodes)", ui.fmt_int(report["n_nodes"]), None),
    ("Account pairs (edges)", ui.fmt_int(report["n_edges"]), None),
    ("Density", ui.fmt_float(report["density"], 6), None),
    ("Weak components", ui.fmt_int(components["n_weakly_connected_components"]), None),
    ("Largest component", ui.fmt_int(components["largest_weakly_connected_component"]), None),
    ("Accounts in small cycles", ui.fmt_int(components["n_accounts_in_small_cycles"]), "Strongly connected groups of 2-50 accounts: money can loop."),
])
if selection.get("truncated_by_max_tx"):
    st.warning(f"The graph covers the first {selection['n_tx_used']:,} of {selection['n_input_tx']:,} transactions (by time).")
st.caption(
    "These graph figures come from the full-data Phase 5 run. They are for display and are not leakage-safe model "
    f"inputs. Betweenness: {report['betweenness']['scope']} (approximate, sampled)."
)

left, right = st.columns(2)
with left:
    st.plotly_chart(charts.distribution_figure(metrics["total_degree"], "Degree distribution", "total degree", log_x=True),
                    key="gr_degree")
with right:
    st.plotly_chart(charts.distribution_figure(metrics["pagerank"], "PageRank distribution", "PageRank", log_x=True),
                    key="gr_pagerank")
st.plotly_chart(charts.metric_scatter(metrics, "total_degree", "pagerank", "PageRank vs degree (random sample)"),
                key="gr_scatter")

st.subheader("Top accounts by a graph metric")
c1, c2 = st.columns(2)
options = ["pagerank", "betweenness", "total_degree", "pass_through_ratio", "in_amount", "out_amount", "reciprocal_partners"]
metric = c1.selectbox("Metric", options, key="gr_metric")
n_rows = c2.selectbox("Rows", [10, 25, 50, 100], index=1, key="gr_rows")

# dict.fromkeys removes duplicates (the chosen metric may already be one of the fixed columns)
columns = list(dict.fromkeys(["account", metric, "total_degree", "pagerank", "pass_through_ratio"]))
top = metrics.nlargest(n_rows, metric)[columns]
risk = data.q("accounts_by_ids", top["account"].astype(str).tolist())
if not risk.empty:
    top = top.merge(risk[["account", "risk_score", "risk_band"]], on="account", how="left")
st.dataframe(top, hide_index=True, key="gr_top_table")
st.caption(
    "A high graph metric is a structural signal (hub, bridge, pass-through), not proof of fraud. "
    "Risk columns come from the leakage-safe model."
)