"""Suspicious Communities: groups of accounts whose money mostly stays inside the group."""
import streamlit as st

from dashboard import charts, data, ui

ui.start("Suspicious Communities", "Communities come from Louvain on the undirected, amount-weighted account graph.")

only_flagged = st.sidebar.checkbox("Flagged communities only", value=False, key="com_flagged")
n_rows = st.sidebar.selectbox("Communities shown", [25, 50, 100, 200], index=1, key="com_rows")
communities = data.q("top_communities", n_rows, flagged_only=only_flagged)

if communities.empty:
    st.info("No communities to show. Build them, then refresh the database:")
    st.code(
        "python -m src.graph.pipeline\npython -m src.graph.community_pipeline\npython -m src.models.scoring_pipeline",
        language="bash",
    )
    st.stop()

with st.expander("How the suspicion score works"):
    st.markdown(
        "The score (0-100) is a **project-defined heuristic**: a relative ranking of communities with at least 3 "
        "accounts by how closed and circular their money flow looks (share of value that stays inside, density, "
        "pass-through behaviour, small cycles). It is **not** a fraud probability. "
        "'Label fraud rate' uses the fraud label and is shown for understanding only."
    )

st.plotly_chart(charts.community_figure(communities), key="com_scatter")

columns = ["community_id", "n_accounts", "suspicion_score", "flagged", "internal_flow_share", "internal_density",
           "mean_pass_through_ratio", "share_in_small_cycle", "merchant_share", "top_account_by_pagerank",
           "label_fraud_rate_pct"]
view = communities[[c for c in columns if c in communities.columns]].reset_index(drop=True)
event = st.dataframe(
    view, hide_index=True, on_select="rerun", selection_mode="single-row", key="com_table",
    column_config={"suspicion_score": st.column_config.ProgressColumn("Suspicion", min_value=0, max_value=100, format="%.1f")},
)
rows = event.selection.rows
chosen = int(view.iloc[rows[0]]["community_id"]) if rows else int(view.iloc[0]["community_id"])

st.subheader(f"Members of community {chosen}")
st.caption("Select a community in the table above (the top one is shown by default).")
members = data.q("community_members", chosen, 200)
selected = ui.accounts_table(members, key="com_members", selectable=True)
if selected:
    st.write(f"Selected: **{selected}**")
    if st.button("Open in Investigation", key="com_open"):
        st.session_state["inv_pending"] = selected
        st.switch_page("pages/investigation.py")