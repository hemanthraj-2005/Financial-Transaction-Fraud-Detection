"""Fraud Detection: how well does the model work on the untouched test window?"""
import streamlit as st

from config import ML_COMPARISON_PATH, ML_REPORT_PATH
from dashboard import charts, data, ui

info = ui.start("Fraud Detection", "Time-based evaluation: trained on the past, judged on the future.")

report = data.read_json(ML_REPORT_PATH)
comparison = data.read_csv(ML_COMPARISON_PATH)
selected = (report or {}).get("selected_model", {})

counts = data.q("confusion_counts", "test")
tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
precision = tp / (tp + fp) if (tp + fp) else 0.0
recall = tp / (tp + fn) if (tp + fn) else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
pr_auc = (selected.get("test") or {}).get("pr_auc")

st.subheader(f"Selected model: {info.get('model_name', 'n/a')}")
st.caption(
    f"Feature set '{info.get('feature_set', 'n/a')}', decision threshold {ui.fmt_float(info.get('threshold'), 4)}, "
    "chosen by validation PR-AUC. The numbers below are from the test window only."
)
ui.kpis([
    ("Precision", ui.fmt_float(precision), "Of the transactions flagged, the share that were fraud."),
    ("Recall", ui.fmt_float(recall), "Of the fraud transactions, the share that were flagged."),
    ("F1", ui.fmt_float(f1), None),
    ("PR-AUC", ui.fmt_float(pr_auc), "Area under the precision-recall curve (primary metric for rare fraud)."),
    ("Fraud caught / missed", f"{tp:,} / {fn:,}", f"{fp:,} legitimate transactions were flagged by mistake."),
])

tab_quality, tab_models, tab_ablation, tab_reasons = st.tabs(
    ["Test-window quality", "Model comparison", "Ablation", "Reason codes"]
)

with tab_quality:
    left, right = st.columns(2)
    with left:
        curve = data.q("pr_curve", "test")
        point = (recall, precision) if (tp + fp + fn) else None
        st.plotly_chart(charts.pr_figure(curve, point), key="fd_pr")
    with right:
        st.plotly_chart(charts.confusion_figure(counts), key="fd_conf")
    st.plotly_chart(charts.risk_hist_figure(data.q("risk_histogram", "test")), key="fd_riskhist")
    st.plotly_chart(charts.importance_figure(selected.get("top_features") or []), key="fd_importance")
    st.caption(
        "Importance shows what the model relies on, not why fraud happens. For logistic regression it is the absolute "
        "standardised coefficient; for tree models it is the built-in importance."
    )

primary = (report or {}).get("primary_feature_set", "tx_graph")
with tab_models:
    if comparison is None:
        st.info("model_comparison.csv not found. Run: python -m src.models.pipeline")
    else:
        rows = comparison[comparison["feature_set"] == primary]
        st.plotly_chart(charts.model_metrics_figure(rows), key="fd_models")
        st.dataframe(rows, hide_index=True, key="fd_models_table")

with tab_ablation:
    if comparison is None or not selected:
        st.info("Ablation results not found. Run: python -m src.models.pipeline")
    else:
        rows = comparison[comparison["model"] == selected.get("name")]
        if rows["feature_set"].nunique() < 2:
            st.info("The ablation was skipped in the last run (--skip-ablation).")
        else:
            order = {"tx_only": 0, "tx_graph": 1, "tx_graph_balance": 2}
            rows = rows.sort_values("feature_set", key=lambda col: col.map(order))
            st.plotly_chart(charts.ablation_figure(rows), key="fd_ablation")
            st.dataframe(rows, hide_index=True, key="fd_ablation_table")
            st.markdown(
                "- **tx_only**: transaction features and past-only account history.\n"
                "- **tx_graph**: adds graph and community features (the primary, honest feature set). "
                "The gap to tx_only is what the graph adds.\n"
                "- **tx_graph_balance**: adds balance columns, which reveal the label in PaySim. Its scores are "
                "**inflated by leakage** and must not be presented as a real result."
            )

with tab_reasons:
    reasons = data.q("reason_code_counts", "test", 61.0)
    st.plotly_chart(charts.reasons_figure(reasons), key="fd_reasons")
    if not reasons.empty:
        st.dataframe(reasons, hide_index=True, key="fd_reasons_table")
    st.caption("Reasons for High+ transactions in the test window.")

if report and report.get("leakage_controls"):
    with st.expander("Leakage controls used in this project"):
        for control in report["leakage_controls"]:
            st.markdown(f"- {control}")