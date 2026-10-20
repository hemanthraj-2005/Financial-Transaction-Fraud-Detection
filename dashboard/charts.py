"""Plotly figure builders for the dashboard. Pure functions (DataFrame in, Figure out); no Streamlit."""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

LEGIT_COLOR = "#2a6fbb"
FRAUD_COLOR = "#d1495b"
NEUTRAL_COLOR = "#6b7280"
TEMPLATE = "plotly_white"
BANDS = ["Low", "Medium", "High", "Critical"]
BAND_COLORS = {"Low": "#7aa6d6", "Medium": "#f2c14e", "High": "#ef8354", "Critical": "#d1495b"}
CLASS_COLORS = {"Legitimate": LEGIT_COLOR, "Fraud": FRAUD_COLOR}


def _style(fig: go.Figure, title: str, height: int = 400) -> go.Figure:
    fig.update_layout(title=title, template=TEMPLATE, height=height, margin=dict(t=70, b=50, l=60, r=25))
    return fig


def empty_figure(message: str = "No data to show", height: int = 320) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper",
                       font=dict(size=14, color=NEUTRAL_COLOR))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(template=TEMPLATE, height=height, margin=dict(t=40, b=40, l=40, r=40))
    return fig


def _is_empty(df) -> bool:
    return df is None or len(df) == 0


# ------------------------------------------------------------------ overview / transactions
def band_bar(counts: dict, title: str = "Risk bands") -> go.Figure:
    values = [int(counts.get(band, 0)) for band in BANDS]
    total = sum(values)
    if total == 0:
        return empty_figure("No scored transactions in this window")
    fig = go.Figure(
        go.Bar(x=BANDS, y=values, marker_color=[BAND_COLORS[b] for b in BANDS],
               text=[f"{v:,} ({100 * v / total:.2f}%)" for v in values], textposition="outside")
    )
    fig.update_yaxes(type="log", title="Transactions (log scale)")
    return _style(fig, title, 380)


def type_figure(df: pd.DataFrame) -> go.Figure:
    if _is_empty(df):
        return empty_figure()
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Transactions per type (log scale)", "Labelled fraud rate per type (%)"))
    fig.add_trace(go.Bar(x=df["type"], y=df["n_transactions"], marker_color=NEUTRAL_COLOR,
                         text=[f"{v:,}" for v in df["n_transactions"]], textposition="outside"), row=1, col=1)
    fig.add_trace(go.Bar(x=df["type"], y=df["fraud_rate_pct"], marker_color=FRAUD_COLOR,
                         text=[f"{r:.2f}% ({int(f):,})" for r, f in zip(df["fraud_rate_pct"], df["n_fraud"])],
                         textposition="outside"), row=1, col=2)
    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_yaxes(title="Fraud rate (%)", row=1, col=2)
    fig.update_layout(showlegend=False)
    return _style(fig, "Fraud by transaction type")


def _count_figure(df: pd.DataFrame, x: str, title: str, xtitle: str) -> go.Figure:
    if _is_empty(df):
        return empty_figure()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df[x], y=df["n_fraud"], name="Labelled fraud", marker_color=FRAUD_COLOR))
    fig.add_trace(go.Bar(x=df[x], y=df["n_alerts"], name="Model alerts (High+)", marker_color=LEGIT_COLOR))
    fig.update_layout(barmode="group", legend=dict(orientation="h", y=1.12))
    fig.update_xaxes(title=xtitle)
    fig.update_yaxes(title="Transactions")
    return _style(fig, title)


def daily_figure(df: pd.DataFrame) -> go.Figure:
    return _count_figure(df, "day", "Labelled fraud vs model alerts per day", "Day (step 1 = day 0, hour 0)")


def hourly_figure(df: pd.DataFrame) -> go.Figure:
    return _count_figure(df, "hour_of_day", "Labelled fraud vs model alerts by hour of day", "Hour of day")


def amount_figure(df: pd.DataFrame) -> go.Figure:
    if _is_empty(df):
        return empty_figure()
    fig = go.Figure()
    for name, sub in df.groupby("class_name"):
        fig.add_trace(go.Bar(x=sub["log10_amount"], y=sub["share_pct"], name=name,
                             marker_color=CLASS_COLORS.get(name, NEUTRAL_COLOR), opacity=0.6))
    fig.update_layout(barmode="overlay", legend=dict(orientation="h", y=1.12))
    fig.update_xaxes(title="log10(1 + amount)")
    fig.update_yaxes(title="% of that class")
    return _style(fig, "Amount distribution (each class normalised to 100%)")


def band_by_type_figure(df: pd.DataFrame) -> go.Figure:
    if _is_empty(df):
        return empty_figure()
    pivot = df.pivot_table(index="type", columns="risk_band", values="n", aggfunc="sum", fill_value=0)
    pivot = pivot.reindex(columns=BANDS, fill_value=0)
    fig = go.Figure()
    for band in BANDS:
        fig.add_trace(go.Bar(name=band, x=pivot.index, y=pivot[band], marker_color=BAND_COLORS[band]))
    fig.update_layout(barmode="stack", barnorm="percent", legend=dict(orientation="h", y=1.12))
    fig.update_yaxes(title="% of the type's transactions")
    return _style(fig, "Risk bands by transaction type")


def reasons_figure(df: pd.DataFrame) -> go.Figure:
    if _is_empty(df):
        return empty_figure("No reason codes in this window")
    top = df.head(12).iloc[::-1]
    fig = go.Figure(go.Bar(x=top["count"], y=top["code"], orientation="h", marker_color=LEGIT_COLOR,
                           hovertext=top["description"], hoverinfo="text+x"))
    fig.update_xaxes(title="Transactions with this reason")
    return _style(fig, "Most common reasons behind Medium+ risk", 420)


# ------------------------------------------------------------------ model quality
def risk_hist_figure(df: pd.DataFrame, alert_risk: float = 61.0) -> go.Figure:
    if _is_empty(df):
        return empty_figure()
    fig = go.Figure()
    for name, sub in df.groupby("class_name"):
        fig.add_trace(go.Bar(x=sub["risk_mid"], y=sub["share_pct"], name=name,
                             marker_color=CLASS_COLORS.get(name, NEUTRAL_COLOR), opacity=0.65))
    fig.add_vline(x=alert_risk, line_dash="dash", line_color=NEUTRAL_COLOR, annotation_text="High+ (model flags)")
    fig.update_layout(barmode="overlay", legend=dict(orientation="h", y=1.12))
    fig.update_xaxes(title="Risk score")
    fig.update_yaxes(title="% of that class", type="log")
    return _style(fig, "Risk score distribution: fraud vs legitimate (log scale)")


def pr_figure(curve: pd.DataFrame, point: tuple | None = None) -> go.Figure:
    if _is_empty(curve):
        return empty_figure("No fraud cases in this window, so no precision-recall curve")
    fig = go.Figure(go.Scatter(x=curve["recall"], y=curve["precision"], mode="lines", line=dict(color=LEGIT_COLOR),
                               name="Precision-recall curve"))
    if point is not None:
        fig.add_trace(go.Scatter(x=[point[0]], y=[point[1]], mode="markers", name="Operating point (High+)",
                                 marker=dict(size=12, color=FRAUD_COLOR, symbol="diamond")))
    fig.update_xaxes(title="Recall", range=[0, 1.02])
    fig.update_yaxes(title="Precision", range=[0, 1.02])
    fig.update_layout(legend=dict(orientation="h", y=1.12))
    return _style(fig, "Precision-recall curve (test window)")


def confusion_figure(counts: dict) -> go.Figure:
    tp, fp, fn, tn = (int(counts.get(k, 0)) for k in ("tp", "fp", "fn", "tn"))
    z = [[fn, tp], [tn, fp]]
    fig = go.Figure(
        go.Heatmap(z=z, x=["Predicted legitimate", "Predicted fraud"], y=["Actually fraud", "Actually legitimate"],
                   text=[[f"{v:,}" for v in row] for row in z], texttemplate="%{text}", colorscale="Blues",
                   showscale=False)
    )
    fig.update_yaxes(autorange="reversed")
    return _style(fig, "Confusion matrix (model flags = High+)", 360)


def importance_figure(items: list) -> go.Figure:
    if not items:
        return empty_figure("No feature importance available")
    ordered = sorted(items, key=lambda item: item["importance"])
    fig = go.Figure(go.Bar(x=[i["importance"] for i in ordered], y=[i["feature"] for i in ordered],
                           orientation="h", marker_color=NEUTRAL_COLOR))
    fig.update_xaxes(title="Importance")
    return _style(fig, "Top features used by the selected model", 480)


_METRICS = [("test_precision", "Precision"), ("test_recall", "Recall"), ("test_f1", "F1"), ("test_pr_auc", "PR-AUC")]


def model_metrics_figure(df: pd.DataFrame) -> go.Figure:
    if _is_empty(df):
        return empty_figure()
    fig = go.Figure()
    for _, row in df.iterrows():
        fig.add_trace(go.Bar(name=str(row["model"]), x=[label for _, label in _METRICS],
                             y=[row[col] for col, _ in _METRICS]))
    fig.update_layout(barmode="group", legend=dict(orientation="h", y=1.12))
    fig.update_yaxes(title="Score", range=[0, 1.05])
    return _style(fig, "Model comparison on the test window (primary feature set)")


def ablation_figure(df: pd.DataFrame) -> go.Figure:
    if _is_empty(df):
        return empty_figure()
    fig = go.Figure()
    for col, label in _METRICS:
        fig.add_trace(go.Bar(name=label, x=df["feature_set"], y=df[col]))
    fig.update_layout(barmode="group", legend=dict(orientation="h", y=1.12))
    fig.update_yaxes(title="Score", range=[0, 1.05])
    return _style(fig, "Ablation: what each group of features adds (test window)")


# ------------------------------------------------------------------ graph / communities
def distribution_figure(values, title: str, xlabel: str, log_x: bool = False, bins: int = 50) -> go.Figure:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if log_x:
        arr = np.log10(arr[arr > 0])
    if len(arr) == 0:
        return empty_figure()
    counts, edges = np.histogram(arr, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    fig = go.Figure(go.Bar(x=centers, y=counts, marker_color=NEUTRAL_COLOR))
    fig.update_yaxes(type="log", title="Accounts (log scale)")
    fig.update_xaxes(title=("log10 of " if log_x else "") + xlabel)
    return _style(fig, title, 360)


def metric_scatter(df: pd.DataFrame, x: str, y: str, title: str, max_points: int = 5000) -> go.Figure:
    if _is_empty(df) or x not in df.columns or y not in df.columns:
        return empty_figure()
    sub = df[(df[x] > 0) & (df[y] > 0)]
    if sub.empty:
        return empty_figure()
    if len(sub) > max_points:
        sub = sub.sample(max_points, random_state=0)
    text = sub["account"].astype(str) if "account" in sub.columns else None
    fig = go.Figure(go.Scatter(x=sub[x], y=sub[y], mode="markers", text=text,
                               marker=dict(size=5, opacity=0.5, color=NEUTRAL_COLOR)))
    fig.update_xaxes(type="log", title=x)
    fig.update_yaxes(type="log", title=y)
    return _style(fig, title, 380)


def community_figure(df: pd.DataFrame) -> go.Figure:
    if _is_empty(df):
        return empty_figure()
    fig = go.Figure()
    for flagged, name, color in ((0, "Not flagged", NEUTRAL_COLOR), (1, "Flagged", FRAUD_COLOR)):
        sub = df[df["flagged"] == flagged]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["n_accounts"], y=sub["suspicion_score"], mode="markers", name=name,
            marker=dict(size=10, color=color, opacity=0.7),
            text=[f"community {int(c)}" for c in sub["community_id"]],
            hovertemplate="%{text}<br>accounts %{x}<br>score %{y}<extra></extra>",
        ))
    fig.update_xaxes(type="log", title="Accounts in community (log scale)")
    fig.update_yaxes(title="Suspicion score (relative heuristic)", range=[-3, 103])
    fig.update_layout(legend=dict(orientation="h", y=1.12))
    return _style(fig, "Communities: size vs suspicion score")


# ------------------------------------------------------------------ investigation
def timeline_figure(tx: pd.DataFrame, account: str) -> go.Figure:
    if _is_empty(tx):
        return empty_figure("No transactions for this account")
    fig = go.Figure()
    for band in BANDS:
        sub = tx[tx["risk_band"] == band]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["step"], y=sub["amount"], mode="markers", name=band,
            marker=dict(color=BAND_COLORS[band], size=11, line=dict(width=1, color="#111"),
                        symbol=["circle" if role == "sent" else "diamond" for role in sub["role"]]),
            text=[f"{r.role}: {r.nameOrig} -> {r.nameDest}<br>risk {r.risk_score}" for r in sub.itertuples()],
            hovertemplate="step %{x}<br>amount %{y:,.2f}<br>%{text}<extra></extra>",
        ))
    fig.update_xaxes(title="Time step (1 step = 1 hour)")
    fig.update_yaxes(type="log", title="Amount (log scale)")
    fig.update_layout(legend=dict(orientation="h", y=1.12))
    return _style(fig, f"Transactions of {account} (circle = sent, diamond = received)")


def network_figure(edges: pd.DataFrame, nodes: pd.DataFrame, center: str) -> go.Figure:
    if _is_empty(edges) or _is_empty(nodes):
        return empty_figure("No connections to draw for this account", 360)
    graph = nx.Graph()
    graph.add_nodes_from(nodes["account"])
    for row in edges.itertuples(index=False):
        graph.add_edge(row.src, row.dst)
    pos = nx.spring_layout(graph, seed=42, iterations=100, k=1.5 / max(1.0, np.sqrt(graph.number_of_nodes())))

    edge_x, edge_y, mid_x, mid_y, mid_text = [], [], [], [], []
    for row in edges.itertuples(index=False):
        if row.src not in pos or row.dst not in pos:
            continue
        (x0, y0), (x1, y1) = pos[row.src], pos[row.dst]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        mid_x.append((x0 + x1) / 2)
        mid_y.append((y0 + y1) / 2)
        mid_text.append(f"{row.src} -> {row.dst}<br>{int(row.n_tx)} tx, total {row.total_amount:,.0f}<br>max risk {row.max_risk}")

    fig = go.Figure(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=1, color="#c3c7cf"),
                               hoverinfo="skip", showlegend=False))
    if len(edges) <= 120:
        for row in edges.itertuples(index=False):
            if row.src in pos and row.dst in pos:
                (x0, y0), (x1, y1) = pos[row.src], pos[row.dst]
                fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y",
                                   showarrow=True, arrowhead=3, arrowsize=1.1, arrowwidth=1,
                                   arrowcolor="#9aa0ab", standoff=10)
    fig.add_trace(go.Scatter(x=mid_x, y=mid_y, mode="markers", marker=dict(size=10, color="rgba(0,0,0,0)"),
                             hovertext=mid_text, hoverinfo="text", showlegend=False))

    info = nodes.set_index("account")
    for band in BANDS:
        members = [a for a in graph.nodes if a in info.index and info.loc[a, "risk_band"] == band]
        if not members:
            continue
        fig.add_trace(go.Scatter(
            x=[pos[a][0] for a in members], y=[pos[a][1] for a in members], mode="markers+text", name=band,
            text=[a if a == center else "" for a in members], textposition="top center",
            marker=dict(size=[26 if a == center else 14 for a in members], color=BAND_COLORS[band],
                        line=dict(width=[3 if a == center else 1 for a in members], color="#111")),
            hovertext=[f"{a}<br>risk {info.loc[a, 'risk_score']} ({band})" for a in members], hoverinfo="text",
        ))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(legend=dict(orientation="h", y=1.05))
    return _style(fig, f"Money flow around {center} (arrow = direction, colour = account risk band)", 540)