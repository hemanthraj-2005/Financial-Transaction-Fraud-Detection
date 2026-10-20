"""Phase 9 tests: dashboard queries, loaders and chart builders.
The Streamlit pages themselves are only checked for syntax (they need a browser to run)."""
import py_compile
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from dashboard import charts
from dashboard.loaders import db_status, open_readonly, read_csv, read_json, run_query
from src.db import analytics, queries
from src.db.database import (
    TRANSACTION_COLUMNS,
    connect,
    create_indexes,
    create_schema,
    insert_dataframe,
    write_run_info,
)
from src.models.risk import build_account_table, risk_band
from src.utils.errors import DatasetNotFoundError, InvalidAccountError

# tx_id, step, type, split, orig, dest, amount, risk, predicted, fraud, reason codes
ROWS = [
    (1, 1, "TRANSFER", "fit", "C1", "C2", 100.0, 90.0, 1, 1, "HIGH_AMOUNT;TYPE_TRANSFER"),
    (2, 2, "CASH_OUT", "fit", "C2", "C3", 90.0, 85.0, 1, 1, "HIGH_AMOUNT"),
    (3, 30, "PAYMENT", "fit", "C4", "M1", 10.0, 5.0, 0, 0, ""),
    (4, 31, "PAYMENT", "val", "C5", "M1", 20.0, 8.0, 0, 0, ""),
    (5, 50, "TRANSFER", "val", "C6", "C7", 5000.0, 70.0, 1, 0, ""),
    (6, 60, "CASH_IN", "test", "C8", "C9", 40.0, 12.0, 0, 0, ""),
    (7, 70, "TRANSFER", "test", "C2", "C1", 300.0, 95.0, 1, 1, "TYPE_TRANSFER"),
    (8, 71, "CASH_OUT", "test", "C1", "C10", 200.0, 40.0, 0, 1, ""),
    (9, 73, "PAYMENT", "test", "C4", "M1", 15.0, 3.0, 0, 0, ""),
    (10, 74, "TRANSFER", "test", "C11", "C12", 60.0, 20.0, 0, 0, ""),
]


def _scored():
    columns = ["tx_id", "step", "type", "split", "nameOrig", "nameDest", "amount", "risk_score", "predicted",
               "isFraud", "reason_codes"]
    df = pd.DataFrame(ROWS, columns=columns)
    df["day"] = (df["step"] - 1) // 24
    df["hour_of_day"] = (df["step"] - 1) % 24
    df["model_score"] = df["risk_score"] / 100.0
    df["reasons"] = df["reason_codes"].str.replace(";", " | ", regex=False)
    return df


def _build_db(path):
    scored = _scored()
    transactions = scored.assign(
        risk_band=risk_band(scored["risk_score"].to_numpy()), label_isFraud=scored["isFraud"]
    )[TRANSACTION_COLUMNS]
    accounts = build_account_table(scored)
    with closing(connect(path)) as conn:
        create_schema(conn)
        insert_dataframe(conn, "transactions", transactions)
        insert_dataframe(conn, "accounts", accounts)
        write_run_info(conn, {"model_name": "test_model", "data_source": "synthetic", "threshold": 0.6})
        create_indexes(conn)
    return path


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    return _build_db(tmp_path_factory.mktemp("dash") / "fraud.db")


# ------------------------------------------------------------------ analytics
def test_analytics_by_type_and_time(db):
    with closing(open_readonly(db)) as conn:
        by_type = analytics.fraud_by_type(conn).set_index("type")
        assert by_type.loc["TRANSFER", ["n_transactions", "n_fraud", "n_alerts"]].tolist() == [4, 2, 3]
        assert by_type.loc["TRANSFER", "fraud_rate_pct"] == pytest.approx(50.0)
        assert by_type.loc["PAYMENT", "n_fraud"] == 0 and by_type.loc["CASH_IN", "n_transactions"] == 1

        test_only = analytics.fraud_by_type(conn, "test").set_index("type")
        assert test_only.loc["TRANSFER", "n_transactions"] == 2 and test_only.loc["TRANSFER", "n_fraud"] == 1
        assert {"CASH_IN", "PAYMENT"} <= set(test_only.index)

        daily = analytics.daily_counts(conn)
        assert daily["day"].tolist() == [0, 1, 2, 3]
        assert daily["n_transactions"].tolist() == [2, 2, 4, 2]
        assert daily["n_fraud"].tolist() == [2, 0, 2, 0]
        assert daily["n_alerts"].tolist() == [2, 0, 2, 0]

        hourly = analytics.hourly_counts(conn).set_index("hour_of_day")
        assert hourly.loc[0, "n_transactions"] == 2 and hourly.loc[1, "n_transactions"] == 3
        assert int(hourly["n_transactions"].sum()) == 10

        bands = analytics.risk_band_by_type(conn)
        assert int(bands["n"].sum()) == 10
        with pytest.raises(ValueError):
            analytics.fraud_by_type(conn, "bogus")


def test_confusion_and_curves(db):
    with closing(open_readonly(db)) as conn:
        assert analytics.confusion_counts(conn) == {"tp": 3, "fp": 1, "fn": 1, "tn": 5}
        assert analytics.confusion_counts(conn, "test") == {"tp": 1, "fp": 0, "fn": 1, "tn": 3}

        curve = analytics.pr_curve(conn, "test")
        assert list(curve.columns) == ["precision", "recall", "threshold"]
        assert ((curve["recall"] == 1.0) & (curve["precision"] == 1.0)).any()  # the test window is perfectly ranked
        assert analytics.pr_curve(conn, "val").empty  # no fraud in the validation window


def test_distributions_sum_to_100(db):
    with closing(open_readonly(db)) as conn:
        amounts = analytics.amount_distribution(conn)
        risks = analytics.risk_histogram(conn)
        assert analytics.amount_distribution(conn, "val").groupby("class_name")["share_pct"].sum().tolist() == [
            pytest.approx(100.0)
        ]
    for frame in (amounts, risks):
        totals = frame.groupby("class_name")["share_pct"].sum()
        assert set(totals.index) == {"Legitimate", "Fraud"}
        assert totals.tolist() == [pytest.approx(100.0), pytest.approx(100.0)]


def test_reason_code_counts(db):
    with closing(open_readonly(db)) as conn:
        counts = analytics.reason_code_counts(conn)
        assert counts.set_index("code")["count"].to_dict() == {"HIGH_AMOUNT": 2, "TYPE_TRANSFER": 2}
        assert counts.set_index("code").loc["HIGH_AMOUNT", "description"] == "Unusually large amount"
        assert analytics.reason_code_counts(conn, "test").set_index("code")["count"].to_dict() == {"TYPE_TRANSFER": 1}
        assert analytics.reason_code_counts(conn, "val").empty


def test_search_and_lookup_queries(db):
    with closing(open_readonly(db)) as conn:
        assert analytics.search_accounts(conn, "C1")["account"].tolist() == ["C1", "C10", "C11", "C12"]
        assert analytics.search_accounts(conn, "%").empty and analytics.search_accounts(conn, "").empty
        assert analytics.search_accounts(conn, "_").empty
        found = analytics.accounts_by_ids(conn, ["C1", "M1", "NOPE"])
        assert set(found["account"]) == {"C1", "M1"}
        assert analytics.accounts_by_ids(conn, []).empty


def test_counterparties_and_neighborhood(db):
    with closing(open_readonly(db)) as conn:
        cp = analytics.counterparties(conn, "C1")
        assert list(zip(cp["counterparty"], cp["direction"])) == [
            ("C2", "received_from"), ("C10", "sent_to"), ("C2", "sent_to"),
        ]
        assert cp["total_amount"].tolist() == [300.0, 200.0, 100.0]

        edges, nodes = analytics.neighborhood(conn, "C1", 5)
        assert set(nodes["account"]) == {"C1", "C2", "C10"}
        assert nodes.set_index("account")["is_center"].to_dict() == {"C1": True, "C2": False, "C10": False}
        assert set(zip(edges["src"], edges["dst"])) == {("C1", "C2"), ("C2", "C1"), ("C1", "C10")}

        no_edges, no_nodes = analytics.neighborhood(conn, "NOPE", 5)
        assert no_edges.empty and no_nodes.empty


# ------------------------------------------------------------------ loaders
def test_loaders_status_and_registry(db, tmp_path):
    missing = tmp_path / "missing.db"
    ok, message = db_status(missing)
    assert not ok and "not found" in message.lower()
    assert not missing.exists()  # checking never creates the file

    ok, message = db_status(db)
    assert ok and "10" in message

    empty = tmp_path / "empty.db"
    with closing(connect(empty)) as conn:
        create_schema(conn)
    assert db_status(empty)[0] is False

    assert run_query(db, "summary")["n_transactions"] == 10
    assert run_query(db, "top_accounts", 1)["account"].tolist() == ["C1"]
    with pytest.raises(ValueError):
        run_query(db, "bogus")
    with pytest.raises(InvalidAccountError):
        run_query(db, "account_detail", "NOPE")
    with pytest.raises(DatasetNotFoundError):
        run_query(missing, "summary")

    assert read_json(tmp_path / "none.json") is None and read_csv(tmp_path / "none.csv") is None
    (tmp_path / "a.json").write_text('{"x": 1}')
    (tmp_path / "bad.json").write_text("{not json")
    assert read_json(tmp_path / "a.json") == {"x": 1} and read_json(tmp_path / "bad.json") is None
    pd.DataFrame({"a": [1]}).to_csv(tmp_path / "a.csv", index=False)
    assert read_csv(tmp_path / "a.csv")["a"].tolist() == [1]


# ------------------------------------------------------------------ charts
def test_charts_build_from_real_queries(db):
    with closing(open_readonly(db)) as conn:
        summary = queries.summary(conn)
        by_type = analytics.fraud_by_type(conn)
        daily = analytics.daily_counts(conn)
        hourly = analytics.hourly_counts(conn)
        amounts = analytics.amount_distribution(conn)
        bands = analytics.risk_band_by_type(conn)
        risks = analytics.risk_histogram(conn, "test")
        curve = analytics.pr_curve(conn, "test")
        counts = analytics.confusion_counts(conn, "test")
        reasons = analytics.reason_code_counts(conn)
        tx = queries.account_transactions(conn, "C1")
        edges, nodes = analytics.neighborhood(conn, "C1", 5)

    comparison = pd.DataFrame(
        {"feature_set": ["tx_only", "tx_graph"], "model": ["m", "m"], "test_precision": [0.5, 0.6],
         "test_recall": [0.4, 0.5], "test_f1": [0.44, 0.54], "test_pr_auc": [0.3, 0.4]}
    )
    communities = pd.DataFrame(
        {"community_id": [0, 1], "n_accounts": [5, 10], "suspicion_score": [80.0, 20.0], "flagged": [1, 0]}
    )
    metrics = pd.DataFrame({"account": ["a", "b"], "pagerank": [0.1, 0.2], "total_degree": [3, 4]})

    figures = [
        charts.band_bar(summary["bands_all"]),
        charts.type_figure(by_type),
        charts.daily_figure(daily),
        charts.hourly_figure(hourly),
        charts.amount_figure(amounts),
        charts.band_by_type_figure(bands),
        charts.risk_hist_figure(risks),
        charts.pr_figure(curve, (1.0, 1.0)),
        charts.confusion_figure(counts),
        charts.reasons_figure(reasons),
        charts.importance_figure([{"feature": "a", "importance": 0.5}, {"feature": "b", "importance": 0.2}]),
        charts.model_metrics_figure(comparison),
        charts.ablation_figure(comparison),
        charts.distribution_figure(np.arange(1, 100), "t", "x", log_x=True),
        charts.metric_scatter(metrics, "total_degree", "pagerank", "t"),
        charts.community_figure(communities),
        charts.timeline_figure(tx, "C1"),
        charts.network_figure(edges, nodes, "C1"),
    ]
    for fig in figures:
        assert isinstance(fig, go.Figure) and len(fig.data) >= 1


def test_charts_handle_empty_data():
    empty = pd.DataFrame()
    figures = [
        charts.band_bar({}),
        charts.type_figure(empty),
        charts.daily_figure(empty),
        charts.hourly_figure(empty),
        charts.amount_figure(empty),
        charts.band_by_type_figure(empty),
        charts.risk_hist_figure(empty),
        charts.pr_figure(empty),
        charts.reasons_figure(empty),
        charts.importance_figure([]),
        charts.model_metrics_figure(empty),
        charts.ablation_figure(empty),
        charts.distribution_figure(np.array([]), "t", "x"),
        charts.metric_scatter(empty, "a", "b", "t"),
        charts.community_figure(empty),
        charts.timeline_figure(empty, "C1"),
        charts.network_figure(empty, empty, "C1"),
    ]
    for fig in figures:
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0 and len(fig.layout.annotations) >= 1


# ------------------------------------------------------------------ pages
def test_dashboard_files_compile():
    files = sorted((Path(__file__).resolve().parents[1] / "dashboard").rglob("*.py"))
    assert len([f for f in files if f.name != "__init__.py"]) >= 12
    for path in files:
        py_compile.compile(str(path), doraise=True)