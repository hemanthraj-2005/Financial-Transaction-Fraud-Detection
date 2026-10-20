"""Phase 8 tests: risk mapping, account aggregation, reason codes, SQLite storage, scoring pipeline."""
import json
from contextlib import closing

import numpy as np
import pandas as pd
import pytest

from src.data.synthetic_generator import generate_synthetic_transactions
from src.db import queries
from src.db.database import (
    ACCOUNT_DB_COLUMNS,
    COMMUNITY_DB_COLUMNS,
    TRANSACTION_COLUMNS,
    connect,
    create_indexes,
    create_schema,
    insert_dataframe,
    write_run_info,
)
from src.features.transaction_features import add_transaction_features
from src.models.explain import ReasonRule, explain_transactions, fit_baselines
from src.models.pipeline import run_ml_pipeline
from src.models.risk import (
    ACCOUNT_RISK_COLUMNS,
    aggregate_accounts,
    build_account_table,
    probability_to_risk,
    risk_band,
)
from src.models.scoring_pipeline import run_scoring_pipeline
from src.preprocessing.cleaning import clean_transactions
from src.utils.errors import InvalidAccountError
from src.utils.io_utils import load_table, save_table


# ------------------------------------------------------------------ helpers
def _scored():
    return pd.DataFrame(
        {
            "tx_id": [1, 2, 3, 4],
            "step": [1, 2, 3, 4],
            "split": ["fit", "fit", "test", "test"],
            "nameOrig": ["C1", "C2", "C1", "C3"],
            "nameDest": ["C2", "M1", "C3", "C1"],
            "amount": [100.0, 50.0, 70.0, 20.0],
            "risk_score": [90.0, 10.0, 65.0, 40.0],
            "predicted": [1, 0, 1, 0],
            "isFraud": [1, 0, 0, 0],
            "reason_codes": ["A;B", "", "B", ""],
            "reasons": ["r1 | r2", "", "r2", ""],
        }
    )


def _transactions_table(scored):
    return scored.assign(
        model_score=lambda d: d["risk_score"] / 100.0,
        risk_band=lambda d: risk_band(d["risk_score"].to_numpy()),
        day=0,
        hour_of_day=0,
        type="TRANSFER",
        label_isFraud=lambda d: d["isFraud"],
    )[TRANSACTION_COLUMNS]


@pytest.fixture(scope="module")
def features_tx():
    raw = generate_synthetic_transactions(n_normal=3000, n_customers=700, seed=5)
    return add_transaction_features(clean_transactions(raw)[0])


# ------------------------------------------------------------------ risk mapping
def test_probability_to_risk_anchors():
    t = 0.4
    p = np.array([0.0, t / 2, t, (1 + t) / 2, 1.0])
    assert probability_to_risk(p, t).tolist() == [0.0, 30.0, 61.0, 81.0, 100.0]


def test_risk_is_monotonic_and_matches_the_threshold():
    p = np.random.default_rng(0).random(2000)
    order = np.argsort(p)
    for t in (0.05, 0.3, 0.5, 0.9):
        risk = probability_to_risk(p, t)
        assert (np.diff(risk[order]) >= 0).all()
        assert ((p >= t) == (risk >= 61.0)).all()  # flagged exactly when risk reaches High


def test_risk_band_edges():
    risks = np.array([0, 30.9, 31, 60.9, 61, 80.9, 81, 100])
    assert list(risk_band(risks)) == ["Low", "Low", "Medium", "Medium", "High", "High", "Critical", "Critical"]


def test_extreme_thresholds_do_not_break_the_mapping():
    p = np.linspace(0, 1, 101)
    for t in (0.0, 1e-9, 0.999999999, 1.0):
        risk = probability_to_risk(p, t)
        assert np.isfinite(risk).all() and risk.min() >= 0 and risk.max() <= 100
        assert (np.diff(risk) >= 0).all()


# ------------------------------------------------------------------ accounts
def test_aggregate_accounts_exact_values():
    acc = aggregate_accounts(_scored())
    assert list(acc.columns) == ACCOUNT_RISK_COLUMNS
    a = acc.set_index("account")
    assert len(a) == 4

    c1 = a.loc["C1"]
    assert (c1.n_transactions, c1.n_sent, c1.n_received) == (3, 2, 1)
    assert (c1.total_sent, c1.total_received) == (170.0, 20.0)
    assert (c1.max_risk, c1.risk_score, c1.risk_band) == (90.0, 90.0, "Critical")
    assert c1.mean_risk == pytest.approx(65.0)
    assert (c1.n_alerts, c1.n_high_risk_tx, c1.riskiest_tx_id, c1.label_fraud_tx) == (2, 2, 1, 1)

    c2 = a.loc["C2"]
    assert (c2.n_sent, c2.n_received, c2.total_sent, c2.total_received) == (1, 1, 50.0, 100.0)
    assert c2.mean_risk == pytest.approx(50.0) and c2.label_fraud_tx == 1

    c3 = a.loc["C3"]
    assert (c3.risk_band, c3.riskiest_tx_id, c3.label_fraud_tx) == ("High", 3, 0)
    assert c3.mean_risk == pytest.approx(52.5)

    m1 = a.loc["M1"]
    assert (m1.is_merchant, m1.n_sent, m1.risk_band, m1.riskiest_tx_id) == (1, 0, "Low", 2)


def test_build_account_table_reasons_and_optional_joins():
    scored = _scored()
    plain = build_account_table(scored)
    assert list(plain.columns) == ACCOUNT_DB_COLUMNS
    p = plain.set_index("account")
    assert (p.loc["C1", "reason_codes"], p.loc["C1", "reasons"]) == ("A;B", "r1 | r2")  # from its riskiest tx
    assert p.loc["C3", "reason_codes"] == "B" and p.loc["M1", "reasons"] == ""
    assert plain["pagerank"].isna().all() and plain["community_id"].isna().all()

    metrics = pd.DataFrame({"account": ["C1"], "total_degree": [5], "pagerank": [0.1],
                            "betweenness": [0.0], "pass_through_ratio": [0.5]})
    membership = pd.DataFrame({"account": ["C1", "C2"], "community_id": [7, 7]})
    stats = pd.DataFrame({"community_id": [7], "suspicion_score": [75.0], "flagged": [1]})
    joined = build_account_table(scored, metrics, membership, stats).set_index("account")
    assert joined.loc["C1", "pagerank"] == pytest.approx(0.1)
    assert np.isnan(joined.loc["C2", "pagerank"])
    assert (joined.loc["C2", "community_id"], joined.loc["C2", "community_suspicion_score"]) == (7, 75.0)
    assert np.isnan(joined.loc["C3", "community_id"])


# ------------------------------------------------------------------ reason codes
_RULES = [
    ReasonRule("AMOUNT_HIGH", "amount", "high", "Unusually large amount"),
    ReasonRule("PASS_HIGH", "pass", "high", "Pass-through account"),
    ReasonRule("IS_T", "is_t", "flag", "Rare type"),
    ReasonRule("IS_COMMON", "is_common", "flag", "Common flag"),
    ReasonRule("MISSING", "not_a_column", "high", "Missing feature"),
]


def _baseline_rows():
    n = 1000
    is_t = np.zeros(n)
    is_t[:100] = 1.0  # 10% prevalence
    is_common = np.zeros(n)
    is_common[::2] = 1.0  # 50% prevalence
    return pd.DataFrame({"amount": np.linspace(1, 100, n), "pass": np.zeros(n), "is_t": is_t, "is_common": is_common})


def test_baselines_skip_common_flags_and_fit_cutoffs():
    baselines = fit_baselines(_baseline_rows(), _RULES)
    assert set(baselines) == {"AMOUNT_HIGH", "PASS_HIGH", "IS_T"}  # common flag and missing feature are inactive
    assert baselines["AMOUNT_HIGH"].cutoff == pytest.approx(np.quantile(np.linspace(1, 100, 1000), 0.95))
    assert baselines["IS_T"].prevalence == pytest.approx(0.1)


def test_explain_orders_and_limits_reasons():
    baselines = fit_baselines(_baseline_rows(), _RULES)
    weights = {"amount": 1.0, "pass": 0.5, "is_t": 0.2}
    features = pd.DataFrame(
        {
            "amount": [1000.0, 50.0, 1000.0],
            "pass": [1.0, 0.0, 0.0],
            "is_t": [1.0, 0.0, 1.0],
            "is_common": [1.0, 1.0, 0.0],
        },
        index=[10, 11, 12],
    )
    out = explain_transactions(features, baselines, weights, top_n=3)
    assert list(out.index) == [10, 11, 12]
    assert out.loc[10, "reason_codes"] == "AMOUNT_HIGH;PASS_HIGH;IS_T"
    assert out.loc[11, "reason_codes"] == "" and out.loc[11, "reasons"] == ""
    assert out.loc[12, "reason_codes"] == "AMOUNT_HIGH;IS_T"
    assert "higher than 99.9% of legitimate activity" in out.loc[10, "reasons"]
    assert "seen in only 10.0% of legitimate transactions" in out.loc[10, "reasons"]

    limited = explain_transactions(features, baselines, weights, top_n=2)
    assert limited.loc[10, "reason_codes"] == "AMOUNT_HIGH;PASS_HIGH"


# ------------------------------------------------------------------ database
def test_database_roundtrip_and_queries(tmp_path):
    scored = _scored()
    membership = pd.DataFrame({"account": ["C1", "C2", "C3"], "community_id": [0, 0, 0]})
    stats = pd.DataFrame({"community_id": [0], "suspicion_score": [75.0], "flagged": [1]})
    accounts = build_account_table(scored, membership=membership, community_stats=stats)
    community = pd.DataFrame(
        [{**{c: 0 for c in COMMUNITY_DB_COLUMNS}, "community_id": 0, "n_accounts": 3, "eligible": 1,
          "suspicion_score": 75.0, "flagged": 1, "top_account_by_pagerank": "C1"}]
    )

    with closing(connect(tmp_path / "t.db")) as conn:
        create_schema(conn)
        create_schema(conn)  # safe to repeat
        assert insert_dataframe(conn, "transactions", _transactions_table(scored)) == 4
        assert insert_dataframe(conn, "accounts", accounts) == 4
        assert insert_dataframe(conn, "communities", community) == 1
        write_run_info(conn, {"model_name": "x", "threshold": 0.4})
        create_indexes(conn)

        s = queries.summary(conn)
        assert (s["n_transactions"], s["n_accounts"], s["n_communities"], s["n_alerts"]) == (4, 4, 1, 2)
        assert s["bands_all"] == {"Low": 1, "Medium": 1, "High": 1, "Critical": 1}
        assert s["bands_test"] == {"Low": 0, "Medium": 1, "High": 1, "Critical": 0}

        assert queries.top_accounts(conn, 2)["account"].tolist() == ["C1", "C2"]
        assert queries.top_accounts(conn, 10, include_merchants=False)["account"].tolist() == ["C1", "C2", "C3"]
        assert queries.top_accounts(conn, 10, min_risk=80)["account"].tolist() == ["C1", "C2"]
        assert queries.top_transactions(conn, 1)["tx_id"].tolist() == [1]
        assert queries.top_transactions(conn, 1, split="test")["tx_id"].tolist() == [3]

        detail = queries.account_detail(conn, "C3")
        assert detail["risk_band"] == "High" and detail["community_id"] == 0
        history = queries.account_transactions(conn, "C1")
        assert history["tx_id"].tolist() == [1, 3, 4]
        assert history["role"].tolist() == ["sent", "sent", "received"]

        assert queries.top_communities(conn)["community_id"].tolist() == [0]
        assert queries.community_members(conn, 0)["account"].tolist() == ["C1", "C2", "C3"]
        assert queries.run_info(conn) == {"model_name": "x", "threshold": 0.4}


def test_database_errors(tmp_path):
    with closing(connect(tmp_path / "e.db")) as conn:
        create_schema(conn)
        with pytest.raises(InvalidAccountError):
            queries.account_detail(conn, "NOPE")
        with pytest.raises(ValueError):
            insert_dataframe(conn, "transactions", pd.DataFrame({"tx_id": [1]}))
        with pytest.raises(ValueError):
            insert_dataframe(conn, "bogus", pd.DataFrame())
        with pytest.raises(ValueError):
            queries.top_accounts(conn, n=0)


# ------------------------------------------------------------------ pipeline
def test_scoring_pipeline_end_to_end(tmp_path, features_tx):
    out = tmp_path / "out"
    save_table(features_tx, out / "transactions_clean")
    run_ml_pipeline(
        tx=features_tx, max_tx=None, n_blocks=4, model_names=("logistic_regression", "random_forest"),
        run_ablation=False, betweenness_k=10, output_dir=out, models_dir=tmp_path / "models",
        report_path=tmp_path / "ml_report.json", comparison_path=tmp_path / "cmp.csv", source="synthetic",
    )
    report = run_scoring_pipeline(
        output_dir=out, models_dir=tmp_path / "models", db_path=tmp_path / "fraud.db",
        report_path=tmp_path / "risk_report.json", source="synthetic",
    )
    predictions = load_table(out / "ml_predictions")

    with closing(connect(tmp_path / "fraud.db")) as conn:
        tx = pd.read_sql_query("SELECT * FROM transactions", conn)
        acc = pd.read_sql_query("SELECT * FROM accounts", conn)
        info = queries.run_info(conn)

        assert len(tx) == len(predictions) == report["counts"]["transactions"]
        assert tx["risk_score"].between(0, 100).all()
        flagged = tx["risk_band"].isin(["High", "Critical"]).astype(int)
        assert (tx["predicted"] == flagged).all()  # High+ exactly when the model flags it

        low = tx[tx["risk_band"] == "Low"]
        assert (low["reasons"].fillna("") == "").all()
        n_codes = tx["reason_codes"].fillna("").map(lambda s: len([c for c in s.split(";") if c]))
        assert n_codes.max() <= 3

        assert (acc["risk_score"] == acc["max_risk"]).all()
        assert set(acc["riskiest_tx_id"]) <= set(tx["tx_id"])
        for account in acc["account"].head(15):  # account risk really is the max over its transactions
            expected = conn.execute(
                "SELECT MAX(risk_score) FROM transactions WHERE nameOrig = ? OR nameDest = ?", (account, account)
            ).fetchone()[0]
            assert acc.loc[acc["account"] == account, "max_risk"].iloc[0] == expected
        assert acc["community_id"].isna().all()  # no Phase 5/6 tables in this temp folder

    assert {"model_name", "threshold", "risk_bands"} <= set(info)
    assert report["synthetic_data_warning"] is True
    assert json.loads((tmp_path / "risk_report.json").read_text())["counts"]["accounts"] == len(acc)