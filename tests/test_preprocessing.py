"""Phase 3 tests: cleaning, features (incl. leakage safety), aggregates, scaling, pipeline.

The Spark parity test at the bottom is skipped automatically when pyspark or
Java is not available.
"""
import json

import numpy as np
import pandas as pd
import pytest

from config import BASE_TIMESTAMP, REQUIRED_COLUMNS, TYPE_CODES, VALID_TRANSACTION_TYPES
from src.analytics.fraud_stats import fraud_stats
from src.data.loader import load_transactions
from src.data.synthetic_generator import generate_synthetic_transactions
from src.features.account_aggregates import build_account_aggregates
from src.features.transaction_feature import add_transaction_features
from src.preprocessing.cleaning import RULE_NAMES, clean_transactions
from src.preprocessing.diagnostics import class_balance_report, inspect_columns
from src.preprocessing.pipeline import resolve_engine, resolve_input, run_preprocessing
from src.preprocessing.scaling import FeatureScaler
from src.preprocessing.schema import ACCOUNT_COLUMNS, HISTORY_FEATURES, OUTPUT_COLUMNS
from src.utils.errors import (
    DatasetNotFoundError,
    EmptyDatasetError,
    FraudProjectError,
    MissingColumnsError,
)
from src.utils.io_utils import load_table, save_table


# ------------------------------------------------------------------ helpers
def _row(**overrides):
    row = dict(step=1, type="PAYMENT", amount=100.0, nameOrig="C1", oldbalanceOrg=500.0,
               newbalanceOrig=400.0, nameDest="M1", oldbalanceDest=0.0, newbalanceDest=0.0,
               isFraud=0, isFlaggedFraud=0)
    row.update(overrides)
    return row


def _frame(rows):
    return pd.DataFrame(rows)[REQUIRED_COLUMNS]


@pytest.fixture(scope="module")
def synth():
    return generate_synthetic_transactions(n_normal=3000, n_customers=700, seed=11)


def _write_dirty_csv(path, base_df):
    """Synthetic rows + 3 invalid rows + 1 exact duplicate (used by e2e and Spark tests)."""
    bad = pd.DataFrame(
        [
            _row(step=5, amount=-10.0, nameOrig="CBAD1", nameDest="CBAD2"),          # negative amount
            _row(step=5, type="BOGUS", nameOrig="CBAD3", nameDest="CBAD4"),          # unknown type
            _row(step=5, nameOrig="CBAD5", nameDest="CBAD5"),                        # self transfer
        ]
    )[REQUIRED_COLUMNS]
    out = pd.concat([base_df, bad, base_df.iloc[[0]]], ignore_index=True)           # + duplicate of row 0
    out.to_csv(path, index=False)
    return path


# ------------------------------------------------------------------ cleaning
def test_each_cleaning_rule_fires_once():
    rows = [
        _row(),                                  # valid
        _row(amount=-5.0),                       # negative_amount
        _row(step=0),                            # invalid_step
        _row(type="BOGUS"),                      # unknown_type
        _row(isFraud=2),                         # invalid_label
        _row(isFlaggedFraud=5),                  # invalid_flag_column
        _row(newbalanceDest=-1.0),               # negative_balance
        _row(nameOrig="C9", nameDest="C9"),      # self_transfer
        _row(nameOrig="   "),                    # blank_account
        _row(amount=np.nan),                     # null_required
    ]
    clean, report = clean_transactions(_frame(rows))
    assert set(report["rule_violations"]) == set(RULE_NAMES)
    assert all(count == 1 for count in report["rule_violations"].values()), report["rule_violations"]
    assert report["n_invalid_removed"] == 9
    assert report["n_clean"] == 1 and len(clean) == 1


def test_duplicates_removed_keep_first():
    clean, report = clean_transactions(_frame([_row(), _row(), _row(), _row(amount=200.0)]))
    assert report["duplicates_removed"] == 2
    assert report["n_clean"] == 2
    assert clean["tx_id"].tolist() == [0, 3]


def test_zero_amount_rows_are_kept_and_counted():
    clean, report = clean_transactions(_frame([_row(amount=0.0), _row(amount=5.0)]))
    assert len(clean) == 2
    assert report["zero_amount_rows"] == 1


def test_tx_id_is_position_in_input():
    clean, _ = clean_transactions(_frame([_row(), _row(amount=-1.0), _row(amount=7.0)]))
    assert clean["tx_id"].tolist() == [0, 2]


def test_cleaning_is_idempotent_and_does_not_mutate_input():
    df = _frame([_row(), _row(amount=-1.0), _row(amount=7.0)])
    before = df.copy()
    once, _ = clean_transactions(df)
    twice, report = clean_transactions(once)
    pd.testing.assert_frame_equal(df, before)
    pd.testing.assert_frame_equal(once, twice)
    assert report["n_invalid_removed"] == 0 and report["duplicates_removed"] == 0


def test_cleaning_rejects_empty_and_missing_columns():
    with pytest.raises(EmptyDatasetError):
        clean_transactions(pd.DataFrame(columns=REQUIRED_COLUMNS))
    with pytest.raises(MissingColumnsError):
        clean_transactions(pd.DataFrame({"step": [1], "amount": [1.0]}))


def test_cleaning_coerces_text_numbers():
    df = _frame([_row(), _row()])
    df["amount"] = ["100", "oops"]
    clean, report = clean_transactions(df)
    assert report["rule_violations"]["null_required"] == 1
    assert len(clean) == 1


# ------------------------------------------------------------------ features
def test_history_features_exact_values_including_ties():
    rows = [
        _row(step=1, amount=100.0, nameOrig="C1", nameDest="M1"),
        _row(step=3, amount=200.0, nameOrig="C1", nameDest="M1"),
        _row(step=8, amount=300.0, nameOrig="C1", nameDest="M2"),
        _row(step=8, amount=10.0, nameOrig="C1", nameDest="M3"),   # same step, later tx_id
    ]
    feats = add_transaction_features(clean_transactions(_frame(rows))[0])

    c1 = feats[feats["nameOrig"] == "C1"].sort_values("tx_id")
    assert c1["orig_prev_tx_count"].tolist() == [0, 1, 2, 3]
    assert c1["orig_prev_amount_sum"].tolist() == [0, 100, 300, 600]
    assert c1["orig_prev_amount_mean"].tolist() == [0, 100, 150, 200]
    assert c1["orig_steps_since_last_tx"].tolist() == [-1, 2, 5, 0]

    m1 = feats[feats["nameDest"] == "M1"].sort_values("tx_id")
    assert m1["dest_prev_tx_count"].tolist() == [0, 1]
    assert m1["dest_prev_amount_sum"].tolist() == [0, 100]
    assert m1["dest_steps_since_last_tx"].tolist() == [-1, 2]


def test_time_features_and_timestamp():
    rows = [_row(step=s, nameDest=f"M{s}") for s in (1, 25, 30, 31)]
    feats = add_transaction_features(clean_transactions(_frame(rows))[0]).sort_values("step")
    assert feats["hour_of_day"].tolist() == [0, 0, 5, 6]
    assert feats["day"].tolist() == [0, 1, 1, 1]
    assert feats["is_night"].tolist() == [1, 1, 1, 0]
    assert feats["timestamp"].iloc[0] == pd.Timestamp(BASE_TIMESTAMP)
    assert feats["timestamp"].iloc[1] == pd.Timestamp(BASE_TIMESTAMP) + pd.Timedelta(hours=24)


def test_categorical_encoding_is_fixed_and_one_hot():
    rows = [_row(step=i + 1, type=t, nameDest=f"C{i}") for i, t in enumerate(VALID_TRANSACTION_TYPES)]
    feats = add_transaction_features(clean_transactions(_frame(rows))[0])
    for _, r in feats.iterrows():
        assert r["type_code"] == TYPE_CODES[r["type"]]
        assert r[f"type_{r['type']}"] == 1
    onehot = [f"type_{t}" for t in VALID_TRANSACTION_TYPES]
    assert (feats[onehot].sum(axis=1) == 1).all()


def test_amount_merchant_and_balance_features():
    rows = [
        _row(step=1, amount=100.0, oldbalanceOrg=500.0, newbalanceOrig=400.0, nameDest="M1"),
        _row(step=2, type="TRANSFER", amount=500.0, oldbalanceOrg=500.0, newbalanceOrig=0.0,
             nameDest="C2", oldbalanceDest=0.0, newbalanceDest=500.0),
        _row(step=3, amount=50.0, oldbalanceOrg=0.0, newbalanceOrig=0.0, nameDest="C3"),
    ]
    f = add_transaction_features(clean_transactions(_frame(rows))[0]).sort_values("step").reset_index(drop=True)
    assert np.isclose(f.loc[0, "log_amount"], np.log1p(100.0))
    assert f["dest_is_merchant"].tolist() == [1, 0, 0]
    assert f["balance_error_orig"].tolist()[:2] == [0.0, 0.0]
    assert f["balance_error_dest"].tolist()[1] == 0.0
    assert f["orig_balance_emptied"].tolist() == [0, 1, 0]
    assert np.allclose(f["amount_to_oldbalance_ratio"].tolist(), [0.2, 1.0, 0.0])


def test_feature_output_schema_and_no_nulls(synth):
    feats = add_transaction_features(clean_transactions(synth)[0])
    assert list(feats.columns) == OUTPUT_COLUMNS
    assert int(feats.isna().sum().sum()) == 0
    assert feats["step"].is_monotonic_increasing


def test_features_independent_of_input_row_order(synth):
    clean, _ = clean_transactions(synth)
    shuffled = clean.sample(frac=1.0, random_state=3)
    pd.testing.assert_frame_equal(add_transaction_features(clean), add_transaction_features(shuffled))


def test_features_never_use_the_fraud_label(synth):
    clean, _ = clean_transactions(synth)
    flipped = clean.copy()
    flipped["isFraud"] = 1 - flipped["isFraud"]
    cols = [c for c in OUTPUT_COLUMNS if c != "isFraud"]
    pd.testing.assert_frame_equal(add_transaction_features(clean)[cols], add_transaction_features(flipped)[cols])


def test_history_features_never_use_the_future(synth):
    clean, _ = clean_transactions(synth)
    altered = clean.copy()
    late = altered["step"] > 400
    altered.loc[late, "amount"] = altered.loc[late, "amount"] * 10
    a, b = add_transaction_features(clean), add_transaction_features(altered)
    early = a["step"] <= 400
    assert early.any() and (~early).any()
    pd.testing.assert_frame_equal(a.loc[early, HISTORY_FEATURES], b.loc[early, HISTORY_FEATURES])


# ---------------------------------------------------- aggregates and statistics
def test_account_aggregates_exact_values():
    tx = pd.DataFrame(
        {
            "nameOrig": ["A", "A", "B", "C", "C"],
            "nameDest": ["B", "B", "C", "A", "M1"],
            "amount": [100.0, 50.0, 20.0, 10.0, 5.0],
            "step": [1, 2, 3, 4, 5],
            "isFraud": [1, 0, 0, 0, 0],
        }
    )
    acc = build_account_aggregates(tx)
    assert list(acc.columns) == ACCOUNT_COLUMNS
    acc = acc.set_index("account")
    assert len(acc) == 4

    a = acc.loc["A"]
    assert (a.out_tx_count, a.out_amount_sum, a.out_unique_counterparties) == (2, 150.0, 1)
    assert (a.in_tx_count, a.in_amount_sum, a.n_counterparties) == (1, 10.0, 2)
    assert (a.label_fraud_sent, a.first_step, a.last_step, a.total_tx_count) == (1, 1, 4, 3)
    assert np.isclose(a.avg_tx_amount, 160.0 / 3)

    b = acc.loc["B"]
    assert (b.in_tx_count, b.in_unique_counterparties, b.out_tx_count, b.label_fraud_received) == (2, 1, 1, 1)

    assert acc.loc["C"].n_counterparties == 3  # B (in), A and M1 (out)

    m = acc.loc["M1"]
    assert (m.is_merchant, m.in_tx_count, m.out_tx_count, m.out_amount_avg, m.first_step) == (1, 1, 0, 0.0, 5)
    assert int(acc.isna().sum().sum()) == 0


def test_fraud_stats_by_type_and_day():
    rows = [
        _row(type="PAYMENT", step=1, amount=10.0, nameDest="M1"),
        _row(type="TRANSFER", step=2, amount=100.0, isFraud=1, nameDest="C5"),
        _row(type="TRANSFER", step=30, amount=300.0, nameDest="C6"),
        _row(type="PAYMENT", step=30, amount=5.0, nameDest="M2"),
    ]
    feats = add_transaction_features(clean_transactions(_frame(rows))[0])
    by_type, by_day = fraud_stats(feats)
    transfer = by_type.set_index("type").loc["TRANSFER"]
    assert (transfer.n_transactions, transfer.n_fraud, transfer.fraud_rate_pct) == (2, 1, 50.0)
    assert (transfer.total_amount, transfer.fraud_amount) == (400.0, 100.0)
    assert by_type.set_index("type").loc["PAYMENT"].n_fraud == 0
    day0 = by_day.set_index("day").loc[0]
    assert (day0.n_transactions, day0.n_fraud) == (2, 1)
    assert by_day.set_index("day").loc[1].n_fraud == 0


# ------------------------------------------------------------------ diagnostics
def test_class_balance_report():
    severe = class_balance_report(5, 995)
    assert severe["fraud_rate_pct"] == 0.5
    assert "severe" in severe["severity"]
    assert severe["imbalance_ratio_legit_per_fraud"] == 199.0
    assert severe["suggested_scale_pos_weight"] == 199.0
    assert any("accuracy" in text.lower() for text in severe["recommendations"])
    assert "no fraud" in class_balance_report(0, 100)["severity"]
    assert "moderate" in class_balance_report(50, 950)["severity"]


def test_inspect_columns():
    df = pd.DataFrame({"a": [1, 2, None, 2], "b": ["x", "y", "y", "y"]})
    info = inspect_columns(df).set_index("column")
    assert info.loc["a", "null_count"] == 1
    assert info.loc["a", "n_unique"] == 2
    assert info.loc["b", "n_unique"] == 2
    assert list(info.columns) == ["dtype", "null_count", "null_pct", "n_unique", "examples"]


# ------------------------------------------------------------------ scaling
def test_scaler_fits_on_train_only_and_roundtrips(tmp_path):
    train = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [5.0, 5.0, 5.0], "keep": [9, 9, 9]})
    test = pd.DataFrame({"a": [2.0, 10.0], "b": [5.0, 5.0], "keep": [1, 1]})
    scaler = FeatureScaler().fit(train, ["a", "b"])
    mean_before = dict(scaler.mean_)

    scaled = scaler.transform(test)
    assert scaler.mean_ == mean_before                       # transforming never changes learned stats
    assert np.isclose(scaled["a"].iloc[0], 0.0)
    assert np.isclose(scaled["a"].iloc[1], (10 - 2) / np.sqrt(2 / 3))
    assert (scaled["b"] == 0.0).all()                        # constant column: std treated as 1
    assert scaled["keep"].tolist() == [1, 1]                 # other columns untouched
    assert test["a"].tolist() == [2.0, 10.0]                 # input not modified

    path = scaler.save(tmp_path / "scaler.json")
    restored = FeatureScaler.load(path)
    pd.testing.assert_frame_equal(restored.transform(test), scaled)


def test_scaler_errors(tmp_path):
    with pytest.raises(FraudProjectError):
        FeatureScaler(["a"]).transform(pd.DataFrame({"a": [1.0]}))
    with pytest.raises(MissingColumnsError):
        FeatureScaler().fit(pd.DataFrame({"a": [1.0]}), ["zzz"])
    with pytest.raises(FraudProjectError):
        FeatureScaler.load(tmp_path / "missing.json")


# ------------------------------------------------------------------ I/O and pipeline
def test_save_and_load_table_roundtrip(tmp_path, synth):
    feats = add_transaction_features(clean_transactions(synth)[0]).head(200)
    written = save_table(feats, tmp_path / "t")
    assert written.suffix in {".parquet", ".csv"}
    loaded = load_table(tmp_path / "t")
    pd.testing.assert_frame_equal(loaded, feats, check_dtype=False)


def test_load_table_missing_gives_helpful_error(tmp_path):
    with pytest.raises(DatasetNotFoundError) as exc:
        load_table(tmp_path / "nothing")
    assert "pipeline" in str(exc.value)


def test_resolve_input(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        resolve_input("paysim", raw_dir=tmp_path / "raw", synthetic_path=tmp_path / "s.csv")
    path, label = resolve_input("auto", raw_dir=tmp_path / "raw", synthetic_path=tmp_path / "s.csv")
    assert label == "synthetic" and path.exists()
    explicit = tmp_path / "x.csv"
    assert resolve_input("paysim", path=explicit)[0] == explicit
    with pytest.raises(ValueError):
        resolve_input("nonsense")


def test_resolve_engine(tmp_path):
    small = tmp_path / "small.csv"
    small.write_text("a\n1\n")
    assert resolve_engine("auto", small) == "pandas"
    assert resolve_engine("spark", small) == "spark"
    assert resolve_engine("pandas", small) == "pandas"
    with pytest.raises(ValueError):
        resolve_engine("hadoop", small)


def test_pipeline_end_to_end_pandas_with_dirty_data(tmp_path, synth):
    csv = _write_dirty_csv(tmp_path / "dirty.csv", synth)
    report = run_preprocessing(source="paysim", path=csv, engine="pandas",
                               output_dir=tmp_path / "out", report_path=tmp_path / "report.json")

    cleaning = report["cleaning"]
    assert cleaning["n_input"] == len(synth) + 4
    assert cleaning["n_invalid_removed"] == 3
    assert cleaning["duplicates_removed"] == 1
    assert cleaning["n_clean"] == len(synth)

    tx = load_table(tmp_path / "out" / "transactions_clean")
    accounts = load_table(tmp_path / "out" / "account_aggregates")
    assert list(tx.columns) == OUTPUT_COLUMNS and len(tx) == len(synth)
    assert list(accounts.columns) == ACCOUNT_COLUMNS and len(accounts) == report["n_accounts"]
    assert (tmp_path / "out" / "fraud_stats_by_type.csv").exists()
    assert (tmp_path / "out" / "fraud_stats_by_day.csv").exists()

    saved = json.loads((tmp_path / "report.json").read_text())
    assert saved["engine"] == "pandas"
    assert saved["class_balance"]["n_fraud"] == int(synth["isFraud"].sum())
    assert "column_inspection" in saved and "feature_groups" in saved


def test_pipeline_flags_synthetic_source_and_respects_nrows(tmp_path):
    report = run_preprocessing(source="synthetic", nrows=500, engine="pandas",
                               output_dir=tmp_path / "out", report_path=tmp_path / "r.json")
    assert report["synthetic_data_warning"] is True
    assert report["cleaning"]["n_input"] == 500


# ------------------------------------------------------------------ Spark parity
def test_spark_engine_matches_pandas_engine(tmp_path, synth):
    pytest.importorskip("pyspark")
    from src.preprocessing.spark_pipeline import (
        account_aggregates_spark,
        add_features_spark,
        clean_spark,
        finalize_cleaning_report,
        fraud_stats_spark,
        get_spark,
        read_transactions_spark,
    )
    from src.utils.errors import MissingDependencyError

    try:
        spark = get_spark("fraud-tests", master="local[2]", driver_memory="1g", shuffle_partitions=2)
    except MissingDependencyError as exc:
        pytest.skip(str(exc))

    try:
        csv = _write_dirty_csv(tmp_path / "dirty.csv", synth)

        # ---- Pandas reference
        raw = load_transactions("paysim", path=csv)
        p_clean, p_report = clean_transactions(raw)
        p_feat = add_transaction_features(p_clean)
        p_acc = build_account_aggregates(p_feat)
        p_type, p_day = fraud_stats(p_feat)

        # ---- Spark
        s_clean, s_partial = clean_spark(read_transactions_spark(spark, csv))
        s_feat = add_features_spark(s_clean)
        s_report = finalize_cleaning_report(s_partial, s_feat.count())
        s_feat_pdf = s_feat.toPandas().sort_values("tx_id").reset_index(drop=True)
        s_acc_pdf = account_aggregates_spark(s_feat).toPandas().sort_values("account").reset_index(drop=True)
        s_type, s_day = fraud_stats_spark(s_feat)

        assert s_report == p_report

        assert list(s_feat_pdf.columns) == OUTPUT_COLUMNS
        assert s_feat_pdf["tx_id"].tolist() == p_feat.sort_values("tx_id")["tx_id"].tolist()
        p_sorted = p_feat.sort_values("tx_id").reset_index(drop=True)
        for column in OUTPUT_COLUMNS:
            if column in ("type", "nameOrig", "nameDest"):
                assert s_feat_pdf[column].tolist() == p_sorted[column].tolist(), column
            elif column != "timestamp":
                assert np.allclose(s_feat_pdf[column].astype(float), p_sorted[column].astype(float),
                                   rtol=1e-7, atol=1e-6), column

        assert s_acc_pdf["account"].tolist() == p_acc["account"].tolist()
        for column in ACCOUNT_COLUMNS[1:]:
            assert np.allclose(s_acc_pdf[column].astype(float), p_acc[column].astype(float),
                               rtol=1e-7, atol=1e-6), column

        for s_stats, p_stats, key in ((s_type, p_type, "type"), (s_day, p_day, "day")):
            assert s_stats[key].tolist() == p_stats[key].tolist()
            for column in ("n_transactions", "n_fraud", "fraud_rate_pct", "total_amount", "fraud_amount"):
                assert np.allclose(s_stats[column].astype(float), p_stats[column].astype(float),
                                   rtol=1e-7, atol=1e-6), column
    finally:
        spark.stop()