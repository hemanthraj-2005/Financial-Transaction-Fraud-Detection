"""Phase 2 tests: synthetic generator, loader, validation and error handling."""
import pandas as pd
import pytest

from config import REQUIRED_COLUMNS
from src.data.loader import describe_dataset, find_paysim_file, load_transactions
from src.data.synthetic_generator import generate_synthetic_transactions, save_synthetic
from src.utils.errors import (
    DatasetNotFoundError,
    EmptyDatasetError,
    InvalidValuesError,
    MissingColumnsError,
)
from src.utils.validation import transaction_quality_report


@pytest.fixture(scope="module")
def small_df():
    return generate_synthetic_transactions(n_normal=2000, n_customers=600, seed=7)


# ---------------------------- synthetic generator ----------------------------
def test_generator_schema_and_size(small_df):
    assert list(small_df.columns) == REQUIRED_COLUMNS
    assert len(small_df) > 2000


def test_generator_is_reproducible():
    a = generate_synthetic_transactions(n_normal=500, n_customers=400, seed=1)
    b = generate_synthetic_transactions(n_normal=500, n_customers=400, seed=1)
    c = generate_synthetic_transactions(n_normal=500, n_customers=400, seed=2)
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(c)


def test_fraud_only_in_transfer_and_cash_out(small_df):
    fraud_types = set(small_df.loc[small_df["isFraud"] == 1, "type"])
    assert fraud_types <= {"TRANSFER", "CASH_OUT"}


def test_fraud_is_rare_but_present(small_df):
    rate = small_df["isFraud"].mean()
    assert 0 < rate < 0.2


def test_no_self_transfers_or_negative_amounts(small_df):
    assert (small_df["nameOrig"] == small_df["nameDest"]).sum() == 0
    assert (small_df["amount"] > 0).all()


def test_sorted_by_step(small_df):
    assert small_df["step"].is_monotonic_increasing


def test_chain_structure_exists(small_df):
    """Layering chains: some fraud receivers must also be fraud senders."""
    fraud = small_df[small_df["isFraud"] == 1]
    assert len(set(fraud["nameDest"]) & set(fraud["nameOrig"])) > 0


def test_repeated_transfers_exist(small_df):
    fraud = small_df[(small_df["isFraud"] == 1) & (small_df["type"] == "TRANSFER")]
    pair_counts = fraud.groupby(["nameOrig", "nameDest"]).size()
    assert (pair_counts >= 3).any()


def test_generator_rejects_too_few_customers():
    with pytest.raises(ValueError):
        generate_synthetic_transactions(n_normal=100, n_customers=20)


# --------------------------------- loader ------------------------------------
def test_load_synthetic_roundtrip(tmp_path, small_df):
    path = save_synthetic(small_df, tmp_path / "s.csv", tmp_path / "meta.json")
    df = load_transactions("synthetic", path=path)
    assert len(df) == len(small_df)
    assert df.attrs["source"] == "synthetic"
    assert (tmp_path / "meta.json").exists()
    assert "SYNTHETIC" in (tmp_path / "meta.json").read_text()


def test_nrows_limits_rows(tmp_path, small_df):
    path = save_synthetic(small_df, tmp_path / "s.csv", tmp_path / "m.json")
    assert len(load_transactions("synthetic", path=path, nrows=100)) == 100


def test_missing_paysim_raises_with_instructions(tmp_path):
    with pytest.raises(DatasetNotFoundError) as exc:
        load_transactions("paysim", raw_dir=tmp_path)
    assert "kaggle" in str(exc.value).lower()


def test_auto_falls_back_to_synthetic(tmp_path):
    df = load_transactions("auto", raw_dir=tmp_path / "raw", synthetic_path=tmp_path / "syn.csv")
    assert df.attrs["source"] == "synthetic"
    assert (tmp_path / "syn.csv").exists()


def test_find_paysim_file_picks_csv(tmp_path):
    assert find_paysim_file(tmp_path) is None
    (tmp_path / "anything.csv").write_text("x\n1\n")
    assert find_paysim_file(tmp_path).name == "anything.csv"


def test_empty_file_raises(tmp_path):
    bad = tmp_path / "empty.csv"
    bad.write_text("")
    with pytest.raises(EmptyDatasetError):
        load_transactions("paysim", path=bad)


def test_header_only_file_raises(tmp_path):
    bad = tmp_path / "header.csv"
    bad.write_text(",".join(REQUIRED_COLUMNS) + "\n")
    with pytest.raises(EmptyDatasetError):
        load_transactions("paysim", path=bad)


def test_missing_columns_raises(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("step,type,amount\n1,PAYMENT,10\n")
    with pytest.raises(MissingColumnsError) as exc:
        load_transactions("paysim", path=bad)
    assert "nameOrig" in str(exc.value)


def _row(**overrides):
    row = dict(step=1, type="PAYMENT", amount=100.0, nameOrig="C1", oldbalanceOrg=500.0,
               newbalanceOrig=400.0, nameDest="M1", oldbalanceDest=0.0, newbalanceDest=0.0,
               isFraud=0, isFlaggedFraud=0)
    row.update(overrides)
    return row


def test_invalid_values_reported_and_strict_raises(tmp_path):
    df = pd.DataFrame([_row(), _row(amount=-5.0), _row(type="BOGUS"), _row(isFraud=3)])
    path = tmp_path / "invalid.csv"
    df.to_csv(path, index=False)

    loaded = load_transactions("paysim", path=path)  # non-strict: loads and reports
    report = loaded.attrs["quality_report"]
    assert report["negative_amounts"] == 1
    assert report["unknown_types"] == 1
    assert report["invalid_labels"] == 1

    with pytest.raises(InvalidValuesError):
        load_transactions("paysim", path=path, strict=True)


def test_non_numeric_amount_becomes_nan(tmp_path):
    df = pd.DataFrame([_row(), _row(amount="abc")])
    path = tmp_path / "nan.csv"
    df.to_csv(path, index=False)
    loaded = load_transactions("paysim", path=path)
    assert loaded["amount"].isna().sum() == 1
    assert loaded.attrs["quality_report"]["null_counts"]["amount"] == 1


def test_quality_report_class_balance(small_df):
    report = transaction_quality_report(small_df)
    assert report["n_fraud"] + report["n_legitimate"] == report["n_rows"]
    assert report["imbalance_ratio"] > 1


def test_describe_dataset_labels_synthetic(small_df):
    small_df.attrs["source"] = "synthetic"
    assert "SYNTHETIC" in describe_dataset(small_df)


def test_invalid_source_name_rejected():
    with pytest.raises(ValueError):
        load_transactions("nonsense")