"""Phase 7 tests: time split, leakage safety of graph features, feature sets, models, pipeline."""
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src.data.synthetic_generator import generate_synthetic_transactions
from src.features.feature_sets import (
    BALANCE_FEATURES,
    FEATURE_SETS,
    GRAPH_FEATURES,
    NEVER_FEATURES,
    PRIMARY_FEATURE_SET,
)
from src.features.graph_features import GRAPH_FEATURE_COLUMNS, GRAPH_INPUT_COLUMNS, compute_graph_features
from src.features.time_split import assign_blocks, assign_split, split_boundaries
from src.features.transaction_features import add_transaction_features
from src.models.pipeline import run_ml_pipeline
from src.models.train import build_model, evaluate, to_matrix, tune_threshold
from src.preprocessing.cleaning import clean_transactions
from src.utils.errors import FraudProjectError
from src.utils.io_utils import load_table

N_BLOCKS = 5
FIT_END = 400


@pytest.fixture(scope="module")
def features_tx():
    raw = generate_synthetic_transactions(n_normal=3000, n_customers=700, seed=5)
    return add_transaction_features(clean_transactions(raw)[0])


@pytest.fixture(scope="module")
def graph_inputs(features_tx):
    return features_tx[GRAPH_INPUT_COLUMNS].copy()


@pytest.fixture(scope="module")
def base_features(graph_inputs):
    return compute_graph_features(graph_inputs, n_blocks=N_BLOCKS, fit_end=FIT_END, betweenness_k=10)


def _blocks(df):
    steps = df["step"]
    return assign_blocks(steps, int(steps.min()), int(steps.max()) - int(steps.min()) + 1, N_BLOCKS)


# ------------------------------------------------------------------ split and blocks
def test_split_boundaries_and_assignment():
    steps = pd.Series(range(1, 744))
    bounds = split_boundaries(steps, 0.56, 0.14)
    assert (bounds["fit_end"], bounds["val_end"]) == (416, 520)
    split = assign_split(steps, bounds)
    assert split.iloc[0] == "fit" and split.iloc[415] == "fit"
    assert split.iloc[416] == "val" and split.iloc[519] == "val"
    assert split.iloc[520] == "test"
    assert split.value_counts().to_dict() == {"fit": 416, "val": 104, "test": 223}


def test_split_validation_errors():
    with pytest.raises(ValueError):
        split_boundaries(pd.Series(range(1, 100)), 0.8, 0.3)
    with pytest.raises(FraudProjectError):
        split_boundaries(pd.Series(range(1, 5)), 0.5, 0.2)


def test_blocks_cover_the_range_in_order():
    blocks = assign_blocks(pd.Series(np.arange(1, 101)), 1, 100, 4)
    assert blocks.iloc[0] == 0 and blocks.iloc[-1] == 3
    assert blocks.is_monotonic_increasing
    assert blocks.value_counts().sort_index().tolist() == [25, 25, 25, 25]


# ------------------------------------------------------------------ leakage safety
def test_graph_feature_schema_and_warmup(graph_inputs, base_features):
    assert list(base_features.columns) == ["tx_id", "block", *GRAPH_FEATURE_COLUMNS]
    assert len(base_features) == len(graph_inputs) and base_features["tx_id"].is_unique
    warmup = base_features[base_features["block"] == 0]
    assert len(warmup) > 0 and (warmup[GRAPH_FEATURE_COLUMNS] == 0).all().all()
    later = base_features[base_features["block"] >= 1]
    assert (later["graph_history_available"] == 1).all()
    assert np.isfinite(base_features[GRAPH_FEATURE_COLUMNS].to_numpy()).all()


def test_features_never_use_the_future(graph_inputs, base_features):
    altered = graph_inputs.copy()
    late = _blocks(altered) >= 3

    # Change only future transactions.
    # Do not change nameOrig/nameDest because those identify the
    # transaction for which the graph features are being generated.
    altered.loc[late, "amount"] = altered.loc[late, "amount"] * 10
    altered.loc[late, "isFraud"] = 1 - altered.loc[late, "isFraud"]

    other = compute_graph_features(
        altered,
        n_blocks=N_BLOCKS,
        fit_end=FIT_END,
        betweenness_k=10,
    )

    # Blocks 0-3 use history from blocks before them.
    # Changing blocks 3+ must not affect their historical graph features.
    pd.testing.assert_frame_equal(
        base_features[base_features["block"] <= 3].reset_index(drop=True),
        other[other["block"] <= 3].reset_index(drop=True),
    )


def test_labels_after_the_fit_window_are_never_used(graph_inputs, base_features):
    altered = graph_inputs.copy()
    after = altered["step"] > FIT_END
    altered.loc[after, "isFraud"] = 1 - altered.loc[after, "isFraud"]
    other = compute_graph_features(altered, n_blocks=N_BLOCKS, fit_end=FIT_END, betweenness_k=10)
    pd.testing.assert_frame_equal(base_features, other)


def test_a_rows_own_block_labels_never_reach_its_features(graph_inputs, base_features):
    altered = graph_inputs.copy()
    late = _blocks(altered) >= 2
    altered.loc[late, "isFraud"] = 1 - altered.loc[late, "isFraud"]
    other = compute_graph_features(altered, n_blocks=N_BLOCKS, fit_end=FIT_END, betweenness_k=10)

    # block <= 2 rows only see labels of blocks < 2: unchanged
    pd.testing.assert_frame_equal(
        base_features[base_features["block"] <= 2].reset_index(drop=True),
        other[other["block"] <= 2].reset_index(drop=True),
    )
    # later rows DO use earlier labels, so the label-based feature must change
    rate_cols = ["s_comm_fraud_rate", "r_comm_fraud_rate"]
    changed = base_features.loc[base_features["block"] >= 3, rate_cols].to_numpy() != other.loc[
        other["block"] >= 3, rate_cols
    ].to_numpy()
    assert changed.any()


def test_feature_sets_are_clean():
    for name, cols in FEATURE_SETS.items():
        assert len(cols) == len(set(cols)), name
        assert not set(cols) & set(NEVER_FEATURES), name
        assert not [c for c in cols if c.startswith("label_")], name
    assert not set(FEATURE_SETS["tx_only"]) & set(GRAPH_FEATURES)
    assert not set(FEATURE_SETS[PRIMARY_FEATURE_SET]) & set(BALANCE_FEATURES)
    assert set(BALANCE_FEATURES) <= set(FEATURE_SETS["tx_graph_balance"])


# ------------------------------------------------------------------ training helpers
def test_tune_threshold_and_evaluate_exact_values():
    y = np.array([0, 0, 0, 1, 1])
    s = np.array([0.1, 0.4, 0.35, 0.8, 0.6])
    threshold, f1 = tune_threshold(y, s)
    assert threshold == pytest.approx(0.6) and f1 == pytest.approx(1.0)
    m = evaluate(y, s, threshold)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 0, 0, 3)
    assert m["precision"] == 1.0 and m["recall"] == 1.0 and m["f1"] == 1.0
    assert m["pr_auc"] == pytest.approx(1.0) and m["roc_auc"] == pytest.approx(1.0)


def test_evaluate_without_positives_and_threshold_error():
    m = evaluate(np.zeros(5, dtype=int), np.linspace(0, 1, 5), 0.5)
    assert m["pr_auc"] is None and m["roc_auc"] is None and m["recall"] == 0.0
    with pytest.raises(FraudProjectError):
        tune_threshold(np.zeros(5, dtype=int), np.linspace(0, 1, 5))


def _smoke(name):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 4))
    y = (X[:, 0] > 1.0).astype(int)
    model = build_model(name, scale_pos_weight=5.0, seed=1)
    model.fit(X, y)
    p = model.predict_proba(X)[:, 1]
    assert p.shape == (400,) and ((p >= 0) & (p <= 1)).all()
    assert p[y == 1].mean() > p[y == 0].mean()


@pytest.mark.parametrize("name", ["logistic_regression", "random_forest"])
def test_models_fit_and_score(name):
    _smoke(name)


def test_xgboost_fits_and_scores():
    pytest.importorskip("xgboost")
    _smoke("xgboost")


def test_unknown_model_and_matrix_cleaning():
    with pytest.raises(ValueError):
        build_model("bogus")
    df = pd.DataFrame({"a": [1.0, np.inf, np.nan], "b": [2, 3, 4]})
    assert to_matrix(df, ["a", "b"]).tolist() == [[1.0, 2.0], [0.0, 3.0], [0.0, 4.0]]


# ------------------------------------------------------------------ pipeline
def test_pipeline_end_to_end(tmp_path, features_tx):
    report = run_ml_pipeline(
        tx=features_tx, max_tx=None, n_blocks=4,
        model_names=("logistic_regression", "random_forest"), run_ablation=True, betweenness_k=10,
        output_dir=tmp_path / "out", models_dir=tmp_path / "models",
        report_path=tmp_path / "ml_report.json", comparison_path=tmp_path / "comparison.csv",
        source="synthetic",
    )
    split = report["split"]
    assert split["fit_end"] < split["val_end"] < split["max_step"]
    assert report["synthetic_data_warning"] is True

    comparison = pd.read_csv(tmp_path / "comparison.csv")
    assert set(comparison["feature_set"]) == set(FEATURE_SETS)
    assert set(comparison["model"]) <= {"logistic_regression", "random_forest"}

    predictions = load_table(tmp_path / "out" / "ml_predictions")
    assert set(predictions["split"]) == {"fit", "val", "test"}
    assert predictions["score"].between(0, 1).all()
    assert (predictions.loc[predictions["split"] == "fit", "step"] <= split["fit_end"]).all()
    assert (predictions.loc[predictions["split"] == "test", "step"] > split["val_end"]).all()
    assert (predictions["block"] >= 1).all()  # warm-up block dropped

    saved = joblib.load(tmp_path / "models" / "best_model.joblib")
    assert saved["feature_columns"] == FEATURE_SETS[PRIMARY_FEATURE_SET]
    assert "isFlaggedFraud" not in saved["feature_columns"]
    metadata = json.loads((tmp_path / "models" / "model_metadata.json").read_text())
    assert metadata["model_name"] == report["selected_model"]["name"]
    assert (tmp_path / "ml_report.json").exists()