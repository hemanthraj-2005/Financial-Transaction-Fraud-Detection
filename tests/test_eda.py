"""Phase 4 tests: EDA figures and the runner."""
import json

import pytest

pytest.importorskip("plotly")

import plotly.graph_objects as go  # noqa: E402

from src.analytics.eda import FIGURES, fig_top_accounts, run_eda, summary_numbers  # noqa: E402
from src.data.synthetic_generator import generate_synthetic_transactions  # noqa: E402
from src.features.account_aggregates import build_account_aggregates  # noqa: E402
from src.features.transaction_features import add_transaction_features  # noqa: E402
from src.preprocessing.cleaning import clean_transactions  # noqa: E402
from src.utils.errors import MissingColumnsError  # noqa: E402


@pytest.fixture(scope="module")
def data():
    raw = generate_synthetic_transactions(n_normal=2000, n_customers=600, seed=5)
    tx = add_transaction_features(clean_transactions(raw)[0])
    return tx, build_account_aggregates(tx)


def test_every_figure_builds_with_data(data):
    tx, acc = data
    for name, build in FIGURES.items():
        fig = build(tx)
        assert isinstance(fig, go.Figure), name
        assert len(fig.data) > 0, name
    assert len(fig_top_accounts(acc).data) >= 1


def test_figures_survive_data_without_fraud(data):
    tx, acc = data
    legit_tx = tx[tx["isFraud"] == 0]
    legit_acc = build_account_aggregates(legit_tx)
    for build in FIGURES.values():
        assert isinstance(build(legit_tx), go.Figure)
    assert isinstance(fig_top_accounts(legit_acc), go.Figure)


def test_summary_numbers(data):
    tx, acc = data
    s = summary_numbers(tx, acc)
    assert s["n_transactions"] == len(tx)
    assert s["n_fraud"] == int(tx["isFraud"].sum())
    assert s["n_accounts"] == len(acc)
    leak = s["leakage_check_pct_balance_emptied"]
    assert leak["fraud"] > leak["legit"]  # synthetic fraud drains balances, like PaySim


def test_run_eda_writes_files(tmp_path, data):
    tx, acc = data
    summary = run_eda(tmp_path / "eda", tx=tx, acc=acc)
    assert len(summary["charts"]) == len(FIGURES) + 1
    for path in summary["charts"].values():
        assert (tmp_path / "eda" / path.split("/")[-1].split("\\")[-1]).exists()
    saved = json.loads((tmp_path / "eda" / "eda_summary.json").read_text())
    assert saved["n_fraud"] == summary["n_fraud"]


def test_run_eda_rejects_missing_columns(tmp_path, data):
    tx, acc = data
    with pytest.raises(MissingColumnsError):
        run_eda(tmp_path, tx=tx.drop(columns=["log_amount"]), acc=acc)