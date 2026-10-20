"""Phase 10 tests: step selection and orchestration (the real steps are tested in their own phases)."""
import json

import pytest

import run_pipeline as rp
from config import GRAPH_MAX_TX, ML_MAX_TX, ML_N_BLOCKS
from src.utils.errors import FraudProjectError


def _args(*extra):
    return rp.build_parser().parse_args(list(extra))


def _fake_runners(calls, fail_at=None, boom_at=None):
    def make(name):
        def run(args):
            calls.append(name)
            if name == fail_at:
                raise FraudProjectError(f"{name} failed on purpose")
            if name == boom_at:
                raise RuntimeError("unexpected")
            return {"step": name}
        return run
    return {name: make(name) for name in rp.STEP_NAMES}


def _quiet(*_args, **_kwargs):
    return None


def test_select_steps_keeps_pipeline_order():
    assert rp.select_steps() == rp.STEP_NAMES
    assert rp.select_steps(only=["ml", "eda"]) == ["eda", "ml"]
    assert rp.select_steps(from_step="graph") == ["graph", "communities", "ml", "score"]
    assert "eda" not in rp.select_steps(skip=["eda"])
    assert rp.select_steps(from_step="graph", skip=["communities"]) == ["graph", "ml", "score"]


def test_select_steps_errors():
    with pytest.raises(ValueError):
        rp.select_steps(only=["bogus"])
    with pytest.raises(ValueError):
        rp.select_steps(only=["ml"], from_step="eda")
    with pytest.raises(ValueError):
        rp.select_steps(skip=list(rp.STEP_NAMES))


def test_run_all_runs_every_step_in_order_and_writes_a_report(tmp_path):
    calls = []
    report = rp.run_all(_args("--source", "synthetic"), runners=_fake_runners(calls),
                        report_path=tmp_path / "run.json", log=_quiet)
    assert calls == rp.STEP_NAMES and report["status"] == "ok"
    saved = json.loads((tmp_path / "run.json").read_text())
    assert [s["name"] for s in saved["steps"]] == rp.STEP_NAMES
    assert all(s["status"] == "ok" for s in saved["steps"])
    assert saved["args"]["source"] == "synthetic"


def test_run_all_stops_at_the_first_failure(tmp_path):
    calls = []
    report = rp.run_all(_args(), runners=_fake_runners(calls, fail_at="graph"),
                        report_path=tmp_path / "run.json", log=_quiet)
    assert calls == ["preprocess", "eda", "graph"]
    assert report["status"] == "failed"
    assert [s["status"] for s in report["steps"]] == ["ok", "ok", "failed", "not_run", "not_run", "not_run"]
    assert "on purpose" in report["steps"][2]["error"]
    assert (tmp_path / "run.json").exists()


def test_run_all_records_unexpected_errors_without_crashing(tmp_path):
    calls = []
    report = rp.run_all(_args(), runners=_fake_runners(calls, boom_at="eda"),
                        report_path=tmp_path / "run.json", log=_quiet)
    assert report["status"] == "error" and calls == ["preprocess", "eda"]
    assert report["steps"][1]["status"] == "error" and "RuntimeError" in report["steps"][1]["error"]
    assert [s["status"] for s in report["steps"][2:]] == ["not_run"] * 4


def test_dry_run_runs_nothing_and_writes_nothing(tmp_path):
    calls = []
    report = rp.run_all(_args("--dry-run", "--from", "ml"), runners=_fake_runners(calls),
                        report_path=tmp_path / "run.json", log=_quiet)
    assert calls == [] and report["status"] == "planned"
    assert [s["name"] for s in report["steps"]] == ["ml", "score"]
    assert not (tmp_path / "run.json").exists()


def test_parser_defaults_and_limits():
    args = _args()
    assert (args.source, args.engine, args.blocks) == ("auto", "auto", ML_N_BLOCKS)
    assert args.models == rp.MODEL_CHOICES and args.skip_ablation is False
    assert rp._limits(args) == (GRAPH_MAX_TX, ML_MAX_TX)
    assert rp._limits(_args("--max-tx", "100")) == (100, 100)
    assert rp._limits(_args("--graph-max-tx", "5", "--ml-max-tx", "7")) == (5, 7)
    assert rp._limits(_args("--max-tx", "0")) == (None, None)  # 0 = no limit


def test_every_step_has_a_runner():
    assert set(rp.RUNNERS) == set(rp.STEP_NAMES)