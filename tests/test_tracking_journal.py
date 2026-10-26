"""MLflow safety net: journal, offline stages, local backup store and later sync."""

import json

import pytest

from tir_jde import tracking


def test_journal_records_calls_and_the_first_missed_one(tmp_path):
    j = tracking.Journal(tmp_path)
    assert j.append("log_metrics", [{"a": 1.0}], {"step": 2}) == 0
    assert j.append("set_tags", [{"t": "x"}], {}) == 1
    j.mark_missed(1)
    j.mark_missed(5)  # a later miss does not move the resend point
    assert [e["op"] for e in j.entries()] == ["log_metrics", "set_tags"]
    assert j.entries(1)[0]["args"] == [{"t": "x"}]
    assert j.state()["primary"]["missed_from"] == 1


def test_numpy_numbers_stay_numbers_in_the_journal(tmp_path):
    np = pytest.importorskip("numpy")
    j = tracking.Journal(tmp_path)
    j.append("log_metrics", [{"m": np.float32(0.5), "n": np.int64(3)}], {})
    assert j.entries()[0]["args"][0] == {"m": 0.5, "n": 3}


def test_offline_logging_goes_to_the_journal_only(tmp_path):
    j = tracking.Journal(tmp_path)
    safe = tracking.SafeMlflow(None, j)
    safe.log_metrics({"val.HOTA": 0.4})
    safe.log_artifact("/x/y.txt", artifact_path="logs")
    assert safe.active_run().info.run_id is None
    assert [e["op"] for e in j.entries()] == ["log_metrics", "log_artifact"]
    assert j.state()["primary"]["missed_from"] == 0


def test_a_server_error_only_warns(tmp_path, capsys):
    class Broken:
        def log_metrics(self, *a, **k):
            raise ConnectionError("dagshub down")

    j = tracking.Journal(tmp_path)
    tracking.SafeMlflow(Broken(), j).log_metrics({"a": 1.0})
    assert "mlflow-sync" in capsys.readouterr().out
    assert j.state()["primary"]["missed_from"] == 0


# --------------------------------------------------------------------------- with MLflow (CI)
def _runs(uri, cfg):
    import mlflow

    mlflow.set_tracking_uri(uri)
    return mlflow.search_runs(experiment_names=[cfg["tracking"]["experiment"]])


@pytest.fixture
def oracle(synthetic_cfg, monkeypatch):
    pytest.importorskip("mlflow")
    pytest.importorskip("trackeval")
    from test_pipeline import OracleAdapter  # tests/ is on sys.path under pytest (rootdir import mode)
    from tir_jde.modeling import adapters

    monkeypatch.setitem(adapters.ADAPTERS, "oracle", OracleAdapter)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("DAGSHUB_TOKEN", raising=False)
    monkeypatch.delenv("TIRJDE_MLFLOW_BACKUP", raising=False)
    return {**synthetic_cfg, "model": {**synthetic_cfg["model"], "adapter": "oracle", "python": "python"}}


def test_every_stage_is_copied_to_the_backup_store(oracle):
    from tir_jde import pipeline

    pipeline.stage_train(oracle)
    pipeline.stage_evaluate(oracle)
    backup = _runs(tracking.backup_uri(oracle), oracle)
    primary = _runs(tracking.tracking_uri(oracle), oracle)
    assert len(backup) == 1 and len(primary) == 1
    for runs in (backup, primary):
        assert runs.iloc[0]["metrics.val.HOTA"] == pytest.approx(1.0)
        assert runs.iloc[0]["params.model.name"] == "strong_baseline"
        assert runs.iloc[0]["tags.stage.last"] == "evaluate"


def test_an_unreachable_server_does_not_stop_training_and_is_synced_later(oracle, monkeypatch):
    from tir_jde import pipeline
    from tir_jde.experiment import run_dir

    monkeypatch.setattr(tracking, "_start_primary", lambda *a, **k: None)  # server down
    pipeline.stage_train(oracle)
    pipeline.stage_evaluate(oracle)

    rdir = run_dir(oracle)
    assert json.loads((rdir / "mlflow_sync.json").read_text())["primary"]["missed_from"] == 0
    assert _runs(tracking.tracking_uri(oracle), oracle).empty               # nothing reached the server
    assert _runs(tracking.backup_uri(oracle), oracle).iloc[0]["metrics.val.HOTA"] == pytest.approx(1.0)

    monkeypatch.undo()
    tracking.sync_primary(oracle)                                           # server back
    run = _runs(tracking.tracking_uri(oracle), oracle).iloc[0]
    assert run["metrics.val.HOTA"] == pytest.approx(1.0)
    assert run["params.model.name"] == "strong_baseline"
    assert json.loads((rdir / "mlflow_sync.json").read_text())["primary"]["missed_from"] is None
