"""End-to-end pipeline run with a fake model and a local MLflow store.

The fake adapter "tracks" by returning the ground truth, so the evaluator must
report a perfect score and every stage must land in the same MLflow run.
"""

import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("mlflow")
pytest.importorskip("trackeval")

import mlflow  # noqa: E402

from tir_jde import pipeline, tracking  # noqa: E402
from tir_jde.modeling import adapters  # noqa: E402
from tir_jde.modeling.adapters.base import Adapter  # noqa: E402


class OracleAdapter(Adapter):
    name = "oracle"

    def ensure_repo(self) -> str:
        return "0" * 40

    def train(self):
        w = self.run_dir / "best.pt"
        w.write_bytes(b"weights")
        results = self.run_dir / "results.csv"
        results.write_text("epoch,train/box_loss\n1,0.5\n2,0.3\n")
        return {"detector": str(w), "detector_results": str(results)}

    def track(self, weights):
        out = self.run_dir / "predictions"
        shutil.copytree(self.data_root / "gt" / "val", out, dirs_exist_ok=True)
        for f in out.glob("*.len"):
            f.unlink()
        return {"pred_dir": str(out), "frames": 19, "seconds": 1.0, "fps": 19.0}


@pytest.fixture
def oracle_cfg(synthetic_cfg, monkeypatch, tmp_path):
    monkeypatch.setitem(adapters.ADAPTERS, "oracle", OracleAdapter)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("DAGSHUB_TOKEN", raising=False)
    cfg = {**synthetic_cfg, "model": {**synthetic_cfg["model"], "adapter": "oracle", "python": "python"}}
    return cfg


def test_train_evaluate_visualize_share_one_mlflow_run(oracle_cfg):
    pipeline.stage_train(oracle_cfg)
    result = pipeline.stage_evaluate(oracle_cfg)
    figures = pipeline.stage_visualize(oracle_cfg, max_frames=5)

    assert result["overall"]["HOTA"] == pytest.approx(1.0)
    assert all(Path(f).exists() for f in figures)

    mlflow.set_tracking_uri(tracking.tracking_uri(oracle_cfg))
    runs = mlflow.search_runs(experiment_names=[oracle_cfg["tracking"]["experiment"]])
    assert len(runs) == 1
    run = runs.iloc[0]
    assert run["metrics.val.HOTA"] == pytest.approx(1.0)
    assert run["metrics.val.FPS_end_to_end"] == pytest.approx(19.0)
    assert run["params.model.name"] == "strong_baseline"
    assert run["params.seed"] == "0"
    assert run["tags.data.split_hash"]
    assert run["tags.stage.last"] == "visualize"


def test_changed_settings_cannot_reuse_a_run(oracle_cfg):
    pipeline.stage_train(oracle_cfg)
    changed = json.loads(json.dumps(oracle_cfg))
    changed["model"]["detector"]["epochs"] = 999
    with pytest.raises(RuntimeError, match="different configuration"):
        pipeline.stage_train(changed)


def test_compare_aggregates_seeds(oracle_cfg, tmp_path, monkeypatch):
    # results.json for two seeds of one experiment
    for seed, hota in ((0, 0.5), (1, 0.7)):
        cfg = {**oracle_cfg, "seed": seed}
        d = Path(cfg["paths"]["work_dir"]) / "runs" / f"{cfg['name']}-s{seed}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "results.json").write_text(json.dumps({"overall": {"HOTA": hota, "IDF1": 0.6}, "fps": 30}))
    out = pipeline.stage_compare(
        ["strong_baseline_bsb"], [0, 1], [f"paths.work_dir={oracle_cfg['paths']['work_dir']}"], tmp_path / "cmp"
    )
    text = out.read_text()
    assert "0.6000 ± 0.1000" in text  # HOTA mean ± std
    assert (tmp_path / "cmp" / "comparison.png").exists()


class DetectorOnlyAdapter(OracleAdapter):
    """train-detector must call train_detector() only and log the frame selection."""

    def train(self):
        raise AssertionError("ReID must not train in a detector-only run")

    def train_detector(self):
        weights = OracleAdapter.train(self)
        frames = self.run_dir / "detector_frames.json"
        frames.write_text(json.dumps({"train": {"step": 2, "frames": 5, "of": 10, "hash": "abc123"},
                                      "val": {"step": 10, "frames": 1, "of": 10, "hash": "def456"}}))
        val = self.run_dir / "detector_val_full.json"
        val.write_text(json.dumps({"metrics": {"metrics/mAP50(B)": 0.5, "metrics/recall(B)": 0.7}}))
        return {**weights, "detector_frames": str(frames), "detector_val_full": str(val)}


def test_detector_only_training_logs_frames_and_full_validation(oracle_cfg, monkeypatch):
    monkeypatch.setitem(adapters.ADAPTERS, "oracle", DetectorOnlyAdapter)
    pipeline.stage_train(oracle_cfg, detector_only=True)

    mlflow.set_tracking_uri(tracking.tracking_uri(oracle_cfg))
    run = mlflow.search_runs(experiment_names=[oracle_cfg["tracking"]["experiment"]]).iloc[0]
    assert run["tags.data.train_frames_hash"] == "abc123"
    assert run["metrics.data.train_step"] == 2
    assert run["metrics.det_val_full.metrics.mAP50_B"] == pytest.approx(0.5)
    assert run["metrics.det_val_full.metrics.recall_B"] == pytest.approx(0.7)


class ZeroShotAdapter(OracleAdapter):
    def evaluate_init_weights(self):
        return {"weights": "/w/SOT_yolov12l.pt", "metrics": {"metrics/mAP50(B)": 0.42}}


def test_zero_shot_check_logs_the_starting_weights(oracle_cfg, monkeypatch):
    monkeypatch.setitem(adapters.ADAPTERS, "oracle", ZeroShotAdapter)
    pipeline.stage_eval_detector({**oracle_cfg, "tag": "zeroshot"})

    mlflow.set_tracking_uri(tracking.tracking_uri(oracle_cfg))
    run = mlflow.search_runs(experiment_names=[oracle_cfg["tracking"]["experiment"]]).iloc[0]
    assert run["metrics.det_val_zero_shot.metrics.mAP50_B"] == pytest.approx(0.42)
    assert run["tags.detector.init_weights"] == "SOT_yolov12l.pt"
