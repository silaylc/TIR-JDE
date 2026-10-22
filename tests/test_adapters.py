"""Adapters are tested without GPUs or baseline environments.

- Command/job construction is checked directly.
- The runner script is executed for real against a tiny fake ``ultralytics``
  package, which also checks the guard against a pip-installed package.
"""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tir_jde import dataset
from tir_jde.experiment import load_config
from tir_jde.modeling.adapters import get_adapter
from tir_jde.modeling.adapters.base import RUNNER


def test_strong_baseline_trains_from_scratch_with_the_run_seed(synthetic_cfg):
    cfg = {**synthetic_cfg, "seed": 2}
    adapter = get_adapter(cfg, dataset.prepare(cfg))
    job = adapter.detector_job()
    assert job["model"] == "ultralytics/cfg/models/v12/yolov12n.yaml"
    assert job["lib_root"].endswith("BoT-SORT/yolov12")
    assert job["train_args"]["seed"] == 2
    assert job["train_args"]["single_cls"] is True
    assert job["train_args"]["data"].endswith("yolo/dataset.yaml")


def test_strong_baseline_tracking_uses_its_own_reid(synthetic_cfg):
    adapter = get_adapter(synthetic_cfg, dataset.prepare(synthetic_cfg))
    cmd = adapter.track_command({"detector": "/w/best.pt", "reid": "/w/reid.pth"}, Path("/out"))
    assert cmd[1] == "tools/predict_track3.py"
    assert cmd[cmd.index("--fast-reid-weights") + 1] == "/w/reid.pth"
    assert cmd[cmd.index("--source") + 1].endswith("track/val/")
    assert cmd[cmd.index("--img-size") + 1] == "1600"


def test_dist_tracker_uses_the_fork_and_author_settings(tmp_path):
    cfg = load_config("dist_tracker_bsb", [f"paths.baselines_dir={tmp_path}"])
    adapter = get_adapter(cfg, {"root": str(tmp_path / "data")})
    train = adapter.detector_job()
    assert train["lib_root"] == str(tmp_path / "dist_tracker")
    assert train["cwd"] == str(tmp_path / "dist_tracker" / "ultralytics")
    assert train["model"] == "./cfg/models/12/yolo12-UAV.yaml"
    assert (train["train_args"]["epochs"], train["train_args"]["imgsz"]) == (50, 640)
    track = adapter.track_job({"detector": "best.pt"})
    assert track["track_args"] == {"tracker": "botsort.yaml", "imgsz": 1280, "conf": 0.0, "device": "0"}


FAKE_ULTRALYTICS = textwrap.dedent(
    '''
    class _T:
        def __init__(self, v): self.v = v
        def cpu(self): return self
        def numpy(self): return self
        def astype(self, t):
            return _T([int(x) for x in self.v])
        def __iter__(self): return iter(self.v)

    class _Boxes:
        def __init__(self, f):
            self.xyxy = _T([[10.0 + f, 20.0, 18.0 + f, 26.0], [40.0, 30.0, 46.0, 35.0]])
            self.id = _T([1, 2])
            self.conf = _T([0.9, 0.8])

    class _Result:
        def __init__(self, f): self.boxes = _Boxes(f)

    class YOLO:
        def __init__(self, weights): self.weights = weights
        def track(self, source, stream, verbose, **kw):
            import os
            n = len([p for p in os.listdir(source) if p.endswith(".jpg")])
            return (_Result(f) for f in range(1, n + 1))
    '''
)


def _fake_lib(root: Path) -> Path:
    pkg = root / "ultralytics"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(FAKE_ULTRALYTICS)
    return root


def _frames(root: Path) -> Path:
    for seq, n in (("C_02", 4), ("T_02", 3)):
        d = root / seq
        d.mkdir(parents=True)
        for i in range(1, n + 1):
            (d / f"{i:05d}.jpg").write_bytes(b"x")
    return root


def test_runner_writes_mot_files_from_the_fork(tmp_path):
    lib = _fake_lib(tmp_path / "fork")
    job = {
        "mode": "track", "lib_root": str(lib), "cwd": str(tmp_path), "weights": "best.pt",
        "frames_root": str(_frames(tmp_path / "frames")), "out_dir": str(tmp_path / "pred"),
        "track_args": {}, "result_file": str(tmp_path / "result.json"),
    }
    (tmp_path / "job.json").write_text(json.dumps(job))
    subprocess.run([sys.executable, str(RUNNER), str(tmp_path / "job.json")], check=True)

    result = json.loads((tmp_path / "result.json").read_text())
    assert result["frames"] == 7
    lines = (tmp_path / "pred" / "C_02.txt").read_text().splitlines()
    assert len(lines) == 8
    assert lines[0] == "1,1,11.00,20.00,8.00,6.00,0.9000,-1,-1,-1"


def test_runner_refuses_a_shadowing_pip_package(tmp_path):
    shadow = _fake_lib(tmp_path / "site")
    fork = tmp_path / "fork"  # no ultralytics package inside -> import falls through to `shadow`
    fork.mkdir()
    job = {"mode": "track", "lib_root": str(fork), "cwd": str(tmp_path), "weights": "w",
           "frames_root": str(tmp_path), "out_dir": str(tmp_path / "o"), "track_args": {},
           "result_file": str(tmp_path / "r.json")}
    (tmp_path / "job.json").write_text(json.dumps(job))
    proc = subprocess.run([sys.executable, str(RUNNER), str(tmp_path / "job.json")],
                          capture_output=True, text=True, env={"PYTHONPATH": str(shadow)})
    assert proc.returncode != 0
    assert "Wrong ultralytics package" in proc.stderr


def test_epoch_metrics_are_read_from_results_csv(tmp_path):
    from tir_jde.modeling.adapters.base import Adapter

    csv_file = tmp_path / "results.csv"
    csv_file.write_text("epoch, train/box_loss, metrics/mAP50(B)\n1, 0.9, 0.10\n2, 0.7, 0.25\n")
    rows = Adapter.epoch_metrics(csv_file)
    assert rows == [(1, {"train.box_loss": 0.9, "metrics.mAP50_B": 0.10}),
                    (2, {"train.box_loss": 0.7, "metrics.mAP50_B": 0.25})]
    assert Adapter.epoch_metrics(tmp_path / "missing.csv") == []


@pytest.mark.parametrize("name", ["strong_baseline_bsb", "dist_tracker_bsb"])
def test_every_experiment_has_an_adapter(name, tmp_path):
    cfg = load_config(name, [f"paths.work_dir={tmp_path}"])
    assert get_adapter(cfg, {"root": str(tmp_path)}).name == cfg["model"]["adapter"]


def test_epoch_metrics_skip_nan_values(tmp_path):
    from tir_jde.modeling.adapters.base import Adapter

    csv_file = tmp_path / "results.csv"
    csv_file.write_text("epoch, train/box_loss, train/cls_loss\n1, nan, 2.5\n")
    assert Adapter.epoch_metrics(csv_file) == [(1, {"train.cls_loss": 2.5})]


def test_extra_detector_arguments_reach_ultralytics(tmp_path):
    for name in ("dist_tracker_bsb", "strong_baseline_bsb"):
        cfg = load_config(name, [f"paths.baselines_dir={tmp_path}",
                                 "model.detector.extra.fraction=0.1", "model.detector.extra.conf=0.1"])
        args = get_adapter(cfg, {"root": str(tmp_path / "data")}).detector_job()["train_args"]
        assert (args["fraction"], args["conf"]) == (0.1, 0.1)
        assert "fraction" not in get_adapter(load_config(name, [f"paths.baselines_dir={tmp_path}"]),
                                             {"root": str(tmp_path)}).detector_job()["train_args"]
