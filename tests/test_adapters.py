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
    assert track["track_args"] == {"tracker": "botsort.yaml", "imgsz": 1280, "conf": 0.0, "device": "0",
                                   "save": False, "save_frames": False, "save_txt": False, "show": False}


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
    assert Adapter.epoch_metrics(csv_file) == [(1, {"train.box_loss_nan": 1.0, "train.cls_loss": 2.5})]


def test_fastreid_losses_are_read_per_iteration(tmp_path):
    from tir_jde.pipeline import _fastreid_metrics

    f = tmp_path / "metrics.json"
    f.write_text('{"iteration": 199, "loss_cls": 3.2, "loss_triplet": 0.8, "lr": 3.5e-05, "time": 0.4}\n'
                 '{"iteration": 399, "loss_cls": 2.9, "loss_triplet": NaN, "total_loss": 3.5}\n'
                 '{"iteration": 59')
    assert _fastreid_metrics(f) == [(199, {"loss_cls": 3.2, "loss_triplet": 0.8, "lr": 3.5e-05}),
                                    (399, {"loss_cls": 2.9, "total_loss": 3.5})]


def test_extra_detector_arguments_reach_ultralytics(tmp_path):
    for name in ("dist_tracker_bsb", "strong_baseline_bsb"):
        cfg = load_config(name, [f"paths.baselines_dir={tmp_path}",
                                 "model.detector.extra.fraction=0.1", "model.detector.extra.conf=0.1"])
        args = get_adapter(cfg, {"root": str(tmp_path / "data")}).detector_job()["train_args"]
        assert (args["fraction"], args["conf"]) == (0.1, 0.1)
        assert "fraction" not in get_adapter(load_config(name, [f"paths.baselines_dir={tmp_path}"]),
                                             {"root": str(tmp_path)}).detector_job()["train_args"]


def test_regenerated_files_are_restored_but_code_changes_are_refused(tmp_path):
    from tir_jde.modeling.adapters.base import Adapter

    repo = tmp_path / "repo"
    (repo / "lib").mkdir(parents=True)
    (repo / "lib" / "rank_cy.c").write_text("original\n")
    (repo / "code.py").write_text("x = 1\n")
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-qm", "init"], check=True)

    class Fake(Adapter):
        repo_dir = repo

        def __init__(self):
            self.model_cfg = {"generated_files": ["lib/*.c"]}
            self.logs_dir = tmp_path / "logs"
            self.logs_dir.mkdir(exist_ok=True)
            self.python = "python"

    (repo / "lib" / "rank_cy.c").write_text("regenerated\n")
    Fake().ensure_repo()
    assert (repo / "lib" / "rank_cy.c").read_text() == "original\n"

    (repo / "code.py").write_text("x = 2\n")
    with pytest.raises(RuntimeError, match="local changes"):
        Fake().ensure_repo()


FAKE_TRACKING = textwrap.dedent(
    '''
    import glob, os

    class _T:
        def __init__(self, v): self.v = v
        def cpu(self): return self
        def numpy(self): return self
        def astype(self, t): return _T([int(x) for x in self.v])
        def __iter__(self): return iter(self.v)

    class _Boxes:
        def __init__(self, rows):
            self.xyxy = _T([r[:4] for r in rows])
            self.conf = _T([r[4] for r in rows])
            self.id = _T(list(range(1, len(rows) + 1))) if rows else None

    class _Result:
        def __init__(self, path, rows): self.path, self.boxes = path, _Boxes(rows)
        def update(self, boxes): self.boxes = _Boxes(boxes.rows)

    class _Predictor:
        pass

    class YOLO:
        """Detector finds one box per frame; every callback runs, in registration order."""
        def __init__(self, weights): self.callbacks = []
        def add_callback(self, event, func): self.callbacks.append(func)
        def track(self, source, stream, verbose, **kw):
            for path in sorted(glob.glob(os.path.join(source, "*.*"))):
                pred = _Predictor()
                pred.results = [_Result(path, [[50.0, 60.0, 58.0, 66.0, 0.5, 0.0]])]
                for cb in self.callbacks:
                    cb(pred)
                yield pred.results[0]
    '''
)

FAKE_TORCH = textwrap.dedent(
    '''
    float32 = "float32"

    class _Tensor:
        def __init__(self, rows): self.rows = [list(r) for r in rows]
        def reshape(self, *shape): return self
        def clone(self): return _Tensor(self.rows)

    def tensor(rows, dtype=None):
        return _Tensor(rows)
    '''
)


def _tracking_job(tmp_path, first_frame_dir):
    lib = tmp_path / "fork"
    (lib / "ultralytics").mkdir(parents=True)
    (lib / "ultralytics" / "__init__.py").write_text(FAKE_TRACKING)
    (lib / "torch").mkdir()
    (lib / "torch" / "__init__.py").write_text(FAKE_TORCH)
    frames = _frames(tmp_path / "frames")  # C_02: 4 frames, T_02: 3 frames
    job = {"mode": "track", "lib_root": str(lib), "cwd": str(tmp_path), "weights": "best.pt",
           "frames_root": str(frames), "out_dir": str(tmp_path / "pred"), "track_args": {},
           "first_frame_dir": first_frame_dir, "result_file": str(tmp_path / "result.json")}
    (tmp_path / "job.json").write_text(json.dumps(job))
    subprocess.run([sys.executable, str(RUNNER), str(tmp_path / "job.json")], check=True)
    return {s: (tmp_path / "pred" / f"{s}.txt").read_text().splitlines() for s in ("C_02", "T_02")}


def test_first_frame_uses_the_given_boxes_then_the_detector(tmp_path):
    first = tmp_path / "first"
    first.mkdir()
    (first / "C_02.txt").write_text("1,1,10,20,8,6,1,1,1.0\n1,2,40,30,6,5,1,1,1.0\n")
    (first / "T_02.txt").write_text("")  # empty first frame (like MultiUAV-068)
    out = _tracking_job(tmp_path, str(first))

    c = [ln for ln in out["C_02"] if ln.startswith("1,")]
    assert c == ["1,1,10.00,20.00,8.00,6.00,1.0000,-1,-1,-1", "1,2,40.00,30.00,6.00,5.00,1.0000,-1,-1,-1"]
    assert all(ln.endswith("0.5000,-1,-1,-1") for ln in out["C_02"] if not ln.startswith("1,"))
    assert len(out["C_02"]) == 2 + 3  # frames 2-4 come from the detector
    assert not any(ln.startswith("1,") for ln in out["T_02"])  # no track on an empty first frame
    assert len(out["T_02"]) == 2


def test_without_first_frame_boxes_tracking_is_unchanged(tmp_path):
    out = _tracking_job(tmp_path, None)
    assert out["C_02"][0] == "1,1,50.00,60.00,8.00,6.00,0.5000,-1,-1,-1"
    assert len(out["C_02"]) == 4 and len(out["T_02"]) == 3


def test_dist_tracker_first_frame_switch(tmp_path):
    def job(*overrides):
        cfg = load_config("dist_tracker_bsb", [f"paths.baselines_dir={tmp_path}", *overrides])
        return get_adapter(cfg, {"root": str(tmp_path / "data")}).track_job({"detector": "best.pt"})

    assert job()["first_frame_dir"] is None  # off until verified on Colab with the real Ultralytics
    assert job("model.tracker.first_frame_gt=true")["first_frame_dir"].endswith("track/val/TestLabels_FirstFrameOnly")


SHARED_TRAIN_ARGS = ("epochs", "batch", "imgsz", "multi_scale", "single_cls", "amp", "optimizer",
                     "warmup_bias_lr", "copy_paste", "close_mosaic", "seed", "deterministic")


def test_controlled_experiments_train_and_track_with_the_same_settings(tmp_path):
    ov = [f"paths.baselines_dir={tmp_path}"]
    sb = get_adapter(load_config("strong_baseline_bsb_ctrl", ov), {"root": str(tmp_path / "d")})
    dist = get_adapter(load_config("dist_tracker_bsb_ctrl", ov), {"root": str(tmp_path / "d")})
    a, b = sb.detector_job()["train_args"], dist.detector_job()["train_args"]
    assert {k: a[k] for k in SHARED_TRAIN_ARGS} == {k: b[k] for k in SHARED_TRAIN_ARGS}
    assert (a["imgsz"], a["multi_scale"], a["copy_paste"]) == (1280, False, 0.0)
    assert sb.detector_job()["model"].endswith("yolov12n.yaml") and dist.detector_job()["model"] == "yolo12n.yaml"

    cmd = sb.track_command({"detector": "w.pt", "reid": "r.pth"}, Path("/o"))
    trk = dist.track_job({"detector": "w.pt"})
    assert cmd[cmd.index("--img-size") + 1] == str(trk["track_args"]["imgsz"]) == "1280"
    assert cmd[cmd.index("--conf-thres") + 1] == str(trk["track_args"]["conf"]) == "0.01"
    assert trk["first_frame_dir"]  # BSB protocol on in the controlled setting


def test_author_recipes_stay_unchanged(tmp_path):
    ov = [f"paths.baselines_dir={tmp_path}"]
    sb = get_adapter(load_config("strong_baseline_bsb", ov), {"root": str(tmp_path / "d")})
    dist = get_adapter(load_config("dist_tracker_bsb", ov), {"root": str(tmp_path / "d")})
    assert "--conf-thres" not in sb.track_command({"detector": "w.pt"}, Path("/o"))  # script default 0.09
    assert dist.detector_job()["model"] == "./cfg/models/12/yolo12-UAV.yaml"  # scale 's'
    assert (dist.detector_job()["train_args"]["imgsz"], dist.track_job({"detector": "w"})["track_args"]["conf"]) == (640, 0.0)
