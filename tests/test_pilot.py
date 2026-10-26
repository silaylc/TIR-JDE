"""Detector pilot: frame thinning, full validation, COCO start, settings and report."""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tir_jde import dataset
from tir_jde.experiment import load_config
from tir_jde.modeling.adapters import get_adapter
from tir_jde.modeling.adapters.base import RUNNER, Adapter


def test_every_kth_frame_of_each_sequence_starting_with_frame_1():
    imgs = [Path(f"/x/{s}_{i:05d}.jpg") for s in ("B_01", "A_01") for i in range(1, 8)]
    kept = dataset.select_frames(list(reversed(imgs)), 3)
    assert [p.name for p in kept] == ["A_01_00001.jpg", "A_01_00004.jpg", "A_01_00007.jpg",
                                      "B_01_00001.jpg", "B_01_00004.jpg", "B_01_00007.jpg"]
    assert dataset.select_frames(imgs, 1) == sorted(imgs)
    # the hash names the frames, not where they are stored
    assert dataset.frames_hash(kept) == dataset.frames_hash([Path("/other") / p.name for p in kept])
    assert dataset.frames_hash(kept) != dataset.frames_hash(imgs)


def test_common_detector_and_pilot_settings(tmp_path):
    ov = [f"paths.baselines_dir={tmp_path}", f"paths.work_dir={tmp_path}", "model.detector.init=scratch"]
    pilot = load_config("common_detector_bsb_pilot", ov)
    det = pilot["model"]["detector"]
    args = get_adapter(pilot, {"root": str(tmp_path / "d")}).detector_job()["train_args"]
    assert (det["arch"], args["epochs"], args["imgsz"], args["batch"], args["nbs"]) == ("yolov12l", 10, 640, 16, 64)
    assert (args["optimizer"], args["lr0"], args["warmup_epochs"], args["close_mosaic"]) == ("SGD", 0.003, 1.0, 2)
    assert (args["copy_paste"], args["hsv_h"], args["hsv_s"], args["multi_scale"]) == (0.0, 0.0, 0.0, False)
    assert (args["conf"], args["single_cls"], args["amp"]) == (0.01, True, True)
    assert (det["frame_step"], det["val_frame_step"], pilot["model"]["reid"]["enabled"]) == (2, 10, False)
    assert args["data"] == str(tmp_path / "runs" / "common_detector_bsb_pilot-s0" / "detector_data" / "dataset.yaml")
    assert args["workers"] >= 1  # auto = CPU count

    main = load_config("common_detector_bsb", ov)
    main_args = get_adapter(main, {"root": str(tmp_path / "d")}).detector_job()["train_args"]
    assert main_args["close_mosaic"] == 10
    assert main_args["data"] == str(tmp_path / "d" / "yolo" / "dataset.yaml")  # all frames
    assert load_config("common_detector_bsb", [])["model"]["detector"]["init"] == "sot"


def test_frame_lists_are_written_with_their_hash(synthetic_cfg):
    cfg = {**synthetic_cfg, "model": {**synthetic_cfg["model"], "detector": {
        **synthetic_cfg["model"]["detector"], "frame_step": 2, "val_frame_step": 4}}}
    adapter = get_adapter(cfg, dataset.prepare(cfg))
    info = adapter.write_detector_data()

    lists = adapter.run_dir / "detector_data"
    train = (lists / "train.txt").read_text().splitlines()
    val = (lists / "val.txt").read_text().splitlines()
    # train: C_01 (12), T_01 (8), BB2P_01 (7) -> 6 + 4 + 4; val: C_02 (10), T_02 (9) -> 3 + 3
    assert (info["train"]["frames"], info["val"]["frames"]) == (14, 6)
    assert info["train"]["hash"] == dataset.frames_hash([Path(p) for p in train])
    assert all(int(Path(p).stem.rsplit("_", 1)[1]) % 2 == 1 for p in train)
    assert all(Path(p.replace("/images/", "/labels/")).with_suffix(".txt").exists() for p in train + val)
    yaml_text = (lists / "dataset.yaml").read_text()
    assert f"train: {lists / 'train.txt'}" in yaml_text and "nc: 1" in yaml_text
    assert json.loads((adapter.run_dir / "detector_frames.json").read_text()) == info


def _fake_download(monkeypatch, content=b"pt"):
    import urllib.request

    calls = []

    def fake(url, dest):
        calls.append(url)
        Path(dest).write_bytes(content)

    monkeypatch.setattr(urllib.request, "urlretrieve", fake)
    return calls


def test_sot_start_downloads_once_and_checks_the_hash(tmp_path, monkeypatch):
    import hashlib

    calls = _fake_download(monkeypatch)
    ov = [f"paths.baselines_dir={tmp_path}", f"paths.work_dir={tmp_path}",
          f"model.detector.pretrained.sot.sha256={hashlib.sha256(b'pt').hexdigest()}"]
    adapter = get_adapter(load_config("common_detector_bsb", ov), {"root": str(tmp_path / "d")})
    assert adapter.detector_job()["model"] == str(tmp_path / "weights" / "strong_baseline_sot_SOT_yolov12l.pt")
    adapter.detector_job()
    assert len(calls) == 1 and calls[0].endswith("/a97bba3252814543a48286b74ef182e731f38221/BoT-SORT/yolov12/"
                                                   "weights/SOT_yolov12l.pt")
    # the detector runs in the author's code at the commit that matches the weights
    assert adapter.repo_dir.name == "common_detector"
    assert adapter.detector_job()["lib_root"].endswith("common_detector/BoT-SORT/yolov12")


def test_sot_start_refuses_a_changed_file_or_another_architecture(tmp_path, monkeypatch):
    _fake_download(monkeypatch, b"something else")
    ov = [f"paths.baselines_dir={tmp_path}", f"paths.work_dir={tmp_path}"]
    with pytest.raises(RuntimeError, match="SHA-256"):
        get_adapter(load_config("common_detector_bsb", ov), {"root": str(tmp_path / "d")}).detector_job()
    with pytest.raises(ValueError, match="yolov12l"):
        get_adapter(load_config("common_detector_bsb", [*ov, "model.detector.arch=yolov12n"]),
                    {"root": str(tmp_path / "d")}).detector_job()


def test_coco_url_follows_the_architecture(tmp_path, monkeypatch):
    calls = _fake_download(monkeypatch)
    cfg = load_config("strong_baseline_bsb", [f"paths.baselines_dir={tmp_path}", f"paths.work_dir={tmp_path}",
                                              "model.detector.init=coco", "model.detector.arch=yolov12s"])
    get_adapter(cfg, {"root": str(tmp_path / "d")}).detector_job()
    assert calls == ["https://github.com/sunsmarterjie/yolov12/releases/download/turbo/yolov12s.pt"]


def test_zero_shot_validation_uses_the_starting_weights(synthetic_cfg, monkeypatch):
    _fake_download(monkeypatch)
    det = {**synthetic_cfg["model"]["detector"], "init": "coco"}
    cfg = {**synthetic_cfg, "model": {**synthetic_cfg["model"], "detector": det}}
    adapter = get_adapter(cfg, dataset.prepare(cfg))
    adapter._run_job = lambda job, name: {"metrics": {"metrics/mAP50(B)": 0.3}, "job": job, "name": name}
    result = adapter.evaluate_init_weights()
    assert result["name"] == "val_zero_shot" and result["job"]["weights"] == result["weights"]
    assert result["job"]["val_args"]["data"].endswith("yolo/dataset.yaml")
    assert (adapter.run_dir / "detector_val_zero_shot.json").exists()


def test_epoch_seconds_from_cumulative_time(tmp_path):
    f = tmp_path / "results.csv"
    f.write_text("epoch,time,metrics/mAP50(B)\n1,100,0.1\n2,250,0.2\n")
    rows = Adapter.epoch_metrics(f)
    assert [m["epoch_seconds"] for _, m in rows] == [100.0, 150.0]


def test_thinned_validation_is_followed_by_a_full_one(synthetic_cfg):
    cfg = {**synthetic_cfg, "model": {**synthetic_cfg["model"], "detector": {
        **synthetic_cfg["model"]["detector"], "val_frame_step": 5}}}
    adapter = get_adapter(cfg, dataset.prepare(cfg))
    jobs = []

    def fake_job(job, log_name):
        jobs.append((log_name, job))
        if job["mode"] == "train":
            return {"best": "/w/best.pt", "save_dir": "/w"}
        return {"metrics": {"metrics/mAP50(B)": 0.4, "metrics/recall(B)": 0.6}}

    adapter._run_job = fake_job
    weights = adapter.train_detector()
    assert [n for n, _ in jobs] == ["train_detector", "val_detector"]
    val_args = jobs[1][1]["val_args"]
    assert val_args["data"].endswith("yolo/dataset.yaml") and jobs[1][1]["weights"] == "/w/best.pt"
    assert json.loads(Path(weights["detector_val_full"]).read_text())["metrics"]["metrics/recall(B)"] == 0.6


FAKE_VAL = textwrap.dedent(
    '''
    class YOLO:
        def __init__(self, weights): self.weights = weights
        def val(self, **kw):
            class M:
                results_dict = {"metrics/mAP50(B)": 0.5, "metrics/recall(B)": 0.7}
                speed = {"inference": 1.5}
            return M()
    '''
)


def test_runner_validates_with_the_fork(tmp_path):
    pkg = tmp_path / "fork" / "ultralytics"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(FAKE_VAL)
    job = {"mode": "val", "lib_root": str(tmp_path / "fork"), "cwd": str(tmp_path), "weights": "best.pt",
           "val_args": {"data": "d.yaml"}, "check_weights": False, "result_file": str(tmp_path / "r.json")}
    (tmp_path / "job.json").write_text(json.dumps(job))
    subprocess.run([sys.executable, str(RUNNER), str(tmp_path / "job.json")], check=True)
    result = json.loads((tmp_path / "r.json").read_text())
    assert result == {"metrics": {"metrics/mAP50(B)": 0.5, "metrics/recall(B)": 0.7}, "speed_ms": {"inference": 1.5}}


def _fake_pilot_run(base: Path, name: str, seed: int, tag: str, map50: float, recall: float) -> None:
    d = base / f"{name}-s{seed}-{tag}"
    (d / "detector" / "train").mkdir(parents=True)
    (d / "detector_val_full.json").write_text(json.dumps(
        {"metrics": {"metrics/mAP50(B)": map50, "metrics/mAP50-95(B)": 0.1, "metrics/recall(B)": recall}}))
    (d / "detector" / "train" / "results.csv").write_text("epoch,time\n1,600\n2,1200\n")
    (d / "gpu_summary.json").write_text(json.dumps({"gpu.memory_peak_mib": 20480, "gpu.utilization_mean_pct": 80}))


def test_pilot_report_applies_the_noise_rule(tmp_path):
    from tir_jde.pipeline import stage_pilot_report

    cfg = load_config("common_detector_bsb_pilot", [f"paths.work_dir={tmp_path}"])
    runs = tmp_path / "runs"
    for seed, tag, m in ((0, "R0a", 0.50), (1, "R0b", 0.51), (0, "R2", 0.53), (0, "R3", 0.515)):
        _fake_pilot_run(runs, cfg["name"], seed, tag, m, 0.6)
    text = stage_pilot_report(cfg).read_text()
    rows = {ln.split("|")[1].strip(): ln for ln in text.splitlines() if ln.startswith("| R")}
    assert "benimse" in rows["R2"]           # +0.03 > 2 × 0.01
    assert "varsayılan kalır" in rows["R3"]   # +0.015 < 0.02
    assert "referans" in rows["R0a"] and "gürültü" in rows["R0b"]
    assert "10.0000" in rows["R0a"] and "20.0000" in rows["R0a"]  # epoch minutes, peak GiB
    assert "eşik 2× = 0.0200" in text


def test_pilot_report_can_use_the_coco_run_as_reference(tmp_path):
    from tir_jde.pipeline import stage_pilot_report

    cfg = load_config("common_detector_bsb_pilot", [f"paths.work_dir={tmp_path}"])
    for seed, tag, m in ((0, "R0a", 0.50), (1, "R0b", 0.51), (0, "R1", 0.60), (0, "R2", 0.61)):
        _fake_pilot_run(tmp_path / "runs", cfg["name"], seed, tag, m, 0.6)
    rows = {ln.split("|")[1].strip(): ln for ln in stage_pilot_report(cfg, baseline="R1").read_text().splitlines()
            if ln.startswith("| R")}
    assert "referans" in rows["R1"] and "varsayılan kalır" in rows["R2"]  # +0.01 vs R1 < 0.02


FAKE_MISMATCH = textwrap.dedent(
    '''
    class _T:
        def __init__(self, *shape): self.shape = shape

    class _Net:
        yaml = {"nc": 1}
        def float(self): return self
        def state_dict(self): return {"model.6.attn.qkv.conv.weight": _T(768, 256, 1, 1)}

    class YOLO:
        def __init__(self, weights): self.model = _Net()
        def train(self, **kw): raise AssertionError("must not train")
        def val(self, **kw): raise AssertionError("must not validate")
    '''
)
FAKE_TASKS = textwrap.dedent(
    '''
    from ultralytics import _T

    class DetectionModel:
        def __init__(self, cfg, nc, verbose): pass
        def state_dict(self):
            return {"model.6.attn.qk.conv.weight": _T(512, 256, 1, 1), "model.6.attn.v.conv.weight": _T(256, 256, 1, 1)}
    '''
)


@pytest.mark.parametrize("mode", ["train", "val"])
def test_runner_refuses_weights_saved_by_other_code(tmp_path, mode):
    pkg = tmp_path / "fork" / "ultralytics"
    (pkg / "nn").mkdir(parents=True)
    (pkg / "__init__.py").write_text(FAKE_MISMATCH)
    (pkg / "nn" / "__init__.py").write_text("")
    (pkg / "nn" / "tasks.py").write_text(FAKE_TASKS)
    job = {"mode": mode, "lib_root": str(tmp_path / "fork"), "cwd": str(tmp_path), "model": "SOT_yolov12l.pt",
           "weights": "SOT_yolov12l.pt", "train_args": {}, "val_args": {}, "result_file": str(tmp_path / "r.json")}
    (tmp_path / "job.json").write_text(json.dumps(job))
    proc = subprocess.run([sys.executable, str(RUNNER), str(tmp_path / "job.json")], capture_output=True, text=True)
    assert proc.returncode != 0
    assert "uyumsuz" in proc.stderr and "qkv" in proc.stderr
