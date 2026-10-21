"""Strong Baseline (YOLOv12 + BoT-SORT-ReID), run from its own repository.

Detector: the authors' ``BoT-SORT/yolov12`` package trained with the same
``YOLO(...).train(...)`` call as their ``train.py`` (data, epochs, batch, image
size, single class), plus the run's seed.
ReID: the authors' FastReID ``train_net.py`` with their SBS-S50 config, on the
crops exported by ``tir_jde.dataset``.
Tracking: the authors' ``tools/predict_track3.py``.
"""

from __future__ import annotations

import time
from pathlib import Path

from tir_jde.modeling.adapters.base import Adapter


class StrongBaselineAdapter(Adapter):
    name = "strong_baseline"

    @property
    def botsort_dir(self) -> Path:
        return self.repo_dir / "BoT-SORT"

    @property
    def yolo_dir(self) -> Path:
        return self.botsort_dir / "yolov12"

    # ------------------------------------------------------------------ train
    def _model_spec(self) -> str:
        det = self.model_cfg["detector"]
        init = det.get("init", "scratch")
        if init == "scratch":
            return f"ultralytics/cfg/models/v12/{det['arch']}.yaml"
        return str(init)

    def detector_job(self) -> dict:
        det = self.model_cfg["detector"]
        return {
            "mode": "train",
            "lib_root": str(self.yolo_dir),
            "cwd": str(self.yolo_dir),
            "model": self._model_spec(),
            "train_args": {
                "data": str(self.data_root / "yolo" / "dataset.yaml"),
                "epochs": det["epochs"],
                "patience": det["epochs"],      # as in the authors' train.py (patience = epochs)
                "batch": det["batch"],
                "imgsz": det["imgsz"],
                "device": 0,
                "workers": det.get("workers", 8),
                "single_cls": True,
                "seed": self.cfg.get("seed", 0),
                "deterministic": True,
                "project": str(self.run_dir / "detector"),
                "name": "train",
                "exist_ok": True,
            },
        }

    def reid_config(self) -> Path:
        """The authors' SBS-S50 config, fetched from their Hugging Face repo if missing."""
        cfg_file = self.botsort_dir / self.model_cfg["reid"]["config"]
        if not cfg_file.exists():
            code = (
                "from huggingface_hub import hf_hub_download; import shutil, os; "
                f"os.makedirs({str(cfg_file.parent)!r}, exist_ok=True); "
                "shutil.copy(hf_hub_download('wish44165/YOLOv12-BoT-SORT-ReID', 'config.yaml'), "
                f"{str(cfg_file)!r})"
            )
            self._run([self.python, "-c", code], "reid_config")
        return cfg_file

    def train(self) -> dict[str, str]:
        weights: dict[str, str] = {}
        result = self._run_job(self.detector_job(), "train_detector")
        weights["detector"] = result["best"]
        weights["detector_results"] = str(Path(result["save_dir"]) / "results.csv")

        reid = self.model_cfg.get("reid", {})
        if reid.get("enabled", True) and not reid.get("weights"):
            out = self.run_dir / "reid"
            self._run(
                [self.python, "fast_reid/tools/train_net.py", "--config-file", str(self.reid_config()),
                 "MODEL.DEVICE", "cuda:0", "OUTPUT_DIR", str(out),
                 "DATASETS.NAMES", '("MOT20",)', "DATASETS.TESTS", '("MOT20",)'],
                "train_reid", cwd=self.botsort_dir,
                env={"FASTREID_DATASETS": str(self.data_root / "reid")},
            )
            weights["reid"] = str(out / "model_final.pth")
        elif reid.get("weights"):
            weights["reid"] = str(reid["weights"])
        return weights

    # ------------------------------------------------------------------ track
    def track_command(self, weights: dict[str, str], out_dir: Path) -> list[str]:
        trk = self.model_cfg["tracker"]
        cmd = [
            self.python, "tools/predict_track3.py",
            "--weights", weights["detector"],
            "--source", str(self.data_root / "track" / "val") + "/",
            "--img-size", str(trk["img_size"]),
            "--track_buffer", str(trk["track_buffer"]),
            "--device", "0",
            "--save_path_answer", str(out_dir),
            "--project", str(self.run_dir / "track_frames"),
            "--hide-labels-name", "--nosave",
        ]
        if trk.get("agnostic_nms", True):
            cmd.append("--agnostic-nms")
        if weights.get("reid") and self.model_cfg.get("reid", {}).get("enabled", True):
            cmd += ["--with-reid", "--fast-reid-config",
                    str(self.botsort_dir / self.model_cfg["reid"]["config"]),
                    "--fast-reid-weights", weights["reid"]]
        return cmd

    def track(self, weights: dict[str, str]) -> dict:
        out_dir = self.run_dir / "predictions"
        out_dir.mkdir(parents=True, exist_ok=True)
        if weights.get("reid"):
            self.reid_config()  # make sure the authors' ReID config is present
        frames = sum(1 for _ in (self.data_root / "track" / "val" / "Test_imgs").glob("*/*"))
        t0 = time.perf_counter()
        self._run(self.track_command(weights, out_dir), "track", cwd=self.botsort_dir)
        seconds = time.perf_counter() - t0
        return {"pred_dir": str(out_dir), "frames": frames, "seconds": seconds,
                "fps": frames / seconds if seconds else 0.0}
