"""Dist-Tracker, run from its own repository.

Training and tracking use the authors' Ultralytics fork through the same calls
as their ``train.py`` / ``track.py`` (``yolo12-UAV.yaml``, 640 px training,
1280 px tracking with ``conf=0`` and their ``botsort.yaml``). Only the data
paths, device and seed come from the pipeline; the fork's code is untouched.

The fork is imported from the repository, never from a pip-installed
``ultralytics`` (that would silently replace the authors' NWD/SSQ loss).
"""

from __future__ import annotations

from tir_jde.modeling.adapters.base import Adapter


class DistTrackerAdapter(Adapter):
    name = "dist_tracker"

    def detector_job(self) -> dict:
        det = self.model_cfg["detector"]
        return {
            "mode": "train",
            "lib_root": str(self.repo_dir),
            "cwd": str(self.repo_dir / "ultralytics"),
            "model": det["cfg"],
            "train_args": {
                "data": str(self.data_root / "yolo" / "dataset.yaml"),
                "task": "detect",
                "epochs": det["epochs"],
                "batch": det["batch"],
                "imgsz": det["imgsz"],
                "device": "0",
                "seed": self.cfg.get("seed", 0),
                "deterministic": True,
                "project": str(self.run_dir / "detector"),
                "name": "train",
                "exist_ok": True,
                # extra Ultralytics arguments, only for short checks (e.g. the smoke test)
                **(det.get("extra") or {}),
            },
        }

    def train(self) -> dict[str, str]:
        result = self._run_job(self.detector_job(), "train_detector")
        return {"detector": result["best"], "detector_results": f"{result['save_dir']}/results.csv"}

    def track_job(self, weights: dict[str, str]) -> dict:
        trk = self.model_cfg["tracker"]
        return {
            "mode": "track",
            "lib_root": str(self.repo_dir),
            "cwd": str(self.repo_dir / "ultralytics"),
            "weights": weights["detector"],
            "frames_root": str(self.data_root / "track" / "val" / "Test_imgs"),
            "out_dir": str(self.run_dir / "predictions"),
            "track_args": {
                "tracker": trk["cfg"],
                "imgsz": trk["imgsz"],
                "conf": trk["conf"],
                "device": "0",
                # output only: the fork's defaults write every annotated frame to disk, which
                # would be counted in the measured FPS. Boxes and IDs are unaffected; the
                # runner writes the MOTChallenge files itself (Strong Baseline runs --nosave).
                "save": False, "save_frames": False, "save_txt": False, "show": False,
            },
        }

    def track(self, weights: dict[str, str]) -> dict:
        return self._run_job(self.track_job(weights), "track")
