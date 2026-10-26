"""Runs inside a baseline's own Python environment (not the pipeline's).

Called as ``<baseline python> _ultralytics_runner.py <job.json>``. The job names
the folder that contains the baseline's own ``ultralytics`` package
(``lib_root``); it is put first on ``sys.path`` and the import is checked, so a
pip-installed ``ultralytics`` can never silently replace the authors' fork.

Modes:
- ``train``: ``YOLO(model).train(**train_args)``; prints the save directory.
- ``val``: ``YOLO(weights).val(...)``; returns the detection metrics.
- ``track``: ``YOLO(weights).track(...)`` on every sequence folder, writes one
  MOTChallenge file per sequence and the end-to-end frames-per-second.
  With ``first_frame_dir`` (BSB protocol), the detections of each sequence's
  first frame are replaced by the given first-frame boxes before the tracker
  sees them; every later frame runs exactly as ``model.track`` does.

Only the standard library is imported at module level so the file works on
the Python versions the baselines pin (3.8+).
"""

import glob
import json
import os
import sys
import time
from pathlib import Path


def _import_fork(lib_root: str):
    lib_root = str(Path(lib_root).resolve())
    sys.path.insert(0, lib_root)
    for mod in [m for m in sys.modules if m == "ultralytics" or m.startswith("ultralytics.")]:
        del sys.modules[mod]
    import ultralytics  # noqa: E402
    from ultralytics import YOLO  # noqa: E402

    loaded = str(Path(ultralytics.__file__).resolve())
    if not loaded.startswith(lib_root):
        raise RuntimeError(
            f"Wrong ultralytics package imported: {loaded} (expected the fork under {lib_root}). "
            "Uninstall the pip package in this environment: pip uninstall -y ultralytics"
        )
    print(f"[runner] ultralytics from {loaded}", flush=True)
    return YOLO


def _train(job: dict) -> dict:
    YOLO = _import_fork(job["lib_root"])
    model = YOLO(job["model"])
    model.train(**job["train_args"])
    save_dir = Path(model.trainer.save_dir)
    return {"save_dir": str(save_dir), "best": str(save_dir / "weights" / "best.pt")}


def _read_first_frame(path: Path) -> list:
    """``TestLabels_FirstFrameOnly`` file (``frame,id,x,y,w,h,...``) -> rows ``[x1, y1, x2, y2, 1.0, 0.0]``."""
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            v = line.strip().split(",")
            if len(v) >= 6:
                x, y, w, h = (float(t) for t in v[2:6])
                rows.append([x, y, x + w, y + h, 1.0, 0.0])
    return rows


def _first_frame_callback(first_image: str, gt_rows: list):
    """``on_predict_postprocess_end`` callback that must run before the tracker's.

    On the sequence's first image (recognised by file name, the same order the
    loader reads: sorted glob) the detections are replaced by the given boxes
    with confidence 1. An empty file gives no boxes, so no track starts there.
    """
    import torch

    gt = torch.tensor(gt_rows, dtype=torch.float32).reshape(-1, 6)

    def callback(predictor) -> None:
        for result in predictor.results:
            if Path(result.path).name == first_image:
                result.update(boxes=gt.clone())

    return callback


def _track(job: dict) -> dict:
    YOLO = _import_fork(job["lib_root"])
    out_dir = Path(job["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    n_frames, seconds = 0, 0.0
    for seq_dir in sorted(Path(job["frames_root"]).iterdir()):
        if not seq_dir.is_dir():
            continue
        model = YOLO(job["weights"])  # fresh tracker state (and callbacks) per sequence
        if job.get("first_frame_dir"):
            images = sorted(glob.glob(os.path.join(str(seq_dir), "*.*")))  # the loader's own order
            if images:
                gt_rows = _read_first_frame(Path(job["first_frame_dir"]) / f"{seq_dir.name}.txt")
                # registered before model.track() adds the tracker's callback -> runs first
                model.add_callback("on_predict_postprocess_end",
                                   _first_frame_callback(Path(images[0]).name, gt_rows))
        rows = []
        t0 = time.perf_counter()
        for frame_idx, result in enumerate(
            model.track(source=str(seq_dir), stream=True, verbose=False, **job["track_args"]), start=1
        ):
            boxes = result.boxes
            if boxes is None or boxes.id is None:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            ids = boxes.id.cpu().numpy().astype(int)
            conf = boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), tid, s in zip(xyxy, ids, conf):
                rows.append(
                    f"{frame_idx},{tid},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},{s:.4f},-1,-1,-1"
                )
        seconds += time.perf_counter() - t0
        n_frames += len(list(seq_dir.glob("*.*")))
        (out_dir / f"{seq_dir.name}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
        print(f"[runner] {seq_dir.name}: {len(rows)} boxes", flush=True)
    return {"pred_dir": str(out_dir), "frames": n_frames, "seconds": seconds,
            "fps": n_frames / seconds if seconds else 0.0}


def _val(job: dict) -> dict:
    """``YOLO(weights).val(**val_args)``: precision, recall, mAP50, mAP50-95 on every val frame."""
    YOLO = _import_fork(job["lib_root"])
    metrics = YOLO(job["weights"]).val(**job["val_args"])
    return {"metrics": {k: float(v) for k, v in metrics.results_dict.items()},
            "speed_ms": {k: float(v) for k, v in getattr(metrics, "speed", {}).items()}}


def main() -> None:
    job = json.loads(Path(sys.argv[1]).read_text())
    os.chdir(job.get("cwd", os.getcwd()))
    result = {"train": _train, "track": _track, "val": _val}[job["mode"]](job)
    Path(job["result_file"]).write_text(json.dumps(result))


if __name__ == "__main__":
    main()
