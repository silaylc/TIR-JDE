"""One evaluator for every model: TrackEval (HOTA, CLEAR, Identity) on the same split.

Inputs are plain MOTChallenge text files, one per sequence::

    gt_dir/<seq>.txt     frame, id, x, y, w, h, 1, 1, 1.0   (+ <seq>.len with the frame count)
    pred_dir/<seq>.txt   frame, id, x, y, w, h, score, -1, -1, -1

They are laid out the way TrackEval's MotChallenge2DBox reader expects in a
temporary folder, so models never need to know about TrackEval.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

_COMBINED = "COMBINED_SEQ"
_CLASS = "pedestrian"  # TrackEval's MOT class id 1; our UAV labels use class 1


def _numpy_compat() -> None:
    """TrackEval still uses aliases removed in NumPy 1.24 (np.float, np.int, np.bool)."""
    for name, typ in (("float", float), ("int", int), ("bool", bool)):
        if name not in np.__dict__:
            setattr(np, name, typ)


def _seq_length(gt_dir: Path, seq: str) -> int:
    length_file = gt_dir / f"{seq}.len"
    if length_file.exists():
        return int(length_file.read_text().strip())
    frames = np.loadtxt(gt_dir / f"{seq}.txt", delimiter=",", ndmin=2)[:, 0]
    return int(frames.max()) if len(frames) else 1


def _layout(gt_dir: Path, pred_dir: Path, seqs: list[str], tracker: str, root: Path) -> dict:
    gt_root, trk_root = root / "gt", root / "trackers"
    for seq in seqs:
        d = gt_root / seq
        (d / "gt").mkdir(parents=True, exist_ok=True)
        (d / "gt" / "gt.txt").write_text((gt_dir / f"{seq}.txt").read_text())
        (d / "seqinfo.ini").write_text(
            f"[Sequence]\nname={seq}\nseqLength={_seq_length(gt_dir, seq)}\n"
        )
        pred = pred_dir / f"{seq}.txt"
        out = trk_root / tracker / "data" / f"{seq}.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(pred.read_text() if pred.exists() else "")  # missing = no detections
    seqmap = root / "seqmap.txt"
    seqmap.write_text("name\n" + "\n".join(seqs) + "\n")
    return {"gt": gt_root, "trackers": trk_root, "seqmap": seqmap}


def evaluate(gt_dir: str | Path, pred_dir: str | Path, seqs: list[str] | None = None,
             tracker: str = "model", metrics: tuple[str, ...] = ("HOTA", "CLEAR", "Identity")) -> dict:
    """Returns overall and per-sequence tracking metrics.

    Overall keys: HOTA, DetA, AssA, LocA, MOTA, MOTP, IDF1, IDP, IDR, IDSW, FP, FN, Frag.
    """
    _numpy_compat()
    import trackeval

    gt_dir, pred_dir = Path(gt_dir), Path(pred_dir)
    seqs = sorted(seqs or [p.stem for p in gt_dir.glob("*.txt")])
    if not seqs:
        raise ValueError(f"No ground-truth files in {gt_dir}")

    with tempfile.TemporaryDirectory() as tmp:
        paths = _layout(gt_dir, pred_dir, seqs, tracker, Path(tmp))
        eval_cfg = trackeval.Evaluator.get_default_eval_config()
        eval_cfg.update(PRINT_RESULTS=False, PRINT_CONFIG=False, TIME_PROGRESS=False,
                        OUTPUT_SUMMARY=False, OUTPUT_DETAILED=False, PLOT_CURVES=False,
                        USE_PARALLEL=False, DISPLAY_LESS_PROGRESS=True)
        ds_cfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
        ds_cfg.update(
            GT_FOLDER=str(paths["gt"]), TRACKERS_FOLDER=str(paths["trackers"]),
            TRACKERS_TO_EVAL=[tracker], SEQMAP_FILE=str(paths["seqmap"]), SKIP_SPLIT_FOL=True,
            TRACKER_SUB_FOLDER="data", OUTPUT_FOLDER=str(Path(tmp) / "out"), CLASSES_TO_EVAL=[_CLASS],
            BENCHMARK="UAV", SPLIT_TO_EVAL="val", DO_PREPROC=False, PRINT_CONFIG=False,
            GT_LOC_FORMAT="{gt_folder}/{seq}/gt/gt.txt",
        )
        metric_objs = [getattr(trackeval.metrics, m)({"PRINT_CONFIG": False}) for m in metrics]
        evaluator = trackeval.Evaluator(eval_cfg)
        res, msg = evaluator.evaluate([trackeval.datasets.MotChallenge2DBox(ds_cfg)], metric_objs)

    status = msg["MotChallenge2DBox"][tracker]
    if status != "Success":
        raise RuntimeError(f"TrackEval failed: {status}")
    per_seq = res["MotChallenge2DBox"][tracker]
    return {
        "overall": _summarise(per_seq[_COMBINED][_CLASS]),
        "per_sequence": {s: _summarise(per_seq[s][_CLASS]) for s in seqs},
    }


def _summarise(r: dict) -> dict:
    out = {}
    if "HOTA" in r:
        for k in ("HOTA", "DetA", "AssA", "LocA"):
            out[k] = float(np.mean(r["HOTA"][k]))
    if "CLEAR" in r:
        for k in ("MOTA", "MOTP"):
            out[k] = float(r["CLEAR"][k])
        for k, name in (("IDSW", "IDSW"), ("CLR_FP", "FP"), ("CLR_FN", "FN"), ("Frag", "Frag")):
            out[name] = int(r["CLEAR"][k])
    if "Identity" in r:
        for k in ("IDF1", "IDP", "IDR"):
            out[k] = float(r["Identity"][k])
    return out
