"""Checks that a tracker followed the BSB protocol.

BSB gives every test sequence its first-frame boxes; all compared methods start
their tracker from them. The first frame of a method's output must therefore
reproduce those boxes. ``first_frame_check`` measures that on the validation
sequences, so every evaluation proves (or flags) protocol compliance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _frame_boxes(path: Path, frame: int = 1) -> np.ndarray:
    """``x, y, w, h`` of one frame from a MOTChallenge file (empty if missing)."""
    if not path.exists() or path.stat().st_size == 0:
        return np.zeros((0, 4))
    data = np.loadtxt(path, delimiter=",", ndmin=2)
    if data.size == 0:
        return np.zeros((0, 4))
    return data[data[:, 0] == frame][:, 2:6]


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2) - np.maximum(a[:, 0, None], b[:, 0]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2) - np.maximum(a[:, 1, None], b[:, 1]), 0, None)
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + b[:, 2] * b[:, 3] - inter
    return inter / np.maximum(union, 1e-9)


def first_frame_check(gt_dir: Path, pred_dir: Path, seqs: list[str], min_iou: float = 0.99) -> dict:
    """How well frame 1 of the predictions reproduces frame 1 of the ground truth.

    Returns ``match`` (share of first-frame GT boxes found with IoU >= ``min_iou``;
    1.0 when there are none), ``extra`` (predicted first-frame boxes without a GT
    match; should be 0) and the sequences that do not comply.
    """
    total = matched = extra = 0
    bad: list[str] = []
    for seq in seqs:
        gt = _frame_boxes(Path(gt_dir) / f"{seq}.txt")
        pred = _frame_boxes(Path(pred_dir) / f"{seq}.txt")
        hit = 0
        if len(gt) and len(pred):
            from scipy.optimize import linear_sum_assignment

            iou = _iou(gt, pred)
            rows, cols = linear_sum_assignment(-iou)
            hit = int((iou[rows, cols] >= min_iou).sum())
        total += len(gt)
        matched += hit
        extra += len(pred) - hit
        if hit < len(gt) or len(pred) > hit:
            bad.append(seq)
    return {"match": matched / total if total else 1.0, "extra": int(extra), "gt_boxes": int(total),
            "failed_sequences": bad}
