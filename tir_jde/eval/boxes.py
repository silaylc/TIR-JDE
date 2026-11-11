"""Box I/O and matching helpers shared by the analyses.

Formats (frames start at 1, ``x, y`` is the top-left corner):

- MOT ground truth / tracker output: ``frame, id, x, y, w, h, conf, cls, vis`` (or ``..., score, -1, -1, -1``)
- dumped detections: ``frame, x1, y1, x2, y2, score``
"""

from __future__ import annotations

import os

import numpy as np
from scipy.optimize import linear_sum_assignment


def load_mot(path) -> np.ndarray:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return np.zeros((0, 10))
    return np.loadtxt(path, delimiter=",", ndmin=2)


def load_dets(path) -> np.ndarray:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return np.zeros((0, 6))
    return np.loadtxt(path, delimiter=",", ndmin=2)


def xywh2xyxy(b) -> np.ndarray:
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    return np.stack([b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]], 1)


def iou_matrix(a, b) -> np.ndarray:
    """(n, 4) x (m, 4) xyxy boxes -> (n, m) IoU."""
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match_frame(gt_xyxy, pr_xyxy, thr: float = 0.5):
    """Hungarian matching on IoU within one frame. Returns ``[(gt_idx, pred_idx, iou)]`` with IoU >= thr."""
    M = iou_matrix(gt_xyxy, pr_xyxy)
    if M.size == 0:
        return []
    r, c = linear_sum_assignment(-M)
    return [(int(i), int(j), float(M[i, j])) for i, j in zip(r, c) if M[i, j] >= thr]


def centers(a) -> np.ndarray:
    """Centres of MOT rows (columns 2..5 are x, y, w, h)."""
    a = np.asarray(a, float)
    return np.c_[a[:, 2] + a[:, 4] / 2, a[:, 3] + a[:, 5] / 2] if len(a) else np.zeros((0, 2))


def nearest_other_gt(g) -> np.ndarray:
    """Per GT box: distance (px) to the nearest other GT centre in the same frame; inf if alone."""
    c = centers(g)
    if len(c) < 2:
        return np.full(len(c), np.inf)
    D = np.linalg.norm(c[:, None] - c[None], axis=-1)
    np.fill_diagonal(D, np.inf)
    return D.min(1)
