"""Post-processing of dumped detections.

``center_nms`` was evaluated in D2b and **not adopted**: it removes the
duplicate boxes of the P2 detector but does not change HOTA and costs some
small-target recall. It is kept for reproducibility and is off by default.
"""

from __future__ import annotations

import numpy as np


def center_nms(dets: np.ndarray, tau: float = 3.0) -> np.ndarray:
    """Per frame, drop lower-scored boxes whose centre is closer than ``tau`` px to a kept box.

    ``dets`` rows are ``frame, x1, y1, x2, y2, score``. The order and values of the
    kept rows are unchanged; ties in score keep the earlier row.
    """
    dets = np.asarray(dets, float)
    if len(dets) == 0:
        return dets.reshape(0, 6)
    keep_all = np.zeros(len(dets), bool)
    for f in np.unique(dets[:, 0]):
        idx = np.where(dets[:, 0] == f)[0]
        b = dets[idx]
        order = np.argsort(-b[:, 5], kind="stable")
        c = np.c_[(b[order, 1] + b[order, 3]) / 2, (b[order, 2] + b[order, 4]) / 2]
        keep = np.ones(len(order), bool)
        for i in range(len(order)):
            if keep[i]:
                d = np.hypot(c[i + 1:, 0] - c[i, 0], c[i + 1:, 1] - c[i, 1])
                keep[i + 1:] &= ~(d < tau)
        keep_all[idx[order[keep]]] = True
    return dets[keep_all]
