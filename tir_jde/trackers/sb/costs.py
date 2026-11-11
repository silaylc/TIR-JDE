"""Association costs for tiny boxes (numpy only; no tracker import needed).

NWD (Normalized Wasserstein Distance, Wang et al. 2021) models a box as a 2-D Gaussian, so two
4-pixel boxes that are one pixel apart still get a graded similarity, whereas their IoU collapses to
zero. Used as a drop-in replacement for BoT-SORT's IoU distance (E1: HOTA +0.067 on val).
"""

from __future__ import annotations

import numpy as np

NWD_C = 15.0  # normalising constant in pixels (chosen before E1)


def to_cwh(x) -> list[float]:
    """Centre-width-height of an ``x1, y1, x2, y2`` array or of an object exposing ``tlwh``."""
    if isinstance(x, np.ndarray):
        x1, y1, x2, y2 = x[:4]
        return [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]
    t = x.tlwh
    return [t[0] + t[2] / 2, t[1] + t[3] / 2, t[2], t[3]]


def nwd_cost(a, b, C: float = NWD_C) -> np.ndarray:
    """Cost matrix ``1 - exp(-W2 / C)`` between two lists of boxes/tracks (0 = identical)."""
    A = np.array([to_cwh(t) for t in a], float).reshape(-1, 4)
    B = np.array([to_cwh(d) for d in b], float).reshape(-1, 4)
    if len(A) == 0 or len(B) == 0:
        return np.zeros((len(A), len(B)))
    d2 = ((A[:, None, :2] - B[None, :, :2]) ** 2).sum(-1) + (((A[:, None, 2:] - B[None, :, 2:]) / 2) ** 2).sum(-1)
    return 1.0 - np.exp(-np.sqrt(d2) / C)
