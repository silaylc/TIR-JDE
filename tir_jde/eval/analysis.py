"""Error analyses behind the experiment decisions (see ``docs/experiments.md``).

- identity switches split into close-pair / long-gap / other (E1b, E5a)
- re-identification after long gaps (E5a)
- recall by target size (D1, D2)
- duplicate detections, i.e. pairs with centres closer than a few pixels (D2b)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tir_jde.eval.boxes import match_frame, nearest_other_gt, xywh2xyxy

SIZE_BINS = ((0, 8, "<8"), (8, 16, "8-16"), (16, 1e9, ">=16"))
CLOSE_PX = 12.7  # median nearest-pair distance in the EDA


def assignments(gt: np.ndarray, pr: np.ndarray, thr: float = 0.5) -> pd.DataFrame:
    """GT -> prediction matches per frame: columns ``f, gid, pid, nn_dist``."""
    rows = []
    for f in np.unique(gt[:, 0]).astype(int):
        g = gt[gt[:, 0] == f]
        p = pr[pr[:, 0] == f] if len(pr) else np.zeros((0, 10))
        nn = nearest_other_gt(g)
        pxy = xywh2xyxy(p[:, 2:6]) if len(p) else np.zeros((0, 4))
        for gi, pj, _ in match_frame(xywh2xyxy(g[:, 2:6]), pxy, thr):
            rows.append((int(f), int(g[gi, 1]), int(p[pj, 1]), float(nn[gi])))
    return pd.DataFrame(rows, columns=["f", "gid", "pid", "nn_dist"])


def idsw_groups(gt, pr, close_px: float = CLOSE_PX, long_gap: int = 10, thr: float = 0.5) -> pd.DataFrame:
    """Identity switches grouped as close pair (nn <= close_px), long gap (> long_gap frames) or other.

    The count approximates TrackEval's IDSW; official totals come from TrackEval.
    """
    A = assignments(gt, pr, thr)
    events = []
    for gid, g in A.sort_values("f").groupby("gid"):
        last_pid = last_f = None
        for r in g.itertuples():
            if last_pid is not None and r.pid != last_pid:
                gap = int(r.f - last_f - 1)
                group = "close pair" if r.nn_dist <= close_px else "long gap" if gap > long_gap else "other"
                events.append((int(r.f), int(gid), group, gap, float(r.nn_dist)))
            last_pid, last_f = r.pid, r.f
    return pd.DataFrame(events, columns=["f", "gid", "group", "gap", "nn_dist"])


def long_gap_returns(gt, pr, long_gap: int = 10, thr: float = 0.5) -> pd.DataFrame:
    """Every gap longer than ``long_gap`` frames in a GT track's matches, and whether the same id came back."""
    A = assignments(gt, pr, thr)
    rows = []
    for gid, g in A.sort_values("f").groupby("gid"):
        fs, ps = g.f.values, g.pid.values
        for k in range(1, len(fs)):
            gap = int(fs[k] - fs[k - 1] - 1)
            if gap > long_gap:
                rows.append((int(gid), gap, bool(ps[k] == ps[k - 1])))
    return pd.DataFrame(rows, columns=["gid", "gap", "same_id"])


def recall_by_size(gt, dets, score_min: float = 0.09, thr: float = 0.5, bins=SIZE_BINS) -> pd.DataFrame:
    """Detection recall per size bin (side = sqrt(w*h)); dets are ``frame, x1, y1, x2, y2, score``."""
    dets = dets[dets[:, 5] >= score_min] if len(dets) else dets
    side = np.sqrt(gt[:, 4] * gt[:, 5])
    hit = np.zeros(len(gt), bool)
    for f in np.unique(gt[:, 0]):
        gi = np.where(gt[:, 0] == f)[0]
        d = dets[dets[:, 0] == f] if len(dets) else np.zeros((0, 6))
        for a, _, _ in match_frame(xywh2xyxy(gt[gi, 2:6]), d[:, 1:5], thr):
            hit[gi[a]] = True
    rows = []
    for lo, hi, label in bins:
        m = (side >= lo) & (side < hi)
        rows.append({"bin": label, "n": int(m.sum()), "matched": int(hit[m].sum())})
    return pd.DataFrame(rows)


def duplicate_pairs(dets, dist_px: float = 3.0, score_min: float = 0.09) -> tuple[int, int]:
    """(number of detections, number of same-frame pairs whose centres are closer than ``dist_px``)."""
    dets = dets[dets[:, 5] >= score_min] if len(dets) else dets
    n_dup = 0
    for f in np.unique(dets[:, 0]) if len(dets) else []:
        b = dets[dets[:, 0] == f]
        if len(b) < 2:
            continue
        c = np.c_[(b[:, 1] + b[:, 3]) / 2, (b[:, 2] + b[:, 4]) / 2]
        D = np.linalg.norm(c[:, None] - c[None], axis=-1)
        n_dup += int(np.triu(D < dist_px, k=1).sum())
    return len(dets), n_dup
