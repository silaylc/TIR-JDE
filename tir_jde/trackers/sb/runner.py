"""Runs ``track.py`` with Strong Baseline's BoT-SORT patched in place (no edits to its source).

Patches, all off unless requested:

- **Cost** (``--cost_mode nwd``, forwarded to track.py): NWD instead of IoU (``costs.nwd_cost``).
- **Coasting** (``--coast_k K``): a lost track is still written for up to K frames after its last match,
  at its Kalman prediction.
- **Kalman noise scale** (``--kf_q``, ``--kf_r``): multiply the process / measurement noise std.
- **Gate** (env ``PATCHES=gate``): Mahalanobis chi2 (95 %, df 4) gate for lost tracks.
  ``PATCHES=gate,conf`` adds exponential confidence decay ``exp(-dt / GATE_TAU)``.
  Note (E5a): combined with the NWD threshold this gate can only *tighten* association.
- **Appearance fusion** (env ``FUSE_W=w`` together with ``--emb_dir``): replaces BoT-SORT's
  ``min(position, appearance)`` with ``(1 - w) * position + w * appearance``.

Environment: ``SB_BOTSORT`` = path to Strong Baseline's ``BoT-SORT`` folder (required);
``TRACK_SCRIPT`` = tracking script (default: ``track.py`` next to this file).
Prints ``PATCH_STATS {...}`` at the end; the counters show which patches actually ran.
"""
import importlib.util
import inspect
import json
import os
from pathlib import Path
import runpy
import sys
import textwrap
import types

import numpy as np

HERE = Path(__file__).resolve().parent
SB = os.environ["SB_BOTSORT"]
TRACK = os.environ.get("TRACK_SCRIPT", str(HERE / "track.py"))
sys.path.insert(0, SB)

_spec = importlib.util.spec_from_file_location("tir_jde_sb_costs", HERE / "costs.py")
costs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(costs)


# ---- Stubs so the tracker imports without FastReID / cython_bbox -----------------------------------
class _NoReID:
    def __init__(self, *a, **k):
        pass

    def inference(self, img, dets):
        raise RuntimeError("FastReID is disabled: with_reid must be False or --emb_dir given")


_pkg = types.ModuleType("fast_reid")
_mod = types.ModuleType("fast_reid.fast_reid_interfece")
_mod.FastReIDInterface = _NoReID
_pkg.fast_reid_interfece = _mod
sys.modules["fast_reid"] = _pkg
sys.modules["fast_reid.fast_reid_interfece"] = _mod
try:
    import cython_bbox  # noqa: F401
except ImportError:
    def _bbox_overlaps(a, b):
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        iw = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]) + 1, 0, None)
        ih = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]) + 1, 0, None)
        inter = iw * ih
        aa = ((a[:, 2] - a[:, 0] + 1) * (a[:, 3] - a[:, 1] + 1))[:, None]
        bb = ((b[:, 2] - b[:, 0] + 1) * (b[:, 3] - b[:, 1] + 1))[None, :]
        return inter / (aa + bb - inter)
    _cb = types.ModuleType("cython_bbox")
    _cb.bbox_overlaps = _bbox_overlaps
    sys.modules["cython_bbox"] = _cb

import tracker.kalman_filter as KFM  # noqa: E402
import tracker.matching as M  # noqa: E402
import tracker.mc_bot_sort as MB  # noqa: E402

STATS = {"gate_calls": 0, "gated_pairs": 0, "rejected_pairs": 0, "conf_pairs": 0, "emb_calls": 0, "fuse_calls": 0}
STATE = {"frame": 0}

# ---- Coasting ---------------------------------------------------------------------------------------
_update_orig = MB.BoTSORT.update


def _update_with_coasting(self, *a, **k):
    STATE["frame"] = self.frame_id + 1  # update() increments frame_id
    out, lost = _update_orig(self, *a, **k)
    K = int(getattr(self, "coast_k", 0) or 0)
    if K > 0:
        out = list(out) + [t for t in lost if 1 <= self.frame_id - int(t.frame_id) <= K]
    return out, lost


MB.BoTSORT.update = _update_with_coasting

# ---- Kalman noise scale (track.py sets KF_SCALE from --kf_q / --kf_r; 1.0 = original) -------------
KFM.KF_SCALE = {"Q": 1.0, "R": 1.0}
_KF = KFM.KalmanFilter
_predict0, _project0, _multi_predict0 = _KF.predict, _KF.project, _KF.multi_predict


def _predict(self, mean, covariance):
    m, c = _predict0(self, mean, covariance)
    q = KFM.KF_SCALE["Q"]
    if q != 1.0:
        wp, wv = self._std_weight_position, self._std_weight_velocity
        s = np.r_[wp * mean[2], wp * mean[3], wp * mean[2], wp * mean[3], wv * mean[2], wv * mean[3], wv * mean[2], wv * mean[3]]
        c = c + (q * q - 1.0) * np.diag(np.square(s))
    return m, c


def _multi_predict(self, mean, covariance):
    m, c = _multi_predict0(self, mean, covariance)
    q = KFM.KF_SCALE["Q"]
    if q != 1.0:
        wp, wv = self._std_weight_position, self._std_weight_velocity
        s = np.stack([wp * mean[:, 2], wp * mean[:, 3], wp * mean[:, 2], wp * mean[:, 3],
                      wv * mean[:, 2], wv * mean[:, 3], wv * mean[:, 2], wv * mean[:, 3]], axis=1)
        c = c + (q * q - 1.0) * (np.square(s)[:, :, None] * np.eye(8)[None])
    return m, c


def _project(self, mean, covariance):
    m, c = _project0(self, mean, covariance)
    r = KFM.KF_SCALE["R"]
    if r != 1.0:
        wp = self._std_weight_position
        s = np.r_[wp * mean[2], wp * mean[3], wp * mean[2], wp * mean[3]]
        c = c + (r * r - 1.0) * np.diag(np.square(s))
    return m, c


_KF.predict, _KF.project, _KF.multi_predict = _predict, _project, _multi_predict

# ---- Position cost: IoU or NWD (track.py sets COST["mode"] / COST["C"]) -----------------------------
M.COST = {"mode": "iou", "C": costs.NWD_C}
_iou_distance = M.iou_distance


def _position_cost(a, b):
    if M.COST["mode"] == "nwd":
        return costs.nwd_cost(a, b, M.COST["C"])
    return _iou_distance(a, b)


M.iou_distance = _position_cost

# ---- Gate and confidence decay for lost tracks -------------------------------------------------------
MODE = {x for x in os.environ.get("PATCHES", "").split(",") if x}
TAU = float(os.environ.get("GATE_TAU", "50"))
MIN_MISSED = int(os.environ.get("GATE_MIN_MISSED", "0"))
CHI2_95_DF4 = 9.4877


def _gated_cost(a, b):
    cost = _position_cost(a, b)
    if not (MODE & {"gate", "conf"}) or len(a) == 0 or len(b) == 0 or isinstance(a[0], np.ndarray):
        return cost
    STATS["gate_calls"] += 1
    dets = np.array([[d.tlwh[0] + d.tlwh[2] / 2, d.tlwh[1] + d.tlwh[3] / 2, d.tlwh[2], d.tlwh[3]] for d in b], float)
    for i, t in enumerate(a):
        missed = STATE["frame"] - int(t.frame_id) - 1
        if missed <= 0 or missed < MIN_MISSED:
            continue
        STATS["gated_pairs"] += len(b)
        if "gate" in MODE:
            m, S = t.kalman_filter.project(t.mean, t.covariance)
            r = dets - np.asarray(m, float)[None, :]
            bad = np.einsum("ij,jk,ik->i", r, np.linalg.inv(S), r) > CHI2_95_DF4
            STATS["rejected_pairs"] += int(bad.sum())
            cost[i, bad] = 1.0
        if "conf" in MODE:
            ok = cost[i] < 1.0
            cost[i, ok] = 1 - np.exp(-missed / TAU) * (1 - cost[i, ok])
            STATS["conf_pairs"] += int(ok.sum())
    return cost


if MODE & {"gate", "conf"}:
    M.iou_distance = _gated_cost

# ---- Appearance: our embeddings instead of FastReID, optional weighted fusion -----------------------
class _ReIDProxy:
    """mc_bot_sort was imported with the stub above; track.py installs its embedding bank afterwards.
    The proxy resolves the current interface when BoTSORT is constructed."""

    def __init__(self, *a, **k):
        self._impl = sys.modules["fast_reid.fast_reid_interfece"].FastReIDInterface(*a, **k)

    def inference(self, img, dets):
        return self._impl.inference(img, dets)


MB.FastReIDInterface = _ReIDProxy
_embedding_distance = M.embedding_distance


def _counted_embedding_distance(*a, **k):
    STATS["emb_calls"] += 1
    return _embedding_distance(*a, **k)


M.embedding_distance = _counted_embedding_distance

if os.environ.get("FUSE_W"):
    MB._FUSE_W = float(os.environ["FUSE_W"])

    def _fuse(pos, emb):
        STATS["fuse_calls"] += 1
        return (1.0 - MB._FUSE_W) * pos + MB._FUSE_W * emb

    MB._fuse = _fuse
    _src = textwrap.dedent(inspect.getsource(_update_orig))
    _line = "dists = np.minimum(ious_dists, emb_dists)"
    if _src.count(_line) != 1:
        raise RuntimeError("fusion line not found in BoTSORT.update")
    _ns = {}
    exec(compile(_src.replace(_line, "dists = _fuse(ious_dists, emb_dists)"), MB.__file__, "exec"), MB.__dict__, _ns)
    _update_orig = _ns["update"]  # picked up by _update_with_coasting at call time

sys.argv = [TRACK] + sys.argv[1:]
try:
    runpy.run_path(TRACK, run_name="__main__")
finally:
    print("PATCH_STATS", json.dumps(STATS))
