"""Python entry point for ``runner.py``: one call = one tracker run over a set of sequences."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent

# Frozen tracker configuration (E2): sparseOptFlow camera motion compensation + NWD cost + coasting K=10.
FROZEN = {"cmc": "sparseOptFlow", "cost_mode": "nwd", "match_thresh": 0.8, "coast_k": 10}


def make_data_dir(dets_dir, gt_dir, seqs, root=None) -> Path:
    """The layout track.py expects: ``dets/``, ``gt/`` (symlinks) and ``val.txt``."""
    d = Path(tempfile.mkdtemp(prefix="sbdata_", dir=root))
    (d / "dets").symlink_to(Path(dets_dir).resolve())
    (d / "gt").symlink_to(Path(gt_dir).resolve())
    (d / "val.txt").write_text("\n".join(seqs) + "\n")
    return d


def run_tracker(dets_dir, gt_dir, images, seqs, out_dir, sb_botsort, *, cmc="sparseOptFlow", cost_mode="nwd",
                nwd_c=15.0, match_thresh=0.8, coast_k=10, patches="", gate_tau=50, gate_min_missed=0,
                emb_dir=None, fuse_w=None, extra=()) -> dict:
    """Runs Strong Baseline's BoT-SORT with the requested patches and returns the patch counters.

    ``emb_dir`` turns on appearance (our vectors); ``fuse_w`` replaces BoT-SORT's min fusion with
    ``(1 - w) * position + w * appearance``. Raises if a requested patch did not run.
    """
    if fuse_w is not None and emb_dir is None:
        raise ValueError("fuse_w needs emb_dir")
    data = make_data_dir(dets_dir, gt_dir, seqs)
    cmd = [sys.executable, str(HERE / "runner.py"), "--bot_root", str(sb_botsort), "--data_dir", str(data),
           "--images", str(images), "--out", str(out_dir), "--cmc", cmc, "--cost_mode", cost_mode,
           "--nwd_C", str(nwd_c), "--match_thresh", str(match_thresh), "--coast_k", str(coast_k),
           "--seqs", ",".join(seqs), *map(str, extra)]
    if emb_dir is not None:
        cmd += ["--emb_dir", str(emb_dir)]
    env = {**os.environ, "SB_BOTSORT": str(sb_botsort), "PATCHES": patches,
           "GATE_TAU": str(gate_tau), "GATE_MIN_MISSED": str(gate_min_missed)}
    env.pop("FUSE_W", None)
    if fuse_w is not None:
        env["FUSE_W"] = str(fuse_w)
    try:
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    finally:
        shutil.rmtree(data, ignore_errors=True)
    if r.returncode != 0:
        raise RuntimeError(f"tracker failed:\n{r.stderr[-3000:]}")
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("PATCH_STATS")]
    stats = json.loads(lines[-1].split(" ", 1)[1]) if lines else {}
    if {"gate", "conf"} & set(patches.split(",")) and stats.get("gate_calls", 0) == 0:
        raise RuntimeError(f"gate patch never ran: {stats}")
    if emb_dir is not None and stats.get("emb_calls", 0) == 0:
        raise RuntimeError(f"appearance was not used: {stats}")
    if (fuse_w is not None) != (stats.get("fuse_calls", 0) > 0):
        raise RuntimeError(f"fusion patch did not behave as requested: {stats}")
    return stats
