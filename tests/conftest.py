"""Synthetic BSB-like dataset for pipeline tests (no real data, GPU or baseline)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

SEQS = {"C_01": 12, "C_02": 10, "T_01": 8, "T_02": 9, "BB2P_01": 7}


def _write_sequence(root: Path, seq: str, n_frames: int, n_targets: int = 3) -> None:
    import cv2

    frames = root / "frames" / seq
    frames.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in range(1, n_frames + 1):
        img = np.full((64, 96, 3), 40, np.uint8)
        for k in range(n_targets):
            x, y = 5 + 25 * k + f, 10 + 12 * k
            img[y:y + 6, x:x + 8] = 220
            rows.append(f"{f},{k + 1},{x},{y},8,6,1,1,1.0")
        cv2.imwrite(str(frames / f"{f:05d}.jpg"), img)
    (root / "labels").mkdir(exist_ok=True)
    (root / "labels" / f"{seq}.txt").write_text("\n".join(rows) + "\n")


@pytest.fixture
def synthetic_cfg(tmp_path: Path) -> dict:
    """A full experiment config pointing at a tiny synthetic dataset."""
    from tir_jde.experiment import load_config

    raw = tmp_path / "raw"
    for seq, n in SEQS.items():
        _write_sequence(raw, seq, n)
    splits = tmp_path / "splits"
    splits.mkdir()
    (splits / "train.txt").write_text("\n".join(SEQS) + "\n")
    (splits / "val.txt").write_text("C_02\nT_02\n")
    return load_config(
        "strong_baseline_bsb",
        [
            f"paths.work_dir={tmp_path / 'work'}",
            f"paths.baselines_dir={tmp_path / 'baselines'}",
            f"data.raw_root={raw}",
            "data.frames_dir=frames",
            "data.labels_dir=labels",
            f"data.splits.train={splits / 'train.txt'}",
            f"data.splits.val={splits / 'val.txt'}",
            "data.reid.first_fraction=1.0",
        ],
    )
