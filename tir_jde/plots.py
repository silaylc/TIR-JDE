"""Visual outputs: tracking videos, per-sequence scores and model comparisons."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _color(track_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(int(track_id) * 7919)
    return tuple(int(c) for c in rng.integers(80, 255, 3))


def draw_tracks(frames_dir: Path, pred_file: Path, out_file: Path,
                gt_file: Path | None = None, max_frames: int | None = 300, fps: int = 25) -> Path:
    """Draws predicted boxes and ids (and thin ground-truth boxes) into an MP4."""
    import cv2

    from tir_jde.dataset import read_mot

    frames = sorted(p for p in Path(frames_dir).iterdir() if p.suffix.lower() in {".jpg", ".png", ".bmp"})
    if max_frames:
        frames = frames[:max_frames]
    pred = read_mot(pred_file) if Path(pred_file).exists() else np.zeros((0, 9))
    gt = read_mot(gt_file) if gt_file and Path(gt_file).exists() else np.zeros((0, 9))
    h, w = cv2.imread(str(frames[0])).shape[:2]
    out_file.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_file), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for idx, frame_path in enumerate(frames, start=1):
        img = cv2.imread(str(frame_path))
        for _, _, x, y, bw, bh, *_ in gt[gt[:, 0] == idx]:
            cv2.rectangle(img, (int(x), int(y)), (int(x + bw), int(y + bh)), (140, 140, 140), 1)
        for _, tid, x, y, bw, bh, *_ in pred[pred[:, 0] == idx]:
            c = _color(int(tid))
            cv2.rectangle(img, (int(x), int(y)), (int(x + bw), int(y + bh)), c, 1)
            cv2.putText(img, str(int(tid)), (int(x), max(int(y) - 3, 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, c, 1, cv2.LINE_AA)
        cv2.putText(img, f"{frame_path.parent.name}  #{idx}", (6, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(img)
    writer.release()
    return out_file


def per_sequence_chart(per_sequence: dict[str, dict], metric: str, out_file: Path, title: str = "") -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seqs = sorted(per_sequence, key=lambda s: per_sequence[s].get(metric, 0.0))
    values = [per_sequence[s].get(metric, 0.0) for s in seqs]
    fig, ax = plt.subplots(figsize=(6, max(2.5, 0.28 * len(seqs))))
    ax.barh(seqs, values, color="#4C78A8")
    ax.set_xlabel(metric)
    ax.set_xlim(0, max(1.0, max(values) if values else 1.0))
    ax.set_title(title or f"{metric} per sequence")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    return out_file


def comparison_chart(table: dict[str, dict[str, tuple[float, float]]], metrics: list[str], out_file: Path) -> Path:
    """Grouped bars (mean ± std over seeds) for several models."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = list(table)
    x = np.arange(len(metrics))
    width = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(1.6 * len(metrics) + 2, 3.6))
    for i, m in enumerate(models):
        means = [table[m].get(k, (np.nan, 0))[0] for k in metrics]
        stds = [table[m].get(k, (np.nan, 0))[1] for k in metrics]
        ax.bar(x + i * width - 0.4 + width / 2, means, width, yerr=stds, capsize=3, label=m)
    ax.set_xticks(x, metrics)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    return out_file
