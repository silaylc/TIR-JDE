"""Dataset loading, splits and conversion into the formats each model expects.

Raw input (per sequence ``seq``):
- frames: either a video ``<raw_root>/<videos_dir>/<seq><video_ext>`` or a folder
  ``<raw_root>/<frames_dir>/<seq>/00001.jpg``;
- labels: ``<raw_root>/<labels_dir>/<seq>.txt`` in MOTChallenge format
  ``frame, id, x_left, y_top, w, h, conf, class, visibility`` (frames start at 1).

Outputs under ``<data_dir>/<dataset>-<split_hash>/`` (``data_dir`` defaults to
``<work_dir>/data``):

- ``frames/<seq>/00001.jpg``                    extracted frames (shared cache)
- ``yolo/{train,val}/{images,labels}/``         detector training data; file names
  keep the sequence (``<seq>_00001.jpg``), unlike a flat frame-level dump
- ``yolo/dataset.yaml``                         Ultralytics dataset file
- ``reid/MOT20/bounding_box_{train,test}/``     ReID crops in the folder and file-name
  layout Strong Baseline's FastReID loader parses (``<gid>_<seq>_0_<frame>_acc_data.bmp``);
  train = train sequences, test = validation sequences, so no identity leaks
- ``track/val/Test_imgs/<seq>/`` + ``TestLabels_FirstFrameOnly/<seq>.txt``
                                                tracking input in the Track 3 layout
- ``gt/val/<seq>.txt``                          ground truth used by the evaluator
- ``manifest.json``                             sequences, counts and the split hash
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tir_jde.experiment import resolve_path


# --------------------------------------------------------------------------- labels
@dataclass
class Sequence:
    name: str
    labels: np.ndarray  # (N, 9): frame, id, x, y, w, h, conf, class, vis


def read_mot(path: Path) -> np.ndarray:
    """Reads a MOTChallenge label file; returns an (N, 9) float array."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().replace(" ", "").split(",")
            if len(parts) < 6 or not parts[0]:
                continue
            values = [float(v) for v in parts[:9]]
            values += [1.0] * (9 - len(values))
            rows.append(values)
    return np.asarray(rows, dtype=float).reshape(-1, 9)


def write_mot(path: Path, rows: np.ndarray, gt: bool = False) -> None:
    """Writes MOTChallenge rows (frame, id, x, y, w, h, ...)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            if gt:
                f.write(f"{int(r[0])},{int(r[1])},{r[2]:.2f},{r[3]:.2f},{r[4]:.2f},{r[5]:.2f},1,1,1.0\n")
            else:
                f.write(f"{int(r[0])},{int(r[1])},{r[2]:.2f},{r[3]:.2f},{r[4]:.2f},{r[5]:.2f},{r[6]:.4f},-1,-1,-1\n")


# --------------------------------------------------------------------------- splits
def read_split(path: str | Path) -> list[str]:
    p = resolve_path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"Split list not found: {p}. Create it under data/splits/ (one sequence name per line)."
        )
    names = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()]
    return [n for n in names if n and not n.startswith("#")]


def resolve_splits(data_cfg: dict) -> dict[str, list[str]]:
    """Train = BSB train minus the validation group and the excluded sequences."""
    train_all = read_split(data_cfg["splits"]["train"])
    val = read_split(data_cfg["splits"]["val"])
    excluded = set(data_cfg.get("exclude") or [])
    unknown = sorted(set(val) - set(train_all))
    if unknown:
        raise ValueError(f"Validation sequences not in the training list: {unknown}")
    if excluded & set(val):
        raise ValueError(f"Excluded sequences in the validation group: {sorted(excluded & set(val))}")
    train = [s for s in train_all if s not in set(val) and s not in excluded]
    return {"train": train, "val": val}


def read_scene_groups(path: str | Path | None) -> dict[str, str]:
    """Sequence -> scene group from a CSV with ``bsb`` and ``scene_group`` columns."""
    if not path:
        return {}
    import csv
    p = resolve_path(path)
    rows = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln and not ln.startswith("#")]
    return {r["bsb"]: str(r["scene_group"]) for r in csv.DictReader(rows)}


def split_hash(splits: dict[str, list[str]], data_cfg: dict) -> str:
    """Identifies the exact data a model saw: sequence lists + conversion settings."""
    payload = {
        "dataset": data_cfg.get("name"),
        "splits": {k: sorted(v) for k, v in splits.items()},
        "reid": data_cfg.get("reid", {}),
        "image_ext": data_cfg.get("image_ext", ".jpg"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def family_of(seq: str) -> str | None:
    """BSB sequence names look like ``<family>_<number>`` (e.g. ``BB2P_02``)."""
    m = re.fullmatch(r"(.+)_(\d+)", seq)
    return m.group(1) if m else None


def make_val_split(
    train: list[str], families: dict[str, str], per_family: int, seed: int,
    fraction: float | None = None, groups: dict[str, str] | None = None,
) -> list[str]:
    """Picks validation sequences from every family (scene type).

    - ``fraction`` given: ``round(fraction * family size)`` sequences per family,
      at least one, so the validation mix follows the family proportions of the
      training data;
    - otherwise: ``per_family`` sequences from every family.

    Every family always keeps at least one sequence for training. ``families``
    maps sequence name -> family; sequences without a family are never chosen.
    ``groups`` (sequence -> scene group) keeps near-identical recordings
    together: a group goes to validation or to training as a whole.
    Deterministic for a given seed.
    """
    rng = np.random.default_rng(seed)
    by_family: dict[str, list[str]] = {}
    for seq in sorted(train):
        if seq in families:
            by_family.setdefault(families[seq], []).append(seq)
    val: list[str] = []
    for fam in sorted(by_family):
        seqs = by_family[fam]
        want = max(1, round(fraction * len(seqs))) if fraction is not None else per_family
        k = min(want, max(len(seqs) - 1, 0))  # always leave one for training
        if not groups:
            val.extend(sorted(rng.choice(seqs, size=k, replace=False).tolist()) if k else [])
            continue
        units: dict[str, list[str]] = {}
        for seq in seqs:  # sequences without a group are their own unit
            units.setdefault(groups.get(seq, f"seq:{seq}"), []).append(seq)
        order = [sorted(units)[i] for i in rng.permutation(len(units))]
        chosen: list[str] = []
        for u in order:
            if len(chosen) >= k:
                break
            if len(chosen) + len(units[u]) <= k and len(seqs) - len(chosen) - len(units[u]) >= 1:
                chosen += units[u]
        if not chosen and k:  # every unit larger than the quota: take the smallest that leaves training data
            fits = [u for u in order if len(seqs) - len(units[u]) >= 1]
            if fits:
                chosen = units[min(fits, key=lambda u: len(units[u]))]
        val.extend(sorted(chosen))
    return sorted(val)


# --------------------------------------------------------------------------- frames
def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src.resolve(), dst)
    except OSError:  # file systems without symlinks (e.g. some Drive mounts)
        shutil.copy2(src, dst)


def extract_frames(data_cfg: dict, seq: str, out_dir: Path) -> list[Path]:
    """Returns the frame files of a sequence, extracting them from video if needed."""
    import cv2

    root = resolve_path(data_cfg["raw_root"])
    ext = data_cfg.get("image_ext", ".jpg")
    if data_cfg.get("frames_dir"):
        src = root / data_cfg["frames_dir"] / seq
        frames = sorted(p for p in src.iterdir() if p.suffix.lower() in {".jpg", ".png", ".bmp"})
        if not frames:
            raise FileNotFoundError(f"No frames in {src}")
        return frames

    seq_dir = out_dir / seq
    done = seq_dir / ".complete"
    if done.exists():
        return sorted(seq_dir.glob(f"*{ext}"))
    video = root / data_cfg["videos_dir"] / f"{seq}{data_cfg.get('video_ext', '.mp4')}"
    if not video.is_file():
        raise FileNotFoundError(f"Video not found: {video}")
    seq_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        cv2.imwrite(str(seq_dir / f"{n:05d}{ext}"), frame)
    cap.release()
    if n == 0:
        raise RuntimeError(f"Could not read any frame from {video}")
    done.write_text(str(n))
    return sorted(seq_dir.glob(f"*{ext}"))


def load_labels(data_cfg: dict, seq: str) -> np.ndarray:
    root = resolve_path(data_cfg["raw_root"])
    return read_mot(root / data_cfg["labels_dir"] / f"{seq}.txt")


# --------------------------------------------------------------------------- converters
def to_yolo_lines(rows: np.ndarray, width: int, height: int) -> list[str]:
    """MOT boxes of one frame -> YOLO lines ``0 xc yc w h`` normalised to [0, 1]."""
    lines = []
    for _, _, x, y, w, h, *_ in rows:
        x1, y1 = max(x, 0.0), max(y, 0.0)
        x2, y2 = min(x + w, width), min(y + h, height)
        if x2 <= x1 or y2 <= y1:
            continue
        xc, yc = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
        bw, bh = (x2 - x1) / width, (y2 - y1) / height
        lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return lines


def export_yolo(seqs: list[str], part: str, data_cfg: dict, frames_dir: Path, out: Path) -> int:
    import cv2

    n_images = 0
    for seq in seqs:
        frames = extract_frames(data_cfg, seq, frames_dir)
        labels = load_labels(data_cfg, seq)
        h, w = cv2.imread(str(frames[0])).shape[:2]
        for idx, frame_path in enumerate(frames, start=1):
            name = f"{seq}_{idx:05d}"
            _link(frame_path, out / part / "images" / f"{name}{frame_path.suffix}")
            rows = labels[labels[:, 0] == idx]
            label_file = out / part / "labels" / f"{name}.txt"
            label_file.parent.mkdir(parents=True, exist_ok=True)
            label_file.write_text("\n".join(to_yolo_lines(rows, w, h)) + ("\n" if len(rows) else ""))
            n_images += 1
    return n_images


def select_frames(images: list[Path], step: int) -> list[Path]:
    """Every ``step``-th frame of each sequence, starting with frame 1.

    YOLO image names are ``<seq>_<frame:05d>``; the frame number decides, so the
    selection does not depend on the listing order and is the same on every machine.
    """
    if step <= 1:
        return sorted(images)
    keep = []
    for img in sorted(images):
        frame = int(img.stem.rsplit("_", 1)[1])
        if (frame - 1) % step == 0:
            keep.append(img)
    return keep


def frames_hash(images: list[Path]) -> str:
    """Identifies a frame selection by its image names (not the folder it sits in)."""
    names = "\n".join(sorted(p.name for p in images))
    return hashlib.sha256(names.encode()).hexdigest()[:12]


def reid_crop_name(gid: int, seq: str, frame: int) -> str:
    """File name parsed by FastReID's MOT20 loader in Strong Baseline.

    The loader uses ``(\\d+)_\\w+_\\d+_(\\d+)_acc_data\\.bmp`` (pid, camid), so the sequence
    name is reduced to word characters and followed by a fixed ``_0`` field.
    """
    safe = re.sub(r"\W", "", seq) or "seq"
    return f"{gid:07d}_{safe}_0_{frame:07d}_acc_data.bmp"


def export_reid(seqs: list[str], part: str, data_cfg: dict, frames_dir: Path, out: Path, id_offset: int) -> tuple[int, int]:
    """Crops every labelled box; identities get globally unique ids (offset per sequence)."""
    import cv2

    reid_cfg = data_cfg.get("reid", {})
    fraction = float(reid_cfg.get("first_fraction", 1.0))
    min_size = int(reid_cfg.get("min_size", 2))
    folder = out / "MOT20" / ("bounding_box_train" if part == "train" else "bounding_box_test")
    folder.mkdir(parents=True, exist_ok=True)
    n_crops = 0
    for seq in seqs:
        frames = extract_frames(data_cfg, seq, frames_dir)
        labels = load_labels(data_cfg, seq)
        last = max(1, int(round(len(frames) * fraction)))
        max_id = 0
        for idx in range(1, last + 1):
            rows = labels[labels[:, 0] == idx]
            if not len(rows):
                continue
            img = cv2.imread(str(frames[idx - 1]))
            H, W = img.shape[:2]
            for _, tid, x, y, bw, bh, *_ in rows:
                x1, y1 = int(max(x, 0)), int(max(y, 0))
                x2, y2 = int(min(x + bw, W)), int(min(y + bh, H))
                if x2 - x1 < min_size or y2 - y1 < min_size:
                    continue
                gid = int(tid) + id_offset
                cv2.imwrite(str(folder / reid_crop_name(gid, seq, idx)), img[y1:y2, x1:x2])
                n_crops += 1
                max_id = max(max_id, int(tid))
        id_offset += max_id
    return n_crops, id_offset


def export_tracking_input(seqs: list[str], data_cfg: dict, frames_dir: Path, out: Path, gt_out: Path) -> None:
    """Track 3 layout (Test_imgs + first-frame labels) and the evaluation ground truth."""
    for seq in seqs:
        frames = extract_frames(data_cfg, seq, frames_dir)
        labels = load_labels(data_cfg, seq)
        for idx, frame_path in enumerate(frames, start=1):
            _link(frame_path, out / "Test_imgs" / seq / f"{idx:05d}{frame_path.suffix}")
        write_mot(out / "TestLabels_FirstFrameOnly" / f"{seq}.txt", labels[labels[:, 0] == 1], gt=True)
        write_mot(gt_out / f"{seq}.txt", labels, gt=True)
        (gt_out / f"{seq}.len").write_text(str(len(frames)))


def prepare(cfg: dict, parts: tuple[str, ...] = ("yolo", "reid", "track")) -> dict:
    """Runs the conversions and returns the manifest (cached by split hash)."""
    data_cfg = cfg["data"]
    splits = resolve_splits(data_cfg)
    h = split_hash(splits, data_cfg)
    data_dir = Path(cfg["paths"].get("data_dir") or Path(cfg["paths"]["work_dir"]) / "data")
    root = data_dir / f"{data_cfg['name']}-{h}"
    frames_dir = data_dir / f"{data_cfg['name']}-frames"
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "dataset": data_cfg["name"], "split_hash": h, "root": str(root),
        "sequences": splits, "counts": {}, "done": [],
    }

    if "yolo" in parts and "yolo" not in manifest["done"]:
        yolo = root / "yolo"
        manifest["counts"]["yolo_train_images"] = export_yolo(splits["train"], "train", data_cfg, frames_dir, yolo)
        manifest["counts"]["yolo_val_images"] = export_yolo(splits["val"], "val", data_cfg, frames_dir, yolo)
        (yolo / "dataset.yaml").write_text(
            f"path: {yolo}\ntrain: train/images\nval: val/images\nnc: 1\nnames: ['uav']\n"
        )
        manifest["done"].append("yolo")

    if "reid" in parts and "reid" not in manifest["done"]:
        reid = root / "reid"
        n_tr, offset = export_reid(splits["train"], "train", data_cfg, frames_dir, reid, 0)
        n_te, _ = export_reid(splits["val"], "val", data_cfg, frames_dir, reid, offset)
        manifest["counts"].update(reid_train_crops=n_tr, reid_test_crops=n_te)
        manifest["done"].append("reid")

    if "track" in parts and "track" not in manifest["done"]:
        export_tracking_input(splits["val"], data_cfg, frames_dir, root / "track" / "val", root / "gt" / "val")
        manifest["done"].append("track")

    root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest
