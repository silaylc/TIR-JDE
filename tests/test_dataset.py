import re
from pathlib import Path

import numpy as np
import pytest

from tir_jde import dataset

# Regex used by Strong Baseline's FastReID MOT20 loader (BSB version)
SB_REID_PATTERN = re.compile(r"(\d+)_\w+_\d+_(\d+)_acc_data\.bmp")


def test_validation_group_is_removed_from_training(synthetic_cfg):
    splits = dataset.resolve_splits(synthetic_cfg["data"])
    assert splits["val"] == ["C_02", "T_02"]
    assert not set(splits["val"]) & set(splits["train"])
    assert len(splits["train"]) == 3


def test_unknown_validation_sequence_is_rejected(synthetic_cfg, tmp_path):
    bad = tmp_path / "bad_val.txt"
    bad.write_text("NOT_A_SEQ\n")
    data_cfg = {**synthetic_cfg["data"], "splits": {**synthetic_cfg["data"]["splits"], "val": str(bad)}}
    with pytest.raises(ValueError):
        dataset.resolve_splits(data_cfg)


def test_split_hash_changes_with_the_split(synthetic_cfg):
    data_cfg = synthetic_cfg["data"]
    a = {"train": ["C_01", "T_01"], "val": ["C_02"]}
    b = {"train": ["C_01", "C_02"], "val": ["T_01"]}
    assert dataset.split_hash(a, data_cfg) == dataset.split_hash({"val": ["C_02"], "train": ["T_01", "C_01"]}, data_cfg)
    assert dataset.split_hash(a, data_cfg) != dataset.split_hash(b, data_cfg)


def test_yolo_lines_are_normalised_and_clipped():
    rows = np.array([[1, 1, 10, 20, 8, 6, 1, 1, 1], [1, 2, -4, 0, 8, 6, 1, 1, 1]], float)
    lines = dataset.to_yolo_lines(rows, width=100, height=50)
    assert lines[0] == "0 0.140000 0.460000 0.080000 0.120000"
    assert lines[1] == "0 0.020000 0.060000 0.040000 0.120000"  # clipped at the left border


def test_prepare_builds_every_format(synthetic_cfg):
    m = dataset.prepare(synthetic_cfg)
    root = Path(m["root"])

    # YOLO: one image and label per frame, file names keep the sequence
    train_imgs = sorted((root / "yolo" / "train" / "images").iterdir())
    assert m["counts"]["yolo_train_images"] == len(train_imgs) == 12 + 8 + 7
    assert all(p.name.split("_")[0] in {"C", "T", "BB2P"} for p in train_imgs)
    assert (root / "yolo" / "train" / "labels" / "C_01_00001.txt").read_text().count("\n") == 3
    assert not any("C_02" in p.name for p in train_imgs)
    assert "train: train/images" in (root / "yolo" / "dataset.yaml").read_text()

    # ReID: parseable by Strong Baseline, identities disjoint between train and test
    train_ids, test_ids = set(), set()
    for part, ids in (("bounding_box_train", train_ids), ("bounding_box_test", test_ids)):
        files = list((root / "reid" / "MOT20" / part).glob("*.bmp"))
        assert files
        for f in files:
            pid, _ = SB_REID_PATTERN.search(str(f)).groups()
            ids.add(int(pid))
    assert not train_ids & test_ids

    # Tracking input in the Track 3 layout + ground truth for the evaluator
    first = (root / "track" / "val" / "TestLabels_FirstFrameOnly" / "C_02.txt").read_text().splitlines()
    assert len(first) == 3 and all(line.startswith("1,") for line in first)
    assert len(list((root / "track" / "val" / "Test_imgs" / "C_02").iterdir())) == 10
    assert (root / "gt" / "val" / "C_02.len").read_text() == "10"


def test_prepare_is_cached(synthetic_cfg):
    first = dataset.prepare(synthetic_cfg)
    second = dataset.prepare(synthetic_cfg)
    assert first == second


def test_reid_names_survive_sequences_with_dashes():
    name = dataset.reid_crop_name(313, "MultiUAV-003", 59)
    assert SB_REID_PATTERN.search("/x/y/" + name).groups() == ("0000313", "0000059")


def test_family_and_validation_split_are_deterministic():
    train = ["C_01", "C_02", "C_03", "T_01", "T_02", "BB2P_01", "Takeoff_01"]
    families = {s: dataset.family_of(s) for s in train}
    assert families["BB2P_01"] == "BB2P"
    a = dataset.make_val_split(train, families, per_family=1, seed=0)
    assert a == dataset.make_val_split(train, families, per_family=1, seed=0)
    assert {dataset.family_of(s) for s in a} == {"C", "T"}  # single-sequence families stay in training


def test_proportional_validation_split_follows_family_sizes():
    train = [f"T_{i:02d}" for i in range(20)] + [f"C_{i:02d}" for i in range(6)] + ["L_00", "L_01", "Takeoff_00"]
    families = {s: dataset.family_of(s) for s in train}
    val = dataset.make_val_split(train, families, per_family=1, seed=0, fraction=0.15)
    count = {f: sum(dataset.family_of(s) == f for s in val) for f in ("T", "C", "L", "Takeoff")}
    assert count == {"T": 3, "C": 1, "L": 1, "Takeoff": 0}
    assert val == dataset.make_val_split(train, families, per_family=1, seed=0, fraction=0.15)


def test_scene_groups_are_never_split_and_excluded_sequences_drop_out(tmp_path):
    train = [f"T_{i:02d}" for i in range(20)] + ["C_00", "C_01", "C_02"]
    families = {s: dataset.family_of(s) for s in train}
    groups = {"T_00": "g1", "T_01": "g1", "T_02": "g1", "T_03": "g2", "T_04": "g2"}
    for seed in range(10):
        val = dataset.make_val_split(train, families, 1, seed, fraction=0.15, groups=groups)
        for g in ("g1", "g2"):
            members = {s for s, x in groups.items() if x == g}
            assert members <= set(val) or not (members & set(val))  # whole group or nothing
        assert any(dataset.family_of(s) == "C" for s in val) and any(dataset.family_of(s) == "C" for s in set(train) - set(val))

    (tmp_path / "train.txt").write_text("\n".join(train) + "\n")
    (tmp_path / "val.txt").write_text("T_05\n")
    cfg = {"splits": {"train": str(tmp_path / "train.txt"), "val": str(tmp_path / "val.txt")}, "exclude": ["T_06"]}
    splits = dataset.resolve_splits(cfg)
    assert "T_06" not in splits["train"] and splits["val"] == ["T_05"]
    cfg["exclude"] = ["T_05"]
    with pytest.raises(ValueError, match="Excluded"):
        dataset.resolve_splits(cfg)


def test_first_frame_file_is_written_even_when_frame_one_is_empty(tmp_path):
    labels = np.array([[2, 1, 10, 20, 8, 6, 1, 1, 1.0], [3, 1, 11, 20, 8, 6, 1, 1, 1.0]])
    out = tmp_path / "TestLabels_FirstFrameOnly" / "E_00.txt"
    dataset.write_mot(out, labels[labels[:, 0] == 1], gt=True)
    assert out.exists() and out.read_text() == ""  # like MultiUAV-068: the file exists, empty


def test_first_frame_labels_match_the_first_image_of_the_sequence(tmp_path):
    """GT frame 1 must be the first image the tracker reads, also with BSB's 0-based file names."""
    import cv2

    raw = tmp_path / "raw"
    (raw / "frames" / "Z_00").mkdir(parents=True)
    (raw / "labels").mkdir()
    rows = []
    for f in range(1, 6):  # label frames 1..5 <-> files 00000..00004, as in BSB TrainImages
        img = np.full((64, 96, 3), 40, np.uint8)
        x = 5 + 12 * (f - 1)  # the target moves 12 px per frame: each frame is recognisable
        img[20:26, x:x + 8] = 220
        cv2.imwrite(str(raw / "frames" / "Z_00" / f"{f - 1:05d}.jpg"), img)
        rows.append(f"{f},1,{x},20,8,6,1,1,1.0")
    (raw / "labels" / "Z_00.txt").write_text("\n".join(rows) + "\n")
    data_cfg = {"raw_root": str(raw), "frames_dir": "frames", "labels_dir": "labels", "image_ext": ".jpg"}

    out = tmp_path / "track"
    dataset.export_tracking_input(["Z_00"], data_cfg, tmp_path / "cache", out, tmp_path / "gt")

    first_image = sorted((out / "Test_imgs" / "Z_00").glob("*.*"))[0]  # the order the tracker reads
    x, y, w, h = (int(float(v)) for v in
                  (out / "TestLabels_FirstFrameOnly" / "Z_00.txt").read_text().split(",")[2:6])
    img = cv2.imread(str(first_image), cv2.IMREAD_GRAYSCALE)
    assert img[y:y + h, x:x + w].mean() > 180          # the frame-1 box sits on the target
    assert img[y:y + h, x + 12:x + 12 + w].mean() < 100  # and not where frame 2 has it
