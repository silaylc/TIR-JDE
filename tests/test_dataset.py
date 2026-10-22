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
