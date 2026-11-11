import pytest

from tir_jde.detector.p2 import NEW_DETECT, P2_LAYERS, p2_config, remap_key


def _v1_cfg():
    backbone = [[-1, 1, "Conv", [64, 3, 2]] for _ in range(9)]
    head = [[-1, 1, "A2C2f", [256]] for _ in range(12)] + [[[14, 17, 20], 1, "Detect", ["nc"]]]
    return {"nc": 1, "scale": "l", "backbone": backbone, "head": head}


def test_p2_config_adds_branch_and_moves_detect():
    cfg = p2_config(_v1_cfg())
    assert len(cfg["backbone"]) + len(cfg["head"]) == NEW_DETECT + 1
    assert cfg["head"][-4:-1] == P2_LAYERS
    assert cfg["head"][-1][0] == [23, 14, 17, 20] and cfg["head"][-1][2] == "Detect"
    assert _v1_cfg()["head"][-1][0] == [14, 17, 20]  # input not modified


def test_p2_config_rejects_other_layouts():
    bad = _v1_cfg()
    bad["head"] = bad["head"][1:]
    with pytest.raises(ValueError):
        p2_config(bad)


@pytest.mark.parametrize("src,dst", [
    ("model.3.cv1.conv.weight", "model.3.cv1.conv.weight"),        # untouched layers
    ("model.21.cv2.0.0.conv.weight", "model.24.cv2.1.0.conv.weight"),  # P3 box branch -> level 1
    ("model.21.cv3.2.2.bias", "model.24.cv3.3.2.bias"),            # P5 class branch -> level 3
    ("model.21.dfl.conv.weight", "model.24.dfl.conv.weight"),
])
def test_remap_key(src, dst):
    assert remap_key(src) == dst


def test_remap_key_rejects_unknown_detect_part():
    with pytest.raises(KeyError):
        remap_key("model.21.unknown.weight")
