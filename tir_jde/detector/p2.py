"""detector-v1-P2: a stride-4 (P2) output branch added to the common YOLOv12-L detector (experiment D2).

Why: TaskAlignedAssigner only takes anchors whose centre lies inside a GT box. At stride 8 (P3) the
centres are 8 px apart, so 38 % of the < 8 px targets got no candidate anchor at all; at stride 4 it is
0.8 %. Result on val (D2): DetA +0.083 over a same-recipe control, < 8 px recall 0.78 -> 0.93.

Construction (layer numbers of the YOLOv12-L yaml):

- layers 0-20 and Detect's P3/P4/P5 branches are copied unchanged from detector-v1;
- new 21 ``Upsample`` (from 14), 22 ``Concat`` with backbone layer 2 (stride 4), 23 ``A2C2f`` (256 ch);
- Detect moves from index 21 to 24 and reads ``[23, 14, 17, 20]``; its P2 branch starts as a copy of
  the P3 branch, with the class bias at Ultralytics' stride-4 prior.

``p2_config`` and ``remap_key`` are pure functions; ``build_p2`` needs torch + Ultralytics (the fork
used for the common detector, a97bba3).
"""

from __future__ import annotations

import copy
import math

P2_LAYERS = [
    [14, 1, "nn.Upsample", [None, 2, "nearest"]],  # 21: P3 output -> stride 4
    [[-1, 2], 1, "Concat", [1]],                  # 22: with backbone P2/4 (layer 2)
    [-1, 2, "A2C2f", [256, False, -1]],           # 23: P2 block (same structure as layer 14, 256 ch)
]
OLD_DETECT, NEW_DETECT = 21, 24


def p2_config(cfg: dict) -> dict:
    """detector-v1 model yaml (dict) -> P2 model yaml (dict)."""
    det = cfg["head"][-1]
    if len(cfg["backbone"]) + len(cfg["head"]) != OLD_DETECT + 1 or list(det[0]) != [14, 17, 20] or det[2] != "Detect":
        raise ValueError(f"unexpected detector layout: {len(cfg['backbone']) + len(cfg['head'])} layers, Detect {det}")
    out = copy.deepcopy(cfg)
    out["head"] = out["head"][:-1] + copy.deepcopy(P2_LAYERS) + [[[23, 14, 17, 20], det[1], det[2], copy.deepcopy(det[3])]]
    return out


def remap_key(key: str, old: int = OLD_DETECT, new: int = NEW_DETECT) -> str:
    """detector-v1 state_dict key -> P2 model key (Detect moves; its cv2/cv3 level index shifts by one)."""
    p = key.split(".")
    if p[0] == "model" and p[1] == str(old):
        p[1] = str(new)
        if p[2] in ("cv2", "cv3"):
            p[3] = str(int(p[3]) + 1)  # P3, P4, P5 -> Detect inputs 1, 2, 3 (0 = P2)
        elif p[2] != "dfl":
            raise KeyError(f"unexpected Detect key: {key}")
    return ".".join(p)


def build_p2(detector_weights, seed: int = 0, imgsz: int = 640, check: bool = True):
    """Builds the P2 model from detector-v1 weights. Returns ``(v1, p2)`` (both float, eval).

    With ``check=True`` it verifies that P3/P4/P5 outputs equal detector-v1's on a random input.
    """
    import torch
    from ultralytics.nn.modules import Detect
    from ultralytics.nn.tasks import DetectionModel, attempt_load_one_weight

    v1, _ = attempt_load_one_weight(str(detector_weights))
    v1 = v1.cpu().float().eval()
    cfg = p2_config(v1.yaml)
    torch.manual_seed(seed)
    p2 = DetectionModel(copy.deepcopy(cfg), nc=v1.model[-1].nc, verbose=False).float().eval()
    det = p2.model[-1]
    if not (isinstance(det, Detect) and det.nl == 4 and p2.stride.tolist() == [4.0, 8.0, 16.0, 32.0]):
        raise RuntimeError("P2 model was not built as expected")

    src, dst = v1.state_dict(), p2.state_dict()
    mapped = {}
    for k, v in src.items():
        nk = remap_key(k)
        if nk not in dst or dst[nk].shape != v.shape:
            raise RuntimeError(f"weight transfer failed for {k} -> {nk}")
        mapped[nk] = v.clone()
    head = {}
    for k in set(dst) - set(mapped):
        if k.startswith((f"model.{NEW_DETECT}.cv2.0.", f"model.{NEW_DETECT}.cv3.0.")):
            head[k] = mapped[k.replace(".cv2.0.", ".cv2.1.", 1).replace(".cv3.0.", ".cv3.1.", 1)].clone()
    cls_bias = f"model.{NEW_DETECT}.cv3.0.2.bias"
    head[cls_bias] = torch.full_like(dst[cls_bias], math.log(5 / det.nc / (640 / 4) ** 2))
    p2.load_state_dict({**dst, **mapped, **head}, strict=True)
    p2 = p2.float().eval()

    if check:
        x = torch.rand(1, 3, imgsz, imgsz)
        with torch.no_grad():
            _, f1 = v1(x)
            _, f2 = p2(x)
        for i in range(3):
            if (f1[i] - f2[i + 1]).abs().max().item() > 1e-4:
                raise RuntimeError(f"P{i + 3} output differs from detector-v1: transfer is wrong")
    return v1, p2
