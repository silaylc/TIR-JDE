"""Head-only identity probe on the frozen detector-v1-P2 (E3). Requires torch, torchvision, Ultralytics.

Features: RoIAlign 2x2 on the neck outputs P2 (layer 23, stride 4) and P3 (layer 14, stride 8),
concatenated. Head: MLP ``D -> 512 -> E`` with L2-normalised output, trained with a cosine classifier
(CE, scale 16) and optionally batch-hard triplet (margin 0.3), PK sampling 32 x 4.

E3 result (val): CE with E = 128 was chosen; on close pairs at a 10-frame gap appearance is 78 %
top-1 vs 86 % for position (chance 5 %). Added to the tracker with a fixed weight it did not improve
HOTA (E7a) - the embedding needs joint training (B2).
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import roi_align

from tir_jde.eval.boxes import xywh2xyxy

FEATURE_LAYERS = {"P2": 23, "P3": 14}
STRIDES = {"P2": 4, "P3": 8}


def letterbox_top_left(img: np.ndarray, size: int = 640):
    """Resize the long side to ``size`` and pad right/bottom (value 114) to a multiple of 32.

    Box coordinates map with a single scale factor ``r`` (no offset).
    """
    h, w = img.shape[:2]
    r = size / max(h, w)
    nh, nw = round(h * r), round(w * r)
    out = np.full((math.ceil(nh / 32) * 32, math.ceil(nw / 32) * 32, 3), 114, np.uint8)
    out[:nh, :nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, r


class FeatureExtractor:
    """Forward hooks on the neck layers; ``__call__(images_bgr, boxes_xywh_per_image)`` -> (N, D) features."""

    def __init__(self, model, imgsz: int = 640, roi: int = 2, device: str = "cuda"):
        self.model, self.imgsz, self.roi, self.device = model.to(device).eval(), imgsz, roi, device
        self._feats = {}
        self._hooks = [model.model[i].register_forward_hook(lambda m, inp, out, k=k: self._feats.__setitem__(k, out))
                       for k, i in FEATURE_LAYERS.items()]

    def close(self):
        for h in self._hooks:
            h.remove()

    @torch.no_grad()
    def __call__(self, images, boxes):
        lb = [letterbox_top_left(im, self.imgsz) for im in images]
        if len({im.shape for im, _ in lb}) != 1:
            raise ValueError("images in one call must share a resolution")
        x = torch.from_numpy(np.stack([im for im, _ in lb])[..., ::-1].copy()).permute(0, 3, 1, 2)
        x = x.to(self.device).float().div_(255)
        with torch.autocast(self.device, dtype=torch.float16, enabled=self.device == "cuda"):
            self.model(x)
        rois = np.concatenate([np.c_[np.full(len(b), i), xywh2xyxy(b) * r] for i, (b, (_, r)) in enumerate(zip(boxes, lb))])
        rois = torch.tensor(rois, dtype=torch.float32, device=self.device)
        parts = []
        for k, s in STRIDES.items():
            fm = self._feats[k].float()
            if x.shape[-1] / fm.shape[-1] != s:
                raise RuntimeError(f"{k}: stride {x.shape[-1] / fm.shape[-1]} != {s}")
            parts.append(roi_align(fm, rois, output_size=self.roi, spatial_scale=1 / s, sampling_ratio=2, aligned=True).flatten(1))
        return torch.cat(parts, 1).half().cpu().numpy()


class Head(nn.Module):
    def __init__(self, d_in: int, d_out: int = 128, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True), nn.Linear(hidden, d_out))

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)


class CosineClassifier(nn.Module):
    def __init__(self, d: int, n_ids: int, scale: float = 16.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_ids, d) * 0.01)
        self.scale = scale

    def forward(self, z):
        return self.scale * z @ F.normalize(self.W, dim=1).t()


def batch_hard_triplet(z, y, margin: float = 0.3):
    """Batch-hard triplet loss on L2-normalised embeddings (distance from the dot product: no NaN at 0)."""
    d = (2 - 2 * z @ z.t()).clamp_min(1e-12).sqrt()
    same = y[:, None] == y[None]
    pos = (d * same).max(1).values
    neg = d.masked_fill(same, float("inf")).min(1).values
    return F.relu(pos - neg + margin).mean()


def pk_indices(rng: np.random.Generator, by_id: list, P: int = 32, K: int = 4) -> np.ndarray:
    ids = rng.choice(len(by_id), P, replace=False)
    return np.concatenate([rng.choice(by_id[i], K, replace=len(by_id[i]) < K) for i in ids])


def train_head(X: torch.Tensor, y: torch.Tensor, dim: int = 128, triplet: bool = False, iters: int = 3000,
               lr: float = 1e-3, wd: float = 5e-4, seed: int = 0) -> Head:
    """Trains the probe head on standardised features ``X`` (N, D) with identity labels ``y`` (0..n-1)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    yc = y.cpu().numpy()
    by_id = [np.where(yc == i)[0] for i in range(int(yc.max()) + 1)]
    head = Head(X.shape[1], dim).to(X.device)
    cls = CosineClassifier(dim, len(by_id)).to(X.device)
    opt = torch.optim.AdamW(list(head.parameters()) + list(cls.parameters()), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    head.train()
    for _ in range(iters):
        idx = torch.from_numpy(pk_indices(rng, by_id)).to(X.device)
        z = head(X[idx])
        loss = F.cross_entropy(cls(z), y[idx])
        if triplet:
            loss = loss + batch_hard_triplet(z, y[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sch.step()
    return head.eval()
