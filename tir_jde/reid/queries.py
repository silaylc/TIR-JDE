"""Identity probe protocol (E3), numpy only.

Query: a GT box at frame ``t`` whose nearest other GT centre is <= ``close_px`` (a "close pair").
Gallery: the GT boxes at frame ``t - gap`` (at least two). Correct if the most similar gallery box
has the query's identity. Two references put the accuracy in context: position (nearest centre,
no motion model) and chance (1 / gallery size).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tir_jde.eval.boxes import nearest_other_gt


def build_queries(boxes: pd.DataFrame, gap: int, close_px: float = 12.7, close_only: bool = True) -> list:
    """``boxes``: columns ``seq, f, gid, x, y, w, h`` (index = row id into the embedding array).

    Returns ``[(seq, query_row, gallery_rows, index_of_correct_in_gallery)]``.
    """
    Q = []
    for s, g in boxes.groupby("seq", sort=True):
        by_f = {f: gg for f, gg in g.groupby("f")}
        for f, cur in by_f.items():
            prev = by_f.get(f - gap)
            if prev is None or len(prev) < 2:
                continue
            nn = nearest_other_gt(np.c_[cur.f.values, cur.gid.values, cur[["x", "y", "w", "h"]].values])
            pg = prev.gid.values
            for j, (row, gid) in enumerate(zip(cur.index.values, cur.gid.values)):
                if close_only and not nn[j] <= close_px:
                    continue
                hit = np.where(pg == gid)[0]
                if len(hit) == 1:
                    Q.append((s, row, prev.index.values, int(hit[0])))
    return Q


def _centres(boxes: pd.DataFrame, rows) -> np.ndarray:
    b = boxes.loc[rows, ["x", "y", "w", "h"]].values
    return np.c_[b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2]


def score_queries(Q: list, boxes: pd.DataFrame, emb: np.ndarray | None = None) -> pd.DataFrame:
    """Per-sequence top-1 accuracy (``acc``) and chance level (``chance``).

    ``emb`` (rows L2-normalised, aligned with ``boxes.index``) -> cosine similarity;
    ``None`` -> position reference (nearest centre).
    """
    res = {}
    for s, q, gal, k in Q:
        if emb is None:
            pick = int(np.argmin(np.linalg.norm(_centres(boxes, gal) - _centres(boxes, [q]), axis=1)))
        else:
            pick = int(np.argmax(emb[gal] @ emb[q]))
        r = res.setdefault(s, [0, 0, 0.0])
        r[0] += int(pick == k)
        r[1] += 1
        r[2] += 1.0 / len(gal)
    return pd.DataFrame(
        {s: {"correct": v[0], "n": v[1], "acc": v[0] / v[1], "chance": v[2] / v[1]} for s, v in res.items()}
    ).T.astype(float)
