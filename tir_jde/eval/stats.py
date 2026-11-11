"""Paired, sequence-level statistics used for every experiment decision.

All comparisons in the experiment log are paired over the same validation
sequences: the difference is the mean of per-sequence differences
(``new - base``), its 95 % interval comes from a sequence bootstrap, and
families of comparisons are corrected with Holm-Bonferroni.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def bootstrap_diff(base: Sequence[float], new: Sequence[float], B: int = 10_000, seed: int = 0):
    """Paired sequence bootstrap of ``mean(new - base)``.

    Returns ``(mean difference, (ci_low, ci_high), two-sided p)``.
    """
    base = np.asarray(base, float)
    new = np.asarray(new, float)
    if base.shape != new.shape or base.ndim != 1 or len(base) == 0:
        raise ValueError(f"paired 1-D inputs of equal length expected, got {base.shape} and {new.shape}")
    d = new - base
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(B, len(d)))].mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return float(d.mean()), (float(lo), float(hi)), float(min(1.0, p))


def holm(pvals: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values, in the input order."""
    p = np.asarray(pvals, float)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def compare(base_table, new_table, metric: str, seqs: Sequence[str], B: int = 10_000) -> dict:
    """Paired comparison of one metric between two per-sequence tables (pandas, index = sequence)."""
    b = base_table.loc[list(seqs), metric].astype(float).values
    n = new_table.loc[list(seqs), metric].astype(float).values
    m, ci, p = bootstrap_diff(b, n, B=B)
    return {"metric": metric, "diff": round(m, 4), "ci95": (round(ci[0], 4), round(ci[1], 4)), "p": round(p, 4)}
