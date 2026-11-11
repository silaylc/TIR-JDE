import numpy as np
import pandas as pd
import pytest

from tir_jde.reid.queries import build_queries, score_queries


def _boxes():
    rows = []
    for f in range(1, 31):
        rows.append(("S", f, 1, 100 + f, 100, 6, 6))  # two targets moving together, 5 px apart
        rows.append(("S", f, 2, 104 + f, 103, 6, 6))
        rows.append(("S", f, 3, 300, 300, 6, 6))      # a far, static one
    return pd.DataFrame(rows, columns=["seq", "f", "gid", "x", "y", "w", "h"])


def test_queries_close_pairs_only():
    b = _boxes()
    assert len(build_queries(b, gap=10, close_only=True)) == 2 * 20
    assert len(build_queries(b, gap=10, close_only=False)) == 3 * 20


def test_scores_identity_position_and_chance():
    b = _boxes()
    Q = build_queries(b, gap=10)
    perfect = score_queries(Q, b, np.eye(4)[b.gid.values])
    assert perfect.acc.iloc[0] == 1.0
    pos = score_queries(Q, b)
    assert pos.acc.iloc[0] == 0.5          # 10 px of motion makes position pick the neighbour half the time
    assert pos.chance.iloc[0] == pytest.approx(1 / 3)
