import numpy as np

from tir_jde.detector.postprocess import center_nms
from tir_jde.eval.analysis import duplicate_pairs


def test_center_nms_removes_lower_scored_neighbour_only():
    d = np.array([[1, 10, 10, 14, 14, 0.9], [1, 11, 10, 15, 14, 0.5], [1, 30, 30, 34, 34, 0.4],
                  [2, 10, 10, 14, 14, 0.3]])
    k = center_nms(d, 3.0)
    assert len(k) == 3 and not any((k == d[1]).all(1))


def test_center_nms_random_invariants():
    rng = np.random.default_rng(0)
    for _ in range(100):
        n = int(rng.integers(0, 60))
        xy = rng.uniform(0, 40, (n, 2))
        d = np.c_[rng.integers(1, 5, n), xy, xy + rng.uniform(3, 8, (n, 2)), np.round(rng.uniform(0.05, 1, n), 2)]
        k = center_nms(d, 3.0)
        idx = [int(np.where((d == r).all(1))[0][0]) for r in k]
        assert idx == sorted(idx)                                # kept rows keep their order
        assert duplicate_pairs(k, dist_px=3.0, score_min=0)[1] == 0  # no close pair is left


def test_center_nms_empty():
    assert center_nms(np.zeros((0, 6))).shape == (0, 6)
