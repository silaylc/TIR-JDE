import numpy as np
import pytest

from tir_jde.eval.analysis import duplicate_pairs, idsw_groups, long_gap_returns, recall_by_size
from tir_jde.eval.boxes import iou_matrix, match_frame, nearest_other_gt


def _row(f, i, x, y, w=6, h=6):
    return [f, i, x, y, w, h, 1, 1, 1, 0]


def test_iou_and_matching():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [50, 50, 60, 60]])
    np.testing.assert_allclose(iou_matrix(a, b)[0], [1.0, 50 / 150, 0.0], atol=1e-6)
    assert match_frame(a, b, 0.5) == [(0, 0, pytest.approx(1.0))]


def test_nearest_other_gt():
    g = np.array([_row(1, 1, 0, 0), _row(1, 2, 3, 4), _row(1, 3, 100, 0)])
    np.testing.assert_allclose(nearest_other_gt(g), [5.0, 5.0, np.hypot(97, 4)])
    assert np.isinf(nearest_other_gt(g[:1])).all()


def test_idsw_groups_close_pair_and_long_gap():
    gt, pr = [], []
    for f in range(1, 41):
        gt += [_row(f, 1, 100, 100), _row(f, 2, 106, 100), _row(f, 3, 400, 400)]
        # close pair: ids 1 and 2 swap at frame 20
        pr += [_row(f, 11 if f < 20 else 12, 100, 100), _row(f, 12 if f < 20 else 11, 106, 100)]
        # long gap: target 3 is missed for frames 10..25 and comes back with a new id
        if f < 10:
            pr.append(_row(f, 13, 400, 400))
        elif f > 25:
            pr.append(_row(f, 14, 400, 400))
    ev = idsw_groups(np.array(gt, float), np.array(pr, float))
    assert sorted(ev.group.tolist()) == ["close pair", "close pair", "long gap"]
    ret = long_gap_returns(np.array(gt, float), np.array(pr, float))
    assert len(ret) == 1 and ret.gap.iloc[0] == 16 and not ret.same_id.iloc[0]


def test_recall_by_size():
    gt = np.array([_row(1, 1, 0, 0, 4, 4), _row(1, 2, 50, 50, 10, 10), _row(1, 3, 100, 100, 20, 20)], float)
    dets = np.array([[1, 0, 0, 4, 4, 0.9], [1, 100, 100, 120, 120, 0.05]])  # big one below the score threshold
    r = recall_by_size(gt, dets).set_index("bin")
    assert r.loc["<8", "matched"] == 1 and r.loc["8-16", "matched"] == 0 and r.loc[">=16", "matched"] == 0


def test_duplicate_pairs():
    dets = np.array([[1, 10, 10, 14, 14, 0.9], [1, 11, 10, 15, 14, 0.5], [1, 30, 30, 34, 34, 0.4],
                     [2, 10, 10, 14, 14, 0.3]])
    assert duplicate_pairs(dets, dist_px=3.0) == (4, 1)
