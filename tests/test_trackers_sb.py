from types import SimpleNamespace

import numpy as np
import pytest

from tir_jde.eval.boxes import iou_matrix
from tir_jde.trackers.sb.api import FROZEN
from tir_jde.trackers.sb.costs import nwd_cost


def test_nwd_is_zero_for_identical_boxes():
    b = np.array([10.0, 10.0, 14.0, 14.0])
    assert nwd_cost([b], [b])[0, 0] == pytest.approx(0.0)


def test_nwd_stays_graded_where_iou_collapses():
    """Two 4 px boxes 4 px apart: IoU is 0, NWD still ranks the nearer box first."""
    a = np.array([10.0, 10.0, 14.0, 14.0])
    near, far = np.array([14.0, 10.0, 18.0, 14.0]), np.array([30.0, 10.0, 34.0, 14.0])
    assert iou_matrix(a[None], near[None])[0, 0] == 0
    c = nwd_cost([a], [near, far])[0]
    assert 0 < c[0] < c[1] < 1


def test_nwd_accepts_track_objects():
    track = SimpleNamespace(tlwh=np.array([10.0, 10.0, 4.0, 4.0]))
    det = np.array([10.0, 10.0, 14.0, 14.0])
    assert nwd_cost([track], [det])[0, 0] == pytest.approx(0.0)


def test_nwd_empty():
    assert nwd_cost([], [np.zeros(4)]).shape == (0, 1)


def test_frozen_config():
    assert FROZEN == {"cmc": "sparseOptFlow", "cost_mode": "nwd", "match_thresh": 0.8, "coast_k": 10}
