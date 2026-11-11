import numpy as np
import pandas as pd
import pytest

from tir_jde.eval.stats import bootstrap_diff, compare, holm


def test_bootstrap_detects_consistent_improvement():
    base = np.linspace(0.5, 0.7, 16)
    m, (lo, hi), p = bootstrap_diff(base, base + 0.05)
    assert m == pytest.approx(0.05)
    assert lo > 0 and p < 0.05


def test_bootstrap_null_difference_covers_zero():
    rng = np.random.default_rng(1)
    base = rng.uniform(0.4, 0.8, 16)
    m, (lo, hi), p = bootstrap_diff(base, base + rng.normal(0, 0.02, 16))
    assert lo < 0 < hi and p > 0.05


def test_bootstrap_rejects_unpaired_input():
    with pytest.raises(ValueError):
        bootstrap_diff([1, 2, 3], [1, 2])


def test_holm_matches_hand_computation():
    adj = holm([0.01, 0.04, 0.03])
    np.testing.assert_allclose(adj, [0.03, 0.06, 0.06])


def test_holm_is_monotone_and_capped():
    adj = holm([0.5, 0.001, 0.9, 0.2])
    order = np.argsort([0.5, 0.001, 0.9, 0.2])
    assert np.all(np.diff(adj[order]) >= 0) and adj.max() <= 1.0


def test_compare_uses_requested_sequences():
    seqs = ["a", "b", "c", "d"]
    base = pd.DataFrame({"HOTA": [0.5, 0.6, 0.7, 0.8]}, index=seqs)
    new = base + 0.1
    r = compare(base, new, "HOTA", seqs[:3])
    assert r["metric"] == "HOTA" and r["diff"] == pytest.approx(0.1) and r["ci95"][0] > 0
