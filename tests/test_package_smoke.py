"""Smoke test: every declared package stub should at least be importable.

This is deliberately trivial while the packages are empty skeletons; it
exists so `pytest` has something to run from day one, and so a later
import-breaking change (bad __init__.py, circular import, etc.) is caught
immediately instead of silently.
"""
import importlib

import pytest

PACKAGES = [
    "tir_jde",
    "tir_jde.config",
    "tir_jde.dataset",
    "tir_jde.features",
    "tir_jde.plots",
    "tir_jde.modeling",
    "tir_jde.modeling.train",
    "tir_jde.modeling.predict",
    "tir_jde.detector",
    "tir_jde.detector.heads",
    "tir_jde.detector.losses",
    "tir_jde.degradation",
    "tir_jde.trackers",
    "tir_jde.trackers.simple",
    "tir_jde.eval",
    "tir_jde.eval.ablations",
    "tir_jde.tools",
]


@pytest.mark.parametrize("package", PACKAGES)
def test_package_imports(package):
    importlib.import_module(package)
