"""Model adapters: one per compared method, all with the same train/track interface."""

from tir_jde.modeling.adapters.base import Adapter
from tir_jde.modeling.adapters.dist_tracker import DistTrackerAdapter
from tir_jde.modeling.adapters.strong_baseline import StrongBaselineAdapter

ADAPTERS: dict[str, type[Adapter]] = {
    StrongBaselineAdapter.name: StrongBaselineAdapter,
    DistTrackerAdapter.name: DistTrackerAdapter,
}


def get_adapter(cfg: dict, manifest: dict) -> Adapter:
    key = cfg["model"]["adapter"]
    if key not in ADAPTERS:
        raise KeyError(f"Unknown adapter '{key}'. Available: {sorted(ADAPTERS)}")
    return ADAPTERS[key](cfg, manifest)
