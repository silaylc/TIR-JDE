"""Strong Baseline's BoT-SORT run on dumped detections, with the patches evaluated in the experiments."""

from tir_jde.trackers.sb.api import FROZEN, run_tracker  # noqa: F401
from tir_jde.trackers.sb.costs import nwd_cost  # noqa: F401
