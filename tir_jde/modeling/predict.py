"""Inference + evaluation entry point (cookiecutter convention).

Same as ``python -m tir_jde.pipeline evaluate``: runs the trained model on the
validation sequences, writes MOTChallenge files and scores them with the
shared TrackEval evaluator::

    python -m tir_jde.modeling.predict strong_baseline_bsb seed=1
"""

import sys

from tir_jde.pipeline import main

if __name__ == "__main__":
    main(["evaluate", *sys.argv[1:]])
