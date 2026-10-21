"""Training entry point (cookiecutter convention).

Same as ``python -m tir_jde.pipeline train``::

    python -m tir_jde.modeling.train strong_baseline_bsb seed=1

Every compared model is trained through its adapter in
``tir_jde/modeling/adapters/``; the TIR-JDE model will be added there too.
"""

import sys

from tir_jde.pipeline import main

if __name__ == "__main__":
    main(["train", *sys.argv[1:]])
