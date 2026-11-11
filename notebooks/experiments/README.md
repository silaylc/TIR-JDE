# Experiment notebooks

Experiment notebooks run on Colab (RTX PRO 6000), outputs cleared. Results and decisions: [`docs/experiments.md`](../../docs/experiments.md).

| Notebook | Contents |
|---|---|
| `D2_p2_head_and_followups.ipynb` | D2 (P2 output: construction, weight transfer and checks, training, evaluation), D2b (centre-distance NMS), E5a.0 (IDSW diagnosis), E3 (identity probe), E7a (appearance fusion) |

Every experiment follows the same order: a markdown definition cell, a pre-registration cell that locks the decision rule, the run, the evaluation (decision, figures, MLflow) and a report. Cell text and comments are in Turkish.

Reusable code from the notebooks lives in the package: `tir_jde/detector/p2.py`, `tir_jde/trackers/sb/`, `tir_jde/eval/`, `tir_jde/reid/`.
