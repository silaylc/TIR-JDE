# Experiment notebooks

Experiment notebooks run on Colab (RTX PRO 6000). They are being cleaned up and will be added here with outputs cleared. Results and decisions: [`docs/experiments.md`](../../docs/experiments.md).

| Notebook | Contents |
|---|---|
| E0–E2 | Protocol, tracker (NWD, coasting, CMC) — *to be added* |
| D0–D1 | SAHI, NWD box loss — *to be added* |
| D2 and follow-ups | D2 (P2 output), D2b (centre-distance NMS), E5a.0 (IDSW diagnosis), E3 (identity probe), E7a (appearance fusion) — *to be added* |

Every experiment follows the same order: a markdown definition cell, a pre-registration cell that locks the decision rule, the run, the evaluation (decision, figures, MLflow) and a report. Cell text and comments are in Turkish.

Reusable code from the notebooks lives in the package: `tir_jde/detector/p2.py`, `tir_jde/trackers/sb/`, `tir_jde/eval/`, `tir_jde/reid/`.
