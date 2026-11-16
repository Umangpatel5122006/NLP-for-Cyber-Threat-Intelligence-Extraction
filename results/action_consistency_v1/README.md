# Fit-only action consistency comparison

**Selected: current baseline.** Omitting ambiguous fit inputs raises training-CV
exact F1 only from 0.132261 to 0.133239; also deduplicating lowers it to 0.130532.
Neither meets the fixed minimum-gain policy. Original validation/gold labels and
active models are unchanged. Dev/test were not evaluated, and no full-train
candidate artifact was created.

- `cv.json`: original-gold metrics, report folds and fit-policy inventories.
- `selection.json`: frozen train-only budgets and eligibility decision.
- `summary.json`: baseline retention, unchanged files and evaluation scope.
- `integrity_audit.json`: independent fit inventories, prediction/count checks
  and verification that every original validation row remains in scoring.

All 350 WSL tests pass. Detailed caches, OOF predictions and logs stay ignored.
The human review workbook remains pending and is not modified by the experiment.
See [the checkpoint](../../checkpoints/ACTION_CONSISTENCY_EXPERIMENT.md) for the
method, actual results, interpretation limits and reproduction prerequisites.
