# Classical action-ending comparison

**Release status: held after failed dev confirmation. Default models unchanged.**

The learned word-pair profile passes training-CV selection: exact F1 improves
from 0.132261 to 0.137671 with wins in all three folds. Dev F1 then falls from
0.155108 to 0.138032, with worse recall and more background findings. No final
test was evaluated and the candidate was not activated.

- `cv.json`: fixed candidate/fold metrics and provenance.
- `selection.json`: training-only decision, budgets and eligibility checks,
  written before dev access.
- `dev.json`: current/selected comparison and all three failed confirmation checks.
- `summary.json`: release hold, actual dev result and unchanged active-file hashes.
- `integrity_audit.json`: independent count/cache/fit-UID/selection checks and
  identical dev predictions after candidate reload.

The candidate binary/manifest, per-fold caches, OOF/dev predictions and training/
test/audit logs stay local and ignored. See [the checkpoint](../../checkpoints/ACTION_ENDING_EXPERIMENT.md)
for the full method, limitations, prerequisite caches and reproduction commands.
