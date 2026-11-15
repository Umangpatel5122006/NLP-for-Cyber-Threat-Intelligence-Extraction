# Action CRF background-training comparison

Three fixed training policies retain the same classical action CRF features:
relevant-only, capped per-report background sampling, and all background.
Selection uses three report-disjoint folds of the official train split.

**Selected: relevant only.** Capped/all background reduce false action spans
but lower exact F1 from 0.132261 to 0.114200 / 0.092669 and exceed the recall-loss
budget. Both candidates win zero folds. Dev/test were not evaluated; no candidate
artifact was created, and the default app/models/configuration stay unchanged.

- `cv.json`: fixed training-fold metrics, sampling inventories and provenance.
- `selection.json`: train-only eligibility checks and the frozen decision.
- `summary.json`: final outcome, any permitted dev confirmation, and unchanged
  active artifact/configuration identities.
- `integrity_audit.json`: independent held-out prediction recount, sampling and
  selection checks, and active-file integrity checks.

Local per-fold caches/OOF predictions are under ignored
`models/action_background_v1/`. Detailed training/test logs stay ignored.
Model binaries and raw dataset copies are not included in Git.

See [the checkpoint](../../checkpoints/ACTION_BACKGROUND_EXPERIMENT.md) for
results, scope, prerequisite caches and review/reproduction commands.
