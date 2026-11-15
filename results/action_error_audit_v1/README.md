# Train-only action diagnostics

`summary.json` records strict exact-first error counts, a separate per-gold
any-overlap diagnostic, boundary types, lengths, report-level counts, gold-status
transitions, fold-specific phrase/sentence exposure and source annotation checks.
`integrity_audit.json` records unchanged inputs/models and agreement with the
frozen training-CV exact counts. It is not an annotation-quality approval.

The current model has 559 exact actions, 1,702 partial overlaps and 2,192 actions
with no overlap. End-boundary errors and entirely missed actions both need work.
Regenerating all training BIO reproduces the saved data exactly. Twenty-one
identical-token groups have conflicting action tags and need contextual review.

Local-only review files are ignored:

- `annotation_review.csv`: all 43 records from 21 conflicting-action groups,
  with source/context, existing tags and a notes column.
- `review_examples.csv`: 48 bounded deterministic examples of missed/partial
  actions and exact predictions lost by the rejected candidates.

No new model, label correction, split change or dev/test evaluation was performed.
These are descriptive diagnostics of development data, not independent accuracy.
See [the checkpoint](../../checkpoints/ACTION_ERROR_AUDIT.md) for methods, actual
results, interpretation limits, review guidance and local reproduction.
