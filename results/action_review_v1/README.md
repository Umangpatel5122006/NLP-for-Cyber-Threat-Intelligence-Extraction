# Train-only action annotation review

`summary.json` records 21 conflict groups / 43 rows, their structural difference
types, same-context conflicts, aggregate frozen-prediction coverage and provenance.
All 21 decisions remain **pending**. No model, label or split was changed and
no dev/test evaluation was performed.

Ignored local review files:

- `casebook.md`: all 43 source/context records and readable original spans.
- `decisions.csv`: one editable finding per group. Regeneration preserves
  existing decisions and rejects changed evidence or invalid metadata.

Run `python -m cti.action_review` in the prepared project runtime. Model binaries
are unnecessary; unavailable frozen OOF caches omit optional coverage counts.
The decision tool does not apply corrections or export training data.

See [the guide](../../docs/ACTION_ANNOTATION_REVIEW.md) and
[checkpoint](../../checkpoints/ACTION_REVIEW_PACK.md). Full WSL suite: **341 tests
pass**. Native generation succeeds and preserves the workbook. Logs/source
excerpts/decisions remain ignored; this summary is descriptive, not an accuracy gain.
