# Train-only action error and annotation audit

Date: 2026-11-15 (Asia/Calcutta)

## Findings

The saved training BIO data reproduce exactly with the current converter.
The action model still has substantial missed-action and boundary problems.
This audit identifies the errors; it does not improve or replace a model.

The current profile's 4,453 gold action spans divide as follows on the frozen
report-grouped training-fold predictions:

| Per-gold diagnostic | Count | Meaning |
|---|---:|---|
| Exact | 559 | A predicted action has identical token boundaries. |
| Overlap only | 1,702 | Some predicted action overlaps, with no exact match. |
| No overlap | 2,192 | No predicted action overlaps the gold action. |

These are per-gold **any-overlap** categories. A merged prediction can overlap
several gold actions; these counts are not relaxed recall/F1 or extra correct
predictions. The separate exact-first, one-to-one pairing conserves both sides:

| Policy | Exact pairs | Boundary pairs | Unpaired gold | Unpaired predictions |
|---|---:|---:|---:|---:|
| Relevant only (current) | 559 | 1,409 | 2,485 | 2,032 |
| Capped background (rejected) | 380 | 967 | 3,106 | 855 |
| All background (rejected) | 268 | 640 | 3,545 | 423 |

For the current model's 1,409 boundary pairs, **745 predictions contain the gold
span plus extra tokens**, 544 lie inside the gold span, and 120 cross its boundary.
There are 877 pairs with the correct start but only 235 with the correct ending.
Predictions average 6.06 tokens versus 4.97 for gold spans. This supports testing
better action-ending features; it does not establish that trimming predictions
would improve the model. Almost half the gold actions have no overlap at all,
so boundary changes alone cannot resolve the problem.

Gold spans range from 1 to 138 tokens (median 4; nearest-rank p90 10). Only 1 of
63 spans of length 20+ is exact under the current model. However, there are also
543 one-token actions with no overlap. The problem is not limited to long spans.
The local examples include nominal actions such as `initial access`; action
coverage cannot be reduced to recognizing a fixed list of verbs.

## Why background training was rejected

The gold-level transition counts show the cost behind the preceding rejection:

- Capped background retains 328 of the baseline's 559 exact actions, changes 50
  to partial overlap and loses all overlap for 181. It gains only 52 exact actions
  from the baseline's non-exact cases, giving 380 exact matches overall.
- All background retains 230 exact actions, changes 39 to partial overlap and
  loses all overlap for 290. It gains 38 exact actions from non-exact cases,
  giving 268 overall.

This comparison uses the same gold spans and disjoint training folds. No new
cutoff, feature candidate or training policy was selected during this audit.

## Source and annotation checks

All **8,635 training sentences** were regenerated in memory with the existing
blank-tokenizer/rule-based BIO converter, using the action layer only:

- Zero invalid action character offsets, zero annotation-text/source-slice
  mismatches, zero conversion errors, and zero token/tag differences from saved BIO.
- The 4,470 raw action annotations project to 4,453 retained gold spans. All 17
  removed unique projected spans overlap retained spans under the converter's
  existing longest-span overlap policy. There are zero duplicate projections,
  empty projections or unexplained retained spans. No converter rule was changed.
- No illegal IOB2 inside-start tags occur in gold or predictions.

There are **132 groups / 327 rows with identical case-sensitive token sequences**
within train. Of those groups, 49 span multiple reports, 21 have different action
BIO labels, and 11 have different relevance labels. The action/relevance groups
can overlap; these are not additive counts of bad examples. Context may explain
some differences, so they are review items rather than automatic corrections.

The existing folds contain 79 validation sentences with identical tokens in
their complementary fit reports (30 / 30 / 19 by fold). All 79 have a matching
tag sequence in fit; none has a conflicting tag sequence there. Report grouping
therefore does not remove every repeated sentence. This is a limitation of the
existing CV estimates, not permission to edit splits after seeing results.

Phrase exposure was checked separately against **each fold's fit-relevant gold
phrases only**, case-folding tokens. Of the 4,453 validation gold spans, 627
phrases were seen and 3,826 unseen. Baseline exact counts are 123 / 436,
respectively. These are descriptive diagnostics, not independent generalization
or causal evidence. No dev/test overlap search was performed.

## Review files and next step

The ignored local file `results/action_error_audit_v1/annotation_review.csv`
contains **all 43 rows from all 21 conflicting-action groups**, including stable
group identifiers, UIDs, report IDs, preceding context, source text, existing
BIO tags and an empty `review_notes` column. Review the groups together and
record whether differing boundaries/labels are explained by context. Do not
overwrite official annotations or silently copy one variant to the others.

`results/action_error_audit_v1/review_examples.csv` contains **48 deterministic
examples**: 12 baseline no-overlap actions, 12 baseline partial overlaps, and 12
exact actions lost by each rejected candidate. Original character offsets and
source excerpts are preserved, alongside predictions from all three policies.
These examples are development review material and cannot become a new holdout.

The next justified model step is a predeclared classical boundary-feature
comparison, guided by reviewed start/end and nominal-action examples. Retain
the current model until training-fold selection and dev confirmation support a
replacement. Annotation conflicts should be reviewed before changing training
labels; any approved correction needs a separate versioned data snapshot and
new cache provenance. The audit does not establish that those 21 groups explain
the model's overall low accuracy. No dataset download is required for this review.

## Validation, scope and reproduction

All **299 WSL tests pass**. Eleven new synthetic tests cover strict BIO behavior,
split/merge diagnostics, adjacent boundaries, exact-first pairing, conservation,
length buckets, transition counts and duplicate annotation checks. Exhaustive
small legal BIO sequences verify the residual pairing against an independent
maximum-matching search. The audit's strict counts reproduce the frozen training
CV metrics for all three policies, and upstream hashes and active-file integrity
checks pass. The integrity report is not an annotation-quality approval.

No model was fitted. Active entity/action/SMS models, held SMS candidate, app,
training configuration, labels and splits are unchanged. No dev/test evaluation
or new browser/remote CI run is claimed. Native Windows spaCy remains blocked
as previously recorded; no security settings changed. Trained Hindi/Hinglish
support remains unavailable; existing phrase rules are unaffected.

Run from the project root in the prepared WSL runtime:

```powershell
wsl -d Ubuntu-24.04 --exec bash -c '~/.cache/cti-classical-py312/bin/python scripts/build_action_error_audit.py'
wsl -d Ubuntu-24.04 --exec bash -c '~/.cache/cti-classical-py312/bin/python -m unittest discover -s tests -v'
```

The builder requires the completed, hash-matching local
[background experiment](ACTION_BACKGROUND_EXPERIMENT.md), including its ignored
OOF/cache files and integrity audit. Those historical prerequisites are absent
from a fresh clone; this command verifies the local snapshot and refuses drift.
Detailed CSVs and logs remain ignored; only compact counts/provenance are in
[the versioned results](../results/action_error_audit_v1/README.md).
No commit or push was made.
