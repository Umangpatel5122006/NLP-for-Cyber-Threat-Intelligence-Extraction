# Fit-only action consistency comparison

Date: 2026-11-16 (Asia/Calcutta)

## Outcome

Keep the current action model and original training configuration. Six planned
candidate fold fits completed. Omitting ambiguous fit inputs improves training-
CV exact F1 only from **0.132261 to 0.133239**, below the fixed 0.005 gain budget.
Adding deduplication lowers F1 to **0.130532**. Neither candidate qualifies for
dev confirmation. No full-train candidate was fitted or saved, and dev/test
examples were not evaluated. This step does not improve deployed accuracy.

All **350 WSL tests pass**, and the independent inventory/count audit passes.
Original labels, official splits, active models and app behavior are unchanged.
All 21 semantic review decisions remain pending; no review finding or corrected
gold label was invented to make the candidates trainable.

## Why this test can run before semantic adjudication

The review pack identifies 43 conflicting records across all train sentences.
The current action model fits only relevant sentences. Among those actual fit
examples, 11 exact-token groups / 22 rows have conflicting BIO labels. Background
all-O labels are not used to call a relevant fit input ambiguous: background
sentences are excluded before grouping, as in the existing training policy.

Two policies were fixed in `configs/action_consistency_experiment.yaml` before
fitting:

1. `consistent_inputs`: within each fold's fit-relevant, nonempty rows, omit all
   members of groups with identical case-sensitive tokens and different tags.
   No majority label is chosen and no tag is rewritten.
2. `unique_consistent_inputs`: apply the same omission, then keep one occurrence
   of each remaining identical-token/tag group, choosing the smallest stable UID.

The policies inspect **fit rows only**. Held-out labels/text do not decide which
fit rows are ambiguous, and the available model predictions are not used to
make that decision. Every original validation sentence and gold span remains
in scoring. This is a fitting policy comparison, not a cleaned evaluation set.

The features, CRF c1/c2=1, 200 iterations, three report-disjoint train folds and
seed 42 remain fixed. Baseline OOF predictions reproduce the previous strict
counts under source/config/data/runtime hash guards. The selection policy is
unchanged: pooled exact F1 gain >=0.005, two fold wins, recall loss <=0.01 and
background predictions no more than 5% above baseline.

## Actual fit inventories and training-CV results

| Fold | Baseline relevant fit rows | Omit ambiguous: selected / omitted | Also dedup: selected / duplicate rows omitted |
|---|---:|---:|---:|
| 1 | 1,896 | 1,892 / 4 | 1,890 / 2 |
| 2 | 2,036 | 2,016 / 20 | 1,995 / 21 |
| 3 | 2,080 | 2,060 / 20 | 2,039 / 21 |

The deduplication column additionally omits the ambiguous rows from the preceding
column. If applied to all train data, these policies would retain 2,984 / 2,962
of 3,006 relevant rows. Those full-train inventories were audited without fitting
full-train candidates; they are not new trained artifacts.

All scores below use the original 4,453 gold actions across all 8,635 held-out
training sentences:

| Policy | Exact matches | Predicted spans | Exact recall | Pooled exact F1 | Background predictions | Fold wins |
|---|---:|---:|---:|---:|---:|---:|
| Current baseline | 559 | 4,000 | 0.125533 | 0.132261 | 1,544 | Reference |
| Omit ambiguous inputs | 565 | 4,028 | 0.126881 | 0.133239 | 1,552 | 2 |
| Also deduplicate | 553 | 4,020 | 0.124186 | 0.130532 | 1,561 | 0 |

Both candidates satisfy the recall/background budgets but fail the minimum gain.
The first gains only 0.000978 F1; the second worsens F1 in all folds. The baseline
decision is written before any possible dev access. Budgets were not loosened
after observing the results.

These results do not establish that consistent annotation or deduplication is
generally unhelpful. They give little support for these particular fit filters
as a sufficient fix under the current representation. The original action
definition still needs review, and most prior exact-span errors occur outside
the 43 known conflict rows. Repeated development comparisons are not independent
generalization evidence. Current dev F1 remains 0.2596 (entities) / 0.1551 (actions).

## Verification and reproduction

Nine new synthetic tests cover ambiguity removal without label selection,
stable deduplication, held-out text/label independence, exclusion of background
labels from ambiguity decisions, invalid/empty fit sets, selected fit inputs,
configuration budgets and rejection without dev access or active-model prediction.
The full WSL suite now passes **350 tests**.

`scripts/audit_action_consistency.py` independently rebuilds each fit inventory,
verifies UID/prediction hashes and report separation, recounts original-gold metrics
and rechecks rejection. All validation rows are retained, and raw/BIO data and
active model/configuration hashes match the frozen reference. The auditor does
not independently refit CRFs. No new browser or remote CI run is claimed.

```powershell
wsl -d Ubuntu-24.04 --exec bash -c '~/.cache/cti-classical-py312/bin/python -m cti.experiment_action_consistency'
wsl -d Ubuntu-24.04 --exec bash -c '~/.cache/cti-classical-py312/bin/python scripts/audit_action_consistency.py'
wsl -d Ubuntu-24.04 --exec bash -c '~/.cache/cti-classical-py312/bin/python -m unittest discover -s tests -v'
```

The runner needs the compatible local audited background-experiment snapshot and
ignored baseline caches. They are absent from a fresh clone; normal default-model
training alone does not recreate historical experiment metadata. It refuses
drift rather than overwriting active models. Compact evidence is under
[the result folder](../results/action_consistency_v1/README.md); fold/OOF caches
and logs remain ignored. No dataset download, commit or push was made.

Further minor action-feature/filter comparisons should not be treated as a
substitute for deciding the annotation target and obtaining trustworthy coverage.
The review workbook remains available for actual findings. The next development
priority should reflect the team's demo goal: scam-message detection and English
report entity/action extraction are separate tasks with different data needs.
