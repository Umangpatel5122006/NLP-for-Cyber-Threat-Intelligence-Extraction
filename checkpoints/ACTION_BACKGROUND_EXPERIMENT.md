# Action CRF background-training comparison

Date: 2026-11-15 (Asia/Calcutta)

## Outcome

**Retain the current relevant-only action model.** All six planned candidate
fold fits completed successfully. Neither added-background policy wins any fold
or meets the fixed gain/recall requirements. The selection was written before
dev access; no full candidate fit, dev evaluation or candidate binary was needed.
No official test or SMS holdout was evaluated.

These are pooled strict exact-span metrics on held-out **training folds**, not
dev or test accuracy:

| Training policy | Exact precision | Exact recall | Exact F1 | Background action spans | Correct exact action spans |
|---|---:|---:|---:|---:|---:|
| Relevant only (current) | 0.139750 | 0.125533 | **0.132261** | 1,544 | 559 |
| Capped background | 0.172570 | 0.085336 | 0.114200 | 621 | 380 |
| All background | 0.201352 | 0.060184 | 0.092669 | 303 | 268 |

All candidates are evaluated against the same 4,453 gold action spans. Capped
background reduces background predictions by 59.8%, and all background by 80.4%,
but the accompanying recall losses exceed the permitted 0.01. More background
data alone is not a successful upgrade under this feature/regularization profile.
This does not establish that every possible negative-training approach fails.

App behavior, entity/action/SMS artifacts, held SMS candidate and normal training
configuration are unchanged. Existing dev exact F1 remains **0.2596 (entities) /
0.1551 (actions)**. This step does not improve the deployed model's accuracy;
action extraction still needs substantial work.

## Scope

Compare fixed training-data policies for the existing English action boundary
CRF. This experiment writes isolated caches/results and never replaces the
app's active artifacts or normal training configuration. It does not introduce
deep learning, add a new language model or change frontend/backend behavior.

The preceding [relevance-filter experiment](ACTION_RELEVANCE_EXPERIMENT.md)
did not meet its gain/recall/coverage budgets. This step tests learning directly
from background examples instead of adding that filter.

## Fixed protocol

The official train split has **70 reports / 8,635 nonempty sentences**: 3,006
relevant sentences and 5,629 background sentences. Every background sentence
has all-O action annotations; the runner checks this before fitting and never
rewrites annotations to create negatives. Relevant sentences with all-O action
annotations are also retained, just as in the existing baseline.

`configs/action_background_experiment.yaml` fixes the current morphology/
boundary-dictionary feature profile, CRF c1/c2=1, 200 iterations, seed 42 and
three report-disjoint folds. Only the training-data policy varies:

1. **Relevant only:** current profile, using verified held-out predictions from
   the preceding boundary-feature experiment.
2. **Capped background:** keep every nonempty relevant fit sentence; within each
   fit report, add at most one all-O background sentence per relevant sentence.
   Order background candidates by SHA-256 of `seed:uid`, with UID as tie-breaker.
   Reports without relevant sentences contribute no negatives under this policy.
3. **All background:** retain every nonempty fit sentence, including background
   reports without relevant sentences.

Sampling happens after splitting reports. Held-out report text, annotations and
counts cannot influence the fitted boundary dictionary or background selection.
Each fold cache records fit report IDs, held-out report IDs, positive/background
counts and a digest of selected UIDs. The runner verifies the baseline's original
source/config/data/runtime signatures, cache hashes and token alignment, and
reproduces its exact CV metrics before using the reference predictions.

Selection requires pooled strict IOB2 exact-span F1 gain **>=0.005**, wins in at
least **two folds**, exact recall loss **<=0.01**, and at least **10% fewer action
spans on background sentences**. These requirements were fixed before fitting.
There is no sentence gate in this experiment, so skipped-sentence counts are zero.

`selection.json` is saved before any dev access. Only a qualifying train-CV
winner may be refitted on the full training set and checked against the current
active model on dev. Dev confirmation requires gain >=0.002, recall loss <=0.01
and background reduction >=10%, with identical predictions after saving/reloading
the isolated candidate. No automatic activation is performed.

## Verification

All **288 tests pass** in the prepared Ubuntu-24.04 WSL Python 3.12 environment.
Nine added synthetic tests cover per-report caps, stable sampling, held-out
exclusion, contradictory annotations, selection budgets, rejection without dev
access, and a qualifying synthetic winner saved before dev confirmation/reload
checks. No real official test annotations are needed for these tests.

`scripts/audit_action_background.py` independently reconstructs the sampling
inventories, compares every cached held-out prediction and recounts strict spans
with public seqeval metrics. It rechecks eligibility without importing the
experiment's scoring/selection/sampling functions. It does not independently
refit CRFs. Dev is inspected only if the completed experiment already selected
and confirmed a training-CV winner.

The audit passes for this completed run: 8,635 training sentences / 70 reports,
all three policies and all fold sampling inventories are verified, with unchanged
active model/configuration hashes. No dev file or candidate binary exists.
Test success verifies implementation behavior; it does not establish detection
accuracy. The saved audit recounts predictions and does not independently refit
the six CRFs.

No new native Windows NER pass, browser run or remote CI run is claimed. WSL was
used because native spaCy remains blocked as previously recorded. No Windows
security setting was changed.

## Reproduce and review

From the project root with this snapshot's compatible local prerequisite caches:

```powershell
wsl -d Ubuntu-24.04 --exec bash -c '~/.cache/cti-classical-py312/bin/python -m cti.experiment_action_background'
wsl -d Ubuntu-24.04 --exec bash -c '~/.cache/cti-classical-py312/bin/python scripts/audit_action_background.py'
wsl -d Ubuntu-24.04 --exec bash -c '~/.cache/cti-classical-py312/bin/python -m unittest discover -s tests -v'
```

Like the preceding relevance comparison, this runner requires the original
`models/action_features_v1/action_boundaries_fold_*.json` files and the matching
active action artifact with its selection metadata. These ignored historical
files are absent from a fresh clone. `--train-models` alone does not recreate
them. A fresh retraining needs a new matching upstream experiment snapshot and
reference; missing/mismatched prerequisites are refused without replacing the
current app's artifacts.

Compact CV, selection, summary and audit reports are under
[results/action_background_v1](../results/action_background_v1/README.md).
OOF predictions, per-fold caches, training/test logs and any candidate binaries
remain ignored. No additional dataset was downloaded. No commit or push was made.

These figures describe frozen processed BIO tokens and dataset sentence
boundaries, not raw whole-report segmentation, Hindi/Hinglish learning or
statistically established generalization. The official CTI test and SMS holdouts
remain reserved. Hindi/Hinglish phrase rules remain available; learned support
still needs independently reviewed real data.

The next justified step is a train-only action error audit: distinguish missed
actions from boundary mismatches and examine how the background policies change
them across reports. Use that evidence to predeclare another classical feature/
training comparison rather than loosening the failed selection budgets. That
follow-up audit/experiment has not been completed in this step.
