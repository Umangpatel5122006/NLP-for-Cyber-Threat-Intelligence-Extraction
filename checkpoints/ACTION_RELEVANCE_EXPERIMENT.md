# Train-only action relevance experiment

Date: 2026-11-15 (Asia/Calcutta)

## Outcome

Keep the current action model and pipeline. None of the four predeclared
TF-IDF/Logistic Regression relevance cutoffs passes the selection policy.
This is a completed negative experiment, not a model accuracy upgrade.

The default app, entity/action/SMS artifacts, held SMS candidate and Hindi/
Hinglish phrase rules are unchanged. No dev text or annotations were evaluated
by this experiment; the official CTI test and SMS holdouts remain reserved.
No frontend restart or dataset download is needed to review this step.

## Method and selection

The existing action CRF was trained only on relevant sentences, but its
report-grouped training-CV predictions contain 1,544 action spans on sentences
labeled irrelevant. Test whether a separately learned sentence relevance check
can suppress enough of these predictions without losing important context.

The experiment uses 8,635 sentences from the 70 official **train** reports,
including 3,006 relevant sentences. Three report-disjoint folds use seed 42.
Each new word unigram/bigram TF-IDF vectorizer and balanced Logistic Regression
classifier fits only the complementary training reports. Shared text cleaning
and defanging run before vectorization. Scores are sentence relevance scores;
they are not calibrated threat probabilities.

The action predictions come from the previously frozen boundary-feature CRF's
three held-out training folds. The runner checks their source/config/data/
runtime provenance, fold membership, prediction hashes and token alignment,
matches the active action profile, and reproduces the prior CV metrics before
fitting relevance models. It does not reuse full-training action predictions
as held-out predictions. No CRF is refitted in this step.

`configs/action_relevance_experiment.yaml` was fixed before fitting: cutoffs
0.1, 0.2, 0.3 and 0.5; minimum pooled exact-span F1 gain **0.005**; wins in at
least two folds; maximum exact recall loss **0.01**; and maximum skipped
relevant-sentence rate **0.03**. A sentence below the cutoff has all action
predictions suppressed. Relevant-sentence coverage and exact action recall
have different denominators and are checked separately. Budgets were not
loosened after inspecting results.

## Actual training-CV results

These are pooled strict IOB2 exact-span metrics on held-out **training folds**,
not dev or final-test results.

| Cutoff | Exact F1 | Exact recall | Predicted spans on irrelevant sentences | Relevant sentences skipped | Decision |
|---|---:|---:|---:|---:|---|
| No filter | 0.132261 | 0.125533 | 1,544 | 0 / 3,006 | Retain |
| 0.1 | 0.132261 | 0.125533 | 1,544 | 2 / 3,006 (0.07%) | Insufficient F1 gain |
| 0.2 | 0.133079 | 0.125533 | 1,494 | 25 / 3,006 (0.83%) | Insufficient F1 gain |
| 0.3 | 0.133950 | 0.123512 | 1,341 | 167 / 3,006 (5.56%) | Fails gain and coverage |
| 0.5 | 0.128784 | 0.105098 | 738 | 937 / 3,006 (31.17%) | Fails gain, recall and coverage |

At 0.2, all three folds improve slightly and 50 background predictions disappear,
but pooled F1 gains only 0.000819. At 0.5, suppressing many predictions also
discards almost a third of relevant sentences. This does not justify adding a
required relevance model to the default app. The runner writes `selection.json`
before any possible dev access; since the baseline wins, it skips full fitting,
dev confirmation and artifact creation entirely.

Existing English dev exact F1 remains **0.2596 for entities / 0.1551 for actions**.
Action extraction still needs substantial improvement. This result does not
establish statistical significance, robust detection or multilingual learned
accuracy. Evaluation uses frozen processed BIO tokens and dataset sentence
boundaries; it does not measure whole-report segmentation or end-to-end UI quality.

## Verification and review

- All **279 tests pass** in the prepared Ubuntu-24.04 WSL Python 3.12 environment.
  Nine added synthetic tests check gating, strict counts, coverage budgets,
  cache validation and that a rejected run never loads dev data or invokes the
  full action model. Split-lock tests do not need official test annotations.
- `scripts/audit_action_relevance.py` passes. It reloads train only, checks every
  UID/report/gold/relevance/token alignment and cached action prediction,
  reconstructs report-disjoint folds and independently recounts every candidate
  and fold. Counts also agree with public strict seqeval metrics. It independently
  rechecks the selection policy, but does not refit the relevance classifiers.
- All four existing entity/action/SMS/candidate model hashes match the preceding
  checkpoint. No new relevance artifact or dev result exists for this experiment.
- Local OOF rows, training logs and test logs remain ignored by Git. Only compact
  selections, summary, audit and documentation are included for review.
- No new browser run or remote CI run is claimed. Application code did not change.
  Native Windows spaCy remains blocked as previously recorded; security settings
  were not changed. The 279-test result is from WSL.

From the project root in the prepared runtime:

```powershell
wsl -d Ubuntu-24.04 --exec bash -c '~/.cache/cti-classical-py312/bin/python -m cti.experiment_action_relevance'
wsl -d Ubuntu-24.04 --exec bash -c '~/.cache/cti-classical-py312/bin/python scripts/audit_action_relevance.py'
wsl -d Ubuntu-24.04 --exec bash -c '~/.cache/cti-classical-py312/bin/python -m unittest discover -s tests -v'
```

These commands verify this local snapshot. The runner requires the compatible
active action artifact with its selection metadata and three fold caches from
[the boundary-feature experiment](ACTION_BOUNDARY_FEATURES.md), at their original
`models/action_features_v1/` paths. They are ignored and absent from a fresh
clone. The audit also expects the preceding checkpoint's exact local model
hashes. `--train-models` alone does not recreate those historical prerequisites.
The earlier checkpoint describes an isolated retraining comparison; its separate
check directories do not automatically satisfy this snapshot's paths or hashes.
A fresh retraining needs a new matching experiment snapshot/reference. This
runner refuses missing or mismatched prerequisites rather than silently
recreating them or replacing active artifacts.

Review [the compact results](../results/action_relevance_v1/README.md),
`configs/action_relevance_experiment.yaml`, `src/cti/experiment_action_relevance.py`,
`scripts/audit_action_relevance.py` and `tests/test_action_relevance.py` before
committing. No commit or push was made.

The next justified model experiment is to train the action CRF with background
negative sentences, selecting the training policy on train-report folds. That
would address false action predictions during learning rather than relying on
a separate filter. It has not been run in this step. Hindi/Hinglish learned
classifiers still need independently reviewed real labeled data; the supplied
mixed-origin download remains a development diagnostic, not a trusted new split.
