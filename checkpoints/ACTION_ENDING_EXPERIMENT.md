# Classical action-ending feature comparison

Date: 2026-11-15 (Asia/Calcutta)

## Outcome

**Hold the candidate; retain the current action model.** The learned word-pair
profile passes the predeclared training-CV policy, but fails all three dev
confirmation requirements. This step is a completed comparison, not an upgrade
to the app's accuracy. No selection budget was loosened after seeing results.

Six candidate fold fits and one selected full-train fit completed. The selected
artifact reloads with identical dev predictions. Existing entity/action/SMS
artifacts, held SMS candidate, app configuration, training configuration, labels
and splits remain unchanged. No final CTI test or SMS holdout was evaluated.

## Methods and protocol

The [training error audit](ACTION_ERROR_AUDIT.md) found frequent action-ending
errors alongside entirely missed actions. This comparison keeps the current
CRF hyperparameters, morphology, boundary dictionary and relevant-only training,
adding two isolated classical profiles:

- **Ending cues:** fixed clause-marker, punctuation, closing-token,
  coordination/verb-cue, three-token function-class and nominal-suffix features.
  These are token rules, not trained POS/dependency predictions. They do not
  trim spans, assign labels, or require actions to begin with a known verb.
- **Ending edges:** the same fixed cues plus stem-pair statistics for action
  endings and following starts. Counts come only from each fold's fit-relevant
  sentences. All occurrences from the current training sentence are subtracted
  when constructing its training features. A remaining count of at least three
  is required; rates use additive smoothing. Prediction never updates dictionaries.

`ActionEndCRFTagger` is an isolated subclass of the existing classical boundary
CRF. The earlier feature implementation, shared artifact loader and default
training CLI are unchanged. Experimental persistence records profile metadata
and a source hash, and the model checks that hash even when the ordinary NER
loader validates it. `load_candidate` additionally verifies the declared profile.
Changes to the new feature source require retraining; existing default artifacts
do not acquire a new dependency.

The protocol in `configs/action_ending_experiment.yaml` was fixed before fitting:
three report-disjoint train folds, seed 42, CRF c1/c2=1 and 200 iterations.
Each fit uses only the complementary reports' relevant nonempty sentences;
validation includes all sentences. There are 70 train reports / 8,635 validation
sentences across folds and 3,006 relevant sentences for full-train fitting.
Reference predictions are the verified current-model OOF caches, not predictions
from the full-training artifact on its own training data.

Selection requires pooled strict exact-span F1 gain >=0.005, wins in at least two
folds, exact recall loss <=0.01, and background action predictions no more than
5% above baseline. The decision is saved before dev access. Only that winner may
be refitted and confirmed on dev, with F1 gain >=0.002, recall loss <=0.01 and
the same 5% background-increase budget. Artifact creation does not activate it.

## Actual training-CV results

These are development diagnostics on held-out **training reports**, not dev or
final-test accuracy. Existing duplicate-sentence/annotation limitations from the
preceding audit still apply.

| Profile | Fold 1 F1 | Fold 2 | Fold 3 | Pooled F1 | Exact recall | Background action spans |
|---|---:|---:|---:|---:|---:|---:|
| Current baseline | 0.1314 | 0.1419 | 0.1231 | 0.132261 | 0.125533 | 1,544 |
| Ending cues | 0.1374 | 0.1413 | 0.1275 | 0.135546 | 0.131148 | 1,642 |
| Ending edges | 0.1374 | 0.1502 | 0.1249 | **0.137671** | 0.131148 | 1,575 |

Both candidates produce 584 exact matches versus 559 for baseline. Fixed cues
win two folds but fail the minimum gain and background budget. Learned edges
win all three folds, gain 0.005410 pooled F1 and add 2.01% background predictions,
passing the frozen policy. Only learned edges proceed to dev confirmation.

## Actual dev confirmation and release hold

Full-train fitting uses the 3,006 relevant train sentences. The selected candidate
and current active model are compared on all 1,238 official dev sentences, against
the same 608 gold actions:

| Metric | Current model | Selected candidate |
|---|---:|---:|
| Exact matches | 104 | 94 |
| Predicted spans | 733 | 754 |
| Exact precision | 0.141883 | 0.124668 |
| Exact recall | 0.171053 | 0.154605 |
| Exact F1 | **0.155108** | 0.138032 |
| Background action spans | **323** | 343 |

F1 decreases, recall drops by 0.016447, and background findings increase by
6.19%. Every dev confirmation check fails. The release status is
`held_failed_dev_confirmation`; the current action artifact remains active.
The rejected fixed-cue profile was not scored on dev. No additional threshold,
feature profile or retraining was chosen to rescue the failed dev outcome.

The held candidate SHA-256 is
`585a2bbebcb9f9db68fd64a71e6f6883562e4374ee56630d6f75a5c466357c07`.
It is retained under ignored `models/action_ending_v1/candidate_action.joblib`
for reproducibility, with its manifest, fold caches and detailed predictions.
Current dev exact F1 remains **0.2596 for entities / 0.1551 for actions**.

## Verification and review

- All **310 WSL tests pass**, including eleven new synthetic tests for cue
  construction, sentence-excluded edge counts, immutable prediction dictionaries,
  empty positions, input scope, source-drift rejection, persistence round trips,
  selection budgets, fit-only row filtering, no-dev rejection and selection
  frozen before a qualifying synthetic winner's dev access.
- `scripts/audit_action_ending.py` passes. It uses the preceding audit's public
  seqeval-based counter, independently reconstructs all fold fit-UID hashes,
  recounts every policy/fold and rechecks selection/confirmation. It verifies
  actual candidate and baseline dev predictions after loading. It does not
  independently refit CRFs or establish accuracy beyond these development data.
- Existing model/configuration hashes match the preceding experiment. The
  experimental artifact and all large caches/logs remain ignored by Git.
- No new browser or remote CI run is claimed; the failed candidate was not
  integrated into the app. Native Windows spaCy remains blocked as previously
  recorded. No Windows security setting changed.

From the project root with compatible local upstream prerequisites:

```powershell
wsl -d Ubuntu-24.04 --exec bash -c '~/.cache/cti-classical-py312/bin/python -m cti.experiment_action_ending'
wsl -d Ubuntu-24.04 --exec bash -c '~/.cache/cti-classical-py312/bin/python scripts/audit_action_ending.py'
wsl -d Ubuntu-24.04 --exec bash -c '~/.cache/cti-classical-py312/bin/python -m unittest discover -s tests -v'
```

The runner needs the audited local background experiment and its ignored
current-model OOF reference, matching active action artifact and original
action-feature cache hashes. These historical prerequisites are absent from
a fresh clone; `--train-models` alone does not recreate this experiment snapshot.
A fresh reproduction needs matching regenerated upstream snapshots/references.
The runner refuses drift instead of replacing the default model.

Review the new feature class, experiment/config, synthetic tests and
[compact results](../results/action_ending_v1/README.md). No commit or push was made.
No dataset download is needed for this review. Hindi/Hinglish learned support
and the 21 unresolved action-annotation groups remain unchanged.

A possible next classical comparison is explicit beginning/inside/end/unit
sequence labels, converted back to the same BIO spans for evaluation. This
would test end-state modeling rather than adding more handcrafted cues. It
requires a separate frozen protocol and has not been implemented or evaluated
here. Repeated dev comparisons are development work; the official test remains
reserved for a final frozen evaluation.
