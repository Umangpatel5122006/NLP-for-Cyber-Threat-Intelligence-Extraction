# BILOU comparison completed after WSL recovery

Date: 2026-11-16 (Asia/Calcutta)

## Outcome

WSL starts successfully again. The prepared Python 3.12.3 / spaCy 3.8.16
environment imports correctly, all **327 tests pass**, and the workbench
readiness check selects WSL with both **Entities + IOCs** and **IOCs only** ready.
The existing English SMS classifier is also ready. Native spaCy still encounters
Application Control; the working launcher uses WSL. The user reported recovery;
the exact system repair and its cause were not established by this run.

The three planned BILOU training-fold fits completed, and the independent
post-run audit passes. **Retain the current BIO action model.** BILOU improves
each fold slightly but fails the frozen minimum-gain and background budgets.
No dev data were evaluated, no full-train candidate was fitted, and no candidate
artifact was created. The final CTI test and SMS holdouts remain reserved.

Default entity/action/SMS artifacts, the previously held candidates, pipeline/
training configuration, annotations and splits are unchanged. Existing recorded
dev exact F1 remains **0.2596 for entities / 0.1551 for actions**. This experiment
does not upgrade the app's detection accuracy.

## Method retained from preparation

The [preparation checkpoint](ACTION_BILOU_PREPARATION.md) froze the protocol
before any BILOU fit. The candidate uses the same current morphology/boundary
dictionary features and relevant-only training, c1/c2=1, 200 iterations, and
three report-disjoint train folds with seed 42. Only internal labels change:
beginning/inside/last/unit/outside. Gold BIO -> BILOU -> BIO preserves all 4,453
actions across 8,635 train sentences / 70 reports / 223,042 tokens.

The CRF makes an unconstrained BILOU prediction. A fixed, predeclared open-span
adapter returns BIO for the existing strict-span evaluator. All **8,635 internal
OOF sequences are already valid BILOU**; no invalid path required normalization.
This result provides no evidence that changing invalid-path repair would improve
this candidate.

Each fold fits only relevant sentences from its complementary train reports.
The learned dictionary excludes the current sentence during training feature
construction. Baseline OOF predictions and all inputs are checked against the
audited prior snapshot. No full-training predictions are reused as held-out
training predictions. No deep learning or LLM API is introduced.

## Actual training-CV results

These are strict IOB2 exact-span metrics after conversion back to BIO on the
held-out training reports, not dev or final-test accuracy.

| Metric | Current BIO | BILOU |
|---|---:|---:|
| Fold 1 exact F1 | 0.131352 | 0.138767 |
| Fold 2 exact F1 | 0.141904 | 0.142510 |
| Fold 3 exact F1 | 0.123111 | 0.123395 |
| Pooled exact precision | 0.139750 | 0.139147 |
| Pooled exact recall | 0.125533 | 0.131148 |
| Pooled exact F1 | 0.132261 | 0.135029 |
| Exact matches | 559 | 584 |
| Predicted spans | 4,000 | 4,197 |
| Background action predictions | 1,544 | 1,653 |

All three folds improve and recall passes its budget. However, pooled F1 gains
only **0.002768**, below the fixed **0.005** minimum. Background findings increase
by **7.06%**, above the **5%** allowance. Both required checks fail. The runner
selects the baseline before any dev access and skips full fitting, dev scoring
and artifact creation. Budgets were not changed after observing the results.

The duplicate-sentence and annotation-consistency limitations from the prior
error audit still apply. These repeated development comparisons do not establish
statistical significance or independent generalization.

## Verification and review

- The expanded **327-test WSL suite passes**, including the three production-
  stack BILOU tests that could not import in native Windows during preparation.
  The previously passing native conversion/toy-CRF checks are part of full
  discovery; production features are now also exercised in the compatible runtime.
- `scripts/audit_action_bilou.py` passes against the actually completed root-level
  CV/selection/summary outputs. It independently reproduces the fixed adapter,
  checks internal/external prediction hashes, verifies fit-UID inventories and
  report separation, recounts all strict metrics and confirms rejection. It
  does not independently refit CRFs.
- Active artifact/configuration hashes match the baseline. No dev report or
  `candidate_action.joblib` exists for this rejected experiment.
- `python scripts/start_workbench.py --check` exits successfully, selects the
  prepared WSL environment and reports both modes and the SMS classifier ready.
  This is a readiness check, not a new browser test or model-accuracy result.
- No new Chrome or remote CI run is claimed. The agent did not change Windows
  security settings, reset Ubuntu, run DISM/SFC or restart Windows.

Compact current evidence is under [the result folder](../results/action_bilou_v1/README.md)
and `recovery/summary.json`. The earlier `preparation/` reports remain historical
records of the blocker and encoding-only checks. Local fold/OOF/internal-label
caches and validation logs remain ignored by Git.

## Reproduce and continue

From the project root in the prepared runtime:

```powershell
wsl -d Ubuntu-24.04 --exec bash -c '~/.cache/cti-classical-py312/bin/python -m unittest discover -s tests -v'
wsl -d Ubuntu-24.04 --exec bash -c '~/.cache/cti-classical-py312/bin/python -m cti.experiment_action_bilou'
wsl -d Ubuntu-24.04 --exec bash -c '~/.cache/cti-classical-py312/bin/python scripts/audit_action_bilou.py'
python scripts/start_workbench.py --check
```

The experiment requires the compatible local background-experiment snapshot
and its ignored reference caches. They are absent from a fresh clone; normal
default-model training alone does not recreate historical experiment metadata.
Manifest and source-drift checks remain enabled.

To use the existing app, start frontend/backend together with:

```powershell
python scripts/start_workbench.py
```

No dataset download, commit or push was made. Review the accumulated prepared
code, tests and completed results before committing. The next priority is the
unresolved action-definition/annotation review and repeated-sentence limitations
identified in the error audit, rather than assuming another endpoint tweak will
make the extractor reliable. Hindi/Hinglish trained support still needs reviewed
real labeled data; the existing phrase rules are unaffected.
