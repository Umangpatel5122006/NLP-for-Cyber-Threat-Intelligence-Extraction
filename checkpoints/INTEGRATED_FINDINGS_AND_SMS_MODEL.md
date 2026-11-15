# Unified review findings and trained classical SMS model

Date: 2026-11-15. Builds on the earlier uncommitted visibility changes, preserving
the user's working tree. No commit/push, neural model, dependency installation,
Windows policy change or official CTI test evaluation.

## Reported failure

The UI highlighted seven rule cues but its Findings list read only CTI
`Document.findings`. The message therefore had an empty review list even though
the risk panel flagged it. This was a result-contract gap across frontend/backend.

## Backend and frontend

- `Document.review_findings` is the combined human-review contract. Existing
  `findings`, `entities` and `overlaps` keep their extraction meanings and IDs.
- The result merger builds typed rows for extraction, positive message rules,
  and positive SMS-model predictions. Rule rows group their original source cues;
  model rows identify the classifier, margin and supporting TF-IDF features.
- The Findings count, list, search, label filter, detail panel and JSON download
  now include all these rows. IOC/entity/action counters still count extraction.
  The supplied message shows **1 review finding with 7 evidence cues**.
- Inline highlights retain exact original code-point spans. The detail panel
  scrolls internally instead of stretching the entire findings area.
- Rule/model disagreements remain visible. A ham prediction does not remove a
  suspicious rule finding. Warning/example context retains a review outcome.
  Normal messages may legitimately have no review findings; absence is not safety.
- Missing/incompatible SMS artifacts and optional prediction failures do not
  discard rule or IOC results. Status exposes SMS readiness. Messages over
  1,000 characters or non-English messages skip this short-message model explicitly.
- Both workbench modes include the model when available. Default NER CLI config
  and `configs/config_message.yaml` also integrate it; the basic IOC config
  remains usable without statistical NLP imports.

## Model and data

The public UCI corpus was downloaded from its official source with a pinned hash.
Attribution, license, grouping, split counts and limitations are in
[SMS_DATASET.md](../docs/SMS_DATASET.md). No extra package was needed.

Three fixed candidates used three training-only template-grouped CV folds:

| Profile | Pooled training CV spam F1 |
|---|---:|
| Word TF-IDF + balanced Logistic Regression | 0.9258 |
| Word/character TF-IDF + balanced Logistic Regression | 0.9449 |
| Word/character TF-IDF + balanced linear SVM | **0.9614** |

Maximum pooled F1 selected the SVM before dev was loaded. Decision threshold
stayed **0**, without tuning to dev or the user's example. The selected final
fit used 3,981 training rows. On **796 dev rows**:

| Spam precision | Spam recall | Spam F1 | TP | FP | FN | TN |
|---:|---:|---:|---:|---:|---:|---:|
| 1.0000 | 0.9626 | **0.9810** | 103 | 0 | 4 | 689 |

This is a development result for UCI ham/spam, not final-test accuracy, verified
scam detection or CTI NER accuracy. The 797-row SMS holdout was not evaluated.
Artifact reload reproduces all dev predictions exactly; an independent rescore
matches the report and verifies hashes, split membership and group separation.

**Known domain miss:** the user's political money/SMS message is predicted ham
by the SVM (margin about -0.2673). English rules catch it, the combined assessment
stays suspicious, and the UI displays the disagreement honestly. No threshold
or training label was changed to force this example. More representative,
independently labeled scam data is needed for stronger domain performance.

Existing CTI models were neither retrained nor replaced: their recorded dev
exact-span F1 remains 0.2596 (entities) and 0.1551 (actions). These remain limited
research baselines and are not covered by the new SMS F1.

## Validation and artifacts

- **193 full-suite tests pass in WSL**, including 10 new model/data tests,
  6 combined-review tests and 3 additional launcher tests:
  `results/integrated_findings_validation.txt`.
- Native Windows subsets: **21 API, 6 review and 8 launcher tests pass**.
  Their logs are `results/integrated_findings_windows_*.txt`.
  No full native NER/SMS training pass is claimed.
- **20 actual Chrome check groups pass**. They cover the populated scam row,
  seven cue cards, counts/filter/search/export, model-only detections, model/rule
  disagreement, contextual warnings, Unicode, mixed IOC/risk evidence, mobile,
  setup recovery, stale pre-upgrade server responses and intentional offline failure.
  An old response without the combined contract now shows restart instructions
  instead of silently rendering an empty list.
- Eight actual API examples (four cases in each mode) are stored under
  `results/integrated_findings_v1/`. Positive rule-only and model-only cases,
  a benign notification and warning context are included.
- `results/message_model_v1/` holds frozen selection, dev metrics/predictions and
  integrity audit. `models/message_spam.joblib` and its checked manifest are ignored.
- The CI workflow now includes the SMS train/dev/reload path. It has been
  prepared; the current changes have **not** passed remote CI until pushed/run.

## Run and review

Stop the old server with Ctrl+C, then:

```powershell
python scripts/start_workbench.py
```

Reload http://127.0.0.1:8765 and analyze the supplied message. The Findings
section should contain the message-pattern row; select it to see seven cues.
The model/rule disagreement is shown in Message assessment.

For a fresh checkout with installed dependencies or prepared WSL:

```powershell
python scripts/start_workbench.py --train-models
```

This trains missing/incompatible NER models as needed and the missing SMS model
from the pinned public corpus. `--train-message-model` trains only the SMS model
when unavailable before starting a compatible ready server. `--check` never
trains or starts. IOC mode prefers a runtime with the SMS model when possible.
No implicit training occurs on ordinary startup.

Direct reproduction in the prepared WSL environment:

```powershell
wsl -d Ubuntu-24.04 --exec bash -c '~/.cache/cti-classical-py312/bin/python -m cti.train_message --download'
wsl -d Ubuntu-24.04 --exec bash -c '~/.cache/cti-classical-py312/bin/python -m cti.analyze --config configs/config_message.yaml --input examples/scam_message.txt --output results/message_review_demo.json'
```

Further work: broader scam-domain labels/evaluation and CTI entity/action
false-positive/boundary improvements. Zero-error detection is not established.
Suggested commit: `feat: unify review findings and add classical SMS classification`.
