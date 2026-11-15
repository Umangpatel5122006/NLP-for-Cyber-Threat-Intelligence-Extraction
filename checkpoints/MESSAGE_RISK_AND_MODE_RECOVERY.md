# Message scam patterns and model-mode recovery

Date: 2026-11-15. Classical NLP/rules only; no additional dependencies,
neural models or LLM APIs. This is a reviewable fix, without commit/push.

## Reported failures and causes

The supplied money/SMS message reproduced **zero extracted findings even with
both CRFs correctly loaded**. The existing system extracted CTI entities,
actions and technical IOCs; it did not classify social-engineering messages.
A phone number alone is not a supported technical IOC or evidence of malice.
The SMAH-CTI NER labels do not supply a general SMS-scam classifier.

The menu disabled unavailable NER options. Missing ignored model files on a
fresh clone, artifacts trained in another runtime, a blocked native import,
or an explicit `--ioc-only` launch can cause this state. Startup failures
were cached permanently; refreshing could not recover after training.
The teammate's exact local cause is not accessible from this checkout.

## Resulting behavior

- `message_risk` is a rule module in all three document configs. It reads the
  untouched original text and writes `metadata.message_risk`. It runs without
  spaCy or saved models, including IOC mode and command-line analysis.
- Local combinations cover money/account/contact lures, credential requests
  with contact targets, and advance-fee lures. Titles such as “PM” are
  supporting authority cues; no person or telephone number is hardcoded.
  Binary searches restrict cue combinations to a request's nearby window.
- The supplied message returns `suspicious` with original evidence for the
  reward, money, bank/credit language, SMS request, number and authority claim.
  It still has zero technical IOC/NER findings, as expected. Risk evidence is
  separate from extraction counts and is retained in JSON exports.
- A prominent banner above the counters shows the risk result. A separate
  panel explains each source excerpt with half-open Unicode code-point spans.
  Ordinary credit notifications return `no_pattern_detected` with an explicit
  statement that this does not establish safety. Some preceding warning/example
  contexts return `review_context`. Local negation guards suppress selected
  negated cues.
- Both menu options remain selectable. Unavailable NER shows its reason,
  setup commands and a retry button; analysis stays disabled for that mode.
  Initial startup can fall back to ready IOC mode, with NER setup still visible.
  Later refreshes preserve the user's selected mode.
- Status refresh retries failed loads while preserving successfully loaded
  pipelines. Missing files, runtime imports, artifact mismatch and explicit
  IOC-only startup have distinct explanations.
- A failed/repeated analysis clears previous results and disables export.
  Editing an uploaded file's text also updates the filename caption.

## Start or repair the local app

From the repository root, stop any old server with Ctrl+C and run:

```powershell
python scripts/start_workbench.py
```

Open http://127.0.0.1:8765. The standard-library launcher checks the project
virtualenv, current Python, and (on Windows) the prepared Ubuntu WSL environment.
It picks a runtime that actually loads the requested models, rather than
silently starting an IOC-only server. On this machine it selected WSL.
It does not install dependencies or change Windows security settings.

For a teammate's fresh clone, model binaries are intentionally ignored by Git.
After installing the pinned dependencies or preparing WSL as described in
[WSL_SETUP.md](../docs/WSL_SETUP.md), generate missing models and start together:

```powershell
python scripts/start_workbench.py --train-models
```

Training is explicit and uses the configured classical train/dev workflow.
Compatible existing artifacts are reused. `--check` checks setup without starting
or training; `--ioc-only` explicitly starts IOC plus scam-pattern checks.
`python -m cti.web` remains available in an already compatible environment.
If a running server's runtime is wrong or it used `--ioc-only`, restart using the
launcher. If missing files are trained in that same runtime, the UI retry can
load them without restarting. Already loaded models still require restart
when their files are replaced.

## Validation

- **174 tests pass in WSL**: `results/message_risk_validation.txt`.
- Native Windows: **10 message-risk, 21 API and 5 launcher tests pass**.
  Their three logs are under `results/message_risk_windows_*_validation.txt`.
  No passing full native NER suite is claimed.
- **15 actual Chrome check groups pass**, including the supplied message in
  both real modes, top warning, evidence, JSON download, benign notification,
  warning context, Unicode upload, desktop/mobile layouts and offline recovery.
- The missing-artifact browser state is simulated at the status HTTP boundary;
  retry and subsequent inference use the real server. Synthetic backend tests
  independently verify recovery after a loader's missing-file failure.
- Actual API responses and readiness are saved in `results/message_risk_v1/`.
  Existing entity/action model hashes match their manifests; the active action
  hash still matches the previous promotion. No model was retrained or replaced.
- Official dataset splits and test annotations were not used for this fix.
  Current changes await their remote CI run after the user's commit/push.

Review the browser behavior by uploading [scam_message.txt](../examples/scam_message.txt),
switching between both modes, and comparing a normal bank credit notification.

## Limits and next work

This is a disclosed English rule baseline, not a statistically validated spam
classifier. It does not fact-check the political claim, verify a sender,
look up phone/domain reputation, or estimate a calibrated attack probability.
Complex negation, quoted text, obfuscation, other scams and languages can be
missed or misread. Rule tests are regression checks, not accuracy measurements.
For measured SMS-scam performance, a separate licensed SMS/ham/spam corpus and
source-aware evaluation are needed; do not report the CTI NER F1 as scam accuracy.

Next model work remains background action false-positive reduction on training
folds. A TF-IDF/linear SMS classifier would be a separate classical task with
its own dataset and evaluation; the official CTI test split stays reserved.

Suggested commit: `fix: detect scam message patterns and recover NER mode setup`.
