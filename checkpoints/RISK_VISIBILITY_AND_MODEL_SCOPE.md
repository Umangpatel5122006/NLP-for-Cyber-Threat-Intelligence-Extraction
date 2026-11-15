# Evidence-map risk visibility and model-scope audit

Historical intermediate snapshot. The later
[integrated Findings/SMS step](INTEGRATED_FINDINGS_AND_SMS_MODEL.md) supersedes
its statement that no SMS classifier exists and adds the combined review list.

Date: 2026-11-15. Follows committed `d053e58`. No model retraining, dependency
changes, commits, pushes or held-out test evaluation in this step.

## Reproduction

The screenshot's 152-character message variant (plain newlines and a final
newline) returns `message_risk.status = suspicious` in both real API modes:
**7 source cues, 0 extracted entities/technical findings**. See
`results/risk_visibility_v1/screenshot_ner.json` and `screenshot_ioc.json`.

The previous evidence map highlighted only entity/IOC findings. It looked empty
while the separate rule verdict was above/below the screenshot. That did not
mean the rule layer considered the message safe. The screenshot's running server
was unavailable during this check; responses were reproduced on a fresh server.

## UI change

The evidence map now repeats the verdict above the source report. Orange cue
highlights use original source offsets; click or keyboard activation focuses
the corresponding explanation in the risk panel. Legend/footer text distinguishes
extraction observations from message-risk cues.

Rule evidence remains separate from model/IOC findings and JSON counts.
Extraction selection retains priority where annotations overlap. Warning/example
context retains review wording. Invalid cue offsets are skipped; highlights
preserve original Unicode text. Edits/failed requests clear the inline verdict.

## Actual model quality

| Component | Task | Recorded evidence |
|---|---|---|
| Entity CRF | CTI span extraction | Dev exact-span F1 **0.2596** |
| Action CRF | CTI action extraction | Dev exact-span F1 **0.1551** |
| English scam rules | Local cue combinations | Regression tests; **no measured SMS-classifier accuracy** |

Sources: `results/ner_v2_entities.json` and
`results/action_features_v1/action_boundaries_dev.json`. These scores are not
SMS-scam classification scores or final test results. Statistical models remain
research baselines with substantial missed/spurious spans. A working UI, passing
software tests and one flagged scam example do not establish strong generalization.
There is currently no trained general SMS-scam classifier.
See `results/risk_visibility_v1/model_scope_audit.json` for the machine-readable audit.

## Next model work

Build a separate classical SMS classifier using word/character TF-IDF with
Logistic Regression or a linear SVM. Use duplicate-aware train/dev splitting,
fit vocabularies on training only, and measure positive-class precision, recall,
F1, confusion counts and false positives. Select thresholds on dev; freeze the
model before final test evaluation.

[UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms%2Bspam%2Bcollection)
is a verified starting corpus with SMS ham/spam labels and CC BY 4.0 licensing.
Spam labels do not provide universal fraud/threat ground truth. Modern scam
detection requires appropriate examples and independent evaluation in the intended
message domain. No SMS corpus was downloaded or trained in this step.
CTI extraction improvements remain separate; its official test split is reserved.

## Verification and review

**17 real Chrome check groups passed**, including the plain-newline screenshot
variant in both modes, inline verdict, cue click/keyboard behavior, overlapping
IOC/risk spans, Unicode, mobile layout, exports and setup recovery.
See `results/risk_visibility_v1/browser/web_browser_validation.json` and PNGs.
The first overlap fixture used a suffix outside the IOC finder's recognized set;
it was corrected to the existing supported `example.com` form and full browser
QA passed. No IOC detector behavior changed.

JavaScript syntax and whitespace checks passed. Python/model code is unchanged.
The previous checkpoint's **174-test WSL** and **36-test Windows subset** logs
remain the suite evidence; no new full-suite run is claimed here.

Stop the old server with Ctrl+C, run `python scripts/start_workbench.py` from the
project root, reload the browser and analyze the message. A separate QA server
was used and stopped afterward.

Suggested commit: `fix: show message-risk verdict and cues inside evidence map`.
