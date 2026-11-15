# Learned SMS expansion: candidate improves coverage but is held

Date: 2026-11-15. Reviewable local step; no commit or push performed.

The new candidate learns word/character TF-IDF + linear SVM weights from
**4,236 training rows**, including 255 additional source-audited messages.
It improves supplemental dev coverage substantially, but creates extra false
alarms on authored diagnostic messages. **The active default model is unchanged.**
This is a measured candidate experiment, not a completed universal scam model.

## Data and selection

[Supplement provenance](../docs/SMS_DOMAIN_DATA.md) records the CC BY 4.0 source,
mirror limitation, byte hashes and exclusion of UCI exact/near overlaps. Of
5,971 source rows, only 427 survive. Train/dev/test are 255/86/86; UCI remains
3,981/796/797. Both test partitions and the official CTI test are reserved.

`configs/message_expansion.yaml` fixes four candidates, three grouped training
CV folds and eligibility budgets. Stratification uses source and original label;
feature vocabularies and weights are fitted only on each fold's training rows.
Selection uses mean per-source binary spam F1, with a UCI false-positive budget,
and is written before dev loading. Threshold remains zero; no dev tuning.

| Training CV candidate | UCI spam F1 | Supplement spam F1 | Mean source F1 | Extra UCI FP | Decision |
|---|---:|---:|---:|---:|---|
| UCI reference | 0.9512 | 0.8075 | 0.8793 | 0 | Reference |
| Expanded raw text | 0.9555 | 0.9029 | 0.9292 | 4 | Selected; wins all 3 folds |
| Expanded normalized text | 0.9500 | 0.9153 | 0.9327 | 9 | Exceeds FP budget |
| Expanded normalized + concepts | 0.9510 | 0.9040 | 0.9275 | 9 | Exceeds FP budget |

The **selected model uses the existing raw word/character feature pipeline**.
Its gain comes from added labeled data. The optional normalized/concept feature
experiments remain isolated and were rejected. No user sample, authored
challenge message, rule verdict or neural method was added to training.
The unchanged legacy factory preserves compatibility with existing artifacts.

## Selected dev comparison

| Partition/method | TP | FP | FN | TN | Spam precision | Spam recall | Spam F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| UCI, active baseline | 103 | 0 | 4 | 689 | 1.0000 | 0.9626 | 0.9810 |
| UCI, candidate | 102 | 0 | 5 | 689 | 1.0000 | 0.9533 | 0.9761 |
| Supplement, baseline | 16 | 1 | 17 | 52 | 0.9412 | 0.4848 | 0.6400 |
| Supplement, candidate | 31 | 1 | 2 | 52 | 0.9688 | 0.9394 | 0.9538 |

Candidate recall on the supplement's original smishing subtype is **21/23**.
All four predefined dev gates pass; the small UCI F1 reduction stays inside
the fixed 0.005 budget. Source-mean dev F1 increases from 0.8105 to 0.9650.
These small, related-source dev partitions do not establish deployment accuracy.

## Release hold and the reported message

The 32 messages in `examples/message_challenge.json` are assistant-authored
synthetic diagnostics, clearly labeled as such. They are read after selection
and dev gating, never fitted, and never used to choose a candidate or threshold.
They are not an independent real-world test set.

| Diagnostic model | TP | FP | FN | TN | F1 |
|---|---:|---:|---:|---:|---:|
| Baseline | 7 | 2 | 7 | 16 | 0.6087 |
| Candidate | 10 | 6 | 4 | 12 | 0.6667 |

The candidate catches three more positives but adds four false alarms, including
delivery/job notifications and a warning. Browser QA also finds a false alarm
on “Your bank account was credited with INR 1000. No action is required.”
The aggregate F1 gain conceals this tradeoff. A **post-experiment release hold**
therefore blocks default activation; it is not presented as a predeclared
selection gate. `eligible_for_activation` in the summary records the quantitative
dev gates only; `deployment_hold` takes precedence and `--activate` rejects it.

The exact reported message containing “in you bank account” still predicts
**ham** with the candidate, including shortened/missing-number variants. The
separate rules v2 continue to flag all those variants in both app modes. A
wording variation with “your bank account” flips to spam with a small positive
margin. That is evidence of a remaining lexical weakness, not a resolved model
failure. `integrity_audit.json` records original/shortened/missing-target probes.

Next data work should add independently labeled legitimate financial/delivery
notifications and context warnings, together with diverse real scams. Collect
and split these before another training comparison. Do not train on the current
challenge or repeatedly optimize against these dev reports.

## Preview and reproduce

The candidate is already trained locally. Start an isolated preview alongside
the default app, then review both model verdicts and the Findings evidence:

```powershell
python scripts/start_workbench.py --port 8766 --message-model models/message_generalization_v1/candidate.joblib
```

The launcher forwards a project-relative path to the compatible WSL runtime.
It refuses to silently substitute a missing candidate. The assessment displays
`expanded_smishing_v1`, while default inference displays `uci_sms_v1`. Entity
and action models are unchanged. Neither preview nor readiness writes models.

For a fresh clone, prepare the runtime in [WSL_SETUP.md](../docs/WSL_SETUP.md).
The experiment requires the existing baseline artifact; train that first if
absent. Run these commands in the **same compatible ML runtime** from the root:

```bash
python -m cti.train_message --download
python -m cti.experiment_message --download
python scripts/audit_message_expansion.py
python -m unittest discover -s tests -v
```

Skip the first command when the saved baseline is already compatible. Generated
binary hashes can differ across runtimes; artifact loaders check runtime and
feature provenance. Reproduction downloads the pinned mirror CSV, not the
403-blocked publisher endpoint. No test evaluation is performed by these commands.

Browser QA (Node + installed Chrome; servers must already be running):

```powershell
node scripts/check_web_ui.mjs 'C:\Program Files\Google\Chrome\Application\chrome.exe' http://127.0.0.1:8765 results/message_generalization_v1/browser_default
node scripts/check_web_ui.mjs 'C:\Program Files\Google\Chrome\Application\chrome.exe' http://127.0.0.1:8766 results/message_generalization_v1/browser_candidate expanded_smishing_v1
```

Default QA retains fixed model regression expectations. Explicit preview QA
checks the displayed candidate result against the real API and **records** its
benign-notification false alarm; passing that functional check does not mean
the candidate passes the default accuracy regression.

## Verification and outputs

- **217 tests pass** in the prepared Ubuntu-24.04 WSL Python 3.12 environment.
- **22 Chrome groups pass** against the default; 22 functional groups pass
  against the explicit preview, with the false alarm retained in observations.
- The portable Windows launcher selects WSL and loads the explicit candidate.
- Native Windows subsets pass 22 API tests and 10 launcher tests.
- Twelve real HTTP checks preserve suspicious rule findings for original,
  shortened and missing targets in both modes against both artifacts.
- Independent audit reproduces all 12 CV fold scores, reloads/rescores all
  882 dev predictions and 32 diagnostics, verifies sources/splits and enforces
  the release hold without modifying the default model.
- Native spaCy remains blocked by Windows Application Control; no policy change.
  No new remote CI run is claimed.

Default artifact SHA-256 remains
`29e05300ac835d169b1ef9d4650dc219e350e11a817b65efe7afb96c61d8a2e3`.
Local candidate SHA-256:
`8e9006bbb6d57eb9a37e9b865329c6c57c5bedecb0b150980af8dfc0d87a9300`.

Only compact reports are tracked in `results/message_generalization_v1/`:
`selection.json`, `dev.json`, `summary.json`, `integrity_audit.json` and README.
Raw datasets, model binaries, per-row predictions, browser outputs and logs
remain ignored. See [the report guide](../results/message_generalization_v1/README.md).
