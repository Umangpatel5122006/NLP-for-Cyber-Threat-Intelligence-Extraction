# Message mutation robustness: phone-independent request patterns

Date: 2026-11-15. Baseline: committed `8a03fa8`. Classical rules only in this
step; existing SMS/NER model weights, thresholds and datasets are unchanged.
No commits, pushes, dependency installations or Windows policy changes.

## Reproduced failure

The money/contact rule required a complete phone-number match. With the same
financial offer and SMS request, deleting one digit changed the rule result
from suspicious to no detection. The frozen SVM predicted ham in both cases,
so the combined review list became empty. Removing the entire number had the
same effect. This was an unnecessarily strict identifier dependency.

## Change

- **Rules v2** match the nearby monetary offer, account/payment context and
  response request without requiring a valid-length phone number or URL.
  Numbers and links remain optional supporting source evidence.
- Separate authority/financial-offer and prize/request combinations cover
  meaningful cases without bank wording. No person, specific amount or number
  is hardcoded.
- Currency forms include prefixed INR/Rs, monetary words and digit-free offers.
  More request inflections are supported. Source offsets still index the
  untouched original text; absent/short/masked targets are not fabricated as phones.
- Directive-context checks distinguish requests from passive SMS mentions.
  Completed-credit notice guards and negation reduce specific rule false positives.
  Sending a secret is distinguished from entering a code in one's own app.
- Warning/example context remains a review outcome. These are bounded English
  heuristics, not a trained semantic parser or an identity/fact-checking service.
- API status and the assessment label expose rule version 2. A new frontend
  receiving a cached v1 backend response shows explicit restart instructions
  instead of silently displaying a missed detection.

## Actual behavior

| Mutation | Rules before | Rules now |
|---|---|---|
| Original full number | Suspicious | Suspicious |
| One digit removed | No detection | Suspicious |
| Four-digit target | No detection | Suspicious |
| Masked target | No detection | Suspicious |
| Target removed entirely | No detection | Suspicious |
| All digits removed | No detection | Suspicious |
| Currency/name/request rephrased | No detection | Suspicious |
| Bank wording removed; authority/offer/request retained | No detection | Suspicious |

The missing-number version produces **one message-rule review finding with six
source cues** in both modes. The standalone SVM still predicts ham for that
domain example; the combined assessment retains the suspicious rule finding
and displays the disagreement. No model retraining or ML accuracy gain is claimed.

## Validation

- **202 full-suite tests pass in WSL**, including 768 generated positive
  combinations of names, currency/amounts, account phrases, requests and targets.
  These are structured regression cases, not 768 independent real-world samples.
- Rule checks also cover completed bank notifications, passive SMS wording,
  negated offers, secret-sharing warnings, own-app OTP entry and contextual examples.
- Native Windows: **17 rule, 22 API and 7 review tests pass**.
- **22 real Chrome check groups pass**. Seven mutations run through each real
  API mode and retain their Findings rows and original text. Desktop/mobile PNGs
  show the numberless result. A simulated v1 response verifies restart guidance.
- **24 real API checks** compare 12 curated cases in both modes with the old
  committed classical rule function. Saved responses and comparison are under
  `results/message_robustness_v1/`.
- The existing **796-row SMS dev partition** was rescored before/after.
  A review alert means a positive model or suspicious/contextual rule evidence.
  Counts remain **103 TP, 0 FP, 4 FN, 689 TN**, F1 **0.9810**. Seventeen dev
  spam rows gain rule support; they were already flagged by the model.
  No new ham rule alerts appeared on this dev partition.
- Neither official CTI test nor reserved SMS holdout was evaluated. No model
  artifacts were replaced. Remote CI for this change has not run yet.

## Limits remain explicit

The number dependency is fixed; this does not establish reliable detection of
every paraphrase, language, obfuscation or scam. The public SMS corpus has limited
coverage of the user's message domain, and the standalone model miss remains.
One curated completed-cashback notice is still predicted spam by the existing
SVM even though the rule guard leaves it unflagged. Rule false-positive checks
must not be presented as proof that the combined model has no false positives.
Broader independently labeled scam-domain data/evaluation remains necessary.

## Review and run

Stop the old server with Ctrl+C, then:

```powershell
python scripts/start_workbench.py
```

Reload the page. Message assessment must show **English rules v2**.
Shorten, mask or delete the target number and analyze again. The monetary-offer/
response wording should still show a suspicious result and a message finding.
No model retraining is needed for this update.

Focused regression command:

```powershell
wsl -d Ubuntu-24.04 --exec bash -c '~/.cache/cti-classical-py312/bin/python -m unittest discover -s tests -p test_message_risk.py -v'
```

Suggested commit: `fix: detect suspicious requests independently of phone length`.
