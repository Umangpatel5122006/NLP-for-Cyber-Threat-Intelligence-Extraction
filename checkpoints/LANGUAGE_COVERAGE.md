# Explicit language support and incomplete-analysis states

Date: 2026-11-15. Reviewable local step; no model retraining or promotion.

**The project currently supports one NLP target language: English.** Language
guessing, script recognition and IOC matching are separate capabilities. None
of them establishes trained multilingual threat detection.

| Component | Current language scope |
|---|---|
| Entity and action CRFs | English only |
| TF-IDF + linear SMS classifier, including the held candidate | English only; <=1,000 characters |
| Social-engineering rules | English lexical patterns; can retain matching English cues in mixed text |
| Relevance classifier | English only |
| IOC rules | Match supported indicator syntax across surrounding languages; no semantic understanding |
| Script/language guess | Unverified Unicode-script heuristic; not a trained language identifier |

Hindi/Marathi guesses use Devanagari plus a small Marathi hint dictionary;
Gujarati uses its script. They do not enable a model in those languages. Latin
text includes English, Spanish, romanized Hindi/Hinglish and many other
languages, so it cannot establish English from script alone. The app now says
**English assumed**, rather than presenting `EN` as verified language detection.

## Fixes

The previous heuristic counted only Latin, Devanagari and Gujarati letters.
Other scripts were ignored, allowing Latin characters in an embedded URL to
make Arabic/Chinese/Cyrillic prose look English. The revised heuristic counts
additional scripts and an `other` bucket, excludes common indicator strings
from the guess, and records method, letter counts, status and `verified=false`.
It uses a fixed 80% dominant-script threshold; other mixtures are `mixed`.
That is a conservative heuristic, not a calibrated language probability.
Original report text and evidence offsets are unchanged by this counting step.

Latin text still follows the existing English model path, with a visible
unverified assumption. It cannot reliably distinguish English from Hinglish or
other Latin-script languages. No multilingual model or new language-ID accuracy
claim is made. Script labels such as Han/Arabic/Cyrillic are not language labels.

`MessageRisk` keeps positive English-pattern evidence even in mixed text and
marks its coverage partial. With no matching pattern and unsupported/uncertain
language, it reports `unsupported_language` rather than a supported negative.
The combined assessment also distinguishes `partial_coverage` when the SMS model
is missing, incompatible, outside its length scope, or the pattern check is
absent. Completed English checks retain the ordinary `no_pattern_detected` state.
No state asserts safety, and existing positive rule/model evidence takes priority.

The API status declares per-component `language_support`. Analysis exports add
versioned `message_assessment.coverage`, including the language assumption and
whether model/rule coverage was available. The frontend displays those notes,
shows amber coverage warnings, preserves IOC findings in either mode, and rejects
an older backend lacking the new coverage contract with restart guidance.

## Review

Restart the server to load the new backend and refresh the browser:

```powershell
python scripts/start_workbench.py
```

Review English and non-English messages, a mixed-script message containing an
IP, and a report longer than the SMS model's 1,000-character limit. Confirm that
IOC evidence stays visible while language/length coverage limits are explicit.
Spanish or Hinglish Latin text must say the English guess is unverified; it
must not be described as a supported new language.

No retraining is needed for this update. The default SMS and held-candidate
artifact hashes remain unchanged. The entity/action models are unchanged too;
no NER or SMS accuracy gain is claimed. The original reported scam wording is
still a learned-model miss retained as suspicious by rules, as documented in
[the learned comparison](LEARNED_MESSAGE_EXPANSION.md).

## Validation

- **253 tests pass** in the prepared Ubuntu-24.04 WSL Python 3.12 environment.
- Native Windows passes **10 language**, **10 combined-review** and **23 API**
  tests. No Windows security setting was changed; native spaCy remains blocked.
- **26 real Chrome check groups pass per server**, against default and
  held-candidate previews.
  They cover unsupported/mixed text in both modes, source-aligned IOCs, Latin
  assumptions, long-report partial coverage and stale-backend recovery, alongside
  the existing English scam, CRF, export, upload and mobile checks.
- The prior independent SMS audit still reproduces frozen scores and enforces
  its release hold. No model weights or official train/dev/test files changed.
- All authorization/language fixtures are synthetic. Neither SMS holdout nor
  the official CTI test is inferred or evaluated. No remote CI run is claimed.

Compact evidence is in [the validation summary](../results/language_coverage_v1/summary.json).
Detailed HTTP/browser outputs, screenshots and logs stay ignored. Reproduce:

```powershell
wsl -d Ubuntu-24.04 --exec bash -c '~/.cache/cti-classical-py312/bin/python -m unittest discover -s tests -v'
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -p test_language.py -v
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -p test_review_findings.py -v
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -p test_web.py -v
node scripts/check_web_ui.mjs 'C:\Program Files\Google\Chrome\Application\chrome.exe' http://127.0.0.1:8765 results/language_coverage_v1/browser_default
```

## Work still needed

This coverage fix does not complete model development. SMS generalization and
legitimate-notification false alarms need the reviewed real data described in
[the collection guide](../docs/SMS_COLLECTION.md). English CTI entity/action
exact-span F1 remains limited at 0.2596/0.1551; further comparisons must select
on training reports before dev checks. Final test evaluation remains deferred
until the pipeline and selection decisions are frozen.

Adding Hindi, Marathi, Gujarati or another language requires a separately
labeled corpus, appropriate classical tokenization/features and evaluation for
each intended task. A language menu or script guess alone does not supply that
support. The existing intake remains English-only; do not relabel other-language
messages as `en` to bypass its scope checks.
