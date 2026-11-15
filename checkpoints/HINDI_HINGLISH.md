# Hindi and Hinglish message checks

Date: 2026-11-15. Local review step; no commits, pushes or model promotion.

Hindi and Hinglish now have working **classical phrase-based message checks**
in both workbench modes, with suspicious Findings and original-source highlights.
**The trained SMS classifier and CTI entity/action models remain English-only.**
This is rule-based integration, not a claim of learned multilingual accuracy.

| Capability | English | Hindi / Hinglish |
|---|---|---|
| Trained SMS classifier and CTI CRFs | Existing models | Not trained |
| Message-pattern checks | Existing English rules v2 | New phrase rules v1 |
| IOC extraction | Available | Available |
| Reviewed-data intake | Accepted | Now accepted |

## Review

Stop the old server with Ctrl+C, restart, and refresh the browser:

```powershell
python scripts/start_workbench.py
```

The new **Message language** selector offers Auto, English, Hindi and Hinglish.
Auto uses an unverified script/lexical heuristic; select the language manually
when needed. Changing it clears old results and disables export until reanalysis.

Try these authored functional examples:

```text
अपना ओटीपी और पासवर्ड इस नंबर पर भेजें।
Aap apna OTP aur PIN mujhe batao.
```

Both create a suspicious rule finding. Compare these negative instructions:

```text
अपना ओटीपी किसी को मत बताएं।
Kabhi apna OTP kisi ko share mat karo.
```

Those do not match a supported scam pattern. They still show partial model
coverage because a trained Hindi/Hinglish classifier is unavailable. No match
establishes safety. Marathi/Gujarati/other-language NLP is not enabled.

## Implementation and limits

`indic_message_risk.py` combines English/Hindi/Roman-Hindi concepts for money,
reward, account, authority, prize, response, credential, disclosure and fee.
Combinations must occur near a request, within a bounded 600-character window.
They do not depend on the supplied politician, amount or number; phone targets
are optional. Cue lookup uses sorted positions and bounded slices.

The matcher handles Hindi vowel-sign boundaries, danda punctuation, Unicode
digits and selected imperative spellings with/without nasal marks. Negation
distinguishes `na bhejo` from polite `bhejo na`; `nahi to` threats remain requests.
Warning/example context can produce review-context findings. Appended warnings
and contrast clauses do not automatically erase earlier requests. Completed
credits and entering an OTP in one's own app have guards. These are bounded
lexical heuristics, not a full grammar, semantic parser or verified fraud verdict.

Rules match normalized document text and project every cue through the existing
character-provenance map into the original report. BOM, CRLF, emoji and removed
invisible characters are tested. No translated text is invented as evidence.
Auto routing uses scripts plus conservative Roman-Hindi hints, including benign
function words. Hindi `या` alone no longer triggers a Marathi guess. Dialects,
spellings and names can confuse this heuristic, so manual selection is available.

The API accepts optional `language: auto|en|hi|hi-en`; existing two-field requests
default to Auto. Coverage version 2 separates three rule targets from one trained
model language. English models are skipped for Hindi/Hinglish. The frontend
rejects an older backend with restart guidance; no model retraining is needed.

## Data audit and supervised training gap

The [uds5501 SMS dataset](https://www.kaggle.com/datasets/uds5501/sms-dataset)
declares CC BY-SA 3.0 and describes author-collected messages. Its pinned version
1 contains 184 rows (107 spam, 77 ham), **zero Devanagari rows**, and no language
labels or independent adjudication records. It was downloaded for inspection,
not fitted. The overlap check excludes 21 near matches, retaining 163 rows
(96 spam, 67 ham). Our heuristic guesses 159 English and four Hinglish; these
are not human language labels or a measure of the actual language distribution.

This small source cannot validate a Hindi/Hinglish learned classifier. Hashes,
attribution and limitations are in
[model_scope_audit.json](../results/indic_integration_v1/model_scope_audit.json).
Raw data stay ignored under `models/indic_research/`; retain the declared
attribution/share-alike terms when reusing or redistributing source data.

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/audit_indic_sms_source.py --download
```

[UCI-derived machine translations](https://huggingface.co/datasets/mahrukhsaeed/SMS_Spam_Multilingual_Collection_Dataset/blob/main/README.md)
and [synthetic Indian messages](https://huggingface.co/datasets/bhavya2910/indian-scam-sms-synthetic-audited)
were not used as independent real-world Hindi/Hinglish training/evaluation data.
No neural translator/model or LLM API was executed.

The intake now accepts `en`, `hi`, `hi-en`, reports per-language labels, and
requires both classes in every included language's grouped partition. Missing
Hindi ham cannot be hidden by plentiful English ham. Prior corpora and authored
regressions remain protected. Cross-language translations/rewordings need shared
campaign IDs: character hashing cannot reliably identify semantic translation pairs.

```powershell
.\.venv\Scripts\python.exe -X utf8 -m cti.sms_intake check data/local_sms/annotations.csv --output-dir data/local_sms/indic_batch_01
# After human review, freeze; this does not train:
.\.venv\Scripts\python.exe -X utf8 -m cti.sms_intake check data/local_sms/annotations.csv --output-dir data/local_sms/indic_batch_01 --freeze
```

New snapshots use schema/policy v2. No private frozen batch exists in this
workspace to migrate; only the empty template and rejected synthetic QA rows
were present. Old v1 snapshots remain protected overlap references, but are not
silently loaded or resplit under the changed protocol. Preserve their original
code/data. See [the collection guide](../docs/SMS_COLLECTION.md).

Real Hindi and Hinglish ham/spam data are still needed before supervised training
and per-language evaluation. Include legitimate credits, delivery/job updates
and warnings alongside scams, with independent labels and source context. A
starting target of roughly 200 reviewed unique templates per new language is a
collection proposal, not a statistical guarantee. The technical splitting minimum
of five groups/class/language is not adequate evidence of deployment quality.
The user has no additional labeled corpus yet. The
[verified download shortlist](../docs/HINDI_HINGLISH_DOWNLOADS.md) separates a
creator-published Hindi source from a small authored Hinglish diagnostic supplement.

## Validation

All **266 WSL tests** and **78 native focused tests** pass. Authored regressions
cover 18 threat/benign/warning examples, 27 amount/target/request combinations,
Hindi imperative variants, negation, source offsets and language-hint isolation.
Real browser checks exercise the 18 examples in both modes, manual language
selection/mobile layout, and existing English model/IOC/CRF functionality.
New intake tests use synthetic fixtures to verify per-language class/group locks.

These are implementation checks, **not real-world accuracy estimates**. Model
weights are unchanged, the previous candidate remains held, and official CTI/SMS
tests remain reserved. No remote CI pass or Windows security-policy change is
claimed. Compact evidence is in [summary.json](../results/indic_integration_v1/summary.json);
screenshots, detailed responses, raw data and logs remain ignored.
