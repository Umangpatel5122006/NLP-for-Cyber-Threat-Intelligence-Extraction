# Hindi/Hinglish request and account-threat checks, version 2

Date: 2026-11-15. Follows [the initial integration](HINDI_HINGLISH.md).

Hindi/Hinglish phrase checks now recognize additional imperative forms such as
`verify karein`, `confirm kijiye`, `form bhariye` and `jayein`. Domain-like
destinations without `https://` can support message evidence. A new combination
requires **identity/account context, blocking/suspension/rejection, verification,
a request and a destination**. It does not flag a PAN/bank mention on its own.

The existing warning and negation checks remain active. Regression examples
include legitimate reminders, negated blocking, own-app instructions, quoted
scam warnings and missing destinations. A bare address is labeled **Domain-like
destination**, with syntax/reputation explicitly unverified. It is not given an
invented scheme or a new URL entity by this rule. The existing IOC library may
independently label a schemeless path as URL; its canonical value stays unavailable.
Full URLs do not get a duplicate host cue, and email-domain fragments are excluded.

Auto language guessing includes the added Roman-Hindi grammatical forms; it is
still an unverified heuristic. Manual language selection remains available.
The API readiness response uses the actual rule-version constant. The frontend
rejects a v1 Hindi/Hinglish backend with restart guidance and export disabled.

Restart the existing server and refresh the browser to review version 2:

```powershell
python scripts/start_workbench.py
```

Try this invented functional example, then remove the blocking claim or negate it:

```text
Aapka account block ho jayega. PAN verify karein: secure-check.example.in/kyc
```

This signals a suspicious textual combination, not verified fraud. Legitimate
messages can use similar wording; sender authenticity and reputation are unknown.
Missing a pattern does not establish safety. Trained Hindi/Hinglish SMS and CTI
models are still unavailable; existing English model weights are unchanged.

## Diagnostic evidence

The user's downloaded JSON contains 3,787 mixed English/authored Hinglish records.
Before this step, protected-corpus/diagnostic overlap exclusion retained 86 rows:
47 supplied positive labels and 39 supplied ham labels. That exact population was
kept for the before/after diagnostic, even as new regression exclusions were added.
The dataset has no per-row language/origin metadata and its labels are not
independently verified fraud outcomes.

| Supplied-label diagnostic on those 86 rows | Before | After |
|---|---:|---:|
| Positive labels with suspicious rule matches | 10 | 13 |
| Positive labels without a match | 37 | 34 |
| Ham labels with suspicious rule matches | 0 | 0 |

The downloaded examples helped identify these gaps. **These are development
diagnostics, not independent accuracy or generalization evidence.** Many supplied
positives remain unmatched; this change does not turn the rules into a general
fraud/spam classifier. No classifier was fitted or threshold tuned. Private
per-row diagnostics remain under ignored `models/indic_research/`; the input
SHA and frozen population ID digest are in the compact summary.

The public authored regression file now has 26 examples and remains excluded
from new real-data intake through the existing protected reference. Official
CTI and both SMS test splits were not inferred or evaluated. Previous reports
remain historical records of their own source/reference hashes.

## Verification

All **270 tests pass in WSL**, with **82 focused tests passing on native Windows**.
The **29 browser check groups per preview** and **104 real HTTP cases** cover
both modes and both existing SMS artifact choices,
including source offsets, mobile layout, legitimate notices, warnings and stale
backend recovery. The [compact report](../results/indic_rules_v2/summary.json)
records completed checks, source/model hashes and diagnostic limitations.
Detailed logs, screenshots and per-row output remain ignored. No new remote CI
pass, Windows policy change, model promotion, commit or push is claimed.
