# Reviewed SMS intake and protected batch freezing

Date: 2026-11-15. Follows [the held learned candidate](LEARNED_MESSAGE_EXPANSION.md).

The missing next input is diverse, independently reviewed real SMS data,
especially legitimate financial/delivery/job notifications and warnings. This
step adds a usable local collection/audit/freeze command, rather than adapting
the classifier to the known examples again. **No new real data were collected,
no model was retrained or promoted, and existing model metrics are unchanged.**

## Changes

- `cti.sms_intake` creates a private CSV template, validates recorded review and
  provenance, quarantines problematic records, and reports source/category/label
  coverage. It does not call a model or invent labels.
- Synthetic/uncertain/out-of-scope records are excluded from training intake.
  Recorded disagreements require a third independent reviewer. The tool cannot
  authenticate human identities, permission or ground truth.
- Exact/near protected-text exclusion covers both whole prior SMS corpora,
  authored diagnostics, the reported message/regressions and prior frozen
  private batches. Matches exclude the whole connected campaign/template group.
  Exact duplicate aliases cannot break campaign grouping after removal.
- `--freeze` writes a new immutable accepted snapshot and 3/1/1 train/dev/test
  group-fold assignment. The loader rejects test before file access. Existing
  official splits remain untouched; earlier frozen batches are not resplit.
- `.gitignore` excludes `data/local_sms/`, including raw text, reviewer notes,
  source metadata, audit/quarantine files and private split manifests. Only the
  empty schema and known public diagnostic exclusions are versioned.
- [The collection guide](../docs/SMS_COLLECTION.md) gives commands, column
  meanings, label criteria, collection targets and primary-source research notes.

## What the team can review now

An empty `data/local_sms/annotations.csv` is already created locally. The
[tracked header](../examples/sms_annotation_template.csv) documents its schema.
Begin with new real legitimate and suspicious messages with source context;
record independent reviewer labels and a short rationale. Avoid repeatedly
rewriting the screenshot message or using model/keyword guesses as ground truth.

```powershell
.\.venv\Scripts\python.exe -X utf8 -m cti.sms_intake check data/local_sms/annotations.csv
# After reviewing the audit, freeze accepted examples; this does not train:
.\.venv\Scripts\python.exe -X utf8 -m cti.sms_intake check data/local_sms/annotations.csv --freeze
```

The empty header alone will correctly fail the check until records are added.
An initial target of roughly 200 independently reviewed unique templates is a
collection proposal, not a release threshold or statistical guarantee. Raw
private files remain local. Do not edit frozen batch files to append records;
prepare a new batch instead.

## Validation

Synthetic fixtures cover source/annotation checks, Unicode/CRLF preservation,
label contradictions, protected overlap and campaign propagation, duplicate
bridges, deterministic freezing, snapshot/protocol drift, cross-batch exclusion,
unchanged repeated CLI calls and test access denial before file reads. No
protected examples are used for split-authorization tests.

The [compact validation report](../results/sms_intake_v1/summary.json) records
the executed suite, native/WSL CLI parity and unchanged artifact identities.
Reproduce in the respective prepared runtimes:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -p test_sms_intake.py -v
wsl -d Ubuntu-24.04 --exec bash -c '~/.cache/cti-classical-py312/bin/python -m unittest discover -s tests -v'
wsl -d Ubuntu-24.04 --exec bash -c '~/.cache/cti-classical-py312/bin/python scripts/audit_message_expansion.py'
```

Native spaCy remains blocked by Application Control; this data-only path does
not require it. No security setting was changed. Frontend/server/model code is
unchanged in this step, so earlier browser results remain historical checks,
not a newly claimed browser run. No remote CI run, commit or push is claimed.

Next: review/collect the new data, inspect label/source balance, freeze it, then
run a new predefined train-only model comparison. Do not consume the existing
dev/challenge cases as training examples or promote the previous held candidate.
