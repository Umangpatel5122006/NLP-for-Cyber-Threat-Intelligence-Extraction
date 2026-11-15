# SMS intake validation

`summary.json` records the executed full WSL suite (239 tests), native intake
subset (22 tests), native/WSL CLI parity, synthetic-row exclusion and unchanged
baseline/candidate artifact hashes. No new real messages, model training or
held-out evaluation are claimed. There was no new remote CI or browser run.

See [the checkpoint](../../checkpoints/SMS_DATA_INTAKE.md) and
[collection guide](../../docs/SMS_COLLECTION.md) for commands and limitations.
The existing learned candidate remains held. Intake metadata cannot itself
authenticate reviewer independence, source permission or label correctness.

Raw annotations, QA fixtures, audits, quarantine lists and private frozen
snapshots remain in ignored `data/local_sms/`. The complete WSL test log is the
ignored `results/sms_intake_validation.txt`. This report contains only aggregate
validation and hashes, not private messages or reviewer/source records.
