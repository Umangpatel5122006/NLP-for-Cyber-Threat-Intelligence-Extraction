# Finalization validation

`summary.json` records the actual environment, test/browser checks, model
preparation and documentation/private-folder audit. `integrity_audit.json`
compares original active/held model and configuration hashes with the preceding
checkpoint. `clean_setup/summary.json` contains compact fresh-setup/HTTP evidence.

Local-only artifacts include dependency-install logs, full unit/browser output,
screenshots, initial native-crash diagnostics and repeated-fit logs. The source
copy, separate training outputs and QA helpers are under ignored `models/`.
The private `learning/` folder is also ignored; no presentation contents are
included in this tracked report.

The fresh check began without production models, passed 368 tests, trained via
the real launcher, reproduced recorded dev results and passed 33 Chrome groups.
A second NER process returned 0 with matching dev metrics. Existing deployed
weights were not replaced. Test partitions were not evaluated. Initial native
process crashes and the bounds of the later checks are disclosed in
[the checkpoint](../../checkpoints/FINALIZATION.md).

Reproduce setup, training and checks using [the portable guide](../../docs/SETUP.md).
Clean training is an integration check, not a new model-selection experiment
or independent accuracy estimate. No remote CI run is claimed.
