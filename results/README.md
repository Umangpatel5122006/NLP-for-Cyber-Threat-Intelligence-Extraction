# Versioned reports and local output

Generated output is ignored by default. The root `.gitignore` explicitly keeps
compact experiment selections, metrics, summaries, diagnostics and these README
files. Source code, tests, configuration, dependency pins, dataset/split definitions,
gazetteers and checkpoint notes remain versioned.

Local-only output includes feature/model caches, screenshots, browser profiles,
API/demo responses, detailed per-row predictions, progress files and validation
logs. Model binaries and downloaded SMS data remain under ignored `models/`.
CI can still generate and upload ignored files as workflow artifacts.

The frozen `ner_v2_action.predictions.json` is a deliberate exception:
`scripts/audit_action_experiment.py` reads it to compare against the original
baseline. Keep this reference; experiment-specific predictions are regenerated
by their runners before auditing.

Files mentioned in checkpoint notes may be local validation artifacts rather
than files present in a fresh clone. Reproduce them using the commands in the
corresponding checkpoint. Current compact robustness evidence is in
`message_robustness_v1/summary.json` and `mutation_comparison.json`.

Removing previously tracked generated files from Git's index keeps their local
copies. The cleanup applies to future commits; it does not rewrite existing
Git history or immediately remove old objects from previous commits.

To retain a new publication result, add a narrow, explicit ignore exception.
Avoid adding entire generated-output directories.
