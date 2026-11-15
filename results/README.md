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
The learned expansion's compact selection, dev comparison, release hold and
independent audit are in [message_generalization_v1/](message_generalization_v1/README.md).
The next [SMS intake validation](sms_intake_v1/README.md) keeps only compact
test/CLI evidence; raw annotations and private snapshots stay in ignored
`data/local_sms/` rather than in public results.
The [language coverage summary](language_coverage_v1/README.md) records explicit
English scope, browser/API checks and unchanged model identities without adding
screenshots or per-case response files to Git.
The [Hindi/Hinglish integration](indic_integration_v1/README.md) retains compact
source/validation audits; downloaded data, browser output and private annotations
remain ignored.
The [Hindi/Hinglish v2 rule report](indic_rules_v2/README.md) preserves only compact
validation and development-diagnostic counts, with their non-independent scope.
The [action relevance comparison](action_relevance_v1/README.md) keeps the rejected
train-only cutoff results and independent recount. Local OOF rows remain ignored;
no default model changed and no dev/test evaluation was needed.
The [action background-training comparison](action_background_v1/README.md)
records six candidate fold fits and their rejection on training-CV gain/recall
criteria. Prediction rows and sampling caches remain local; compact inventories,
decisions and the independent audit are versioned.
The [train-only action error audit](action_error_audit_v1/README.md) records
boundary/miss diagnostics, annotation consistency and exact BIO reproduction.
Source/context review CSVs remain ignored; model weights and labels are unchanged.

Removing previously tracked generated files from Git's index keeps their local
copies. The cleanup applies to future commits; it does not rewrite existing
Git history or immediately remove old objects from previous commits.

To retain a new publication result, add a narrow, explicit ignore exception.
Avoid adding entire generated-output directories.
