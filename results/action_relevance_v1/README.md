# Action relevance filter comparison

The selected policy is **no filter**. None of the predeclared classical relevance
cutoffs meets the training-CV gain, recall and coverage requirements. Default
models and app behavior remain unchanged; no dev or test evaluation was run.

- `selection.json`: all four cutoff comparisons, three-fold and pooled strict
  counts, fixed-policy decisions and source/data/cache provenance.
- `summary.json`: retained baseline, unchanged active action hash and no-dev/
  no-test scope.
- `integrity_audit.json`: independent recount, alignment/fold checks and unchanged
  hashes for all existing entity/action/SMS/candidate artifacts.

Per-sentence OOF scores and predictions are local in ignored
`models/action_relevance_v1/oof.json`. Training and validation logs are ignored.
No model binary or additional dataset is committed.

See [the checkpoint](../../checkpoints/ACTION_RELEVANCE_EXPERIMENT.md) for the
protocol, actual results, limitations, prerequisite caches and reproduction.
