# Action CRF comparison

Full protocol and review commands: [ACTION_EXPERIMENT.md](../../checkpoints/ACTION_EXPERIMENT.md).

| Configuration | Train-only pooled OOF exact F1 | Decision |
|---|---:|---|
| Baseline | **0.1265** | Retained |
| Wider action context | 0.1208 | Failed minimum gain / fold-win rule |
| Sparse action context | 0.1209 | Failed minimum gain / fold-win rule |

Three folds cover 70 training reports and all 8,635 training sentences once for
validation. Training within each fold uses relevant sentences only. Metrics
version 2 is used throughout.

The selected baseline was frozen before dev evaluation. Its full-train refit
reproduces the existing 1,238 dev predictions exactly: all-dev exact F1 0.1379.
The losing variants were not evaluated on dev. Active models remain unchanged;
the official test split was not evaluated.

Read `cv.json` for fold metrics and provenance, `summary.json` for the decision,
`baseline_diagnostics.json` for boundary examples and `integrity_validation.txt`
for independent checks. The conditional paired-report bootstrap is recorded in
`paired_cv_bootstrap.json`; its intervals do not include refitting uncertainty.
