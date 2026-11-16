# BILOU action experiment

**Status: completed; retain the current model. WSL startup is restored.**

BILOU wins all three train-report folds but pooled F1 improves only from
0.132261 to 0.135029, below the fixed minimum gain. Background predictions rise
from 1,544 to 1,653, exceeding the 5% allowance. Dev/test were not evaluated and
no full-train candidate artifact was created. All 327 WSL tests and the independent
post-run audit pass; the default workbench is ready through WSL.

- `cv.json`: actual report-fold metrics, internal-invalid count and provenance.
- `selection.json`: fixed budgets and baseline decision, saved before dev access.
- `summary.json`: completed rejection and unchanged active-file identities.
- `integrity_audit.json`: independent normalization/cache/count/selection checks.
- `recovery/summary.json`: current WSL imports, full-suite and launcher readiness.

Earlier preparation evidence is retained as history:

- `preparation/summary.json`: actual native test results and blocked WSL/native
  production-stack checks. The previous 310-test full-suite run is historical.
- `preparation/integrity_audit.json`: encoding-only check over 8,635 training
  sentences / 4,453 gold action spans, with identical BIO round trips and
  unchanged active model/configuration hashes. It is not a model evaluation.

At preparation time, fourteen new native conversion/toy-CRF tests and 82 existing
native regressions passed. The WSL `HCS/0xc0000139` blocker is now resolved;
production checks run successfully in WSL. Native Application Control remains.

The root-level reports now describe the actual completed experiment. There is
no `dev.json`, because the candidate did not qualify. Logs/caches/binaries remain
ignored.

See [the completed checkpoint](../../checkpoints/ACTION_BILOU_EXPERIMENT.md) for
results, limits and reproduction. The [preparation checkpoint](../../checkpoints/ACTION_BILOU_PREPARATION.md)
preserves the original frozen protocol and earlier blocker.
