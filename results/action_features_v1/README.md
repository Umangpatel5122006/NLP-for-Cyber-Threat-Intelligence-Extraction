# Action boundary-feature experiment

Protocol and review: [ACTION_BOUNDARY_FEATURES.md](../../checkpoints/ACTION_BOUNDARY_FEATURES.md).

| Profile | Train-only pooled exact F1 |
|---|---:|
| Baseline | 0.1265 |
| Morphology | 0.1250 |
| Morphology + boundary dictionary | **0.1323** |

The combined profile passed the fixed minimum-gain/two-fold rule before dev
evaluation. All-dev action exact F1 improves from **0.1379 to 0.1551**; relaxed F1
from 0.4643 to 0.4966. Scores use metrics version 2 and processed BIO labels.
The official test split was not evaluated.

The selected artifact is now the active action model. The old model is backed
up under ignored `models/action_features_v1/`. Restart the running review server
to load the new model. Entity inference and the frontend schema are unchanged.

Read `cv.json` for fold metrics, `summary.json` for the isolated experiment,
`promotion.json` for activation, and `action_boundaries_dev.json` for the selected
dev evaluation. `summary.json` says the isolated experiment did not modify active
files; the later activation is recorded separately in `promotion.json`.

Uncertainty and tradeoffs matter: the conditional CV interval includes zero and
spurious spans rise from 361 to 400. This is a modest development improvement,
not final-test evidence.
