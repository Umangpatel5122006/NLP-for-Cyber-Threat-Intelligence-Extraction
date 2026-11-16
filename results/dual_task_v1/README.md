# Two-task workbench validation

`summary.json` records actual suite, browser and readiness results for equal
message-risk/report-extraction task cards. `integrity_audit.json` verifies the
seven prior model/configuration identities against the preceding experiment.
No model training or official test evaluation occurred.

Local-only evidence includes `../dual_task_tests.txt`, browser screenshots and
`browser/web_browser_validation.json`. These files are ignored. Browser checks
use authored examples and runtime simulations, not held-out accuracy data.

Reproduction, changes and review instructions are in the
[checkpoint](../../checkpoints/DUAL_TASK_WORKBENCH.md).
