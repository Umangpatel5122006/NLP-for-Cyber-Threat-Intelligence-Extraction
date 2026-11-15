# Message robustness evidence

[Checkpoint](../../checkpoints/MESSAGE_ROBUSTNESS.md).

- `mutation_comparison.json` compares old committed rule results against 12
  curated cases in each real API mode. The legacy function is loaded from the
  classical rule file at `8a03fa8`, excluding the registration class.
- `*_ner.json` / `*_ioc.json` preserve actual current API responses.
  Shortened, masked and absent numbers retain positive message-rule Findings.
- `dev_before.json` / `dev_after.json` report the unchanged 796-row SMS dev
  model-or-rule review-alert metrics. No test predictions are included.
- `browser/web_browser_validation.json` records 22 actual Chrome check groups,
  including seven mutation variants per mode and a simulated stale v1 backend.
- `browser/web_numberless_findings.png` and `web_numberless_mobile.png` show
  the missing-number result. The induced offline error is expected.

Full WSL and native Windows subset logs are one directory above. Curated/template
checks establish regression coverage, not universal scam accuracy. Model weights
and thresholds remain unchanged, including the documented SVM domain miss.
