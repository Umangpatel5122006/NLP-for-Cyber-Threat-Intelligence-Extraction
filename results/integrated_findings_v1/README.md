# Integrated model/backend/frontend validation

[Checkpoint](../../checkpoints/INTEGRATED_FINDINGS_AND_SMS_MODEL.md).

- `screenshot_ner.json` / `screenshot_ioc.json`: one message-rule review finding
  with seven source cues; no invented CTI entities/IOCs. Model ham/rule suspicious
  disagreement is explicit.
- `model_only_*.json`: positive trained SMS classifier finding.
- `benign_*.json`: normal notification without review detections.
- `warning_context_*.json`: contextual review, without declaring a live attack.
- `status.json`: readiness includes the checked SMS classifier.
- `browser/web_browser_validation.json`: 20 actual Chrome check groups.
  `web_populated_findings.png` shows the repaired list and evidence details.
  The single offline error is induced intentionally; unexpected errors fail QA.

Full WSL and Windows subset logs are one directory above. Models remain ignored
and current remote CI has not run. Neither official CTI test nor SMS holdout
performance is claimed.
