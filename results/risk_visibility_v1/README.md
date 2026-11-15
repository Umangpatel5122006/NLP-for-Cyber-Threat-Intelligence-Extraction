# Screenshot reproduction and visibility checks

See [the checkpoint](../../checkpoints/RISK_VISIBILITY_AND_MODEL_SCOPE.md).

- `screenshot_ner.json` / `screenshot_ioc.json`: the 152-character screenshot
  variant returns suspicious, seven cues and zero technical findings.
- `model_scope_audit.json`: CTI extraction F1 and rule-only message detection
  are distinct tasks; measured SMS-classifier accuracy is unavailable.
- `browser/web_browser_validation.json`: 17 successful Chrome groups. PNGs
  show the inline verdict and cue highlights on desktop/mobile.
  The offline error is intentional; unexpected browser errors fail QA.

NER artifacts are unchanged. No accuracy improvement, SMS model training or
final test evaluation occurred.
