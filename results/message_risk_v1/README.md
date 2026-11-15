# Message-risk and mode-recovery checks

See [the checkpoint](../../checkpoints/MESSAGE_RISK_AND_MODE_RECOVERY.md).

- `scam_ner_demo.json` / `scam_ioc_demo.json`: real API responses for the
  reported message. Both show suspicious rule patterns with zero extracted
  technical findings. Risk evidence is in `document.metadata.message_risk`.
- `status.json`: actual compatible-runtime readiness; both modes ready.
- `browser/web_browser_validation.json`: 15 Chrome check groups. Missing-model
  status is simulated for UI recovery; real inference and retry are used.
- Browser PNGs show desktop/mobile output and the missing-model setup state.
  The intentionally induced offline request produces the recorded
  `ERR_INTERNET_DISCONNECTED`; no unexpected browser error was accepted.
- The full WSL and Windows subset test logs are one directory above.

No new model accuracy, test-split score, fact verification or reputation result
is claimed by these artifacts.
