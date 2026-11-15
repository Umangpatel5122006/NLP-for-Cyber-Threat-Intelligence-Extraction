# Learned SMS expansion comparison

This directory's compact versioned evidence is:

- `selection.json`: four candidates and 12 training CV fold results; selection
  frozen before dev loading, with configuration/source/split hashes.
- `dev.json`: active baseline versus selected candidate on UCI and supplement.
- `summary.json`: dev gates, authored-diagnostic scores, candidate identity and
  **deployment hold**. Dev eligibility does not override the release hold.
- `integrity_audit.json`: independent score/reload/provenance checks, unchanged
  default model and remaining reported-message misses.

See [the checkpoint](../../checkpoints/LEARNED_MESSAGE_EXPANSION.md) and
[source provenance](../../docs/SMS_DOMAIN_DATA.md) for scope and reproduction.
The candidate is a learned TF-IDF/linear SVM fit with broader data. It is held
because authored diagnostics show more false alarms, despite improved source
dev scores. Official CTI and both SMS test partitions remain unevaluated.

Per-row OOF/dev predictions, `challenge.json`, browser screenshots/results and
validation logs are ignored generated outputs. Recreate them with
`python -m cti.experiment_message --download`, then
`python scripts/audit_message_expansion.py` in the compatible ML runtime.
The baseline artifact must already be present; see the checkpoint for setup.
`models/` contains ignored downloads/artifacts and is never required in a commit.
