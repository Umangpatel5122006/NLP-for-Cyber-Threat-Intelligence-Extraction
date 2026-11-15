# Language and coverage validation

`summary.json` retains compact evidence from the 253-test WSL suite, native
language/review/API tests, default/candidate browser checks, synthetic HTTP
cases, and unchanged model hashes. No new weights, accuracy improvement,
multilingual trained model or held-out evaluation are claimed.

See [the checkpoint](../../checkpoints/LANGUAGE_COVERAGE.md) for the supported
English scope, script-heuristic limits, changed assessment states and review
commands. The expanded SMS candidate remains held for its prior false alarms.

Full browser JSON, screenshots and HTTP responses in this directory are ignored
generated outputs. `results/language_coverage_validation.txt` is also local-only.
Only this README and the compact summary need to be committed.
