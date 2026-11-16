# Finalized classical workbench recovery branch

Date: 2026-11-16 (Asia/Calcutta)

Branch: `recovery/classical-cti-finalized`.

The baseline is the existing `Finalization` commit
`fc0505820dca825075678506002f5f1418bcd8af`, containing the completed classical
workbench, portable setup documentation and finalization evidence.

The original local `main` worktree was in an unfinished merge, with a README
conflict and numerous staged incoming changes. The recovery branch was created
from the saved baseline in a separate worktree. It does not complete that merge,
incorporate its incoming changes, discard local work or rewrite an existing branch.
The only added file over the baseline is this recovery record.

Verification on the isolated checkout:

- All source hashes recorded in `results/finalization_v1/summary.json` match.
- All **368 tests pass**, with process exit 0, in the prepared WSL runtime using
  this checkout's `src` directory first on `PYTHONPATH`.
- Both frontend/browser-QA JavaScript syntax checks pass.
- Root `learning/` remains ignored and untracked. No model binaries, virtual
  environments or private presentation files are tracked in this branch.
- No training, protected-test evaluation or model/configuration change occurred
  during branch recovery. The previous finalization evidence retains its scope.

Use the root README and `docs/SETUP.md` to install a clone of this branch.
Model binaries must be prepared locally as documented. The original worktree's
unfinished merge remains separate and should be reviewed before committing there.
