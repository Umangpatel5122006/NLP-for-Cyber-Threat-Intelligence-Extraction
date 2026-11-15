# Browser report-review workbench

Date: 2026-11-15 (Asia/Calcutta)

## Run frontend and backend together

From the project root in PowerShell, using the prepared environment with the
saved entity/action models:

```powershell
wsl -d Ubuntu-24.04 --exec bash -c '~/.cache/cti-classical-py312/bin/python -m cti.web'
```

Open **http://127.0.0.1:8765** in your browser. Use **Load example**, then
**Analyze report**. Press Ctrl+C in the server terminal to stop.
An alternative port can be selected with `--port 8766`.

For IOC-only use in the existing native Windows environment:

```powershell
.\.venv\Scripts\python.exe -m cti.web --ioc-only
```

The complete model runtime remains WSL because of the earlier Windows native
dependency restriction and the Python-version checks on saved models. No
Windows security setting was changed. See [WSL_SETUP.md](../docs/WSL_SETUP.md)
for the prepared environment; train models first on a fresh checkout.

## Architecture decision

The user approved the simpler Python-served implementation after asking about
Next.js/Turborepo and a single launch command. The frontend is **HTML, CSS and
JavaScript**, served with a small Python HTTP API. Python runs the existing
NLP pipeline. One command starts both; there is no Node/npm runtime requirement,
Next.js scaffold or separate frontend development server.

The frontend uses the Sites skill's local-development workflow within this
repository. This is a local review step for the user to commit.

- `src/cti/web/server.py`: loopback server, model readiness, analysis API,
  fixed static routes and request validation.
- `src/cti/web/static/`: responsive workspace and browser interactions.
- `python -m cti.web`: the single entry point.
- `pyproject.toml`: includes the three frontend assets as package data.

Runtime still uses the editable project checkout, its configs/examples and local
saved models. Wheel inclusion of the frontend assets is checked; a standalone
installation without project resources is not the supported launch setup.

## What works

- Paste a report, load the synthetic example, or choose a UTF-8 `.txt`/`.md` file.
- Select **Entities + IOCs** or **IOCs only**. Readiness comes from the server,
  not hard-coded UI state. Missing models leave IOC mode available.
- See actual primary-finding, IOC, entity and action counts.
- Inspect highlights in the **original** report, retaining defanged spelling.
  Highlight segmentation supports overlapping findings and Unicode code-point
  offsets without duplicating source text.
- Filter by label or search finding text, canonical values and sources.
- Expand host components, select findings/highlights and inspect detector
  evidence, original excerpts, canonical values and overlap links.
- Export the full document JSON with its raw observations and source evidence.
- Inspect method notes and see explicit English-NER support status.

The sample still has 13 raw observations, 9 findings and 7 primary findings.
The UI runs the saved classical CRFs and rule-based IOC modules; it does not
reimplement detection in JavaScript or introduce neural methods.

Scores are per source. IOC `1.0` is displayed as a rule match, NER confidence as
unavailable, and no aggregate attack probability is invented. The default
saved-NER config continues to run without hard relevance gating.

Loading or editing another input clears the previous results. File uploads
retain a UTF-8 BOM, CRLF and non-BMP characters: the browser stores the untouched
decoded input separately from the textarea's newline-normalized display until
an edit occurs. Report markup is rendered with text nodes, not HTML templates.
Error, empty, busy and disconnected states are explicit.

## API contract

| Route | Behavior |
|---|---|
| `GET /api/status` | Schema version, model-mode readiness and report length limit |
| `GET /api/example` | The tracked synthetic merge report; no arbitrary file path |
| `POST /api/analyze` | JSON `{"mode":"ner","text":"..."}` or `mode: "ioc"` |
| `/`, `/app.js`, `/styles.css` | The local static frontend |

Analysis returns `{"document": <existing Document JSON>, "analysis":
{"mode": "...", "elapsed_ms": ...}}`. Downloaded JSON is the document alone,
consistent with the CLI output. The API reuses the existing `Pipeline`, with
absolute model paths anchored to the checkout.

Pipelines are cached per mode and run under a lock; another concurrent analysis
gets an explicit busy response. Mode initialization failures are cached, so
restart the server after training/replacing models or correcting the environment.
The demo accepts up to 100,000 Unicode code points and 1 MiB JSON requests.

The server binds to localhost only, checks Host/Origin, serves fixed assets,
returns JSON errors and does not enable cross-origin requests. It writes no
report access log or analysis history. This is the project's local research
review server, not a public multi-user deployment.

## Validation completed

- **138 tests pass in WSL**, including 17 new API/server tests:
  `results/web_validation.txt`.
- **17 API/server tests pass on native Windows Python 3.11**, using real IOC
  extraction without loading the blocked native NER stack:
  `results/web_windows_validation.txt`.
- Actual headless Chrome QA passed: real CRF example analysis, source details,
  component toggle, search, label filtering, method dialog, JSON download, input
  invalidation, literal markup, Unicode highlighting, real BOM/CRLF file upload,
  IOC mode, unsupported-language messaging, mobile layout and network failure.
  See `results/web_browser_validation.json`. The recorded offline network error
  is intentionally induced by that check; unexpected browser errors fail QA.
- Desktop and 390px mobile screenshots:
  `results/web_desktop.png` and `results/web_mobile.png`.
- The wheel built and contains all three static assets:
  `results/web_packaging_validation.txt`. Temporary tracked egg-info changes
  from the build were restored.
- JavaScript syntax, workflow shell/Python syntax and real-model API assertions
  were checked locally. Existing model binaries, dependency pins, datasets and
  earlier result artifacts remain unchanged. The official test split was not
  evaluated.

A combined background server/browser launch was rejected by automatic approval
review with only “blocked by policy” as its reason. A safer foreground localhost
server and isolated browser QA with private pipes succeeded. No remote-debugging
port or security-policy change was needed. QA-owned processes were stopped
after checks.

The preceding merge commit (`3041904`) has a successful
[GitHub Actions run](https://github.com/Umangpatel5122006/NLP-for-Cyber-Threat-Intelligence-Extraction/actions/runs/43680752877).
This step adds JavaScript syntax and real saved-model API checks to CI; the
updated workflow awaits a remote run after commit/push. Local browser QA is
recorded separately and is not claimed as a remote CI browser run.

## Review and next step

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -p test_web.py -v
git diff --check
git status --short
```

To repeat optional browser QA, start the full WSL server first, then use an
installed Chrome/Chromium and Node:

```powershell
node scripts/check_web_ui.mjs 'C:\Program Files\Google\Chrome\Application\chrome.exe'
```

Node and Chrome are optional verification tools, not application dependencies.
The checker uses a separate ignored profile/download area under `models/`.

Suggested commit: `feat: add local browser workbench for CTI report review`.
Commits and pushes remain with the user.

After review, return to backend/model quality: inspect action boundary errors
on train/dev, compare a targeted classical feature/rule change with this frozen
baseline, then work toward technique classification. The UI helps inspect
evidence; it does not change model accuracy or complete the remaining research.
