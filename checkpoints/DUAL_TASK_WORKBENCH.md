# Message risk and report extraction workbench

Date: 2026-11-16 (Asia/Calcutta)

## Result

Both project goals now have equal space in the workbench. Message risk has its
own assessment, positive-finding count and inspection button. Report extraction
has separate entity/action/IOC counts and execution coverage. A suspicious
message remains flagged when extraction returns zero entities or IOCs.
An extracted address or action does not become a message threat verdict.

The Findings scope selector supports All tasks, Message risk and Report
extraction alongside the existing search, label and host-component controls.
Inspection buttons filter and move keyboard focus to the corresponding evidence.
Highlight navigation can reveal a finding across task scopes. Editing input,
changing mode/language or a failed request clears both task summaries.

The source panel offers separate message and report examples. The authored
message is a demonstration, not training or independent accuracy evidence.
Hindi/Hinglish rule Findings now identify their phrase-rule source rather than
being labeled English rules. The skip link is clipped until keyboard focus and
appears at the viewport top when used.

## Backend and export

`metadata.task_summary.version = 1` is included in the API document and JSON
download. Its message status mirrors the existing message assessment, preserving
model/rule disagreement. It never combines scores into a threat probability.
Extraction counts exclude host components, which are counted separately.

Extraction status distinguishes completed English model/IOC execution, IOC-only
mode, unsupported entity/action language, and missing detector coverage. Zero
findings is not used to infer either completion or safety. `/api/status` reports
the task-summary version. The browser requests a server restart for old responses
without this contract instead of displaying incomplete cards.

`/api/example` retains the existing report example; `?task=message` selects the
new fixed message file. Invalid, empty or repeated task selectors return 400.
Other legacy parameters remain ignored and cannot select arbitrary files.

## Verification

- **361 tests pass** in Ubuntu-24.04 WSL / Python 3.12.3, including nine new
  synthetic task-summary tests and two additional HTTP tests.
- The nine task-summary tests also pass in native Windows Python.
- **33 actual Chrome check groups pass**, covering real CRF/IOC/SMS execution,
  both task filters, positive messages with zero extraction, independent task
  results, JSON download, stale servers, request failures, keyboard navigation
  and 390px mobile layout without horizontal page overflow.
- All 26 authored Hindi/Hinglish regression cases pass in both modes with
  matching task coverage, original-source cues and correct rule-source labels.
  Long English reports show SMS coverage limits while extraction completes.
- JavaScript syntax and Git whitespace checks pass. Launcher `--check` selects
  prepared WSL and reports both modes plus the English SMS model ready.
- SHA-256 checks match all seven prior model/configuration files. No fitting,
  annotation edits, split changes, official test evaluation or remote CI run
  occurred. Browser diagnostics are development checks, not accuracy estimates.

Native model execution remains unavailable in the checked Windows environments;
the launcher finds the compatible WSL runtime. No security settings changed.
Full local logs and screenshots remain ignored; compact evidence is under
[`results/dual_task_v1`](../results/dual_task_v1/README.md).

Reproduce from the project root:

```powershell
wsl -d Ubuntu-24.04 --exec bash -c '~/.cache/cti-classical-py312/bin/python -m unittest discover -s tests -v'
python scripts/start_workbench.py --check
python scripts/start_workbench.py
```

With the server running, optional browser QA uses Node and installed Chrome:

```powershell
node scripts/check_web_ui.mjs 'C:\Program Files\Google\Chrome\Application\chrome.exe' http://127.0.0.1:8765 results/dual_task_v1/browser
```

## Review and remaining work

Restart an older server using Ctrl+C and `python scripts/start_workbench.py`.
Load each example, analyze, inspect each card and switch the Findings scope.
Try the previously supplied financial-offer message with its destination removed;
its message finding should remain visible even with zero extracted IOCs. Try a
Hindi/Hinglish example in both modes and check the explicit English-model limits.

This step changes presentation and coverage reporting, not trained accuracy.
English remains the only trained NLP language; Hindi/Hinglish use phrase rules.
The English SMS model remains limited to its SMS domain/length. Current CTI dev
exact-span F1 remains 0.2596 for entities and 0.1551 for actions. Real independently
reviewed multilingual messages and decisions on the 21 pending action annotation
conflicts remain useful inputs before further model changes. Final holdouts stay
reserved. Commits and pushes remain with the user.
