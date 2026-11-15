"use strict";

const $ = (id) => document.getElementById(id);
const labels = {
  THREAT_ACTOR: "Threat actor", MALWARE_TOOL: "Malware / tool", INFRASTRUCTURE: "Infrastructure",
  ACTION: "Action", IP: "IP address", DOMAIN: "Domain", URL: "URL", EMAIL: "Email", HASH: "Hash", CVE: "CVE",
};
const iocLabels = new Set(["IP", "DOMAIN", "URL", "EMAIL", "HASH", "CVE"]);
const sourceNames = {ner_entities: "Entity CRF", ner_action: "Action CRF", ioc_finder: "IOC finder", ioc_regex: "IOC regex"};
const overlapNames = {action_overlap: "Action overlap", label_conflict: "Label conflict", contained_overlap: "Contained overlap", crossing_overlap: "Crossing overlap"};
const state = {result: null, selected: null, modes: [], busy: false, refreshing: false, maxCharacters: 100000, rawInput: null};

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}
function tone(label) {
  if (iocLabels.has(label)) return "tone-ioc";
  return {THREAT_ACTOR: "tone-actor", MALWARE_TOOL: "tone-tool", ACTION: "tone-action"}[label] || "tone-infra";
}
function badge(label) { return element("span", "type-badge " + tone(label), labels[label] || label); }
function notice(message = "", error = false) {
  $("notice").textContent = message;
  $("notice").hidden = !message;
  $("notice").classList.toggle("error", error);
}
async function fetchJSON(url, options) {
  const response = await fetch(url, options);
  let body;
  try { body = await response.json(); }
  catch { throw new Error("The local server returned an unreadable response."); }
  if (!response.ok) throw new Error(body.error?.message || "The request could not be completed.");
  return body;
}
function reportText() {
  return state.rawInput && $("report-input").value === state.rawInput.editor
    ? state.rawInput.original : $("report-input").value;
}
function updateControls() {
  const ready = state.modes.some((mode) => mode.id === $("analysis-mode").value && mode.ready);
  $("analyze-button").disabled = state.busy || state.refreshing || !ready || !reportText().trim();
  // Keep unavailable modes selectable so their setup instructions are visible.
  $("analysis-mode").disabled = state.busy || state.refreshing;
  for (const id of ["report-input", "report-file", "example-button", "clear-button"]) $(id).disabled = state.busy;
  $("connection-status").disabled = state.busy || state.refreshing;
  $("retry-models").disabled = state.busy || state.refreshing;
  $("analyze-button").classList.toggle("busy", state.busy);
  $("analyze-button").querySelector("span").textContent = state.busy ? "Analyzing…" : "Analyze report";
  $("character-count").textContent = Array.from(reportText()).length.toLocaleString() + " characters";
}
function setBusy(busy) {
  state.busy = busy;
  $("workspace").setAttribute("aria-busy", String(busy));
  updateControls();
}
function emptyDetail() {
  const wrapper = element("div", "detail-empty");
  wrapper.append(element("span", "detail-symbol", "⌁"), element("h3", "", "Follow the evidence"), element("p", "", "Select a finding to see the original excerpt, detector sources, and related overlaps."));
  $("finding-detail").replaceChildren(wrapper);
}
function resetResults() {
  state.result = null;
  state.selected = null;
  $("message-risk").hidden = true;
  $("risk-alert").hidden = true;
  $("risk-signals").replaceChildren();
  $("report-view").hidden = true;
  $("report-view").replaceChildren();
  $("report-empty").hidden = false;
  for (const name of ["findings", "iocs", "entities", "actions"]) $("stat-" + name).textContent = "—";
  for (const id of ["download-button", "finding-search", "label-filter", "show-components"]) $(id).disabled = true;
  $("finding-search").value = "";
  $("show-components").checked = false;
  $("label-filter").replaceChildren(new Option("All labels", ""));
  $("finding-count").textContent = "0";
  $("result-summary").textContent = "No analysis yet";
  $("analysis-timing").textContent = "Original report";
  $("map-note").textContent = "Highlights will reference the original report.";
  const empty = element("div", "list-empty");
  empty.append(element("span", "empty-plus", "+"), element("p", "", "Findings will appear here."), element("small", "", "Start with your report or load the example above."));
  $("finding-list").replaceChildren(empty);
  emptyDetail();
}
function setInput(text, filename) {
  $("report-input").value = text;
  // Textareas normalize CRLF. Keep the untouched file/example until an edit.
  state.rawInput = {original: text, editor: $("report-input").value};
  $("file-name").textContent = filename || "UTF-8 · .txt or .md";
  resetResults();
  notice();
  updateControls();
}
function showModeSetup() {
  const mode = state.modes.find((item) => item.id === $("analysis-mode").value);
  const ner = state.modes.find((item) => item.id === "ner");
  // Keep NER setup visible after the initial fallback to IOC mode.
  const unavailable = mode && !mode.ready ? mode : ner && !ner.ready ? ner : null;
  $("mode-setup").hidden = !unavailable;
  $("mode-help").textContent = mode && !mode.ready
    ? "This mode needs setup before analysis can run. IOC mode remains available when ready."
    : "Both modes include English scam-pattern checks. Zero extracted findings is not a safety verdict.";
  if (!unavailable) return;
  $("setup-heading").textContent = unavailable.label + " needs setup";
  $("setup-message").textContent = unavailable.message;
  $("setup-steps").replaceChildren(...(unavailable.setup || ["Check the server terminal, then retry loading models."]).map((step) => element("li", "", step)));
}
async function refreshStatus() {
  if (state.refreshing || state.busy) return;
  state.refreshing = true;
  const initial = !state.modes.length;
  updateControls();
  $("connection-status").classList.add("is-loading");
  $("connection-status").classList.remove("is-error");
  $("connection-text").textContent = "Connecting to engine…";
  try {
    const status = await fetchJSON("/api/status");
    state.modes = status.modes;
    state.maxCharacters = status.max_characters;
    for (const option of $("analysis-mode").options) {
      const mode = state.modes.find((item) => item.id === option.value);
      option.disabled = false;
      option.textContent = (mode?.label || option.textContent.replace(" (setup required)", "")) + (mode && !mode.ready ? " (setup required)" : "");
    }
    if (initial && !state.modes.some((mode) => mode.id === $("analysis-mode").value && mode.ready)) {
      const available = state.modes.find((mode) => mode.ready);
      if (available) $("analysis-mode").value = available.id;
    }
    const nerReady = state.modes.some((mode) => mode.id === "ner" && mode.ready);
    const iocReady = state.modes.some((mode) => mode.id === "ioc" && mode.ready);
    $("connection-text").textContent = nerReady ? "NER + IOC engine ready" : iocReady ? "IOC engine ready" : "Engine unavailable";
    $("connection-status").classList.toggle("is-error", !nerReady && !iocReady);
    if (!nerReady) notice(state.modes.find((mode) => mode.id === "ner")?.message || "The engine is unavailable.", !iocReady);
    else notice();
  } catch {
    state.modes = [];
    $("connection-text").textContent = "Server disconnected";
    $("connection-status").classList.add("is-error");
    notice("Could not reach the local server. Start it, then use the connection button to retry.", true);
  } finally {
    state.refreshing = false;
    $("connection-status").classList.remove("is-loading");
    showModeSetup();
    updateControls();
  }
}
function visibleFindings() {
  if (!state.result) return [];
  const search = $("finding-search").value.trim().toLowerCase();
  const label = $("label-filter").value;
  return state.result.document.findings.filter((finding) =>
    ($("show-components").checked || finding.role === "primary") && (!label || finding.label === label) &&
    (!search || [finding.text, finding.original_text, finding.canonical_value || "", ...finding.sources, ...finding.sources.map((source) => sourceNames[source] || source)].join(" ").toLowerCase().includes(search)));
}
function selectFinding(id, reveal = false) {
  if (!state.result) return;
  if (reveal) {
    $("finding-search").value = "";
    $("label-filter").value = "";
    const finding = state.result.document.findings.find((item) => item.id === id);
    if (finding?.role === "component") $("show-components").checked = true;
  }
  state.selected = id;
  renderResults();
}
function renderReport(findings) {
  const original = Array.from(state.result.document.original_text);
  const events = new Map();
  const at = (index) => {
    if (!events.has(index)) events.set(index, {add: [], remove: []});
    return events.get(index);
  };
  at(0); at(original.length);
  for (const finding of findings) {
    at(finding.original_start).add.push(finding);
    at(finding.original_end).remove.push(finding.id);
  }
  const positions = Array.from(events.keys()).sort((a, b) => a - b);
  const active = new Map();
  const fragment = document.createDocumentFragment();
  for (let i = 0; i < positions.length - 1; i++) {
    const position = positions[i], event = events.get(position);
    for (const id of event.remove) active.delete(id);
    for (const finding of event.add) active.set(finding.id, finding);
    const text = original.slice(position, positions[i + 1]).join("");
    if (!active.size) { fragment.append(document.createTextNode(text)); continue; }
    const matches = Array.from(active.values()).sort((a, b) =>
      Number(a.label === "ACTION") - Number(b.label === "ACTION") ||
      (a.end - a.start) - (b.end - b.start) || a.id.localeCompare(b.id));
    const target = active.get(state.selected) || matches[0];
    const mark = element("mark", tone(target.label), text);
    mark.tabIndex = 0;
    mark.setAttribute("role", "button");
    mark.setAttribute("aria-label", (labels[target.label] || target.label) + ": " + target.text + ". View source evidence.");
    mark.title = matches.map((finding) => (labels[finding.label] || finding.label) + ": " + finding.text).join("\n");
    mark.classList.toggle("selected", active.has(state.selected));
    mark.addEventListener("click", () => selectFinding(target.id));
    mark.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") { event.preventDefault(); selectFinding(target.id); }
    });
    fragment.append(mark);
  }
  $("report-view").replaceChildren(fragment);
  $("report-view").hidden = false;
  $("report-empty").hidden = true;
}
function detailSection(title, content) {
  const section = element("section", "detail-section");
  section.append(element("h4", "", title), content);
  return section;
}
function renderDetails() {
  const doc = state.result.document;
  const finding = doc.findings.find((item) => item.id === state.selected);
  if (!finding) { emptyDetail(); return; }
  const detail = $("finding-detail"), header = element("div", "detail-heading");
  header.append(badge(finding.label), element("span", "detail-id", finding.id));
  detail.replaceChildren(header, element("h3", "detail-title", finding.text), element("div", "detail-caption", "Original characters " + finding.original_start + "–" + finding.original_end + " · end excluded"));
  detail.append(detailSection("ORIGINAL EVIDENCE", element("pre", "detail-excerpt", finding.original_text)));
  if (finding.canonical_status === "available") detail.append(detailSection("CANONICAL VALUE", element("p", "canonical-value", finding.canonical_value)));
  const sources = element("div");
  for (const evidence of finding.evidence) {
    const card = element("div", "source-card"), line = element("div", "source-line");
    let score = "Unavailable";
    if (evidence.confidence !== null) score = evidence.source.startsWith("ioc_") && evidence.confidence === 1 ? "Rule match" : "Source score " + evidence.confidence;
    line.append(element("span", "", sourceNames[evidence.source] || evidence.source || "Unnamed source"), element("span", "source-score", score));
    card.append(line, element("div", "source-label", labels[evidence.label] || evidence.label));
    sources.append(card);
  }
  detail.append(detailSection("DETECTOR EVIDENCE · NO COMBINED SCORE", sources));
  const links = element("div"), related = [];
  for (const id of finding.parent_ids) related.push([id, "Host of"]);
  for (const id of finding.component_ids) related.push([id, "Host component"]);
  for (const edge of doc.overlaps) {
    if (edge.kind === "ioc_host_component") continue;
    if (edge.left_id === finding.id) related.push([edge.right_id, overlapNames[edge.kind] || edge.kind]);
    else if (edge.right_id === finding.id) related.push([edge.left_id, overlapNames[edge.kind] || edge.kind]);
  }
  for (const [id, caption] of related) {
    const other = doc.findings.find((item) => item.id === id);
    if (!other) continue;
    const button = element("button", "related-button", caption + " · " + other.text);
    button.type = "button";
    button.addEventListener("click", () => selectFinding(id, true));
    links.append(button);
  }
  if (related.length) detail.append(detailSection("RELATED EVIDENCE", links));
}
function renderResults() {
  const findings = visibleFindings();
  if (!findings.some((finding) => finding.id === state.selected)) state.selected = findings[0]?.id || null;
  const list = $("finding-list"), fragment = document.createDocumentFragment();
  for (const finding of findings) {
    const row = element("button", "finding-row" + (finding.id === state.selected ? " active" : ""));
    row.type = "button";
    row.dataset.finding = finding.id;
    row.setAttribute("aria-pressed", String(finding.id === state.selected));
    const main = element("span", "finding-main");
    main.append(element("span", "finding-text", finding.text), element("span", "finding-meta", finding.evidence.length + (finding.evidence.length === 1 ? " source observation" : " source observations") + (finding.role === "component" ? " · Host component" : "")));
    row.append(badge(finding.label), main, element("span", "row-arrow", "›"));
    row.addEventListener("click", () => selectFinding(finding.id));
    fragment.append(row);
  }
  if (!findings.length) {
    const empty = element("div", "list-empty");
    empty.append(element("p", "", state.result.document.findings.length ? "No findings match these filters." : "No entities or technical indicators were extracted."), element("small", "", "Check the separate message risk result. Missing findings do not establish safety."));
    fragment.append(empty);
  }
  list.replaceChildren(fragment);
  $("result-summary").textContent = findings.length + " shown · " + state.result.document.findings.length + " total findings";
  renderReport(findings);
  renderDetails();
}
function renderMessageRisk(risk) {
  $("message-risk").hidden = false;
  $("message-risk").dataset.status = risk?.status || "unavailable";
  $("risk-status").textContent = {suspicious: "Suspicious message patterns detected", review_context: "Review the message context", no_pattern_detected: "No supported scam pattern detected"}[risk?.status] || "Message risk check unavailable";
  $("risk-summary").textContent = risk?.summary || "Restart the local server to load the message risk check. Extracted findings alone do not establish safety.";
  $("risk-alert").hidden = false;
  $("risk-alert").dataset.status = risk?.status || "unavailable";
  $("risk-headline").textContent = $("risk-status").textContent;
  $("risk-alert-summary").textContent = $("risk-summary").textContent;
  $("risk-limitations").textContent = risk?.limitations || "English rule patterns only. No safety verdict is available.";
  $("risk-signals").replaceChildren(...(risk?.signals || []).map((signal) => {
    const item = element("li", "risk-signal");
    item.append(element("strong", "", signal.label + ": "), element("q", "risk-excerpt", signal.original_text), element("small", "", signal.explanation + " Original characters " + signal.original_start + "\u2013" + signal.original_end + "."));
    return item;
  }));
}
function showAnalysis(result) {
  state.result = result;
  const doc = result.document, primary = doc.findings.filter((finding) => finding.role === "primary");
  renderMessageRisk(doc.metadata.message_risk);
  $("stat-findings").textContent = primary.length;
  $("stat-iocs").textContent = primary.filter((finding) => iocLabels.has(finding.label)).length;
  $("stat-actions").textContent = primary.filter((finding) => finding.label === "ACTION").length;
  $("stat-entities").textContent = primary.filter((finding) => !iocLabels.has(finding.label) && finding.label !== "ACTION").length;
  $("finding-count").textContent = doc.findings.length;
  $("label-filter").replaceChildren(new Option("All labels", ""), ...Array.from(new Set(doc.findings.map((finding) => finding.label))).sort().map((label) => new Option(labels[label] || label, label)));
  for (const id of ["download-button", "finding-search", "label-filter", "show-components"]) $(id).disabled = false;
  $("finding-search").value = "";
  $("show-components").checked = false;
  $("analysis-timing").textContent = (result.analysis.elapsed_ms / 1000).toFixed(2) + "s · " + (doc.metadata.language || "unknown").toUpperCase();
  $("map-note").textContent = doc.entities.length + " detector observations · Source-covering highlights · Select a finding to inspect";
  const unsupported = Object.values(doc.metadata.ner || {}).some((layer) => layer.status === "unsupported_language");
  if (unsupported) notice("Entity and action models support English. IOC findings are shown for this report.");
  else notice();
  renderResults();
}
async function analyze() {
  if (state.busy || state.refreshing) return;
  if (!state.modes.some((mode) => mode.id === $("analysis-mode").value && mode.ready)) { showModeSetup(); notice("Set up the selected mode or choose an available mode.", true); return; }
  const text = reportText();
  if (!text.trim()) { notice("Add report text before analyzing.", true); return; }
  if (Array.from(text).length > state.maxCharacters) { notice("Reports can contain up to " + state.maxCharacters.toLocaleString() + " characters.", true); return; }
  resetResults(); setBusy(true); notice();
  try {
    const result = await fetchJSON("/api/analyze", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({text, mode: $("analysis-mode").value})});
    showAnalysis(result);
  } catch (error) { notice(error.message || "The analysis request failed.", true); }
  finally { setBusy(false); }
}

$("report-input").addEventListener("input", () => { state.rawInput = null; $("file-name").textContent = $("report-input").value ? "Pasted / edited text" : "UTF-8 \u00b7 .txt or .md"; resetResults(); notice(); updateControls(); });
$("analysis-mode").addEventListener("change", () => { resetResults(); notice(); showModeSetup(); updateControls(); });
$("analyze-button").addEventListener("click", analyze);
$("connection-status").addEventListener("click", refreshStatus);
$("retry-models").addEventListener("click", refreshStatus);
$("clear-button").addEventListener("click", () => { setInput(""); $("report-file").value = ""; $("report-input").focus(); });
$("example-button").addEventListener("click", async () => {
  setBusy(true);
  try { const example = await fetchJSON("/api/example"); setInput(example.text, example.filename); }
  catch (error) { notice(error.message || "The example could not be loaded.", true); }
  finally { setBusy(false); }
});
$("report-file").addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  try {
    if (file.size > 1048576) throw new Error("Choose a UTF-8 text file smaller than 1 MB.");
    const text = new TextDecoder("utf-8", {fatal: true, ignoreBOM: true}).decode(await file.arrayBuffer());
    if (Array.from(text).length > state.maxCharacters) throw new Error("Reports can contain up to " + state.maxCharacters.toLocaleString() + " characters.");
    setInput(text, file.name);
  } catch (error) { notice(error instanceof TypeError ? "This file is not valid UTF-8 text." : error.message, true); }
  finally { event.target.value = ""; }
});
for (const id of ["finding-search", "label-filter", "show-components"]) $(id).addEventListener(id === "finding-search" ? "input" : "change", () => { if (state.result) renderResults(); });
$("download-button").addEventListener("click", () => {
  if (!state.result) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(state.result.document, null, 2) + "\n"], {type: "application/json;charset=utf-8"}));
  const link = element("a");
  link.href = url; link.download = "cti-analysis.json"; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
$("method-button").addEventListener("click", () => $("method-dialog").showModal());
$("close-method").addEventListener("click", () => $("method-dialog").close());
$("method-dialog").addEventListener("click", (event) => { if (event.target === $("method-dialog")) $("method-dialog").close(); });
updateControls();
refreshStatus();
