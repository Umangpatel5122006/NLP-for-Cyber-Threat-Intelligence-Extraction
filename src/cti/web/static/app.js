"use strict";

const $ = (id) => document.getElementById(id);
const labels = {
  THREAT_ACTOR: "Threat actor", MALWARE_TOOL: "Malware / tool", INFRASTRUCTURE: "Infrastructure",
  ACTION: "Action", IP: "IP address", DOMAIN: "Domain", URL: "URL", EMAIL: "Email", HASH: "Hash", CVE: "CVE",
  SOCIAL_ENGINEERING: "Message pattern", SMS_SPAM: "SMS spam prediction",
};
const iocLabels = new Set(["IP", "DOMAIN", "URL", "EMAIL", "HASH", "CVE"]);
const sourceNames = {ner_entities: "Entity CRF", ner_action: "Action CRF", ioc_finder: "IOC finder", ioc_regex: "IOC regex", message_risk: "English message rules", message_classifier: "Classical SMS classifier"};
const overlapNames = {action_overlap: "Action overlap", label_conflict: "Label conflict", contained_overlap: "Contained overlap", crossing_overlap: "Crossing overlap"};
const state = {result: null, selected: null, modes: [], busy: false, refreshing: false, maxCharacters: 100000, rawInput: null};

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}
function tone(label) {
  if (["SOCIAL_ENGINEERING", "SMS_SPAM"].includes(label)) return "tone-risk";
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
  for (const id of ["report-input", "report-file", "example-button", "clear-button", "report-language"]) $(id).disabled = state.busy;
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
  $("map-risk").hidden = true;
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
    : "Both modes include English, Hindi and Hinglish message-pattern checks. Zero extracted findings is not a safety verdict.";
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
function reviewFindings() {
  return state.result?.document.review_findings || state.result?.document.findings || [];
}
function visibleFindings() {
  if (!state.result) return [];
  const search = $("finding-search").value.trim().toLowerCase();
  const label = $("label-filter").value;
  return reviewFindings().filter((finding) =>
    ($("show-components").checked || finding.role === "primary") && (!label || finding.label === label) &&
    (!search || [finding.title || "", finding.description || "", finding.text, finding.original_text, finding.canonical_value || "", ...finding.sources, ...finding.sources.map((source) => sourceNames[source] || source)].join(" ").toLowerCase().includes(search)));
}
function selectFinding(id, reveal = false) {
  if (!state.result) return;
  if (reveal) {
    $("finding-search").value = "";
    $("label-filter").value = "";
    const finding = reviewFindings().find((item) => item.id === id);
    if (finding?.role === "component") $("show-components").checked = true;
  }
  state.selected = id;
  renderResults();
}
function riskSignals() {
  const doc = state.result?.document, risk = doc?.metadata.message_risk;
  if (!risk || !["suspicious", "review_context"].includes(risk.status)) return [];
  const original = Array.from(doc.original_text);
  return (risk.signals || []).map((signal, index) => ({...signal, index})).filter((signal) =>
    Number.isInteger(signal.original_start) && Number.isInteger(signal.original_end) &&
    signal.original_start >= 0 && signal.original_end <= original.length && signal.original_start < signal.original_end &&
    original.slice(signal.original_start, signal.original_end).join("") === signal.original_text);
}
function revealRiskSignal(index) {
  const row = reviewFindings().find((item) => item.kind === "message_rule");
  if (row) selectFinding(row.id, true);
  const signal = $("risk-signal-" + index);
  if (!signal) return;
  signal.scrollIntoView({block: "center"});
  signal.focus({preventScroll: true});
}
function renderReport(findings) {
  const original = Array.from(state.result.document.original_text);
  const events = new Map();
  const at = (index) => {
    if (!events.has(index)) events.set(index, {add: [], remove: []});
    return events.get(index);
  };
  at(0); at(original.length);
  const spans = [...findings.filter((finding) => !finding.kind || finding.kind === "extraction"), ...riskSignals().map((signal) => ({
    id: "risk-" + signal.index, label: "MESSAGE_RISK", text: signal.original_text,
    original_start: signal.original_start, original_end: signal.original_end,
    start: signal.original_start, end: signal.original_end, risk_signal: signal,
  }))];
  for (const finding of spans) {
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
      Number(Boolean(a.risk_signal)) - Number(Boolean(b.risk_signal)) ||
      Number(a.label === "ACTION") - Number(b.label === "ACTION") ||
      (a.end - a.start) - (b.end - b.start) || a.id.localeCompare(b.id));
    const target = active.get(state.selected) || matches[0];
    const mark = element("mark", target.risk_signal ? "tone-risk" : tone(target.label), text);
    mark.tabIndex = 0;
    mark.setAttribute("role", "button");
    const caption = (finding) => finding.risk_signal ? "Scam-pattern cue (" + finding.risk_signal.label + ")" : labels[finding.label] || finding.label;
    mark.setAttribute("aria-label", caption(target) + ": " + target.text + ". View source evidence.");
    mark.title = matches.map((finding) => caption(finding) + ": " + finding.text).join("\n");
    if (target.risk_signal) mark.dataset.riskIndex = target.risk_signal.index;
    mark.classList.toggle("selected", active.has(state.selected));
    const activate = target.risk_signal ? () => revealRiskSignal(target.risk_signal.index) : () => selectFinding(target.id);
    mark.addEventListener("click", activate);
    mark.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") { event.preventDefault(); activate(); }
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
  const finding = reviewFindings().find((item) => item.id === state.selected);
  if (!finding) { emptyDetail(); return; }
  if (finding.kind && finding.kind !== "extraction") { renderMessageDetail(finding); return; }
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
function renderMessageDetail(finding) {
  const detail = $("finding-detail"), header = element("div", "detail-heading");
  header.append(badge(finding.label), element("span", "detail-id", finding.kind === "message_rule" ? "Rule finding" : "Model prediction"));
  detail.replaceChildren(header, element("h3", "detail-title", finding.title), element("p", "", finding.description), element("div", "detail-caption", "Original characters " + finding.original_start + "\u2013" + finding.original_end + " \u00b7 end excluded"));
  detail.append(detailSection("ORIGINAL EVIDENCE", element("pre", "detail-excerpt", finding.original_text)));
  const evidence = element("div");
  for (const item of finding.evidence) {
    const card = element("div", "source-card");
    card.append(element("strong", "", item.label === "SOCIAL_ENGINEERING" ? item.explanation : sourceNames[item.source]), element("pre", "detail-excerpt", item.original_text));
    evidence.append(card);
  }
  detail.append(detailSection("SOURCE EVIDENCE", evidence));
  if (finding.matched_rules) detail.append(detailSection("MATCHED RULES", element("p", "", finding.matched_rules.join(", "))));
  if (finding.kind === "message_model") {
    detail.append(detailSection("MODEL DECISION", element("p", "", finding.profile + " \u00b7 margin " + finding.decision_margin.toFixed(4) + " \u00b7 threshold 0. Margin is not a probability.")));
    const features = element("ul");
    for (const item of finding.supporting_features) features.append(element("li", "", item.feature + ": " + item.contribution));
    detail.append(detailSection("SPAM-SUPPORTING LINEAR FEATURES", features));
  }
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
    main.append(element("span", "finding-text", finding.kind && finding.kind !== "extraction" ? finding.title : finding.text), element("span", "finding-meta", finding.evidence.length + (finding.evidence.length === 1 ? " source observation" : " source observations") + (finding.role === "component" ? " · Host component" : "") + (finding.kind === "message_rule" ? " · English rules" : finding.kind === "message_model" ? " · Classical prediction" : "")));
    row.append(badge(finding.label), main, element("span", "row-arrow", "›"));
    row.addEventListener("click", () => selectFinding(finding.id));
    fragment.append(row);
  }
  if (!findings.length) {
    const empty = element("div", "list-empty");
    const limited = ["unsupported_language", "partial_coverage"].includes(state.result?.document?.metadata?.message_assessment?.status);
    empty.append(element("p", "", reviewFindings().length ? "No findings match these filters." : limited ? "No extracted findings; analysis coverage is limited." : "No detections requiring review."), element("small", "", "Missing detections do not establish safety. Check model coverage and message assessment."));
    fragment.append(empty);
  }
  list.replaceChildren(fragment);
  $("result-summary").textContent = findings.length + " shown · " + reviewFindings().length + " total review findings";
  renderReport(findings);
  renderDetails();
}
function renderMessageRisk(risk) {
  $("risk-method").textContent = risk.indic_rule_version ? "Hindi/Hinglish phrase rules v" + risk.indic_rule_version : "Classical model + English rules v" + risk.rule_version;
  const model = state.result.document.metadata.message_classifier;
  const assessment = state.result.document.metadata.message_assessment || risk;
  $("model-verdict").textContent = model?.status === "ok" ? "Classical SMS model: " + (model.label === "spam" ? "spam predicted" : "ham predicted") : "Classical SMS model: " + (model?.status === "not_applicable" ? "outside its scope" : "unavailable");
  $("model-details").textContent = model?.status === "ok" ? model.profile + " \u00b7 " + (model.training_protocol || "uci_sms_v1") + " \u00b7 decision margin " + model.decision_margin.toFixed(4) + ". This score is not a threat probability." : model?.reason || model?.message || "Train with python -m cti.train_message --download in the chosen server runtime.";
  $("model-disagreement").textContent = assessment?.disagreement ? "The classifier predicts ham, but message rules flag suspicious cues. The suspicious finding is retained for review." : "Ham/spam is a separate prediction from scam-pattern rules and CTI entity extraction.";
  $("message-risk").hidden = false;
  $("message-risk").dataset.status = assessment?.status || "unavailable";
  $("risk-status").textContent = {suspicious: "Suspicious message detected for review", review_context: "Review the message context", no_pattern_detected: "No supported message-risk detection", unsupported_language: "Message-rule language coverage unavailable", partial_coverage: "Message analysis coverage is incomplete"}[assessment?.status] || "Message risk check unavailable";
  $("risk-summary").textContent = assessment?.summary || "Restart the local server to load the message risk check. Extracted findings alone do not establish safety.";
  $("language-coverage").textContent = assessment?.coverage?.language_note || "Only English NLP is supported; language has not been verified.";
  $("risk-alert").hidden = false;
  $("risk-alert").dataset.status = assessment?.status || "unavailable";
  $("risk-headline").textContent = $("risk-status").textContent;
  $("risk-alert-summary").textContent = $("risk-summary").textContent;
  $("map-risk").hidden = false;
  $("map-risk").dataset.status = assessment?.status || "unavailable";
  $("map-risk-title").textContent = $("risk-status").textContent;
  $("map-risk-note").textContent = risk?.status === "suspicious"
    ? "This message is flagged by " + (risk.indic_rule_version ? "Hindi/Hinglish phrase" : "English scam-pattern") + " rules. Orange highlights show the cues, even when no entities or technical IOCs are extracted."
    : risk?.status === "review_context"
      ? "Highlighted cues appear in warning or example context. Review the wording before interpreting the message."
      : $("risk-summary").textContent;
  $("risk-limitations").textContent = risk?.limitations || "English rule patterns only. No safety verdict is available.";
  $("risk-signals").replaceChildren(...(risk?.signals || []).map((signal, index) => {
    const item = element("li", "risk-signal");
    item.id = "risk-signal-" + index;
    item.tabIndex = -1;
    item.append(element("strong", "", signal.label + ": "), element("q", "risk-excerpt", signal.original_text), element("small", "", signal.explanation + " Original characters " + signal.original_start + "\u2013" + signal.original_end + "."));
    return item;
  }));
}
function showAnalysis(result) {
  if (!Array.isArray(result?.document?.review_findings)) {
    throw new Error("The running server uses the old Findings format. Stop it with Ctrl+C, run python scripts/start_workbench.py, then reload this page.");
  }
  if (result.document.metadata?.message_risk?.rule_version !== 2) {
    throw new Error("The running server has outdated message rules. Stop it with Ctrl+C, run python scripts/start_workbench.py, then reload this page. No model retraining is needed for this rule update.");
  }
  if (result.document.metadata?.message_assessment?.coverage?.version !== 2) {
    throw new Error("The running server has outdated language-coverage reporting. Stop it with Ctrl+C, run python scripts/start_workbench.py, then reload this page. No model retraining is needed.");
  }
  if (["hi", "hi-en"].includes(result.document.metadata?.language) && result.document.metadata?.message_risk?.indic_rule_version !== 2) {
    throw new Error("The running server has outdated Hindi/Hinglish rules. Stop it with Ctrl+C, run python scripts/start_workbench.py, then reload this page. No model retraining is needed.");
  }
  state.result = result;
  const doc = result.document, rows = reviewFindings(), primary = rows.filter((finding) => finding.role === "primary");
  const extractions = doc.findings.filter((finding) => finding.role === "primary");
  renderMessageRisk(doc.metadata.message_risk);
  $("stat-findings").textContent = primary.length;
  $("stat-iocs").textContent = extractions.filter((finding) => iocLabels.has(finding.label)).length;
  $("stat-actions").textContent = extractions.filter((finding) => finding.label === "ACTION").length;
  $("stat-entities").textContent = extractions.filter((finding) => !iocLabels.has(finding.label) && finding.label !== "ACTION").length;
  $("finding-count").textContent = rows.length;
  $("label-filter").replaceChildren(new Option("All labels", ""), ...Array.from(new Set(rows.map((finding) => finding.label))).sort().map((label) => new Option(labels[label] || label, label)));
  for (const id of ["download-button", "finding-search", "label-filter", "show-components"]) $(id).disabled = false;
  $("finding-search").value = "";
  $("show-components").checked = false;
  const languageName = {en: "English", hi: "Hindi", "hi-en": "Hinglish"}[doc.metadata.language];
  $("analysis-timing").textContent = (result.analysis.elapsed_ms / 1000).toFixed(2) + "s · " + (languageName ? languageName + (doc.metadata.language_detection?.status === "user_selected" ? " selected" : doc.metadata.language === "en" ? " assumed" : " guessed") : (doc.metadata.script || "unknown") + " script");
  const cueCount = riskSignals().length;
  $("map-note").textContent = doc.entities.length + " extraction observations · " + cueCount + " message-risk cues · Select a highlight to inspect its evidence";
  const unsupported = doc.metadata.language !== "en" || Object.values(doc.metadata.ner || {}).some((layer) => layer.status === "unsupported_language");
  if (["hi", "hi-en"].includes(doc.metadata.language)) notice("Hindi/Hinglish phrase checks are active. Trained SMS and entity/action models support English only; IOC evidence remains available.");
  else if (unsupported) notice("Language coverage is unavailable or uncertain. Message rules cover English/Hindi/Hinglish; trained models cover English only. IOC evidence remains available.");
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
    const payload = {text, mode: $("analysis-mode").value};
    if ($("report-language").value !== "auto") payload.language = $("report-language").value;
    const result = await fetchJSON("/api/analyze", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)});
    showAnalysis(result);
  } catch (error) { notice(error.message || "The analysis request failed.", true); }
  finally { setBusy(false); }
}

$("report-input").addEventListener("input", () => { state.rawInput = null; $("file-name").textContent = $("report-input").value ? "Pasted / edited text" : "UTF-8 \u00b7 .txt or .md"; resetResults(); notice(); updateControls(); });
$("analysis-mode").addEventListener("change", () => { resetResults(); notice(); showModeSetup(); updateControls(); });
$("report-language").addEventListener("change", () => { resetResults(); notice(); updateControls(); });
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
