// Optional browser QA: Node + an installed Chromium browser, no npm packages.
// Run against an already running local server; browser control uses private pipes.
import {spawn} from "node:child_process";
import {mkdir, readFile, writeFile} from "node:fs/promises";
import {resolve} from "node:path";
import assert from "node:assert/strict";

const [browserPath, baseURL = "http://127.0.0.1:8765", outputDirectory = "results", previewProtocol] = process.argv.slice(2);
if (!browserPath) throw new Error("Usage: node scripts/check_web_ui.mjs <browser-executable> [local-url] [output-directory] [preview-training-protocol]");
if (!/^http:\/\/(127\.0\.0\.1|localhost):\d+$/.test(baseURL)) throw new Error("Use a loopback review server URL.");
await mkdir(outputDirectory, {recursive: true});
const browser = spawn(browserPath, [
  "--headless=new", "--remote-debugging-pipe", "--no-first-run", "--no-default-browser-check",
  "--disable-background-networking", "--user-data-dir=" + resolve("models/web-qa-profile"), "about:blank",
], {stdio: ["ignore", "ignore", "pipe", "pipe", "pipe"], windowsHide: true});
let sequence = 0, buffer = Buffer.alloc(0), stderr = "";
const pending = new Map(), failures = [];
browser.stderr.on("data", (chunk) => { stderr += chunk.toString(); });
browser.stdio[4].on("data", (chunk) => {
  buffer = Buffer.concat([buffer, chunk]);
  for (let boundary; (boundary = buffer.indexOf(0)) >= 0;) {
    const raw = buffer.subarray(0, boundary).toString();
    buffer = buffer.subarray(boundary + 1);
    if (!raw) continue;
    const packet = JSON.parse(raw);
    if (packet.id && pending.has(packet.id)) {
      const {resolve: done, reject, timer} = pending.get(packet.id);
      clearTimeout(timer);
      pending.delete(packet.id);
      packet.error ? reject(new Error(JSON.stringify(packet.error))) : done(packet.result);
    } else if (packet.method === "Runtime.exceptionThrown") {
      failures.push(packet.params.exceptionDetails.exception?.description || packet.params.exceptionDetails.text);
    } else if (packet.method === "Log.entryAdded" && packet.params.entry.level === "error") {
      failures.push(packet.params.entry.text);
    }
  }
});
browser.on("error", (error) => { for (const item of pending.values()) item.reject(error); });
function command(method, params = {}, sessionId) {
  return new Promise((done, reject) => {
    const id = ++sequence;
    const timer = setTimeout(() => {
      pending.delete(id);
      reject(new Error("Browser command timed out: " + method + "\n" + stderr.slice(-1500)));
    }, 30000);
    pending.set(id, {resolve: done, reject, timer});
    browser.stdio[3].write(JSON.stringify({id, method, params, ...(sessionId ? {sessionId} : {})}) + "\0");
  });
}
let session;
const checks = [];
const modelObservations = [];
const languageObservations = [];
const record = (message) => { checks.push(message); console.log(message); };
async function evaluate(expression) {
  const result = await command("Runtime.evaluate", {expression, awaitPromise: true, returnByValue: true}, session);
  if (result.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
  return result.result.value;
}
async function waitFor(expression) {
  for (let i = 0; i < 100; i++) {
    if (await evaluate(expression)) return;
    await new Promise((done) => setTimeout(done, 100));
  }
  throw new Error("UI condition timed out: " + expression);
}
const setText = (text) => evaluate("document.getElementById('report-input').value=" + JSON.stringify(text) + ";document.getElementById('report-input').dispatchEvent(new Event('input',{bubbles:true}));");
async function click(id) { await evaluate("document.getElementById(" + JSON.stringify(id) + ").click()"); }
async function analyze() {
  await click("analyze-button");
  await waitFor("!state.busy && !!state.result");
}
async function screenshot(name) {
  const metrics = await command("Page.getLayoutMetrics", {}, session);
  const size = metrics.cssContentSize;
  const capture = await command("Page.captureScreenshot", {
    format: "png", captureBeyondViewport: true,
    clip: {x: 0, y: 0, width: size.width, height: size.height, scale: 1},
  }, session);
  await writeFile(resolve(outputDirectory, name), Buffer.from(capture.data, "base64"));
}
try {
  const target = await command("Target.createTarget", {url: "about:blank"});
  session = (await command("Target.attachToTarget", {targetId: target.targetId, flatten: true})).sessionId;
  await command("Runtime.enable", {}, session);
  await command("Page.enable", {}, session);
  await command("Log.enable", {}, session);
  await command("Emulation.setDeviceMetricsOverride", {width: 1440, height: 1120, deviceScaleFactor: 1, mobile: false}, session);
  await command("Page.navigate", {url: baseURL + "/"}, session);
  await waitFor("typeof state !== 'undefined' && state.modes.some(m=>m.ready)");
  assert.equal(await evaluate("document.getElementById('analyze-button').disabled"), true);
  record("Initial workspace, server readiness and empty-input controls passed.");
  await click("example-button");
  await waitFor("!state.busy && document.getElementById('report-input').value.includes('APT29')");
  await analyze();
  const summary = await evaluate("({mode:state.result.analysis.mode,actionArtifact:state.result.document.metadata.ner?.action?.artifact_sha256,findings:state.result.document.review_findings.length,extractions:state.result.document.findings.length,primary:document.getElementById('stat-findings').textContent,map:document.getElementById('report-view').textContent,original:state.result.document.original_text,rows:document.querySelectorAll('.finding-row').length})");
  assert.equal(summary.map, summary.original);
  assert.equal(summary.rows, Number(summary.primary));
  assert.equal(summary.mode, "ner");
  assert.ok(summary.findings > summary.rows);
  record("Real CRF example, primary/component counts and exact original-text map passed.");
  await evaluate("Array.from(document.querySelectorAll('#report-view mark')).find(m=>m.getAttribute('aria-label').startsWith('IP address:')).click()");
  assert.equal(await evaluate("document.querySelector('#finding-detail .detail-title').textContent"), "185.22.10.5");
  await evaluate("document.querySelector('#report-view mark').focus()");
  await command("Input.dispatchKeyEvent", {type: "keyDown", key: "Enter", code: "Enter", windowsVirtualKeyCode: 13}, session);
  await command("Input.dispatchKeyEvent", {type: "keyUp", key: "Enter", code: "Enter", windowsVirtualKeyCode: 13}, session);
  assert.equal(await evaluate("document.querySelector('#finding-detail .detail-title').textContent"), "APT29");
  await evaluate("Array.from(document.querySelectorAll('#report-view mark')).find(m=>m.getAttribute('aria-label').startsWith('IP address:')).click()");
  assert.equal(await evaluate("document.querySelectorAll('#finding-detail .source-card').length"), 2);
  record("Report highlight clicks and keyboard selection open the correct source evidence.");
  await screenshot("web_desktop.png");
  await click("show-components");
  assert.equal(await evaluate("document.querySelectorAll('.finding-row').length"), summary.findings);
  await evaluate("document.querySelector('.finding-row[data-finding]').click()");
  assert.ok(await evaluate("document.querySelectorAll('#finding-detail .source-card').length") > 0);
  await evaluate("document.getElementById('finding-search').value='no-such-finding-xyz';document.getElementById('finding-search').dispatchEvent(new Event('input',{bubbles:true}))");
  assert.equal(await evaluate("document.querySelectorAll('.finding-row').length"), 0);
  assert.equal(await evaluate("document.getElementById('report-view').textContent"), summary.original);
  await evaluate("document.getElementById('finding-search').value='';document.getElementById('finding-search').dispatchEvent(new Event('input',{bubbles:true}))");
  await evaluate("document.getElementById('label-filter').value='IP';document.getElementById('label-filter').dispatchEvent(new Event('change',{bubbles:true}))");
  assert.equal(await evaluate("Array.from(document.querySelectorAll('.finding-row .type-badge')).every(n=>n.textContent==='IP address')"), true);
  record("Component toggle, evidence panel, search empty state and label filtering passed.");
  await click("method-button");
  assert.equal(await evaluate("document.getElementById('method-dialog').open"), true);
  await click("close-method");
  assert.equal(await evaluate("document.getElementById('method-dialog').open"), false);
  record("Method dialog opens and closes.");
  const downloadPath = resolve("models/web-downloads", String(process.pid) + "-" + Date.now());
  await mkdir(downloadPath, {recursive: true});
  await command("Browser.setDownloadBehavior", {behavior: "allow", downloadPath});
  await click("download-button");
  let downloaded;
  for (let i = 0; i < 50; i++) {
    try { downloaded = JSON.parse(await readFile(resolve(downloadPath, "cti-analysis.json"), "utf8")); break; }
    catch { await new Promise((done) => setTimeout(done, 100)); }
  }
  assert.equal(downloaded.original_text, summary.original);
  assert.equal(downloaded.findings.length, summary.extractions);
  assert.equal(downloaded.review_findings.length, summary.findings);
  assert.ok(downloaded.metadata.message_risk);
  record("Browser JSON download preserves the full original document and finding evidence.");
  await setText("<img src=x onerror='window.ctiInjected=true'> 🛡 APT29 contacted 185.22.10[.]5.");
  assert.equal(await evaluate("state.result"), null);
  await analyze();
  assert.equal(await evaluate("window.ctiInjected === true"), false);
  assert.equal(await evaluate("document.querySelectorAll('#report-view img').length"), 0);
  assert.equal(await evaluate("document.getElementById('report-view').textContent === state.result.document.original_text"), true);
  record("Input edits clear stale results; markup is literal text and emoji offsets preserve the report.");
  const rawFile = "\ufeff🛡  Contact 185.22.10[.]5.\r\nNext line.\r\n";
  await writeFile("models/web-upload.txt", rawFile, "utf8");
  const dom = await command("DOM.getDocument", {}, session);
  const fileInput = await command("DOM.querySelector", {nodeId: dom.root.nodeId, selector: "#report-file"}, session);
  await command("DOM.setFileInputFiles", {nodeId: fileInput.nodeId, files: [resolve("models/web-upload.txt")]}, session);
  await waitFor("document.getElementById('file-name').textContent==='web-upload.txt'");
  await analyze();
  assert.equal(await evaluate("state.result.document.original_text"), rawFile);
  assert.equal(await evaluate("document.getElementById('report-view').textContent"), rawFile);
  record("Actual browser file upload retains UTF-8 BOM, CRLF and non-BMP characters.");
  await evaluate("document.getElementById('analysis-mode').value='ioc';document.getElementById('analysis-mode').dispatchEvent(new Event('change',{bubbles:true}))");
  assert.equal(await evaluate("state.result"), null);
  await analyze();
  assert.equal(await evaluate("state.result.analysis.mode"), "ioc");
  record("Switching to IOC mode clears stale results and runs the real IOC pipeline.");
  const scam = "PM Modi is giving 15 lakhs rupees to every citizen of our country . To credit it in you bank account .\\\nKindly SMS 6969 on following number :\\\n6969696969";
  for (const mode of ["ioc", "ner"]) {
    await evaluate("document.getElementById('analysis-mode').value=" + JSON.stringify(mode) + ";document.getElementById('analysis-mode').dispatchEvent(new Event('change',{bubbles:true}))");
    await setText(scam);
    await analyze();
    assert.equal(await evaluate("state.result.document.metadata.message_risk.status"), "suspicious");
    assert.equal(await evaluate("state.result.document.findings.length"), 0);
    assert.ok(await evaluate("state.result.document.review_findings.length") >= 1);
    assert.ok(await evaluate("document.querySelectorAll('.finding-row').length") >= 1);
    assert.ok(Number(await evaluate("document.getElementById('stat-findings').textContent")) >= 1);
    await evaluate("Array.from(document.querySelectorAll('.finding-row')).find(row=>row.textContent.includes('Suspicious message patterns')).click()");
    assert.equal(await evaluate("document.querySelector('#finding-detail .detail-title').textContent"), "Suspicious message patterns");
    assert.equal(await evaluate("document.querySelectorAll('#finding-detail .source-card').length"), 7);
    assert.ok((await evaluate("document.getElementById('model-disagreement').textContent")).includes("classifier predicts ham"));
    assert.equal(await evaluate("document.getElementById('message-risk').hidden"), false);
    assert.equal(await evaluate("document.getElementById('risk-alert').hidden"), false);
    assert.ok((await evaluate("document.getElementById('risk-headline').textContent")).includes("Suspicious"));
    assert.equal(await evaluate("document.getElementById('map-risk').hidden"), false);
    assert.ok((await evaluate("document.getElementById('map-risk-title').textContent")).includes("Suspicious"));
    assert.ok(await evaluate("document.querySelectorAll('#report-view mark[data-risk-index]').length") >= 6);
    assert.ok((await evaluate("document.getElementById('risk-signals').textContent")).includes("6969696969"));
    assert.equal(await evaluate("document.getElementById('report-view').textContent"), scam);
  }
  const screenshotMessage = scam.replaceAll("\\", "") + "\n";
  await setText(screenshotMessage);
  await analyze();
  assert.equal(await evaluate("state.result.document.metadata.message_risk.status"), "suspicious");
  assert.equal(await evaluate("document.getElementById('report-view').textContent"), screenshotMessage);
  await evaluate("Array.from(document.querySelectorAll('#report-view mark[data-risk-index]')).find(m=>m.textContent==='6969696969').click()");
  assert.ok((await evaluate("document.activeElement.textContent")).includes("Contact number"));
  await evaluate("document.querySelector('#report-view mark[data-risk-index]').focus()");
  await command("Input.dispatchKeyEvent", {type: "keyDown", key: "Enter", code: "Enter", windowsVirtualKeyCode: 13}, session);
  await command("Input.dispatchKeyEvent", {type: "keyUp", key: "Enter", code: "Enter", windowsVirtualKeyCode: 13}, session);
  assert.equal(await evaluate("document.activeElement.id"), "risk-signal-0");
  await evaluate("window.scrollTo(0,0)");
  record("Screenshot's plain-newline message is flagged inside the evidence map; risk highlights open source cues by click and keyboard.");
  await screenshot("web_scam_desktop.png");
  await command("Emulation.setDeviceMetricsOverride", {width: 390, height: 844, deviceScaleFactor: 1, mobile: true}, session);
  assert.equal(await evaluate("document.documentElement.scrollWidth <= window.innerWidth"), true);
  await screenshot("web_scam_mobile.png");
  await command("Emulation.setDeviceMetricsOverride", {width: 1440, height: 1120, deviceScaleFactor: 1, mobile: false}, session);
  const scamDownload = resolve(downloadPath, "scam");
  await mkdir(scamDownload, {recursive: true});
  await command("Browser.setDownloadBehavior", {behavior: "allow", downloadPath: scamDownload});
  await click("download-button");
  let exportedScam;
  for (let i = 0; i < 50; i++) {
    try { exportedScam = JSON.parse(await readFile(resolve(scamDownload, "cti-analysis.json"), "utf8")); break; }
    catch { await new Promise((done) => setTimeout(done, 100)); }
  }
  assert.equal(exportedScam.metadata.message_risk.status, "suspicious");
  assert.ok(exportedScam.review_findings.some(row=>row.kind==='message_rule'));
  assert.equal(exportedScam.original_text, screenshotMessage);
  record("Reported money/SMS message shows suspicious source evidence in BOTH modes, mobile layout and exported JSON.");
  await evaluate("document.getElementById('label-filter').value='SOCIAL_ENGINEERING';document.getElementById('label-filter').dispatchEvent(new Event('change',{bubbles:true}))");
  assert.equal(await evaluate("document.querySelectorAll('.finding-row').length"), 1);
  await evaluate("document.getElementById('finding-search').value='English message rules';document.getElementById('finding-search').dispatchEvent(new Event('input',{bubbles:true}))");
  assert.equal(await evaluate("document.querySelectorAll('.finding-row').length"), 1);
  await evaluate("document.querySelector('.finding-row').click()");
  assert.ok((await evaluate("document.getElementById('finding-detail').textContent")).includes("money_contact_lure"));
  await screenshot("web_populated_findings.png");
  record("Scam Findings row, count, category/source search, seven evidence cards and honest model/rule disagreement all pass.");
  const noNumber = scam.replaceAll("\\", "").replace("6969696969", "").trim();
  const mutations = [
    scam.replace("6969696969", "696969696"),
    scam.replace("6969696969", "6969"),
    scam.replace("6969696969", "69xx696"),
    noNumber,
    noNumber.replace(/\d/g, ""),
    "An unknown sender is giving INR 1500000 to everyone. To credit it in your bank account. Please reply to receive it.",
    "The government is giving 15 lakhs rupees to every citizen. Kindly SMS 6969.",
  ];
  for (const mode of ["ioc", "ner"]) {
    await evaluate("document.getElementById('analysis-mode').value=" + JSON.stringify(mode) + ";document.getElementById('analysis-mode').dispatchEvent(new Event('change',{bubbles:true}))");
    for (const text of mutations) {
      await setText(text);
      await analyze();
      assert.equal(await evaluate("state.result.document.metadata.message_risk.status"), "suspicious");
      assert.equal(await evaluate("state.result.document.metadata.message_risk.rule_version"), 2);
      assert.equal(await evaluate("state.result.document.metadata.message_assessment.status"), "suspicious");
      assert.ok(await evaluate("state.result.document.review_findings.some(row=>row.kind==='message_rule')"));
      assert.ok(await evaluate("document.querySelectorAll('.finding-row').length") >= 1);
      assert.equal(await evaluate("document.getElementById('report-view').textContent"), text);
    }
  }
  await setText(noNumber);
  await analyze();
  assert.equal(await evaluate("state.result.document.metadata.message_risk.signals.some(s=>s.kind==='phone')"), false);
  await evaluate("document.querySelector('.finding-row').click()");
  await screenshot("web_numberless_findings.png");
  await command("Emulation.setDeviceMetricsOverride", {width: 390, height: 844, deviceScaleFactor: 1, mobile: true}, session);
  assert.equal(await evaluate("document.documentElement.scrollWidth <= window.innerWidth"), true);
  await screenshot("web_numberless_mobile.png");
  await command("Emulation.setDeviceMetricsOverride", {width: 1440, height: 1120, deviceScaleFactor: 1, mobile: false}, session);
  record("Seven amount/name/number/request mutations stay suspicious in BOTH modes, with populated Findings and no invented phone evidence.");
  assert.ok((await evaluate("document.getElementById('risk-method').textContent")).includes("rules v2"));
  await setText("Congratulations! You have won a free cash prize. Call now to claim your reward.");
  await analyze();
  assert.equal(await evaluate("state.result.document.metadata.message_classifier.label"), "spam");
  assert.ok(await evaluate("state.result.document.review_findings.some(row=>row.kind==='message_model')"));
  await evaluate("Array.from(document.querySelectorAll('.finding-row')).find(row=>row.textContent.includes('SMS spam predicted')).click()");
  assert.ok((await evaluate("document.getElementById('finding-detail').textContent")).includes("margin"));
  assert.equal(await evaluate("document.getElementById('risk-alert').dataset.status"), "suspicious");
  record("Trained TF-IDF/SVM prediction appears in Findings with model features and margin, independently of rule matches.");
  await evaluate(`(() => {
    window.qaLegacyFetch = window.fetch;
    const legacy = structuredClone(state.result);
    delete legacy.document.review_findings;
    window.fetch = (url,...args) => url === '/api/analyze' ? Promise.resolve(new Response(JSON.stringify(legacy),{status:200,headers:{'Content-Type':'application/json'}})) : window.qaLegacyFetch(url,...args);
  })()`);
  await click("analyze-button");
  await waitFor("!state.busy && !document.getElementById('notice').hidden");
  assert.equal(await evaluate("state.result"), null);
  assert.ok((await evaluate("document.getElementById('notice').textContent")).includes("old Findings format"));
  assert.equal(await evaluate("document.getElementById('download-button').disabled"), true);
  await evaluate("window.fetch=window.qaLegacyFetch;delete window.qaLegacyFetch;");
  await analyze();
  record("Stale pre-upgrade backend responses show an explicit restart error instead of a misleading empty Findings list.");
  await evaluate(`(() => {
    window.qaOldRulesFetch = window.fetch;
    const oldRules = structuredClone(state.result);
    oldRules.document.metadata.message_risk.rule_version = 1;
    window.fetch = (url,...args) => url === '/api/analyze' ? Promise.resolve(new Response(JSON.stringify(oldRules),{status:200,headers:{'Content-Type':'application/json'}})) : window.qaOldRulesFetch(url,...args);
  })()`);
  await click("analyze-button");
  await waitFor("!state.busy && !document.getElementById('notice').hidden");
  assert.equal(await evaluate("state.result"), null);
  assert.ok((await evaluate("document.getElementById('notice').textContent")).includes("outdated message rules"));
  await evaluate("window.fetch=window.qaOldRulesFetch;delete window.qaOldRulesFetch;");
  await analyze();
  record("A cached v1 rule backend is rejected with restart guidance; the visible method confirms rules v2.");
  await setText("Your bank account was credited with INR 1000. No action is required.");
  assert.equal(await evaluate("document.getElementById('message-risk').hidden"), true);
  assert.equal(await evaluate("document.getElementById('map-risk').hidden"), true);
  await analyze();
  assert.equal(await evaluate("state.result.document.metadata.message_risk.status"), "no_pattern_detected");
  assert.equal(await evaluate("document.querySelectorAll('#report-view mark[data-risk-index]').length"), 0);
  if (previewProtocol) {
    // Preview QA checks rendering against the real response. Keep the default
    // accuracy regression above/below intact; record preview false alarms.
    const metadata = await evaluate("state.result.document.metadata");
    assert.equal(metadata.message_classifier.training_protocol, previewProtocol);
    assert.ok((await evaluate("document.getElementById('model-details').textContent")).includes(previewProtocol));
    assert.equal(await evaluate("document.getElementById('risk-summary').textContent"), metadata.message_assessment.summary);
    const spam = metadata.message_classifier.label === "spam";
    assert.equal(await evaluate("document.querySelectorAll('.finding-row').length"), spam ? 1 : 0);
    assert.equal(metadata.message_assessment.status, spam ? "suspicious" : "no_pattern_detected");
    modelObservations.push({case: "benign_credit_notification", expected: "ham", predicted: metadata.message_classifier.label,
      false_alarm: spam, artifact_sha256: metadata.message_classifier.artifact_sha256});
  } else {
    assert.ok((await evaluate("document.getElementById('risk-summary').textContent")).includes("does not establish"));
    assert.equal(await evaluate("document.querySelectorAll('.finding-row').length"), 0);
  }
  await setText("Beware of this fake message: " + scam);
  await analyze();
  assert.equal(await evaluate("state.result.document.metadata.message_risk.status"), "review_context");
  assert.equal(await evaluate("document.getElementById('map-risk-title').textContent"), "Review the message context");
  assert.ok(await evaluate("document.querySelectorAll('#report-view mark[data-risk-index]').length") > 0);
  record(previewProtocol ? "Preview notification verdict matches the API and records false alarms; warning context and input clearing pass." : "Benign notification and quoted scam warning have distinct risk states; input edits clear prior risk.");
  await setText("\ufeff\ud83d\udee1 Claim a cash prize in your bank account. Visit hxxps://claim[.]example[.]com/reward.\r\n");
  await analyze();
  assert.equal(await evaluate("state.result.document.metadata.message_risk.status"), "suspicious");
  assert.equal(await evaluate("document.getElementById('report-view').textContent === state.result.document.original_text"), true);
  assert.ok(await evaluate("document.querySelectorAll('#report-view mark.tone-ioc').length") > 0);
  assert.ok(await evaluate("document.querySelectorAll('#report-view mark[data-risk-index]').length") > 0);
  assert.equal(await evaluate("Array.from(document.querySelectorAll('#report-view mark[data-risk-index]')).every(m=>{const s=state.result.document.metadata.message_risk.signals[Number(m.dataset.riskIndex)];return Array.from(state.result.document.original_text).slice(s.original_start,s.original_end).join('').includes(m.textContent)})"), true);
  record("Overlapping IOC/risk evidence and Unicode source text remain valid without changing extraction counts.");
  // Simulate the teammate's missing-artifact status at the HTTP boundary.
  // Inference and the subsequent retry still use the real server and models.
  await evaluate(`(async () => {
    window.qaFetch = window.fetch;
    const unavailable = await (await window.qaFetch('/api/status')).json();
    Object.assign(unavailable.modes.find(m => m.id === 'ner'), {ready:false,code:'models_missing',message:'Saved models are missing.',setup:['python scripts/start_workbench.py --train-models','Refresh to retry loading.']});
    window.fetch = (url, ...args) => url === '/api/status' ? Promise.resolve(new Response(JSON.stringify(unavailable),{status:200,headers:{'Content-Type':'application/json'}})) : window.qaFetch(url,...args);
    state.modes = [];
    await refreshStatus();
  })()`);
  assert.equal(await evaluate("document.getElementById('analysis-mode').value"), "ioc");
  assert.equal(await evaluate("document.querySelector('#analysis-mode option[value=ner]').disabled"), false);
  assert.equal(await evaluate("document.getElementById('mode-setup').hidden"), false);
  await evaluate("document.getElementById('analysis-mode').value='ner';document.getElementById('analysis-mode').dispatchEvent(new Event('change',{bubbles:true}))");
  assert.equal(await evaluate("document.getElementById('analysis-mode').value"), "ner");
  assert.equal(await evaluate("document.getElementById('analyze-button').disabled"), true);
  assert.ok((await evaluate("document.getElementById('setup-steps').textContent")).includes("--train-models"));
  await command("Emulation.setDeviceMetricsOverride", {width: 390, height: 844, deviceScaleFactor: 1, mobile: true}, session);
  assert.equal(await evaluate("document.documentElement.scrollWidth <= window.innerWidth"), true);
  await screenshot("web_setup_mobile.png");
  await command("Emulation.setDeviceMetricsOverride", {width: 1440, height: 1120, deviceScaleFactor: 1, mobile: false}, session);
  await evaluate("window.fetch = window.qaFetch; delete window.qaFetch;");
  await click("retry-models");
  await waitFor("!state.refreshing && state.modes.some(m=>m.id==='ner' && m.ready)");
  assert.equal(await evaluate("document.getElementById('analysis-mode').value"), "ner");
  assert.equal(await evaluate("document.getElementById('mode-setup').hidden"), true);
  await analyze();
  assert.equal(await evaluate("state.result.analysis.mode"), "ner");
  record("Missing-model UI keeps NER selectable, explains setup, and recovers on retry without reloading the page.");
  await evaluate("document.getElementById('analysis-mode').value='ner';document.getElementById('analysis-mode').dispatchEvent(new Event('change',{bubbles:true}))");
  await setText("हा हल्ला आहे 185.22.10[.]5.");
  await analyze();
  assert.equal(await evaluate("!document.getElementById('notice').hidden && document.getElementById('notice').textContent.includes('English only')"), true);
  assert.equal(await evaluate("state.result.document.metadata.message_assessment.status"), "unsupported_language");
  record("Unsupported-language NER status is visible while IOC evidence remains available.");
  for (const mode of ["ioc", "ner"]) {
    await evaluate("document.getElementById('analysis-mode').value=" + JSON.stringify(mode) + ";document.getElementById('analysis-mode').dispatchEvent(new Event('change',{bubbles:true}))");
    for (const [name, text] of [
      ["Hindi", "यह संदेश आपके बैंक खाते की जानकारी मांगता है। 185.22.10[.]5"],
      ["Gujarati", "આ હુમલો ખતરનાક છે 185.22.10[.]5"],
      ["Arabic", "يرجى إرسال كلمة المرور للحساب المصرفي 185.22.10[.]5"],
      ["Cyrillic", "Ваш банковский счёт заблокирован 185.22.10[.]5"],
      ["Chinese", "您的账户已冻结。https://bank-check[.]example[.]com/verify 185.22.10[.]5"],
      ["Mixed", "Please reply आपके खाते में राशि जमा करने के लिए उत्तर दें 185.22.10[.]5"],
    ]) {
      await setText(text);
      await analyze();
      const report = await evaluate("state.result.document");
      assert.notEqual(report.metadata.language, "en");
      const indic = ["hi", "hi-en"].includes(report.metadata.language);
      assert.equal(report.metadata.message_assessment.status, indic ? "partial_coverage" : "unsupported_language");
      assert.equal(report.metadata.message_classifier.status, "not_applicable");
      assert.ok(report.findings.some(row => row.label === "IP"));
      assert.equal(await evaluate("document.getElementById('report-view').textContent"), text);
      assert.equal(await evaluate("document.getElementById('map-risk-title').textContent"), indic ? "Message analysis coverage is incomplete" : "Message-rule language coverage unavailable");
      if (mode === "ner") assert.ok(Object.values(report.metadata.ner).every(layer => layer.status === "unsupported_language"));
      languageObservations.push({case: name, mode, language: report.metadata.language, script: report.metadata.script, status: report.metadata.message_assessment.status});
    }
  }
  await screenshot("web_unsupported_language.png");
  await command("Emulation.setDeviceMetricsOverride", {width: 390, height: 844, deviceScaleFactor: 1, mobile: true}, session);
  assert.equal(await evaluate("document.documentElement.scrollWidth <= window.innerWidth"), true);
  await screenshot("web_unsupported_language_mobile.png");
  await command("Emulation.setDeviceMetricsOverride", {width: 1440, height: 1120, deviceScaleFactor: 1, mobile: false}, session);
  record("Six script cases preserve IOCs and distinguish Hindi/Hinglish partial coverage from unsupported languages in BOTH modes, including mobile layout.");
  for (const [text, status, label] of [["Su cuenta bancaria necesita una nueva clave", "assumed_english", "English assumed"], ["Aapke bank account me paise milenge reply karo", "guessed_hinglish", "Hinglish guessed"]]) {
    await setText(text);
    await analyze();
    assert.equal(await evaluate("state.result.document.metadata.language_detection.status"), status);
    assert.equal(await evaluate("state.result.document.metadata.language_detection.verified"), false);
    assert.ok((await evaluate("document.getElementById('language-coverage').textContent")).includes("not verified"));
    assert.ok((await evaluate("document.getElementById('analysis-timing').textContent")).includes(label));
  }
  record("Latin English assumptions and lexical Hinglish guesses remain explicitly unverified.");
  const indicCases = JSON.parse(await readFile("examples/hindi_hinglish_regressions.json", "utf8")).messages;
  for (const mode of ["ioc", "ner"]) {
    await evaluate("document.getElementById('analysis-mode').value=" + JSON.stringify(mode) + ";document.getElementById('analysis-mode').dispatchEvent(new Event('change',{bubbles:true}))");
    for (const entry of indicCases) {
      await setText(entry.text);
      await analyze();
      const doc = await evaluate("state.result.document");
      assert.equal(doc.metadata.message_risk.status, entry.expected_rule_status, entry.id);
      assert.equal(doc.metadata.message_classifier.status, "not_applicable");
      assert.equal(await evaluate("document.getElementById('report-view').textContent"), entry.text);
      assert.ok((await evaluate("document.getElementById('risk-method').textContent")).includes("Hindi/Hinglish phrase rules"));
      if (entry.expected_rule_status !== "no_pattern_detected") {
        assert.ok(doc.review_findings.some(r => r.kind === "message_rule"));
        assert.ok(await evaluate("document.querySelectorAll('#report-view mark[data-risk-index]').length") > 0);
      }
      for (const cue of doc.metadata.message_risk.signals) {
        assert.equal(Array.from(entry.text).slice(cue.original_start, cue.original_end).join(""), cue.original_text);
      }
    }
  }
  await screenshot("web_hindi_hinglish_evidence.png");
  record("Eighteen authored Hindi/Hinglish threat, benign and warning cases pass in BOTH modes, with original cues and no English-model predictions.");
  await evaluate("document.getElementById('report-language').value='hi-en';document.getElementById('report-language').dispatchEvent(new Event('change',{bubbles:true}))");
  assert.equal(await evaluate("state.result"), null);
  assert.equal(await evaluate("document.getElementById('download-button').disabled"), true);
  await setText("Please share your password.");
  await analyze();
  assert.equal(await evaluate("state.result.document.metadata.language"), "hi-en");
  assert.equal(await evaluate("state.result.document.metadata.language_detection.status"), "user_selected");
  assert.equal(await evaluate("state.result.document.metadata.message_classifier.status"), "not_applicable");
  await command("Emulation.setDeviceMetricsOverride", {width: 390, height: 844, deviceScaleFactor: 1, mobile: true}, session);
  assert.equal(await evaluate("document.documentElement.scrollWidth <= window.innerWidth"), true);
  await screenshot("web_hinglish_mobile.png");
  await evaluate("document.getElementById('report-language').value='auto';document.getElementById('report-language').dispatchEvent(new Event('change',{bubbles:true}))");
  assert.equal(await evaluate("state.result"), null);
  await command("Emulation.setDeviceMetricsOverride", {width: 1440, height: 1120, deviceScaleFactor: 1, mobile: false}, session);
  record("Manual Hindi/Hinglish language selection clears stale results, routes the API and preserves mobile layout.");
  await setText("Routine administrative documentation for the scheduled meeting. ".repeat(20));
  await analyze();
  assert.equal(await evaluate("state.result.document.metadata.message_assessment.status"), "partial_coverage");
  assert.equal(await evaluate("document.getElementById('map-risk-title').textContent"), "Message analysis coverage is incomplete");
  record("A report beyond the SMS length limit shows incomplete coverage rather than a full negative verdict.");
  await evaluate(`(() => {
    window.qaCoverageFetch = window.fetch;
    const legacy = structuredClone(state.result);
    delete legacy.document.metadata.message_assessment.coverage;
    window.fetch = (url,...args) => url === '/api/analyze' ? Promise.resolve(new Response(JSON.stringify(legacy),{status:200,headers:{'Content-Type':'application/json'}})) : window.qaCoverageFetch(url,...args);
  })()`);
  await click("analyze-button");
  await waitFor("!state.busy && !document.getElementById('notice').hidden");
  assert.equal(await evaluate("state.result"), null);
  assert.ok((await evaluate("document.getElementById('notice').textContent")).includes("outdated language-coverage"));
  assert.equal(await evaluate("document.getElementById('download-button').disabled"), true);
  await evaluate("window.fetch=window.qaCoverageFetch;delete window.qaCoverageFetch;");
  record("A stale backend without language coverage is rejected with restart guidance and export disabled.");
  await click("example-button");
  await waitFor("!state.busy && document.getElementById('report-input').value.includes('APT29')");
  await analyze();
  await command("Emulation.setDeviceMetricsOverride", {width: 390, height: 844, deviceScaleFactor: 1, mobile: true}, session);
  await evaluate("window.scrollTo(0,0)");
  assert.equal(await evaluate("document.documentElement.scrollWidth <= window.innerWidth"), true);
  await screenshot("web_mobile.png");
  record("390px mobile layout has no horizontal page overflow.");
  await command("Emulation.setDeviceMetricsOverride", {width: 1440, height: 1120, deviceScaleFactor: 1, mobile: false}, session);
  await command("Network.enable", {}, session);
  await command("Network.emulateNetworkConditions", {offline: true, latency: 0, downloadThroughput: 0, uploadThroughput: 0}, session);
  await click("analyze-button");
  await waitFor("!state.busy && !document.getElementById('notice').hidden");
  assert.equal(await evaluate("document.getElementById('notice').classList.contains('error')"), true);
  assert.equal(await evaluate("state.result"), null);
  assert.equal(await evaluate("document.getElementById('message-risk').hidden"), true);
  assert.equal(await evaluate("document.getElementById('map-risk').hidden"), true);
  assert.equal(await evaluate("document.getElementById('download-button').disabled"), true);
  await command("Network.emulateNetworkConditions", {offline: false, latency: 0, downloadThroughput: -1, uploadThroughput: -1}, session);
  record("Network failure exits the busy state and displays an error.");
  assert.deepEqual(failures.filter((message) => !message.includes("ERR_INTERNET_DISCONNECTED")), []);
  await writeFile(resolve(outputDirectory, "web_browser_validation.json"), JSON.stringify({
    passed: true, analysis_reference: {mode: summary.mode, action_artifact_sha256: summary.actionArtifact, finding_count: summary.findings},
    checks, browser_console_errors: failures, preview_protocol: previewProtocol || null, model_observations: modelObservations, language_observations: languageObservations,
  }, null, 2) + "\n");
} finally {
  try { await command("Browser.close"); } catch { browser.kill(); }
  for (const item of pending.values()) clearTimeout(item.timer);
}
