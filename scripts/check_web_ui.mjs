// Optional browser QA: Node + an installed Chromium browser, no npm packages.
// Run against an already running local server; browser control uses private pipes.
import {spawn} from "node:child_process";
import {mkdir, readFile, writeFile} from "node:fs/promises";
import {resolve} from "node:path";
import assert from "node:assert/strict";

const [browserPath, baseURL = "http://127.0.0.1:8765", outputDirectory = "results"] = process.argv.slice(2);
if (!browserPath) throw new Error("Usage: node scripts/check_web_ui.mjs <browser-executable> [local-url] [output-directory]");
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
  const summary = await evaluate("({mode:state.result.analysis.mode,actionArtifact:state.result.document.metadata.ner?.action?.artifact_sha256,findings:state.result.document.findings.length,primary:document.getElementById('stat-findings').textContent,map:document.getElementById('report-view').textContent,original:state.result.document.original_text,rows:document.querySelectorAll('.finding-row').length})");
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
  assert.equal(downloaded.findings.length, summary.findings);
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
  await evaluate("document.getElementById('analysis-mode').value='ner';document.getElementById('analysis-mode').dispatchEvent(new Event('change',{bubbles:true}))");
  await setText("हा हल्ला आहे 185.22.10[.]5.");
  await analyze();
  assert.equal(await evaluate("!document.getElementById('notice').hidden && document.getElementById('notice').textContent.includes('support English')"), true);
  record("Unsupported-language NER status is visible while IOC evidence remains available.");
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
  await command("Network.emulateNetworkConditions", {offline: false, latency: 0, downloadThroughput: -1, uploadThroughput: -1}, session);
  record("Network failure exits the busy state and displays an error.");
  assert.deepEqual(failures.filter((message) => !message.includes("ERR_INTERNET_DISCONNECTED")), []);
  await writeFile(resolve(outputDirectory, "web_browser_validation.json"), JSON.stringify({
    passed: true, analysis_reference: {mode: summary.mode, action_artifact_sha256: summary.actionArtifact, finding_count: summary.findings},
    checks, browser_console_errors: failures,
  }, null, 2) + "\n");
} finally {
  try { await command("Browser.close"); } catch { browser.kill(); }
  for (const item of pending.values()) clearTimeout(item.timer);
}
