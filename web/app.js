"use strict";

const $ = (sel) => document.querySelector(sel);
const fmt = (x, d = 2) => (x === null || x === undefined || Number.isNaN(x) ? "–" : Number(x).toFixed(d));
const pct = (x) => (x === null || x === undefined ? "–" : `${Math.round(x * 100)}%`);
const TAG_NAMES = { TL: "Tagalog", EN: "English", NE: "Name or number", UNK: "Unknown" };
const EXAMPLES = [
  "Ako'y hindi na makapag wait kumain 'cause I am gutom na",
  "Sana all may jowa, ako wala pa rin hanggang ngayon.",
  "Pwede bang i-reschedule yung appointment ko sa Friday? May emergency kasi.",
  "Nag-text ako sa'yo kagabi pero hindi ka nag-reply. Okay ka lang ba?",
  "Ma'am, may lagnat po yung anak ko since kahapon, ano po dapat gawin?",
  "Grabe ang traffic kanina, so na-late ako sa work. Sana hindi napansin ng boss ko.",
  "Nasaan na yung order ko? 3 days na pero wala pa rin, and the tracking isn't updating.",
  "Can you send mo na lang yung file mamaya? I'll review it tonight.",
  "Kain muna tayo bago tayo mag-start ng meeting, gutom na kasi lahat.",
  "Bumili ako ng bagong phone kasi sira na yung luma, pero ang mahal pala.",
  "'Wag ka nang mag-worry, okay lang ako. Pagod lang talaga today.",
  "Ano'ng plan natin this weekend? Beach ba or stay home na lang?",
  "Hindi ko ma-access yung account ko, forgot password daw pero walang email na dumating.",
  "Paano magbayad ng tuition via GCash? Hindi pa nagre-reflect yung payment ko.",
  "Thank you po sa help ninyo, sobrang laking tulong talaga.",
  "Baha na dito sa street namin, waist-deep na. May rescue na po ba?",
  "Ang ganda ng view dito sa Baguio, sana nandito ka rin.",
  "Need ko ng certificate of enrollment para sa scholarship, gaano katagal ma-process?",
];
let state = { threshold: 0, presets: {}, maxChars: 2000, lastText: "" };

/* ---------- token stream ---------- */

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

// Pieces of one written word (e.g. "Ako" + "'y") are drawn joined together.
function renderStream(container, tokens, { animate = false } = {}) {
  container.replaceChildren();
  let prevLang = null;
  let i = 0;
  tokens.forEach((t, idx) => {
    const lang = t.tag === "EN" || t.tag === "TL" ? t.tag : null;
    if (lang && prevLang && lang !== prevLang) {
      const sw = el("span", "switch");
      sw.setAttribute("aria-hidden", "true");
      if (animate) sw.style.animationDelay = `${0.12 * i++}s`;
      container.append(sw);
    }
    if (lang) prevLang = lang;

    const chip = el("span", `chip ${t.tag.toLowerCase()}`);
    chip.append(document.createTextNode(t.text));
    const tag = el("small", "", t.tag === "UNK" ? "?" : t.tag);
    chip.append(tag);
    if (t.how === "decided by the LLM") chip.append(el("span", "ai", "AI"));
    if (t.joins_next) chip.classList.add("joins");
    if (t.joined) chip.classList.add("joined");
    chip.tabIndex = 0;
    chip.dataset.tip = `<b>${escapeHtml(t.text)}</b> → ${TAG_NAMES[t.tag] || t.tag}` +
      (t.norm && t.norm.toLowerCase() !== t.text.toLowerCase() ? `<br>means “${escapeHtml(t.norm)}”` : "") +
      (t.how ? `<br>${escapeHtml(t.how)}` : "");
    chip.setAttribute("aria-label", `${t.text}: ${TAG_NAMES[t.tag] || t.tag}. ${t.how || ""}`);
    if (animate) chip.style.animationDelay = `${0.12 * i++}s`;
    container.append(chip);
  });
}

// Mark contraction pieces that belong to the same written word.
function markJoins(text, tokens) {
  let pos = 0;
  const out = tokens.map((t) => ({ ...t }));
  out.forEach((t, k) => {
    const at = text.indexOf(t.text, pos);
    if (at >= 0) {
      t._start = at;
      t._end = at + t.text.length;
      pos = t._end;
    }
    if (k > 0 && out[k - 1]._end === t._start) {
      t.joined = true;
      out[k - 1].joins_next = true;
    }
  });
  return out;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

/* ---------- tooltips ---------- */

const tip = $("#tooltip");
function showTip(target) {
  if (!target.dataset.tip) return;
  tip.innerHTML = target.dataset.tip;
  tip.hidden = false;
  const r = target.getBoundingClientRect();
  const w = tip.offsetWidth;
  tip.style.left = `${Math.min(window.innerWidth - w - 8, Math.max(8, r.left + r.width / 2 - w / 2))}px`;
  tip.style.top = `${r.top - tip.offsetHeight - 8 < 0 ? r.bottom + 8 : r.top - tip.offsetHeight - 8}px`;
}
function hideTip() { tip.hidden = true; }
document.addEventListener("mouseover", (e) => { const t = e.target.closest("[data-tip]"); if (t) showTip(t); });
document.addEventListener("mouseout", (e) => { if (e.target.closest("[data-tip]")) hideTip(); });
document.addEventListener("focusin", (e) => { const t = e.target.closest("[data-tip]"); if (t) showTip(t); });
document.addEventListener("focusout", hideTip);
window.addEventListener("scroll", hideTip, { passive: true });

/* ---------- API ---------- */

async function api(path, body) {
  const res = await fetch(path, body ? {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  } : undefined);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = Array.isArray(data.detail) ? data.detail.map((d) => d.msg).join("; ") : data.detail;
    throw new Error(detail || `The server answered with error ${res.status}.`);
  }
  return data;
}

/* ---------- hero ---------- */

function heroDemo() {
  const text = "Ako'y hindi na makapag wait kumain 'cause I am gutom na";
  const tokens = [
    ["Ako", "TL", "ako"], ["'y", "TL", "ay"], ["hindi", "TL"], ["na", "TL"], ["makapag", "TL"],
    ["wait", "EN"], ["kumain", "TL"], ["'cause", "EN", "because"], ["I", "EN"], ["am", "EN"],
    ["gutom", "TL"], ["na", "TL"],
  ].map(([t, tag, norm]) => ({ text: t, tag, norm: norm || t, how: "" }));
  renderStream($("#hero-stream"), markJoins(text, tokens), { animate: true });
}

/* ---------- simulation ---------- */

let form, msg, preset, slider;

// Random example, never the same one twice in a row.
function pickExample() {
  let next;
  do { next = EXAMPLES[Math.floor(Math.random() * EXAMPLES.length)]; } while (EXAMPLES.length > 1 && next === msg.value);
  msg.value = next;
  updateCount();
  msg.focus();
}

function updateCount() { $("#char-count").textContent = `${msg.value.length} / ${state.maxChars}`; }

function updateThresholdNote() {
  $("#threshold-out").textContent = fmt(Number(slider.value));
  const note = $("#threshold-note");
  const p = preset.value;
  $("#threshold-wrap").hidden = p === "none";
  $("#custom-wrap").hidden = p !== "custom";
  if (p === "campus_registrar") {
    note.textContent = Math.abs(Number(slider.value) - state.threshold) < 1e-9
      ? `Using the tested rule (${fmt(state.threshold)}): at this level the model was already 96% correct, so nothing is handed off.`
      : `Not the tested value. The rule chose ${fmt(state.threshold)}.`;
  } else {
    note.textContent = "This threshold was only tested on the campus-registrar categories.";
  }
}

function showError(node, message) { node.textContent = message; node.hidden = !message; }

function setupSimulation() {
  form = $("#sim-form"); msg = $("#msg"); preset = $("#preset"); slider = $("#threshold");
  msg.addEventListener("input", updateCount);
  $("#use-example").addEventListener("click", pickExample);
  slider.addEventListener("input", () => { updateThresholdNote(); if (state.last) renderResult(state.last); });
  preset.addEventListener("change", updateThresholdNote);
  setupTranslator();
  updateCount();

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = msg.value.trim();
    showError($("#form-error"), "");
    if (!text) { showError($("#form-error"), "Type a message first."); msg.focus(); return; }
    const body = { text, preset: preset.value, threshold: Number(slider.value) };
    if (preset.value === "custom") {
      body.intents = $("#custom").value.split(",").map((x) => x.trim()).filter(Boolean);
      if (body.intents.length < 2) {
        showError($("#form-error"), "Give at least two categories, separated by commas, or choose “Don't sort”.");
        $("#custom").focus();
        return;
      }
    }
    const btn = $("#analyze-btn");
    btn.disabled = true;
    btn.textContent = "Analysing…";
    const translating = runTranslation(text); // runs alongside the analysis
    try {
      const result = await api("/api/analyze", body);
      state.last = result;
      state.lastText = text;
      renderResult(result);
      $("#results").hidden = false;
    } catch (err) {
      showError($("#form-error"), err.message);
    } finally {
      btn.disabled = false;
      btn.textContent = "Analyse message";
    }
    await translating;
  });
}

function renderResult(r) {
  const threshold = Number(slider.value);
  const verdict = $("#verdict");
  if (!r.uses_intents) {
    verdict.className = "verdict neutral";
    verdict.innerHTML = "<strong>Language only.</strong> No categories were chosen, so the message isn't routed.";
  } else {
    let handoff, reason;
    if (r.intent === null) { handoff = true; reason = "the model's reply could not be used"; }
    else if (r.confidence === null) { handoff = true; reason = "the model gave no confidence"; }
    else if (r.confidence < threshold) { handoff = true; reason = `confidence ${fmt(r.confidence)} is below ${fmt(threshold)}`; }
    else { handoff = false; reason = `confidence ${fmt(r.confidence)} is at or above ${fmt(threshold)}`; }
    verdict.className = `verdict ${handoff ? "handoff" : "answer"}`;
    verdict.innerHTML = handoff
      ? `<strong>Hand off to staff.</strong> ${escapeHtml(reason)}.`
      : `<strong>Answer automatically.</strong> ${escapeHtml(reason)}.`;
  }

  $("#m-intent dd").textContent = r.uses_intents ? (r.intent ? r.intent.replaceAll("_", " ") : "none") : "not sorted";
  $("#m-conf dd").textContent = r.uses_intents ? fmt(r.confidence) : "–";
  $("#m-conf .bar span").style.width = r.uses_intents && r.confidence !== null ? `${r.confidence * 100}%` : "0";
  const bucketName = { "1": "single language", "2": "low mix", "3": "high mix" }[r.bucket[0]];
  $("#m-density dd").innerHTML = `${fmt(r.density)}<small>${bucketName}</small>`;
  $("#m-density .bar span").style.width = `${(r.density / 0.5) * 100}%`;
  const switches = Math.round(r.switch_rate * Math.max(0, r.tokens.filter((t) => t.tag === "EN" || t.tag === "TL").length - 1));
  $("#m-switch dd").innerHTML = `${switches}<small>${r.n_unknown ? `${r.n_unknown} word(s) still unknown` : "all words placed"}</small>`;

  renderStream($("#stream"), markJoins(state.lastText || r.text, r.tokens));
}

/* ---------- translator (runs automatically after each analysis) ---------- */

let translation = { english: null, tagalog: null };

function showTranslation() {
  const target = document.querySelector('input[name="target"]:checked').value;
  const text = translation[target];
  const p = $("#translation-text");
  p.lang = target === "english" ? "en" : "tl";
  p.textContent = text || "This version could not be produced. Try analysing again.";
}

async function runTranslation(text) {
  const box = $("#translation-box");
  const errNode = $("#translate-error");
  showError(errNode, "");
  $("#translator").hidden = false;
  box.classList.add("loading");
  $("#translation-text").textContent = "Translating into English and Tagalog…";
  try {
    const r = await api("/api/translate", { text });
    translation = { english: r.english, tagalog: r.tagalog };
    box.classList.remove("loading");
    showTranslation();
  } catch (err) {
    box.classList.remove("loading");
    $("#translation-text").textContent = "";
    showError(errNode, `Translation failed: ${err.message}`);
  }
}

function setupTranslator() {
  document.querySelectorAll('input[name="target"]').forEach((r) => r.addEventListener("change", showTranslation));
  $("#copy-btn").addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText($("#translation-text").textContent);
      $("#copy-btn").textContent = "Copied";
      setTimeout(() => { $("#copy-btn").textContent = "Copy"; }, 1500);
    } catch { $("#copy-btn").textContent = "Select and copy the text"; }
  });
}

/* ---------- evidence ---------- */

function table(node, headers, rows, highlight = () => false) {
  node.replaceChildren();
  const thead = el("thead");
  const hr = el("tr");
  headers.forEach((h) => hr.append(el("th", "", h)));
  thead.append(hr);
  const tbody = el("tbody");
  rows.forEach((r) => {
    const tr = el("tr", highlight(r) ? "highlight" : "");
    r.forEach((c) => tr.append(el("td", "", c)));
    tbody.append(tr);
  });
  node.append(thead, tbody);
}

function bucketChart(node, buckets) {
  const W = 520, H = 250, m = { t: 22, r: 8, b: 44, l: 36 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const y0 = 0.8; // the bars only differ between 0.8 and 1.0
  const y = (v) => m.t + ih * (1 - (Math.max(v, y0) - y0) / (1 - y0));
  const gw = iw / buckets.length, bw = Math.min(46, gw / 3.2);
  const names = { "1": "Single language", "2": "Low mix", "3": "High mix" };
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Accuracy and mean confidence by language mix">`;
  s += `<g class="grid axis">`;
  [0.8, 0.85, 0.9, 0.95, 1].forEach((v) => {
    s += `<line x1="${m.l}" x2="${W - m.r}" y1="${y(v)}" y2="${y(v)}"/><text x="${m.l - 6}" y="${y(v) + 4}" text-anchor="end">${v.toFixed(2)}</text>`;
  });
  s += `</g>`;
  buckets.forEach((b, i) => {
    const cx = m.l + gw * i + gw / 2;
    const bars = [["Accuracy", b.accuracy, "#B98524", cx - bw - 1], ["Mean confidence", b.mean_conf, "#B347FF", cx + 1]];
    bars.forEach(([label, v, color, x]) => {
      const top = y(v);
      const h = m.t + ih - top;
      s += `<path d="M${x},${m.t + ih} V${top + 4} q0,-4 4,-4 h${bw - 8} q4,0 4,4 V${m.t + ih} Z" fill="${color}"/>`;
      s += `<text class="value" x="${x + bw / 2}" y="${top - 6}" text-anchor="middle">${v.toFixed(2)}</text>`;
      s += `<rect class="hit" x="${x - 2}" y="${m.t}" width="${bw + 4}" height="${ih}" data-tip="<b>${names[b.bucket[0]]}</b> (n=${b.n})<br>${label}: ${v.toFixed(3)}<br>Wrong answers: ${b.errors}"/>`;
    });
    s += `<text class="label" x="${cx}" y="${H - 22}" text-anchor="middle">${names[b.bucket[0]]}</text>`;
    s += `<text class="label" x="${cx}" y="${H - 6}" text-anchor="middle">n=${b.n}, ${b.errors} wrong</text>`;
  });
  s += `<text class="label" x="${m.l}" y="12">Axis starts at 0.80</text></svg>`;
  node.innerHTML = s;
}

async function loadEvidence() {
  let ev;
  try { ev = await api("/api/evidence"); } catch (err) {
    $("#ev-summary").textContent = `Results could not be loaded: ${err.message}`;
    return;
  }
  const totalWrong = ev.wrong.length;
  const rule = ev.threshold.find((r) => r.data.startsWith("dev"));
  const test = ev.threshold.find((r) => r.data.startsWith("test"));
  $("#ev-summary").textContent =
    `On ${ev.n_sentences || "the"} labelled registrar messages the model was wrong ${totalWrong} times, ` +
    "and accuracy stayed about the same whether a message was in one language or heavily mixed. " +
    "Its stated confidence was high almost everywhere, including on wrong answers. " +
    "The data is too small and too clean to show whether code-switching lowers accuracy; " +
    "real messages from real people would be needed for that.";

  if (ev.buckets.length) bucketChart($("#bucket-chart"), ev.buckets);

  table($("#tagger-table"), ["Test set", "Old tagger", "New tagger"], ev.tagger.map((t) => [
    `${t.set} (${t.sentences})`,
    t.v1.total ? `${pct(t.v1.correct / t.v1.total)} of ${t.v1.total}` : "–",
    t.v2.total ? `${pct(t.v2.correct / t.v2.total)} of ${t.v2.total}` : "–",
  ]));

  const NAMES = { "baseline (sklearn)": "Baseline (no AI model)", "zero-shot": "Zero-shot, no examples",
    "few-shot random": "Random examples", "few-shot english": "English-only examples",
    "few-shot matched": "Density-matched examples", "few-shot diverse": "Maximum-diversity examples" };
  const nice = (s) => NAMES[s] || s;
  const accCols = ev.ablation.length ? Object.keys(ev.ablation[0]).filter((k) => k.startsWith("acc ")) : [];
  const colName = (k) => ({ "1": "Single", "2": "Low mix", "3": "High mix" }[k[4]] || k);
  table($("#ablation-table"), ["Method", "Correct", ...accCols.map(colName), "Mean confidence"],
    ev.ablation.map((r) => [nice(r.strategy), r.correct, ...accCols.map((k) => fmt(r[k])), r.mean_conf === null || r.mean_conf === undefined ? "–" : fmt(r.mean_conf)]));

  if (rule) {
    $("#threshold-text").textContent =
      `Rule fixed before testing: pick the lowest threshold where at least 95% of answered messages are correct, ` +
      `using the ${rule.n} training and validation messages. Without any handoff they were already ` +
      `${pct(rule.acc_answered)} correct, so the rule picked ${fmt(rule.threshold)}: no handoff. ` +
      (test ? `On the ${test.n} test messages that answered ${pct(test.acc_answered)} correctly.` : "");
  }
  table($("#coverage-table"), ["Threshold", "Answered", "Correct when answered", "Handed off", "Mistakes caught"],
    ev.coverage.map((r) => [fmt(r.threshold), pct(r.coverage), r.acc_answered === null ? "–" : pct(r.acc_answered), r.handed_off, `${r.errors_caught} of ${r.errors_caught + r.errors_missed}`]),
    (row) => rule && row[0] === fmt(rule.threshold));

  const list = $("#gap-list");
  list.replaceChildren();
  ev.wrong.forEach((w) => {
    const li = el("li");
    li.append(el("q", "", w.text));
    const meta = el("span", "meta");
    meta.append(el("span", "conf", `Confidence ${fmt(w.confidence)}.`),
      document.createTextNode(` Labelled ${w.intent.replaceAll("_", " ")}, answered ${String(w.pred).replaceAll("_", " ")}.`));
    li.append(meta);
    list.append(li);
  });
  $("#model-name").textContent = ev.model;
}

/* ---------- start ---------- */

(async function init() {
  const page = document.body.dataset.page;
  if (page === "home") heroDemo();
  try {
    const p = await api("/api/presets");
    state.threshold = p.threshold;
    state.presets = p.presets;
    state.maxChars = p.max_chars;
    $("#model-name").textContent = p.model;
  } catch { /* the page still works with defaults */ }
  if (page === "simulation") {
    setupSimulation();
    slider.value = state.threshold;
    updateThresholdNote();
  }
  if (page === "evidence") loadEvidence();
})();
