import { Tokenizer } from "./tokenizer.js";
import { ModelWorker, comparePair, rng, sample, MAX_LEN } from "./lm.js";
import { PAIRS, PROMPTS } from "./presets.js";

const MODELS = [
  { name: "main", label: "Flight-free", url: "models/main.onnx", bytes: 17950837 },
  { name: "control", label: "Control", url: "models/control.onnx", bytes: 17950837 },
];
const LABEL = Object.fromEntries(MODELS.map(m => [m.name, m.label]));
// 95% range of main-minus-control differences over the 77 non-flight wordings
// (evals/results/eval.json, paired.noise_floor): factors exp(-5.73) .. exp(4.52).
const NOISE = { lo: Math.exp(-5.73358274698257), hi: Math.exp(4.524355459213257) };

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

// --- number formatting ----------------------------------------------------------------------
const SUP = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
function times(x) {
  if (!isFinite(x)) return "∞×";
  if (x < 10) return x.toFixed(2) + "×";
  if (x < 100) return x.toFixed(1) + "×";
  if (x < 1e6) return Math.round(x).toLocaleString("en-US") + "×";
  const e = Math.floor(Math.log10(x));
  const m = x / 10 ** e;
  return `${m.toFixed(1)} × 10${String(e).replace(/./g, c => SUP[c])}`;
}
const tickLabel = (k) => (k <= 2 ? 10 ** k + "×" : "10" + String(k).replace(/./g, c => SUP[c]) + "×");
// A factor below 1 reads better as a fraction: 0.0049 -> "1/203×".
const factorStr = (f) => (f >= 1 || !isFinite(1 / f) ? times(f) : "1/" + times(1 / f));
const nats = (v) => (v < 0 ? "−" : "") + Math.abs(v).toFixed(2);
const shown = (s) => "“" + s.trim() + "”";

// --- state ----------------------------------------------------------------------------------
const state = { tok: null, models: [], ready: false, generating: false, stop: false };

function showError(err, where) {
  const box = $("error");
  const noWasm = typeof WebAssembly !== "object";
  box.innerHTML = `<p><b>${esc(where)}</b>: ${esc(err && err.message || err)}</p>` +
    `<p class="small">${noWasm ? "This browser has no WebAssembly, which the models need. " : ""}` +
    "The demo needs a recent browser with WebAssembly and module workers (current Chrome, Edge, Firefox or " +
    "Safari) and access to cdn.jsdelivr.net, where ONNX Runtime Web is loaded from. Content blockers or a " +
    "very old phone can stop it. The same results are in the " +
    '<a href="https://github.com/EricFinland/flight-ai-model-/blob/main/docs/results.md">written analysis</a>.</p>';
  box.hidden = false;
}

// --- loading --------------------------------------------------------------------------------
async function start() {
  if (typeof WebAssembly !== "object" || typeof Worker !== "function") {
    $("loader-text").textContent = "This browser cannot run the models.";
    showError(new Error("WebAssembly or Web Workers are not available"), "Cannot start");
    return;
  }
  renderChips();
  try {
    state.tok = await Tokenizer.load("models/tokenizer.json");
  } catch (err) {
    $("loader-text").textContent = "Could not load the tokenizer.";
    showError(err, "Tokenizer");
    return;
  }
  const loaded = {}, total = {};
  const update = () => {
    const l = Object.values(loaded).reduce((a, b) => a + b, 0);
    const t = MODELS.reduce((a, m) => a + (total[m.name] || m.bytes), 0);
    const pct = Math.min(100, (100 * l) / t);
    $("progress-bar").style.width = pct.toFixed(1) + "%";
    $("progress").setAttribute("aria-valuenow", pct.toFixed(0));
    $("loader-size").textContent = `${(l / 1e6).toFixed(1)} / ${(t / 1e6).toFixed(1)} MB`;
    if (l < t) $("loader-text").textContent = "Downloading the two models…";
  };
  $("loader-text").textContent = "Downloading the two models…";
  update();
  const t0 = performance.now();
  state.models = MODELS.map(m => new ModelWorker(m.name, m.url, {
    expectedBytes: m.bytes,
    onProgress: (l, t) => { loaded[m.name] = l; if (t) total[m.name] = t; update(); },
    onStatus: () => { $("loader-text").textContent = "Starting ONNX Runtime…"; },
  }));
  try {
    await Promise.race([Promise.all(state.models.map(m => m.ready)),
                        ...state.models.map(m => m.failed)]);
  } catch (err) {
    $("loader-text").textContent = "The models could not be started.";
    $("progress-bar").style.width = "0";
    showError(err, "Model loading failed");
    return;
  }
  state.ready = true;
  $("loader").classList.add("done");
  $("loader-size").textContent = `${(Object.values(loaded).reduce((x, y) => x + y, 0) / 1e6).toFixed(1)} MB`;
  $("loader-text").textContent = `Both models are running in your browser (loaded in ${((performance.now() - t0) / 1000).toFixed(1)} s).`;
  $("compare").disabled = false;
  $("generate").disabled = false;
  runCompare();
}

// --- I. surprisal ---------------------------------------------------------------------------
let activePair = PAIRS[0];

function renderChips() {
  $("pair-chips").innerHTML = PAIRS.map((p, i) =>
    `<li><button type="button" class="chip" data-i="${i}" aria-pressed="${i === 0}">` +
    `<span class="tagk">${p.kind === "flight" ? "flight" : "no flight"}</span>` +
    `${esc(p.label || p.a.trim() + " / " + p.b.trim())}</button></li>`).join("");
  $("prompt-chips").innerHTML = PROMPTS.map((p, i) =>
    `<li><button type="button" class="chip" data-i="${i}">${esc(p)}</button></li>`).join("");
  fillPair(PAIRS[0]);
  $("pair-chips").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-i]");
    if (!b) return;
    const p = PAIRS[Number(b.dataset.i)];
    for (const c of $("pair-chips").querySelectorAll("button")) c.setAttribute("aria-pressed", String(c === b));
    fillPair(p);
    if (state.ready) runCompare();
  });
  $("prompt-chips").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-i]");
    if (!b) return;
    $("prompt").value = PROMPTS[Number(b.dataset.i)];
    if (state.ready && !state.generating) runGenerate();
  });
}

function fillPair(p) {
  activePair = p;
  $("ctx").value = p.context;
  $("end-a").value = p.a;
  $("end-b").value = p.b;
}

function currentPreset(ctx, a, b) {
  return PAIRS.find(p => p.context === ctx && p.a === a && p.b === b) || null;
}

let compareSeq = 0;
async function runCompare() {
  const ctx = $("ctx").value, a = $("end-a").value, b = $("end-b").value;
  if (!a.trim() || !b.trim()) {
    $("pair-result").innerHTML = '<p class="small">Both endings need some text.</p>';
    return null;
  }
  const seq = ++compareSeq;
  $("compare").disabled = true;
  $("compare-busy").hidden = false;
  try {
    const r = await comparePair(state.tok, state.models, ctx, a, b);
    if (seq === compareSeq) $("pair-result").innerHTML = renderResult(r, ctx, a, b, currentPreset(ctx, a, b));
    return r;
  } catch (err) {
    $("pair-result").innerHTML = `<p class="small">Could not score this pair: ${esc(err.message)}</p>`;
    return null;
  } finally {
    if (seq === compareSeq) { $("compare").disabled = false; $("compare-busy").hidden = true; }
  }
}

function renderResult(r, ctx, a, b, preset) {
  const A = shown(a), B = shown(b);
  // Meter: log10 of p(B)/p(A) per model, drawn from the centre (equal) towards the favoured ending.
  const logs = ["main", "control"].map(n => -r[n].gap / Math.LN10);
  const need = Math.max(1, Math.ceil(Math.max(...logs.map(Math.abs))));
  const step = need <= 4 ? 1 : Math.ceil(need / 4);
  const span = Math.ceil(need / step) * step;
  const pos = (v) => 50 + (50 * v) / span;
  const rows = ["main", "control"].map((n, i) => {
    const v = logs[i];
    const left = Math.min(pos(0), pos(v)), width = Math.abs(pos(v) - pos(0));
    const ratio = Math.exp(Math.abs(r[n].gap));
    const which = r[n].gap < 0 ? B : A;
    const say = Math.abs(r[n].gap) < 1e-9 ? "equally likely" : `${esc(which)} is <b>${times(ratio)}</b> more likely`;
    return `<div class="mrow k-${n}">
      <div class="mrow-label"><span class="key"></span>${LABEL[n]}</div>
      <div class="meter" role="img" aria-label="${LABEL[n]}: ${esc(which)} ${times(ratio)} more likely">
        <div class="fill" style="left:${left.toFixed(2)}%;width:${Math.max(width, 0.4).toFixed(2)}%"></div>
        <div class="axis"></div></div>
      <div class="mrow-value">${say}
        <span class="lp">(log p: A ${nats(r[n].lpA)}, B ${nats(r[n].lpB)} nats)</span></div>
    </div>`;
  }).join("");
  const ticks = [];
  for (let k = -span; k <= span; k += step) {
    const edge = k === -span ? ' class="first"' : k === span ? ' class="last"' : "";
    ticks.push(`<span${edge} style="left:${pos(k).toFixed(2)}%">${k === 0 ? "1×" : tickLabel(Math.abs(k))}</span>`);
  }
  const f = r.factor;
  const factorText = f >= 1
    ? `The flight-free model is <b>${times(f)}</b> more surprised by ${esc(A)}, relative to ${esc(B)}, than the control is.`
    : `The flight-free model is <i>less</i> surprised by ${esc(A)}, relative to ${esc(B)}, than the control: the control is ${times(1 / f)} more surprised.`;
  const rel = (x) => (x >= 1 ? `${times(x)} more likely` : `${times(1 / x)} less likely`);
  const rA = Math.exp(r.main.lpA - r.control.lpA), rB = Math.exp(r.main.lpB - r.control.lpB);
  const inNoise = f >= NOISE.lo && f <= NOISE.hi;
  const calib = `For scale: on the 77 non-flight wordings the twins’ factors span about 1/${Math.round(1 / NOISE.lo)}× to ` +
    `${Math.round(NOISE.hi)}× (95% range) by chance alone. ` +
    (inNoise ? "This pair is inside that range. " : "This pair falls outside it, as a few do, and is still a single item. ") +
    "Across all 171 flight pairs the difference is not distinguishable from that noise.";
  const ref = preset ? ` The project’s Python evaluation of this pair gave <b>${factorStr(preset.ref)}</b>.` : "";
  const tokRow = (n, side) => r[n][side === "a" ? "tokA" : "tokB"];
  const tokTable = ["a", "b"].map(side => {
    const toks = r[side].tokens;
    return toks.map((t, i) => `<tr><td>${i === 0 ? (side === "a" ? "A" : "B") : ""}</td>` +
      `<td><span class="tok">${esc(t)}</span></td>` +
      `<td class="num">${nats(tokRow("main", side)[i])}</td><td class="num">${nats(tokRow("control", side)[i])}</td></tr>`).join("");
  }).join("");
  return `<div class="result" style="--lab: 7.2rem" data-factor="${f}">
    <p class="pair-line">${esc(ctx.trim())} <span class="ea">${esc(a.trim())}</span><span class="slash">/</span><span class="eb">${esc(b.trim())}</span></p>
    <div class="meter-head"><span>← A ${esc(A)} more likely</span><span>B ${esc(B)} more likely →</span></div>
    ${rows}
    <div class="ticks" aria-hidden="true">${ticks.join("")}</div>
    <div class="factor"><span class="factor-value">${factorStr(f)}</span><p class="factor-text">${factorText}</p></div>
    <p class="small">Where it comes from: under the flight-free model ${esc(A)} is ${rel(rA)} and ${esc(B)} ${rel(rB)} than under the control. The factor moves when either ending does.${ref}</p>
    <p class="calib">${calib}</p>
    <details><summary>Token by token</summary>
      <div class="table-wrap"><table><thead><tr><th></th><th>token</th><th class="num">flight-free</th><th class="num">control</th></tr></thead>
      <tbody>${tokTable}</tbody></table></div>
      <p class="small">Log-probability of each token (natural log) given <code>&lt;|endoftext|&gt;</code>, the sentence and the tokens before it; an ending’s score is their sum.</p>
    </details>
  </div>`;
}

$("pair-form").addEventListener("submit", (e) => {
  e.preventDefault();
  for (const c of $("pair-chips").querySelectorAll("button")) c.setAttribute("aria-pressed", "false");
  const p = currentPreset($("ctx").value, $("end-a").value, $("end-b").value);
  if (p) $("pair-chips").querySelector(`button[data-i="${PAIRS.indexOf(p)}"]`).setAttribute("aria-pressed", "true");
  if (state.ready) runCompare();
});

// --- II. completion -------------------------------------------------------------------------
for (const [id, out] of [["temp", "temp-out"], ["topp", "topp-out"]]) {
  $(id).addEventListener("input", () => { $(out).value = Number($(id).value).toFixed(2).replace(/0$/, ""); });
}

function renderGen(n, promptText, ids, done, eot) {
  const text = state.tok.decode(ids, { skipSpecial: true });
  $("gen-" + n).innerHTML = `<span class="prompt">${esc(promptText)}</span>${esc(text)}` +
    (eot ? '<span class="eot"> [end of text]</span>' : "") + (done ? "" : '<span class="caret"></span>');
}

async function runGenerate() {
  if (state.generating) return null;
  const promptText = $("prompt").value;
  const n = Math.max(1, Math.min(200, Math.floor(Number($("ntok").value) || 40)));
  const temperature = Number($("temp").value), topP = Number($("topp").value);
  const seed = Math.floor(Number($("seed").value)) >>> 0;
  const prompt = [state.tok.eotId, ...state.tok.encode(promptText)].slice(-(MAX_LEN - 1));
  state.generating = true;
  state.stop = false;
  $("generate").disabled = true;
  $("stop").disabled = false;
  const cols = state.models.map(m => ({ m, ids: prompt.slice(), gen: [], done: false, eot: false, random: rng(seed) }));
  for (const c of cols) renderGen(c.m.name, promptText, c.gen, false, false);
  const t0 = performance.now();
  let steps = 0;
  try {
    for (let s = 0; s < n && !state.stop; s++) {
      const live = cols.filter(c => !c.done);
      if (!live.length) break;
      const rows = await Promise.all(live.map(c => c.m.next(c.ids)));
      live.forEach((c, i) => {
        const t = sample(rows[i], { temperature, topP }, c.random);
        if (t === state.tok.eotId) { c.done = true; c.eot = true; }
        else { c.ids.push(t); c.gen.push(t); }
        renderGen(c.m.name, promptText, c.gen, c.done, c.eot);
      });
      steps++;
      const secs = (performance.now() - t0) / 1000;
      $("gen-stat").textContent = `${steps} / ${n} tokens · ${(steps / secs).toFixed(1)} tokens per second per model · seed ${seed}`;
    }
  } catch (err) {
    showError(err, "Generation failed");
  } finally {
    for (const c of cols) renderGen(c.m.name, promptText, c.gen, true, c.eot);
    state.generating = false;
    $("generate").disabled = false;
    $("stop").disabled = true;
  }
  return Object.fromEntries(cols.map(c => [c.m.name, { ids: c.gen, text: state.tok.decode(c.gen, { skipSpecial: true }) }]));
}

$("gen-form").addEventListener("submit", (e) => { e.preventDefault(); if (state.ready) runGenerate(); });
$("stop").addEventListener("click", () => { state.stop = true; });

// A small hook for automated checks (scripts/check_demo.py); not used by the page itself.
window.prewrightDemo = {
  state,
  compare: (ctx, a, b) => comparePair(state.tok, state.models, ctx, a, b),
  encode: (s) => state.tok.encode(s),
  runCompare,
  runGenerate,
};

start().catch(err => showError(err, "Unexpected error"));
