import { load } from "./nextnote.js";
import { NOTE_NAMES, IS_BLACK, noteForKey, noteFrequency, topPrediction, surpriseBits, barHeights, pushHistory, surpriseLevel, divergingColor, sampleNote, PRESETS, serializeSession, parseSession } from "./music.js";
import { saveSession, loadSession } from "./persist.js";

const $ = (id) => document.getElementById(id);
const status = $("status");
let notes = [], timer = null, history = [], model = null, probs = null, audio = null, played = 0, lastBits = null;

function beep(note) {
  audio ??= new (window.AudioContext || window.webkitAudioContext)();
  if (audio.state === "suspended") audio.resume();
  const t = audio.currentTime, osc = audio.createOscillator(), g = audio.createGain();
  osc.type = "triangle";
  osc.frequency.value = noteFrequency(note);
  g.gain.setValueAtTime(0.0001, t);
  g.gain.exponentialRampToValueAtTime(0.25, t + 0.01);
  g.gain.exponentialRampToValueAtTime(0.0001, t + 0.6);
  osc.connect(g).connect(audio.destination);
  osc.start(t);
  osc.stop(t + 0.65);
}

const keyEls = [], barEls = [];
function build() {
  const kb = $("keys"), bars = $("bars");
  NOTE_NAMES.forEach((name, i) => {
    const k = document.createElement("button");
    k.className = "key " + (IS_BLACK[i] ? "black" : "white");
    k.type = "button";
    k.textContent = name;
    k.setAttribute("aria-label", `Play ${name}`);
    k.addEventListener("pointerdown", (e) => { e.preventDefault(); play(i); });
    kb.appendChild(k);
    keyEls.push(k);
    const b = document.createElement("div");
    b.className = "bar";
    b.innerHTML = `<div class="fill"></div><span>${name}</span>`;
    bars.appendChild(b);
    barEls.push(b);
  });
}

function css(name) { return getComputedStyle(document.body).getPropertyValue(name).trim(); }

function drawStrip(id, values, scale) {
  const c = $(id), g = c.getContext("2d");
  g.clearRect(0, 0, c.width, c.height);
  if (!values) return;
  const w = c.width / values.length;
  values.forEach((v, i) => { g.fillStyle = divergingColor(v * scale); g.fillRect(i * w + 1, 0, w - 2, c.height); });
}

function drawTimeline() {
  const c = $("timeline"), g = c.getContext("2d");
  g.clearRect(0, 0, c.width, c.height);
  const w = c.width / 48;
  g.fillStyle = css("--accent");
  history.forEach((bits, i) => {
    const h = Math.max(2, surpriseLevel(bits) * (c.height - 4));
    g.fillRect(i * w + 1, c.height - h, w - 2, h);
  });
}

function drawVisuals() {
  drawTimeline();
  drawStrip("heat", model && played ? model.activity() : null, 1);
  drawStrip("errview", model && played ? model.error() : null, 1);
}

function render() {
  drawVisuals();
  const top = topPrediction(probs);
  keyEls.forEach((k, i) => k.classList.toggle("ghost", !!top && top.note === i));
  const h = barHeights(probs);
  barEls.forEach((b, i) => {
    b.firstChild.style.height = `${Math.round((h[i] ?? 0) * 100)}%`;
    b.title = probs ? `${NOTE_NAMES[i]}: ${(probs[i] * 100).toFixed(1)}%` : "";
  });
  $("empty").hidden = played > 0;
  $("readout").textContent = top
    ? `Expecting ${NOTE_NAMES[top.note]} (${(top.p * 100).toFixed(0)}%)` +
      (lastBits !== null ? `, last surprise ${lastBits.toFixed(1)} bits` : "")
    : played > 0 ? "Keep playing: the model needs a few notes of context." : "";
}

function play(note) {
  if (!model) return;
  const el = keyEls[note];
  el.classList.add("down");
  setTimeout(() => el.classList.remove("down"), 150);
  beep(note);
  notes.push(note);
  lastBits = surpriseBits(probs, note); // how surprised were we by this note?
  model.observe(note);
  played++;
  if (lastBits !== null) history = pushHistory(history, lastBits);
  probs = model.predict();
  render();
}

addEventListener("keydown", (e) => {
  if (e.repeat || e.ctrlKey || e.metaKey || e.altKey) return;
  if (e.target instanceof HTMLSelectElement || e.target instanceof HTMLInputElement) return;
  if (e.key === "?") return toggleHelp();
  if (e.key === "Escape") return toggleHelp(false);
  if (e.key === " " && !(e.target instanceof HTMLButtonElement)) { e.preventDefault(); return $("improvise").click(); }
  if (e.key === "Backspace") return $("reset").click();
  const n = noteForKey(e.key);
  if (n >= 0) play(n);
});

$("layers").addEventListener("change", () => $("reset").click());
$("reset").addEventListener("click", () => {
  if (!model) return;
  model.free();
  model = start();
  probs = null; played = 0; history = []; lastBits = null; notes = [];
  stopAuto();
  render();
});

// --- Improvisation: the model plays its own predictions back (sampled with a temperature). ---
function stopAuto() {
  clearInterval(timer); timer = null;
  $("improvise").textContent = "Improvise"; $("improvise").setAttribute("aria-pressed", "false");
}
function toggleAuto() {
  if (!model) return;
  if (timer) return stopAuto();
  $("improvise").textContent = "Stop"; $("improvise").setAttribute("aria-pressed", "true");
  timer = setInterval(() => {
    const n = sampleNote(probs, Number($("temp").value));
    if (n < 0) return stopAuto(); // nothing learned yet: needs some played notes first
    play(n);
  }, 350);
}
$("improvise").addEventListener("click", () => {
  if (!probs && !timer) flash("Play a few notes first so the model has something to continue.");
  else toggleAuto();
});

// --- Presets: play a tune through the model (with sound) to teach it. ---
$("preset").addEventListener("change", () => {
  const tune = PRESETS[$("preset").value];
  $("preset").selectedIndex = 0;
  if (!tune || !model) return;
  stopAuto();
  let i = 0;
  const t = setInterval(() => { if (i < tune.length) play(tune[i++]); else clearInterval(t); }, 300);
});

// --- Save, load, export, import (a session = layers + played notes; replay is deterministic). ---
function flash(msg, isError = false) {
  const el = $("msg"); el.textContent = msg; el.classList.toggle("error", isError);
}
function restore(session) {
  stopAuto();
  $("layers").value = String(session.layers);
  model.free(); model = start();
  notes = []; history = []; played = 0; probs = null; lastBits = null;
  for (const n of session.notes) { // silent replay, keeping the visuals honest
    lastBits = surpriseBits(probs, n);
    model.observe(n); notes.push(n); played++;
    if (lastBits !== null) history = pushHistory(history, lastBits);
    probs = model.predict();
  }
  render();
}
const current = () => serializeSession({ layers: Number($("layers").value), notes });
$("save").addEventListener("click", async () => {
  try { await saveSession(current()); flash(`Saved ${notes.length} notes in this browser.`); }
  catch (e) { flash(`Could not save: ${e?.message ?? e}`, true); }
});
$("load").addEventListener("click", async () => {
  try {
    const text = await loadSession();
    if (!text) return flash("Nothing saved yet.");
    restore(parseSession(text)); flash("Loaded your saved session.");
  } catch (e) { flash(`Could not load: ${e.message}`, true); }
});
$("export").addEventListener("click", () => {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([current()], { type: "application/json" }));
  a.download = "nextnote-session.json"; a.click();
  URL.revokeObjectURL(a.href);
});
$("import").addEventListener("change", async (e) => {
  const f = e.target.files[0]; e.target.value = "";
  if (!f) return;
  try { restore(parseSession(await f.text())); flash(`Imported ${notes.length} notes.`); }
  catch (err) { flash(`Could not import: ${err.message}`, true); }
});

// --- Help overlay ---
const help = $("help");
function toggleHelp(force) { help.hidden = force ?? !help.hidden; }
$("helpbtn").addEventListener("click", () => toggleHelp());
$("helpclose").addEventListener("click", () => toggleHelp(false));

let wasm;
function start() { return wasm.create(12, 3, 24, 1, Number($("layers").value)); }

build();
render();
try {
  const res = await fetch("nextnote.wasm");
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  wasm = await load(await res.arrayBuffer());
  model = start();
  status.textContent = "Ready";
  document.body.classList.add("ready");
} catch (err) {
  status.textContent = `Could not load the model (${err.message}). The keys will stay silent.`;
  status.classList.add("error");
}
