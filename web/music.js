// Pure helpers for the instrument UI (no DOM, no audio) so they can be unit-tested in Node.
export const NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
export const IS_BLACK = NOTE_NAMES.map((n) => n.includes("#"));

// Computer-keyboard layout: the home row is white keys, the row above is black keys (piano style).
export const KEY_MAP = {
  a: 0, w: 1, s: 2, e: 3, d: 4, f: 5, t: 6, g: 7, y: 8, h: 9, u: 10, j: 11,
};

/** Note index (0..11) for a KeyboardEvent.key, or -1. Case-insensitive. */
export function noteForKey(key) {
  const i = KEY_MAP[String(key).toLowerCase()];
  return i === undefined ? -1 : i;
}

/** Frequency in Hz of a chromatic note index, with C4 = 261.6256 Hz (equal temperament, A4 = 440). */
export function noteFrequency(index, octave = 4) {
  const midi = 12 * (octave + 1) + index;
  return 440 * Math.pow(2, (midi - 69) / 12);
}

/** The most likely next note, or null when there are no predictions yet. */
export function topPrediction(probs) {
  if (!probs || probs.length === 0) return null;
  let best = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
  return { note: best, p: probs[best] };
}

/** Surprise in bits of `note` under `probs`; null when no prediction was available. */
export function surpriseBits(probs, note) {
  if (!probs || !(probs[note] > 0)) return null;
  return -Math.log2(probs[note]);
}

/** Height (0..1) of a probability bar, scaled so the tallest bar is full height. */
export function barHeights(probs) {
  if (!probs) return [];
  const max = Math.max(...probs, 1e-12);
  return Array.from(probs, (p) => p / max);
}

/** Append to a fixed-length history, dropping the oldest entries. Returns a new array. */
export function pushHistory(history, value, max = 48) {
  const next = [...history, value];
  return next.length > max ? next.slice(next.length - max) : next;
}

/** Map surprise (bits) to a bar height in 0..1 on a log2(vocab)-bit scale; null (warm-up) is 0. */
export function surpriseLevel(bits, vocab = 12) {
  if (bits === null || bits === undefined || !(bits >= 0)) return 0;
  return Math.min(1, bits / Math.log2(vocab));
}

/**
 * Diverging colour for a signed value in [-1, 1]: negative is blue, positive is orange,
 * zero is neutral grey. Returns an "rgb(r, g, b)" string. Values are clamped.
 */
export function divergingColor(v) {
  const t = Math.max(-1, Math.min(1, Number.isFinite(v) ? v : 0));
  const mix = (a, b, f) => Math.round(a + (b - a) * f);
  const grey = [128, 128, 138], pos = [242, 163, 58], neg = [70, 120, 220];
  const target = t >= 0 ? pos : neg, f = Math.abs(t);
  return `rgb(${mix(grey[0], target[0], f)}, ${mix(grey[1], target[1], f)}, ${mix(grey[2], target[2], f)})`;
}

/**
 * Sample a note from `probs` after applying a softmax-style temperature (p^(1/T), renormalised).
 * T near 0 approaches argmax, T = 1 samples the distribution as is, T > 1 flattens it.
 * `rand` is a () => [0,1) source (injectable for tests). Returns -1 when there are no predictions.
 */
export function sampleNote(probs, temperature = 1, rand = Math.random) {
  if (!probs || probs.length === 0) return -1;
  const T = Math.max(temperature, 0.05);
  const w = Array.from(probs, (p) => (p > 0 ? Math.pow(p, 1 / T) : 0));
  const total = w.reduce((a, b) => a + b, 0);
  if (!(total > 0)) return -1;
  let r = rand() * total;
  for (let i = 0; i < w.length; i++) {
    r -= w[i];
    if (r < 0) return i;
  }
  return w.length - 1; // floating-point slack
}

/** Small preset tunes (note indices 0..11) to teach the model. */
export const PRESETS = {
  "Twinkle": [0, 0, 7, 7, 9, 9, 7, 5, 5, 4, 4, 2, 2, 0],
  "Ode to Joy": [4, 4, 5, 7, 7, 5, 4, 2, 0, 0, 2, 4, 4, 2, 2],
  "Scale up and down": [0, 2, 4, 5, 7, 9, 11, 9, 7, 5, 4, 2],
  "Arpeggio": [0, 4, 7, 4, 0, 4, 7, 11, 7, 4],
};

export const SESSION_VERSION = 1;

/**
 * A session is the model config plus every note played. Because the model is deterministic per
 * seed, replaying the notes into a fresh model reproduces its weights exactly, so this tiny JSON
 * is a complete save file.
 */
export function serializeSession({ layers, notes }) {
  return JSON.stringify({ app: "nextnote", version: SESSION_VERSION, layers, notes });
}

/** Parse and validate a session; throws an Error with a user-facing message when invalid. */
export function parseSession(text, vocab = 12, maxNotes = 5000) {
  let o;
  try { o = JSON.parse(text); } catch { throw new Error("not valid JSON"); }
  if (!o || o.app !== "nextnote") throw new Error("not a nextnote session");
  if (o.version !== SESSION_VERSION) throw new Error(`unsupported version ${o.version}`);
  if (o.layers !== 1 && o.layers !== 2) throw new Error("layers must be 1 or 2");
  if (!Array.isArray(o.notes) || o.notes.length > maxNotes) throw new Error("notes missing or too long");
  if (!o.notes.every((n) => Number.isInteger(n) && n >= 0 && n < vocab)) throw new Error("notes out of range");
  return { layers: o.layers, notes: o.notes };
}
