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
