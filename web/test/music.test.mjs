import test from "node:test";
import assert from "node:assert/strict";
import { noteForKey, noteFrequency, topPrediction, surpriseBits, barHeights, KEY_MAP, NOTE_NAMES } from "../music.js";

test("key map covers every note exactly once", () => {
  const vals = Object.values(KEY_MAP).sort((a, b) => a - b);
  assert.deepEqual(vals, [...NOTE_NAMES.keys()]);
});

test("noteForKey is case-insensitive and rejects other keys", () => {
  assert.equal(noteForKey("A"), 0);
  assert.equal(noteForKey("j"), 11);
  assert.equal(noteForKey("z"), -1);
  assert.equal(noteForKey("Enter"), -1);
});

test("frequencies follow equal temperament", () => {
  assert.ok(Math.abs(noteFrequency(9, 4) - 440) < 1e-9);
  assert.ok(Math.abs(noteFrequency(0, 4) - 261.6256) < 1e-3);
  assert.ok(Math.abs(noteFrequency(0, 5) / noteFrequency(0, 4) - 2) < 1e-12);
});

test("topPrediction picks the argmax and handles empty input", () => {
  assert.deepEqual(topPrediction([0.1, 0.7, 0.2]), { note: 1, p: 0.7 });
  assert.equal(topPrediction(null), null);
  assert.equal(topPrediction([]), null);
});

test("surpriseBits is -log2 p, null without a prediction", () => {
  assert.equal(surpriseBits([0.5, 0.5], 0), 1);
  assert.equal(surpriseBits(null, 0), null);
  assert.equal(surpriseBits([1, 0], 1), null);
});

test("barHeights normalises to the tallest bar", () => {
  assert.deepEqual(barHeights([0.1, 0.4, 0.2]), [0.25, 1, 0.5]);
  assert.deepEqual(barHeights(null), []);
});

import { pushHistory, surpriseLevel, divergingColor } from "../music.js";

test("pushHistory keeps only the newest entries and does not mutate", () => {
  const h = [1, 2, 3];
  assert.deepEqual(pushHistory(h, 4, 3), [2, 3, 4]);
  assert.deepEqual(h, [1, 2, 3]);
  assert.deepEqual(pushHistory([], 7), [7]);
});

test("surpriseLevel scales to log2(vocab) bits and clamps", () => {
  assert.equal(surpriseLevel(null), 0);
  assert.equal(surpriseLevel(0), 0);
  assert.ok(Math.abs(surpriseLevel(Math.log2(12) / 2) - 0.5) < 1e-12);
  assert.equal(surpriseLevel(40), 1);
  assert.equal(surpriseLevel(-1), 0);
});

test("divergingColor is neutral at 0, warm positive, cool negative, clamped", () => {
  assert.equal(divergingColor(0), "rgb(128, 128, 138)");
  assert.equal(divergingColor(1), "rgb(242, 163, 58)");
  assert.equal(divergingColor(-1), "rgb(70, 120, 220)");
  assert.equal(divergingColor(5), divergingColor(1));
  assert.equal(divergingColor(NaN), divergingColor(0));
});

import { sampleNote, PRESETS, serializeSession, parseSession } from "../music.js";

test("sampleNote: low temperature is argmax, T=1 follows the distribution", () => {
  const p = [0.1, 0.6, 0.3];
  let seed = 7;
  const rand = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32);
  for (let i = 0; i < 50; i++) assert.equal(sampleNote(p, 0.05, rand), 1);
  const counts = [0, 0, 0];
  for (let i = 0; i < 6000; i++) counts[sampleNote(p, 1, rand)]++;
  assert.ok(Math.abs(counts[1] / 6000 - 0.6) < 0.03);
  assert.ok(Math.abs(counts[0] / 6000 - 0.1) < 0.03);
});

test("sampleNote: high temperature flattens, edge cases", () => {
  const p = [0.05, 0.9, 0.05];
  let k = 0;
  const rand = () => [0.01, 0.5, 0.99][k++ % 3];
  const seen = new Set();
  for (let i = 0; i < 9; i++) seen.add(sampleNote(p, 20, rand));
  assert.equal(seen.size, 3);
  assert.equal(sampleNote(null), -1);
  assert.equal(sampleNote([0, 0]), -1);
  assert.equal(sampleNote([0, 1], 1, () => 0.9999999), 1);
});

test("presets only use valid notes", () => {
  for (const notes of Object.values(PRESETS)) assert.ok(notes.length >= 8 && notes.every((n) => n >= 0 && n < 12));
});

test("session round-trips and rejects bad input", () => {
  const s = { layers: 2, notes: [0, 4, 7] };
  assert.deepEqual(parseSession(serializeSession(s)), s);
  assert.throws(() => parseSession("{"), /JSON/);
  assert.throws(() => parseSession('{"app":"x"}'), /not a nextnote/);
  assert.throws(() => parseSession(serializeSession({ layers: 3, notes: [] })), /layers/);
  assert.throws(() => parseSession(serializeSession({ layers: 1, notes: [12] })), /range/);
  assert.throws(() => parseSession(serializeSession({ layers: 1, notes: [1.5] })), /range/);
  assert.throws(() => parseSession(JSON.stringify({ app: "nextnote", version: 9, layers: 1, notes: [] })), /version/);
  assert.throws(() => parseSession(serializeSession({ layers: 1, notes: [0, 0, 0] }), 12, 2), /too long/);
});
