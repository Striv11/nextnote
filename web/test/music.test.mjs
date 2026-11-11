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
