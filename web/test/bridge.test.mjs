import { readFileSync } from "node:fs";
import test from "node:test";
import assert from "node:assert/strict";
import { load } from "../nextnote.js";
import { TUNE, PASSES, fnv } from "./golden.mjs";
const GOLDEN_HASH = 3775367774;

const wasmPath = process.env.NN_WASM ||
  new URL("../../target/wasm32-unknown-unknown/release/nextnote.wasm", import.meta.url);
const nn = await load(readFileSync(wasmPath));

function run() {
  const m = nn.create();
  const f = new Float32Array(1), u = new Uint32Array(f.buffer);
  const bits = [];
  for (let p = 0; p < PASSES; p++) for (const s of TUNE) { f[0] = m.observe(s); bits.push(u[0]); }
  return { m, bits };
}

test("surprise stream is bit-identical to the Rust model", () => {
  const { m, bits } = run();
  assert.equal(fnv(bits), GOLDEN_HASH);
  m.free();
});

test("surprise falls as the tune is learned", () => {
  const m = nn.create();
  const pass = () => TUNE.reduce((t, s) => t + m.observe(s), 0);
  const first = pass();
  for (let i = 0; i < 40; i++) pass();
  assert.ok(pass() < first * 0.5);
  m.free();
});

test("predict is null while warming up, then a distribution", () => {
  const m = nn.create();
  assert.equal(m.predict(), null);
  TUNE.slice(0, 3).forEach((s) => m.observe(s));
  const p = m.predict();
  assert.equal(p.length, 12);
  assert.ok(Math.abs(p.reduce((a, b) => a + b, 0) - 1) < 1e-5);
  m.resetContext();
  assert.equal(m.predict(), null);
  m.free();
});

test("invalid input throws instead of trapping", () => {
  assert.throws(() => nn.create(0), /invalid/);
  const m = nn.create();
  assert.throws(() => m.observe(99), /invalid symbol/);
  m.free();
});

test("activity and error are null until learning, then well-formed", () => {
  const m = nn.create();
  assert.equal(m.activity(), null);
  assert.equal(m.error(), null);
  TUNE.slice(0, 5).forEach((s) => m.observe(s));
  const a = m.activity(), e = m.error();
  assert.equal(a.length, 24);
  assert.equal(e.length, 12);
  assert.ok(a.every((v) => Math.abs(v) <= 1));
  assert.ok(a.some((v) => v !== 0));
  m.free();
});

test("layered model: layers option validated, 2 layers differs from 1", () => {
  assert.throws(() => nn.create(12, 3, 24, 1, 3), /invalid/);
  const a = nn.create(12, 3, 24, 1, 1), b = nn.create(12, 3, 24, 1, 2);
  let diff = false;
  for (const s of TUNE.concat(TUNE)) diff ||= a.observe(s) !== b.observe(s);
  assert.ok(diff);
  a.free(); b.free();
});

test("replaying a note list reproduces the model exactly (basis of save/load)", () => {
  const notes = TUNE.concat(TUNE, TUNE);
  const a = nn.create(12, 3, 24, 1, 2), b = nn.create(12, 3, 24, 1, 2);
  for (const n of notes) a.observe(n);
  for (const n of notes) b.observe(n);
  assert.deepEqual([...a.predict()], [...b.predict()]);
  assert.deepEqual([...a.activity()], [...b.activity()]);
  a.free(); b.free();
});
