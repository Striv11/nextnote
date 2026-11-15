import { readFileSync, statSync } from "node:fs";
import test from "node:test";
import assert from "node:assert/strict";
import { load } from "../nextnote.js";
import { TUNE } from "./golden.mjs";

const wasmUrl = process.env.NN_WASM ||
  new URL("../../target/wasm32-unknown-unknown/release/nextnote.wasm", import.meta.url);
const nn = await load(readFileSync(wasmUrl));

function median(xs) {
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
}

for (const layers of [1, 2]) {
  test(`observe() median stays under 1 ms (${layers} layer${layers > 1 ? "s" : ""})`, () => {
    const m = nn.create(12, 3, 24, 1, layers);
    for (let i = 0; i < 200; i++) m.observe(TUNE[i % TUNE.length]); // warm up
    const times = [];
    for (let i = 0; i < 2000; i++) {
      const t0 = performance.now();
      m.observe(TUNE[i % TUNE.length]);
      times.push(performance.now() - t0);
    }
    const med = median(times);
    m.free();
    assert.ok(med < 1, `median observe took ${med.toFixed(4)} ms`);
  });
}

test("shipped bundle (.wasm + JS) is under 150 KB", () => {
  const web = (f) => new URL(`../${f}`, import.meta.url);
  const wasm = statSync(wasmUrl).size;
  const js = ["app.js", "music.js", "nextnote.js", "persist.js"]
    .reduce((n, f) => n + statSync(web(f)).size, 0);
  assert.ok(wasm + js < 150 * 1024, `bundle is ${wasm + js} bytes`);
});
