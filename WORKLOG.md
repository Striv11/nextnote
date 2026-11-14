# Work log

Newest entries at the bottom. Each entry records what actually changed, what was learned, and what comes next.

## 2026-11-09 (day 1)
- Picked "predictive-coding sequence learner" from the backlog (Rust to WASM, cognition). It is a new domain for the active projects and a browser app with a Rust core.
- Built the core: splitmix64 RNG, small matrix module, and `PcModel` (latent relaxes to explain a window of notes, local error x activity weight updates, softmax readout, surprise in bits).
- 11 tests pass, incl. surprise collapsing >75% on a looped tune and 13+/14 correct next-note predictions. Terminal demo shows ~45 bits falling to ~7.5.
- Surprise floors around 7.5 bits because softmax temperature is fixed; may tune or calibrate on day 5.
- Added CI (fmt, clippy, test), a placeholder landing page and the Pages workflow.
- Next: day 2, WASM C-ABI bridge plus a Node test that matches Rust output exactly.

## 2026-11-10 (day 2)
- Added the C-ABI bridge (`src/ffi.rs`): `nn_new/observe/predict/reset_context/free` plus `nn_alloc/dealloc` for result buffers. Invalid input returns null or a negative code instead of panicking (release uses `panic = "abort"`, which would trap the wasm instance).
- `web/nextnote.js` wraps the ABI; `web/test/bridge.test.mjs` loads the real .wasm in Node.
- Learned: the first golden comparison failed. Native and wasm differed in the last bits of `exp`/`tanh`/`log2` because libm differs per target, so "same seed, same output" was false across platforms. Fixed at the root with `src/det.rs` (pure +,-,*,/ in f64). Now an FNV hash over every per-note surprise bit pattern is identical in Rust (native) and Node (wasm): same golden constant in both tests.
- `check.sh` and CI now build wasm and run the Node test; pages.yml copies the wasm into the site (ignored by git).
- Next: day 3, playable keyboard UI with Web Audio and ghost-key predictions.

## 2026-11-11 (day 3)
- Built the playable instrument: 12-key piano (pointer/touch + computer keys), Web Audio triangle-wave tones, every note goes through the wasm model, the predicted next key gets an orange "ghost" outline, probability bars for all 12 notes, empty state, loading/error state, reset button, light/dark via `prefers-color-scheme`.
- Kept logic (key map, frequencies, argmax, surprise, bar scaling) in `web/music.js` with 6 Node tests; `app.js` is thin DOM glue. `check.sh` runs both Node test files.
- Limitation: I could not drive a real browser in this environment, so the DOM/audio layer is only syntax-checked (`node --check`), not exercised. Playwright e2e (day 4) should cover it.
- Learned: the first `check.sh` run failed only because the wasm target was still being installed; a rerun was green.
- Next: day 4, surprise timeline canvas, latent heatmap, error view, Playwright e2e.

## 2026-11-12 (day 4)
- `PcModel` now keeps the latent activity and newest-slot prediction error of its last learning step (`activity()`, `error()`); exposed over the ABI as `nn_activity`/`nn_error` and in `web/nextnote.js`. Numerics untouched: the golden hash still matches.
- UI: surprise-per-note timeline, latent-activity strip and prediction-error strip (canvas). Pure helpers (`pushHistory`, `surpriseLevel`, `divergingColor`) are unit-tested in Node; Rust and Node bridge tests cover the new calls, including too-small-buffer and null-handle errors.
- Not done: Playwright e2e. No browser or Playwright is installed here, and I did not want to claim coverage I cannot run. The canvas drawing is only syntax-checked.
- Learned: `check.sh` failed at first because the wasm target was missing in this environment; `rustup target add wasm32-unknown-unknown` fixed it.
- Next: finish day 4 with Playwright e2e where a browser is available, then day 5 (two-layer model).

## 2026-11-13 (day 5)
- Added an optional slow top layer to `PcModel` (`Config::slow_decays`): leaky traces of the whole note stream (`t = d*t + onehot`) feed a learned top-down prior (`U`) on the next-note slot. The prior enters inference (explaining away the newest-slot error) and is trained with the same local error x input rule. Empty `slow_decays` is the old model exactly: the golden hash is unchanged.
- Test: a 15-note melody where `[0,1,2]` is followed by 3, 4 or 5 depending on the earlier phrase. The single layer scores <= 13/15 next-note guesses, the two-layer model >= 14/15 (gap >= 2 asserted). Also tests determinism, reset and invalid decays.
- Exposed as `nn_new_layered(..., layers)` (the old `nn_new` is untouched), `create(..., layers)` in JS, and a `Memory` selector in the UI that resets the model. Rust and Node tests cover validation.
- Caveat: the UI selector is not browser-tested (still no browser here), same as day 4. Decays 0.5/0.85 are hand-picked, not tuned.
- Next: finish day 4 Playwright e2e if a browser is available; otherwise day 6 (improvisation, presets, IndexedDB persistence).

## 2026-11-14 (day 6)
- Added improvisation: `sampleNote` samples the model's prediction with a temperature (p^(1/T) renormalised); an `Improvise` button plays the samples back, with a temperature slider. Preset tunes (`PRESETS`) can be taught from a selector.
- Persistence without touching the Rust core: a session is `{layers, notes}`. Since the model is deterministic per seed, replaying the notes rebuilds the exact weights; a new Node bridge test asserts two replays give identical predictions and activity. `parseSession` validates version, layers, note range and length. Stored in IndexedDB (`web/persist.js`) and exported/imported as JSON.
- Shortcuts: `Space` improvise, `Backspace` reset, `?`/`Esc` help overlay.
- Caveat: the DOM, IndexedDB and audio glue is only syntax-checked (no browser here); pure logic has Node tests. Replay cost grows with session length (capped at 5000 notes).
- Learned: the wasm target had to be re-added in this environment before `check.sh` could pass.
- Next: day 7 (benchmark with asserted budget, bundle size check, screenshots, e2e if a browser is available).
