# Plan

Seven-day arc. Every milestone has acceptance criteria that can be checked by a test or a command.

| Day | Milestone | Acceptance criteria | Status |
|-----|-----------|---------------------|--------|
| 1 | Rust core: PC model | `cargo test` passes; model is deterministic per seed; surprise on a looped 14-note tune falls >75% in 120 passes; after training, next-note argmax is right for >= 13/14 contexts; `bash check.sh` green; landing page placeholder | [x] |
| 2 | WASM build and C-ABI bridge | `cargo build --target wasm32-unknown-unknown --release` succeeds; exported functions (`nn_new`, `nn_observe`, `nn_predict`, `nn_free`) callable from a Node test that loads the .wasm and reproduces the Rust surprise numbers exactly; pages.yml builds the wasm into the site | [x] |
| 3 | Playable instrument UI | Responsive piano keyboard (touch, mouse, computer keys) plays notes via Web Audio; every note goes through the model; a "ghost" highlight shows the predicted next key with probability bars; empty state explains what to do; light/dark (UI built; browser behaviour checked manually, automated e2e arrives day 4) | [x] |
| 4 | Surprise and learning visuals | Live surprise-per-note timeline (canvas); latent-activity heatmap; prediction-error view; loading and error states; Playwright e2e covers play, predict, reset (visuals, `nn_activity`/`nn_error` and tests done; Playwright e2e still open, no browser available in the build environment) | [~] |
| 5 | Richer model: two layers and longer memory | Hierarchical (2-level) PC with a slower top layer; test that it disambiguates a melody with repeated 3-note contexts that the 1-layer model cannot (accuracy gap asserted); selectable in UI | [ ] |
| 6 | Auto-play, presets, persistence | Model "improvises" by sampling its predictions (with temperature); preset tunes; save/load weights in IndexedDB; export/import JSON; keyboard shortcuts with help overlay | [ ] |
| 7 | Polish, performance, docs | Benchmark: observe() under 1 ms median at default size, budget asserted in CI; bundle (.wasm + JS) under 150 KB; screenshots in README; full e2e suite; final README/roadmap | [ ] |

## Stretch
- Microphone pitch input (autocorrelation) as a melody source.
- MIDI file import to pre-train.
- Compare against an n-gram baseline on held-out tunes.
