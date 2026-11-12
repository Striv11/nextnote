# nextnote

> A tiny brain-inspired model that learns melodies as you play them, and shows you what it expects to hear next.

![CI](https://github.com/Striv11/nextnote/actions/workflows/ci.yml/badge.svg)

**Live demo:** https://striv11.github.io/nextnote/ (playable keyboard with ghost-key predictions)

## Problem
Most "AI learns music" demos are opaque: you train offline and get output. It is hard to *see* what a model
expects, when it is surprised, and how that changes while it learns. nextnote is for anyone curious about how
brains might learn sequences: play a few notes and watch a model form expectations in real time.

## Motivation
Predictive coding says the brain constantly predicts its inputs and learns from the *errors*. This project
implements that idea from scratch in Rust (no ML crates, no backprop through time): a latent state tries to explain
a window of recent notes, and weights change with purely local error x activity updates. Compiled to WebAssembly,
it runs entirely in your browser: no server, no API keys, nothing leaves your device.

## Architecture
```
 notes ──► window of last K notes (one-hot) ──┐
                                              ▼
          latent z ──tanh──► activity a ──W──► predicted window
              ▲                                   │
              └──── relax z to shrink error ◄─────┘   (inference)

 predict: clamp the K known notes, relax z, read the (K+1)th slot ─► softmax
 learn:   clamp all K+1 notes, relax z, W += lr · error · aᵀ        (local rule)
 surprise = -log2 p(actual note)
```
- `src/det.rs`: `exp`/`tanh`/`log2` from plain IEEE arithmetic, so native and WASM agree bit-for-bit (platform libm does not).
- `src/ffi.rs`: the `nn_*` C ABI (`nn_new`, `nn_observe`, `nn_predict`, `nn_reset_context`, `nn_activity`, `nn_error`, `nn_free`, `nn_alloc`, `nn_dealloc`); `web/nextnote.js` wraps it.
- `src/rng.rs`: deterministic splitmix64 (reproducible, no OS randomness needed in WASM).
- `src/mat.rs`: minimal dense matrix ops.
- `src/model.rs`: the predictive-coding model (`PcModel`).
- `web/`: the static site. `music.js` is pure, unit-tested logic; `app.js` is the DOM/Web Audio glue.

Invariants: same seed and same input gives bit-identical behaviour; predictions always form a probability distribution.

## Setup
Prerequisites: Rust stable (with `rustfmt`, `clippy`; add `wasm32-unknown-unknown` for the web build).
```
bash check.sh        # fmt, clippy, Rust tests, wasm build, Node bridge test (needs Node 20+)
cargo run --release  # terminal demo
```

## Demo
Terminal demo works today (`cargo run --release`): it loops a 14-note tune and prints total surprise per pass; the
bar shrinks from ~45 bits on pass 1 to ~7.5 bits by pass 60 as the model learns it.

The in-browser instrument (`web/`): click, tap or type `A W S E D F T G Y H U J` to play; the model's expected next key is outlined and probability bars show its full prediction. Below the keys: a surprise-per-note timeline, a latent-activity strip and a prediction-error strip for the last note (orange = under-predicted, blue = over-predicted). Browser e2e tests are still open (see [PLAN.md](PLAN.md)). To try locally: build the wasm, copy it to `web/nextnote.wasm`, and serve `web/` (e.g. `python3 -m http.server -d web`).

## Design decisions & trade-offs
- One-hot window of the last K notes instead of a recurrent state: simple, local, and easy to visualize; a second
  slower layer (day 5) adds longer memory.
- Softmax with a fixed temperature over the generative output turns reconstruction into a distribution, instead of
  adding a separate classifier head.
- C ABI rather than wasm-bindgen: zero JS-glue dependencies and a smaller bundle.

## Roadmap
- [x] Predictive-coding core with tests (day 1)
- [x] WASM build and bridge (day 2)
- [x] Playable instrument UI (day 3)
- [~] Surprise and activity visuals (done), e2e tests (open) (day 4)
- [ ] Two-layer model (day 5)
- [ ] Improvisation, presets, persistence (day 6)
- [ ] Benchmarks, screenshots, polish (day 7)

## License
MIT. See [LICENSE](LICENSE).
