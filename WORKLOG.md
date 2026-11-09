# Work log

Newest entries at the bottom. Each entry records what actually changed, what was learned, and what comes next.

## 2026-11-09 (day 1)
- Picked "predictive-coding sequence learner" from the backlog (Rust to WASM, cognition). It is a new domain for the active projects and a browser app with a Rust core.
- Built the core: splitmix64 RNG, small matrix module, and `PcModel` (latent relaxes to explain a window of notes, local error x activity weight updates, softmax readout, surprise in bits).
- 11 tests pass, incl. surprise collapsing >75% on a looped tune and 13+/14 correct next-note predictions. Terminal demo shows ~45 bits falling to ~7.5.
- Surprise floors around 7.5 bits because softmax temperature is fixed; may tune or calibrate on day 5.
- Added CI (fmt, clippy, test), a placeholder landing page and the Pages workflow.
- Next: day 2, WASM C-ABI bridge plus a Node test that matches Rust output exactly.
