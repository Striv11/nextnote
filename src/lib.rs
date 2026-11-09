//! nextnote: a predictive-coding sequence learner.
//!
//! The model never sees labels or gradients through time. It keeps a latent
//! state that tries to *explain* a window of recent symbols, and learns with
//! purely local error-times-activity updates.

pub mod mat;
pub mod model;
pub mod rng;

pub use model::{Config, PcModel, Step};
