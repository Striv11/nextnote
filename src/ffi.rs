//! C-ABI bridge for WebAssembly (and native tests).
//!
//! No wasm-bindgen: JS talks to these `nn_*` functions directly. Handles are
//! opaque pointers; probability vectors are copied into caller-provided
//! buffers obtained from `nn_alloc`. Functions never panic across the
//! boundary: invalid input yields a null handle or a negative return code.

use crate::{Config, PcModel};

/// Returned by fallible calls when the arguments are invalid.
pub const NN_ERR: i32 = -1;

/// Create a model. Returns null if any dimension is zero or too large.
#[no_mangle]
pub extern "C" fn nn_new(vocab: u32, context: u32, hidden: u32, seed: u32) -> *mut PcModel {
    let ok = |n: u32| (1..=256).contains(&n);
    if !(ok(vocab) && ok(context) && ok(hidden)) {
        return std::ptr::null_mut();
    }
    let cfg = Config {
        vocab: vocab as usize,
        context: context as usize,
        hidden: hidden as usize,
        seed: seed as u64,
        ..Config::default()
    };
    Box::into_raw(Box::new(PcModel::new(cfg)))
}

/// Like `nn_new`, plus `layers`: 1 = single layer, 2 = adds the slow top
/// layer (note traces with decays 0.5 and 0.85). Other values return null.
#[no_mangle]
pub extern "C" fn nn_new_layered(
    vocab: u32,
    context: u32,
    hidden: u32,
    seed: u32,
    layers: u32,
) -> *mut PcModel {
    let ok = |n: u32| (1..=256).contains(&n);
    if !(ok(vocab) && ok(context) && ok(hidden)) || !(1..=2).contains(&layers) {
        return std::ptr::null_mut();
    }
    let cfg = Config {
        vocab: vocab as usize,
        context: context as usize,
        hidden: hidden as usize,
        seed: seed as u64,
        slow_decays: if layers == 2 { vec![0.5, 0.85] } else { vec![] },
        ..Config::default()
    };
    Box::into_raw(Box::new(PcModel::new(cfg)))
}

/// Feed one symbol. Returns its surprise in bits (0.0 while warming up), or
/// a negative value if the handle is null or the symbol out of vocabulary.
///
/// # Safety
/// `m` must be null or a live pointer from `nn_new`.
#[no_mangle]
pub unsafe extern "C" fn nn_observe(m: *mut PcModel, symbol: u32) -> f32 {
    match m.as_mut() {
        Some(model) if (symbol as usize) < model.config().vocab => {
            model.observe(symbol as usize).surprise
        }
        _ => NN_ERR as f32,
    }
}

/// Write the current next-symbol distribution (`vocab` floats) into `out`.
/// Returns the number of floats written, 0 while warming up, or `NN_ERR` on
/// a null handle/buffer or a buffer smaller than `vocab`.
///
/// # Safety
/// `m` must be null or from `nn_new`; `out` must be valid for `cap` f32 writes.
#[no_mangle]
pub unsafe extern "C" fn nn_predict(m: *const PcModel, out: *mut f32, cap: u32) -> i32 {
    let Some(model) = m.as_ref() else {
        return NN_ERR;
    };
    if out.is_null() || (cap as usize) < model.config().vocab {
        return NN_ERR;
    }
    match model.next_distribution() {
        Some(p) => {
            std::ptr::copy_nonoverlapping(p.as_ptr(), out, p.len());
            p.len() as i32
        }
        None => 0,
    }
}

unsafe fn copy_out(src: &[f32], out: *mut f32, cap: u32) -> i32 {
    if out.is_null() || (cap as usize) < src.len() {
        return NN_ERR;
    }
    std::ptr::copy_nonoverlapping(src.as_ptr(), out, src.len());
    src.len() as i32
}

/// Copy the latent activity of the last learning step into `out`. Returns
/// the count written (0 before any learning) or `NN_ERR` for a null handle
/// or too small a buffer (`cap` must be >= `hidden`).
///
/// # Safety
/// `m` must be null or from `nn_new`; `out` valid for `cap` f32 writes.
#[no_mangle]
pub unsafe extern "C" fn nn_activity(m: *const PcModel, out: *mut f32, cap: u32) -> i32 {
    match m.as_ref() {
        Some(model) => copy_out(model.activity(), out, cap),
        None => NN_ERR,
    }
}

/// Copy the newest-note prediction error (`vocab` floats) of the last
/// learning step into `out`. Same return convention as `nn_activity`.
///
/// # Safety
/// `m` must be null or from `nn_new`; `out` valid for `cap` f32 writes.
#[no_mangle]
pub unsafe extern "C" fn nn_error(m: *const PcModel, out: *mut f32, cap: u32) -> i32 {
    match m.as_ref() {
        Some(model) => copy_out(model.error(), out, cap),
        None => NN_ERR,
    }
}

/// Forget the recent stream but keep learned weights.
///
/// # Safety
/// `m` must be null or from `nn_new`.
#[no_mangle]
pub unsafe extern "C" fn nn_reset_context(m: *mut PcModel) {
    if let Some(model) = m.as_mut() {
        model.reset_context();
    }
}

/// Destroy a model. Null is a no-op.
///
/// # Safety
/// `m` must be null or from `nn_new`, and not used afterwards.
#[no_mangle]
pub unsafe extern "C" fn nn_free(m: *mut PcModel) {
    if !m.is_null() {
        drop(Box::from_raw(m));
    }
}

/// Allocate `n` f32 slots for JS to read/write. Free with `nn_dealloc`.
#[no_mangle]
pub extern "C" fn nn_alloc(n: u32) -> *mut f32 {
    let mut v = vec![0f32; n as usize];
    let p = v.as_mut_ptr();
    std::mem::forget(v);
    p
}

/// # Safety
/// `p`/`n` must be exactly a pointer and length from `nn_alloc`.
#[no_mangle]
pub unsafe extern "C" fn nn_dealloc(p: *mut f32, n: u32) {
    if !p.is_null() {
        drop(Vec::from_raw_parts(p, n as usize, n as usize));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub const TUNE: [u32; 14] = [0, 0, 4, 4, 5, 5, 4, 3, 3, 2, 2, 1, 1, 0];

    #[test]
    fn activity_and_error_are_exposed_and_validated() {
        unsafe {
            let m = nn_new(12, 3, 24, 1);
            let buf = nn_alloc(24);
            assert_eq!(nn_activity(m, buf, 24), 0, "nothing learned yet");
            assert_eq!(nn_error(m, buf, 24), 0);
            for &n in TUNE.iter().take(6) {
                nn_observe(m, n);
            }
            assert_eq!(nn_activity(m, buf, 24), 24);
            let act = std::slice::from_raw_parts(buf, 24).to_vec();
            assert!(act.iter().all(|a| a.abs() <= 1.0));
            assert!(act.iter().any(|a| a.abs() > 0.0));
            assert_eq!(nn_activity(m, buf, 23), NN_ERR, "buffer too small");
            assert_eq!(nn_error(m, buf, 12), 12);
            assert_eq!(nn_error(m, std::ptr::null_mut(), 12), NN_ERR);
            assert_eq!(nn_error(std::ptr::null(), buf, 12), NN_ERR);
            nn_dealloc(buf, 24);
            nn_free(m);
        }
    }

    #[test]
    fn matches_the_safe_api_bit_for_bit() {
        let mut direct = PcModel::new(Config::default());
        unsafe {
            let m = nn_new(12, 3, 24, 1);
            let buf = nn_alloc(12);
            for pass in 0..5 {
                for &s in &TUNE {
                    let a = direct.observe(s as usize).surprise;
                    let b = nn_observe(m, s);
                    assert_eq!(a.to_bits(), b.to_bits(), "pass {pass} symbol {s}");
                    let Some(want) = direct.next_distribution() else {
                        assert_eq!(nn_predict(m, buf, 12), 0);
                        continue;
                    };
                    assert_eq!(nn_predict(m, buf, 12), 12);
                    let got = std::slice::from_raw_parts(buf, 12);
                    assert!(want
                        .iter()
                        .zip(got)
                        .all(|(a, b)| a.to_bits() == b.to_bits()));
                }
            }
            nn_dealloc(buf, 12);
            nn_free(m);
        }
    }

    /// Same golden as web/test/bridge.test.mjs: FNV-1a over surprise bit patterns.
    #[test]
    fn surprise_stream_matches_the_wasm_golden() {
        let mut m = PcModel::new(Config::default());
        let mut h: u32 = 0x811c9dc5;
        for _ in 0..30 {
            for &s in &TUNE {
                for byte in m.observe(s as usize).surprise.to_bits().to_le_bytes() {
                    h ^= byte as u32;
                    h = h.wrapping_mul(0x01000193);
                }
            }
        }
        assert_eq!(h, 3775367774);
    }

    #[test]
    fn layered_constructor_validates_and_matches_single_layer() {
        unsafe {
            assert!(nn_new_layered(12, 3, 24, 1, 0).is_null());
            assert!(nn_new_layered(12, 3, 24, 1, 3).is_null());
            let (a, b) = (nn_new(12, 3, 24, 1), nn_new_layered(12, 3, 24, 1, 1));
            let two = nn_new_layered(12, 3, 24, 1, 2);
            assert!(!two.is_null());
            let (mut same, mut diff) = (true, false);
            for &s in TUNE.iter().cycle().take(40) {
                let (x, y, z) = (nn_observe(a, s), nn_observe(b, s), nn_observe(two, s));
                same &= x.to_bits() == y.to_bits();
                diff |= x.to_bits() != z.to_bits();
            }
            assert!(same && diff);
            nn_free(a);
            nn_free(b);
            nn_free(two);
        }
    }

    #[test]
    fn invalid_arguments_are_errors_not_panics() {
        unsafe {
            assert!(nn_new(0, 3, 24, 1).is_null());
            assert!(nn_new(12, 3, 100_000, 1).is_null());
            assert!(nn_observe(std::ptr::null_mut(), 0) < 0.0);
            let m = nn_new(12, 3, 24, 1);
            assert!(nn_observe(m, 12) < 0.0);
            let mut small = [0f32; 4];
            assert_eq!(nn_predict(m, small.as_mut_ptr(), 4), NN_ERR);
            let mut out = [0f32; 12];
            assert_eq!(nn_predict(m, out.as_mut_ptr(), 12), 0, "warming up");
            nn_free(m);
            nn_free(std::ptr::null_mut());
        }
    }

    #[test]
    fn reset_context_returns_to_warmup() {
        unsafe {
            let m = nn_new(12, 3, 24, 1);
            let mut out = [0f32; 12];
            for &s in &TUNE[..4] {
                nn_observe(m, s);
            }
            assert_eq!(nn_predict(m, out.as_mut_ptr(), 12), 12);
            nn_reset_context(m);
            assert_eq!(nn_predict(m, out.as_mut_ptr(), 12), 0);
            nn_free(m);
        }
    }
}
