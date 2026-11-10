//! Deterministic `exp`, `tanh` and `log2`.
//!
//! The standard library defers to the platform libm, whose results differ in
//! the last bits between native targets and WebAssembly. These versions use
//! only `+ - * /` and bit manipulation in `f64` (all exactly specified by
//! IEEE 754), so the model behaves bit-identically everywhere.

const LN2: f64 = std::f64::consts::LN_2;

/// e^x, accurate to ~1e-15 relative error for |x| <= 700.
pub fn exp(x: f32) -> f32 {
    exp64(x as f64) as f32
}

fn exp64(x: f64) -> f64 {
    let x = x.clamp(-700.0, 700.0);
    let k = (x / LN2 + if x >= 0.0 { 0.5 } else { -0.5 }) as i64; // round half away
    let r = x - k as f64 * LN2; // |r| <= ~0.35
    let mut term = 1.0;
    let mut sum = 1.0;
    for n in 1..=16 {
        term *= r / n as f64;
        sum += term;
    }
    sum * f64::from_bits(((k + 1023) as u64) << 52)
}

pub fn tanh(x: f32) -> f32 {
    let x = x as f64;
    if x.abs() > 20.0 {
        return x.signum() as f32;
    }
    let e = exp64(2.0 * x);
    ((e - 1.0) / (e + 1.0)) as f32
}

/// Base-2 logarithm of a positive finite value.
pub fn log2(x: f32) -> f32 {
    assert!(x > 0.0 && x.is_finite());
    let bits = (x as f64).to_bits();
    let mut e = ((bits >> 52) & 0x7ff) as i64 - 1023;
    let mut m = f64::from_bits((bits & ((1u64 << 52) - 1)) | (1023u64 << 52)); // [1,2)
    if m > std::f64::consts::SQRT_2 {
        m /= 2.0;
        e += 1;
    }
    let s = (m - 1.0) / (m + 1.0);
    let s2 = s * s;
    let mut pow = s;
    let mut sum = 0.0;
    for n in 0..14 {
        sum += pow / (2 * n + 1) as f64;
        pow *= s2;
    }
    (e as f64 + 2.0 * sum / LN2) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() <= 2e-6 * b.abs().max(1e-3)
    }

    #[test]
    fn exp_matches_std() {
        for i in -400..=100 {
            let x = i as f32 * 0.1;
            assert!(close(exp(x), x.exp()), "exp({x})");
        }
    }

    #[test]
    fn tanh_matches_std_and_saturates() {
        for i in -300..=300 {
            let x = i as f32 * 0.05;
            assert!((tanh(x) - x.tanh()).abs() < 1e-6, "tanh({x})");
        }
        assert_eq!(tanh(50.0), 1.0);
        assert_eq!(tanh(-50.0), -1.0);
        assert_eq!(tanh(0.0), 0.0);
    }

    #[test]
    fn log2_matches_std() {
        for &x in &[1e-9f32, 1e-4, 0.1, 0.5, 0.999, 1.0, 1.5, 2.0, 3.7, 1024.0] {
            assert!((log2(x) - x.log2()).abs() < 1e-5, "log2({x})");
        }
    }
}
