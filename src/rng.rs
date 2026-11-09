//! Tiny deterministic PRNG (splitmix64) so runs are reproducible everywhere,
//! including in WebAssembly where no OS randomness is available.

#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    pub fn uniform(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Uniform in [-scale, scale).
    pub fn symmetric(&mut self, scale: f32) -> f32 {
        (self.uniform() * 2.0 - 1.0) * scale
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_for_a_seed() {
        let mut a = Rng::new(7);
        let mut b = Rng::new(7);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
        assert_ne!(Rng::new(1).next_u64(), Rng::new(2).next_u64());
    }

    #[test]
    fn uniform_is_in_unit_interval_and_roughly_centered() {
        let mut r = Rng::new(42);
        let mut sum = 0.0;
        for _ in 0..10_000 {
            let u = r.uniform();
            assert!((0.0..1.0).contains(&u));
            sum += u;
        }
        assert!((sum / 10_000.0 - 0.5).abs() < 0.02);
    }
}
