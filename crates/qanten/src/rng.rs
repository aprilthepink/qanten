//! A small, fast, deterministic random number generator (SplitMix64), so the
//! crate and its users don't need `rand` for simulations and test data.
//!
//! ```
//! use qanten::rng::SplitMix64;
//! let mut rng = SplitMix64::new(42);
//! let x = rng.uniform(-1.0, 1.0);
//! assert!((-1.0..1.0).contains(&x));
//! assert_eq!(SplitMix64::new(42).uniform(-1.0, 1.0), x); // same seed, same stream
//! ```

use std::f32::consts::TAU;

/// SplitMix64: one `u64` of state, statistically solid for simulation, not
/// for cryptography.
#[derive(Debug, Clone)]
pub struct SplitMix64(u64);

impl SplitMix64 {
    /// A generator; equal seeds give equal streams.
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// Next raw 64-bit value.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Uniform in `[lo, hi)`.
    pub fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_f32()
    }

    /// Uniform integer in `0..n` (`0` for `n == 0`).
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n.max(1) as u64) as usize
    }

    /// Standard normal sample (Box–Muller).
    pub fn next_normal(&mut self) -> f32 {
        let u1 = self.next_f32().max(f32::MIN_POSITIVE);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos()
    }

    /// Shuffles `items` in place (Fisher–Yates).
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            items.swap(i, self.below(i + 1));
        }
    }
}
