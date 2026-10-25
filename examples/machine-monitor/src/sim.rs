//! Synthetic vibration signals of rotating machines.
//!
//! Stands in for an accelerometer on a bearing housing. Each fault leaves the
//! signature a vibration analyst would look for:
//!
//! | Condition | Signature |
//! |---|---|
//! | imbalance | strong 1× shaft speed |
//! | misalignment | strong 2× (and 3×) |
//! | looseness | many harmonics plus half orders (0.5×, 1.5×, …) |
//! | outer race | impacts at BPFO ≈ 3.585× ringing a structural resonance |
//! | inner race | impacts at BPFI ≈ 5.415×, amplitude-modulated at 1× |
//! | cavitation | random, non-periodic bubble-collapse impacts at 2–5 kHz |
//!
//! Cavitation is deliberately **not** one of the classifier's classes: it is
//! the "fault nobody trained for" that the novelty detector has to catch.

use std::f32::consts::TAU;
use std::fmt;
use std::str::FromStr;

/// Accelerometer sample rate in Hz.
pub const SAMPLE_RATE: f32 = 10_240.0;
/// Samples per measurement window (0.4 s).
pub const WINDOW: usize = 4096;
/// Ball-pass frequency of the outer race, in shaft orders (6205-type bearing).
pub const BPFO: f32 = 3.585;
/// Ball-pass frequency of the inner race, in shaft orders.
pub const BPFI: f32 = 5.415;

/// Machine condition: the classes the model predicts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Condition {
    /// No fault.
    Healthy,
    /// Mass imbalance of the rotor.
    Imbalance,
    /// Shaft misalignment at the coupling.
    Misalignment,
    /// Mechanical looseness (loose mounts, worn fits).
    Looseness,
    /// Bearing outer-race defect.
    OuterRace,
    /// Bearing inner-race defect.
    InnerRace,
    /// Pump cavitation. Not a model class (see [`Condition::ALL`]); used to
    /// test detection of faults the classifier has never seen.
    Cavitation,
}

impl Condition {
    /// All conditions, in model output order.
    pub const ALL: [Condition; 6] = [
        Condition::Healthy,
        Condition::Imbalance,
        Condition::Misalignment,
        Condition::Looseness,
        Condition::OuterRace,
        Condition::InnerRace,
    ];

    /// Faults the simulator can produce but the classifier never sees.
    pub const UNSEEN: [Condition; 1] = [Condition::Cavitation];

    /// Position in the model output (`None` for [`Condition::UNSEEN`]).
    pub fn class(self) -> Option<usize> {
        Self::ALL.iter().position(|&c| c == self)
    }

    /// Position in the model output.
    ///
    /// # Panics
    /// For conditions in [`Condition::UNSEEN`].
    pub fn index(self) -> usize {
        match self.class() {
            Some(i) => i,
            None => panic!("{self} is not a model class"),
        }
    }

    /// Short name, also accepted by `FromStr`.
    pub fn label(self) -> &'static str {
        match self {
            Condition::Healthy => "healthy",
            Condition::Imbalance => "imbalance",
            Condition::Misalignment => "misalignment",
            Condition::Looseness => "looseness",
            Condition::OuterRace => "outer-race",
            Condition::InnerRace => "inner-race",
            Condition::Cavitation => "cavitation",
        }
    }

    /// What a maintenance planner should do about it.
    pub fn action(self) -> &'static str {
        match self {
            Condition::Healthy => "none",
            Condition::Imbalance => "schedule rotor balancing",
            Condition::Misalignment => "re-align the coupling",
            Condition::Looseness => "check mounting bolts and fits",
            Condition::OuterRace => "plan bearing replacement",
            Condition::InnerRace => "plan bearing replacement (urgent)",
            Condition::Cavitation => "check suction pressure and flow",
        }
    }
}

impl fmt::Display for Condition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

impl FromStr for Condition {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        let all = Self::ALL.into_iter().chain(Self::UNSEEN);
        all.clone().find(|c| c.label() == s).ok_or_else(|| {
            let names: Vec<_> = all.map(|c| c.label()).collect();
            format!(
                "unknown condition `{s}` (expected one of {})",
                names.join(", ")
            )
        })
    }
}

/// Small deterministic RNG (SplitMix64) with uniform and Gaussian draws.
#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
    /// Seeded generator.
    pub fn new(seed: u64) -> Self {
        Self(seed ^ 0x5DEE_CE66_D1CE_4E5B)
    }

    /// Next raw 64-bit value.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[lo, hi)`.
    pub fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        let u = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        lo + (hi - lo) * u
    }

    /// Standard normal (Box–Muller).
    pub fn gaussian(&mut self) -> f32 {
        let u1 = self.uniform(1e-7, 1.0);
        let u2 = self.uniform(0.0, 1.0);
        (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos()
    }

    /// Uniform index below `n`.
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n.max(1) as u64) as usize
    }
}

/// One machine: its running speed and the parts of its vibration that don't
/// depend on a fault (residual 1×/2×, electrical hum, noise floor, resonance).
#[derive(Debug, Clone)]
pub struct Machine {
    /// Shaft speed in Hz (from a tachometer or the drive).
    pub shaft_hz: f32,
    base_1x: f32,
    base_2x: f32,
    hum: f32,
    noise: f32,
    resonance_hz: f32,
    ring_ms: f32,
    fault_gain: f32,
}

impl Machine {
    /// A random machine running between 15 and 45 Hz (900–2700 rpm).
    pub fn random(seed: u64) -> Self {
        let mut r = Rng::new(seed);
        Self::random_at(r.uniform(15.0, 45.0), seed)
    }

    /// A random machine at a given shaft speed.
    pub fn random_at(shaft_hz: f32, seed: u64) -> Self {
        let mut r = Rng::new(seed.wrapping_mul(31).wrapping_add(7));
        Self {
            shaft_hz,
            base_1x: r.uniform(0.02, 0.06),
            base_2x: r.uniform(0.005, 0.02),
            hum: r.uniform(0.0, 0.03),
            noise: r.uniform(0.015, 0.04),
            resonance_hz: r.uniform(2700.0, 4000.0),
            ring_ms: r.uniform(0.4, 0.9),
            fault_gain: r.uniform(0.7, 1.3),
        }
    }

    /// One [`WINDOW`]-sample acceleration measurement (in g) with `condition`
    /// at `severity` (0 = just starting, 1 = severe).
    pub fn measure(&self, condition: Condition, severity: f32, seed: u64) -> Vec<f32> {
        let mut r = Rng::new(seed);
        let f = self.shaft_hz;
        let s = severity.clamp(0.0, 1.0) * self.fault_gain;
        let dt = 1.0 / SAMPLE_RATE;
        let t0 = r.uniform(0.0, 100.0);
        let phase = |r: &mut Rng| r.uniform(0.0, TAU);

        // Periodic part: (order, amplitude, phase).
        let p1 = phase(&mut r);
        let mut tones = vec![(1.0, self.base_1x, p1), (2.0, self.base_2x, phase(&mut r))];
        match condition {
            Condition::Imbalance => tones[0].1 += s * r.uniform(0.15, 0.3),
            Condition::Misalignment => {
                tones[1].1 += s * r.uniform(0.1, 0.2);
                tones.push((3.0, s * r.uniform(0.04, 0.08), phase(&mut r)));
                tones[0].1 += s * 0.02;
            }
            Condition::Looseness => {
                let a = s * r.uniform(0.03, 0.05);
                for h in 1..=10 {
                    tones.push((h as f32, a / (h as f32).sqrt(), phase(&mut r)));
                }
                for h in [0.5, 1.5, 2.5, 3.5] {
                    tones.push((h, a * 0.6, phase(&mut r)));
                }
            }
            _ => {}
        }
        let hum_phase = phase(&mut r);
        let mut x: Vec<f32> = (0..WINDOW)
            .map(|i| {
                let t = t0 + i as f32 * dt;
                let mut v = self.hum * (TAU * 100.0 * t + hum_phase).sin();
                for &(order, amp, ph) in &tones {
                    v += amp * (TAU * order * f * (t - t0) + ph).sin();
                }
                v + self.noise * r.gaussian()
            })
            .collect();

        // Bearing impacts: decaying rings of the structural resonance.
        let impacts = match condition {
            Condition::OuterRace => Some((BPFO, false)),
            Condition::InnerRace => Some((BPFI, true)),
            _ => None,
        };
        if let Some((order, modulated)) = impacts {
            let period = 1.0 / (order * f);
            let amp = s * r.uniform(0.25, 0.6);
            let tau = self.ring_ms * 1e-3;
            let ring = (8.0 * tau / dt) as usize;
            let mut t = r.uniform(0.0, period);
            let duration = WINDOW as f32 * dt;
            while t < duration {
                let gain = if modulated {
                    0.6 + 0.4 * (TAU * f * t + p1).cos()
                } else {
                    1.0
                } * (1.0 + 0.2 * r.gaussian());
                let start = (t / dt) as usize;
                for k in 0..ring.min(WINDOW.saturating_sub(start)) {
                    let tt = k as f32 * dt;
                    x[start + k] +=
                        amp * gain * (-tt / tau).exp() * (TAU * self.resonance_hz * tt).sin();
                }
                // Rolling elements slip a little, so impacts aren't perfectly periodic.
                t += period * (1.0 + 0.01 * r.gaussian());
            }
        }
        if condition == Condition::Cavitation {
            // Poisson bubble collapses: short, broadband rings at random
            // frequencies, plus some extra low-frequency turbulence noise.
            let rate = 3000.0 + 5000.0 * s;
            let amp = s * r.uniform(0.4, 0.8);
            let mut t = 0.0f32;
            let duration = WINDOW as f32 * dt;
            loop {
                t += -r.uniform(1e-6, 1.0).ln() / rate;
                if t >= duration {
                    break;
                }
                let freq = r.uniform(2000.0, 5000.0);
                let tau = r.uniform(0.15, 0.4) * 1e-3;
                let a = amp * r.uniform(0.3, 1.0);
                let start = (t / dt) as usize;
                let ring = (6.0 * tau / dt) as usize + 1;
                for k in 0..ring.min(WINDOW - start) {
                    let tt = k as f32 * dt;
                    x[start + k] += a * (-tt / tau).exp() * (TAU * freq * tt).sin();
                }
            }
            let turbulence = s * 0.02;
            x.iter_mut().for_each(|v| *v += turbulence * r.gaussian());
        }
        x
    }
}
