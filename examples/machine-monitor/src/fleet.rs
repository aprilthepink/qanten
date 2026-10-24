//! A simulated plant: machines that wear over time, measured every hour, and
//! alarm logic on top of the classifier.

use crate::sim::{Condition, Machine, Rng};

/// A fault that starts at some hour and grows over `ramp_hours`.
#[derive(Debug, Clone, Copy)]
pub struct Degradation {
    /// What goes wrong.
    pub condition: Condition,
    /// Hour the fault starts.
    pub onset_hour: usize,
    /// Hours from onset to full severity.
    pub ramp_hours: usize,
}

/// One monitored asset.
#[derive(Debug, Clone)]
pub struct Asset {
    /// Tag like `pump-03`.
    pub name: String,
    /// Its vibration behaviour.
    pub machine: Machine,
    /// The fault it will develop, if any (the hidden ground truth).
    pub degradation: Option<Degradation>,
}

impl Asset {
    /// Ground truth at `hour`: condition and severity.
    pub fn state_at(&self, hour: usize) -> (Condition, f32) {
        match self.degradation {
            Some(d) if hour >= d.onset_hour => {
                let s = (hour - d.onset_hour) as f32 / d.ramp_hours.max(1) as f32;
                (d.condition, s.min(1.0))
            }
            _ => (Condition::Healthy, 0.0),
        }
    }

    /// The measurement taken at `hour`.
    pub fn measure(&self, hour: usize, seed: u64) -> Vec<f32> {
        let (c, s) = self.state_at(hour);
        self.machine
            .measure(c, s, seed ^ (hour as u64).wrapping_mul(0x9E37))
    }
}

/// A plant of `n` assets; about a third develop a fault during `hours`.
pub fn plant(n: usize, hours: usize, seed: u64) -> Vec<Asset> {
    let kinds = ["pump", "fan", "compressor", "conveyor", "mixer"];
    let mut rng = Rng::new(seed);
    let mut count = [0usize; 5];
    (0..n)
        .map(|i| {
            let k = rng.below(kinds.len());
            count[k] += 1;
            let degradation = (i % 3 == 1).then(|| Degradation {
                condition: Condition::ALL[1 + rng.below(Condition::ALL.len() - 1)],
                onset_hour: (hours as f32 * rng.uniform(0.1, 0.5)) as usize,
                ramp_hours: (hours as f32 * rng.uniform(0.3, 0.6)) as usize,
            });
            Asset {
                name: format!("{}-{:02}", kinds[k], count[k]),
                machine: Machine::random(rng.next_u64()),
                degradation,
            }
        })
        .collect()
}

/// What the alarm logic decided for one asset at one hour.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Event {
    /// Nothing new.
    None,
    /// A fault became likely enough to raise an alarm.
    Alarm {
        /// Most likely fault.
        condition: Condition,
        /// Smoothed probability of that fault.
        confidence: f32,
    },
    /// The suspected fault changed while the alarm was active.
    Reclassified {
        /// New most likely fault.
        condition: Condition,
        /// Smoothed probability of that fault.
        confidence: f32,
    },
}

/// Smooths the hourly probabilities of one asset and raises an alarm when a
/// fault stays likely for a few hours in a row (one noisy reading doesn't
/// page anyone).
#[derive(Debug, Clone)]
pub struct AlarmTracker {
    smoothed: [f32; 6],
    streak: usize,
    /// Fault currently alarmed, if any.
    pub active: Option<Condition>,
}

impl Default for AlarmTracker {
    fn default() -> Self {
        let mut smoothed = [0.0; 6];
        smoothed[Condition::Healthy.index()] = 1.0;
        Self {
            smoothed,
            streak: 0,
            active: None,
        }
    }
}

/// Exponential smoothing factor per hour.
const ALPHA: f32 = 0.35;
/// Smoothed fault probability that counts as "likely".
const THRESHOLD: f32 = 0.6;
/// Consecutive likely hours before the alarm fires.
const CONFIRM_HOURS: usize = 3;

impl AlarmTracker {
    /// Feeds one hour of probabilities (in [`Condition::ALL`] order).
    pub fn update(&mut self, probs: &[f32]) -> Event {
        for (s, p) in self.smoothed.iter_mut().zip(probs) {
            *s = (1.0 - ALPHA) * *s + ALPHA * p;
        }
        let (fault, confidence) = self.top_fault();
        let fault_prob = 1.0 - self.smoothed[Condition::Healthy.index()];
        self.streak = if fault_prob > THRESHOLD {
            self.streak + 1
        } else {
            0
        };
        match self.active {
            None if self.streak >= CONFIRM_HOURS => {
                self.active = Some(fault);
                Event::Alarm {
                    condition: fault,
                    confidence,
                }
            }
            Some(current) if current != fault && confidence > THRESHOLD => {
                self.active = Some(fault);
                Event::Reclassified {
                    condition: fault,
                    confidence,
                }
            }
            _ => Event::None,
        }
    }

    /// Smoothed health score: probability that the machine is healthy.
    pub fn health(&self) -> f32 {
        self.smoothed[Condition::Healthy.index()]
    }

    fn top_fault(&self) -> (Condition, f32) {
        Condition::ALL[1..]
            .iter()
            .map(|&c| (c, self.smoothed[c.index()]))
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap_or((Condition::Imbalance, 0.0))
    }
}
