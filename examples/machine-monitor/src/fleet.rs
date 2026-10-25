//! A simulated plant: machines that wear over time, measured every hour, and
//! alarm logic on top of the classifier.

use std::fmt;

use crate::model::Assessment;
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
/// Every fourth fault is cavitation, which the classifier has never seen.
pub fn plant(n: usize, hours: usize, seed: u64) -> Vec<Asset> {
    let kinds = ["pump", "fan", "compressor", "conveyor", "mixer"];
    let mut rng = Rng::new(seed);
    let mut count = [0usize; 5];
    let mut faults = 0;
    (0..n)
        .map(|i| {
            let k = rng.below(kinds.len());
            count[k] += 1;
            let degradation = (i % 3 == 1).then(|| Degradation {
                condition: {
                    faults += 1;
                    if faults % 4 == 2 {
                        Condition::Cavitation
                    } else {
                        Condition::ALL[1 + rng.below(Condition::ALL.len() - 1)]
                    }
                },
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

/// What an alarm says is wrong.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Diagnosis {
    /// One of the classifier's known faults.
    Fault(Condition),
    /// Vibration unlike every known condition (novelty score high).
    Unrecognised,
}

impl Diagnosis {
    /// Recommended action.
    pub fn action(self) -> &'static str {
        match self {
            Diagnosis::Fault(c) => c.action(),
            Diagnosis::Unrecognised => "unknown pattern: send an analyst / inspect on site",
        }
    }
}

impl fmt::Display for Diagnosis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Diagnosis::Fault(c) => write!(f, "{c}"),
            Diagnosis::Unrecognised => f.write_str("unrecognised"),
        }
    }
}

/// What the alarm logic decided for one asset at one hour.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Event {
    /// Nothing new.
    None,
    /// A fault became likely enough to raise an alarm.
    Alarm {
        /// What is wrong.
        diagnosis: Diagnosis,
        /// Smoothed fault probability, or smoothed novelty score for `Unrecognised`.
        confidence: f32,
    },
    /// The diagnosis changed while the alarm was active.
    Reclassified {
        /// New diagnosis.
        diagnosis: Diagnosis,
        /// As in `Alarm`.
        confidence: f32,
    },
}

/// Smooths the hourly assessments of one asset and raises an alarm when a
/// fault (or an unrecognised pattern) persists for a few hours in a row, so
/// one noisy reading doesn't page anyone. An unrecognised pattern wins over
/// the classifier: outside its training data, its probabilities mean little.
#[derive(Debug, Clone)]
pub struct AlarmTracker {
    smoothed: [f32; 6],
    novelty: f32,
    streak: usize,
    novelty_streak: usize,
    /// Diagnosis currently alarmed, if any.
    pub active: Option<Diagnosis>,
}

impl Default for AlarmTracker {
    fn default() -> Self {
        let mut smoothed = [0.0; 6];
        smoothed[Condition::Healthy.index()] = 1.0;
        Self {
            smoothed,
            novelty: 0.0,
            streak: 0,
            novelty_streak: 0,
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
/// Smoothed novelty score that counts as "unrecognised".
const NOVELTY_THRESHOLD: f32 = 1.0;

impl AlarmTracker {
    /// Feeds one hour's assessment.
    pub fn update(&mut self, a: &Assessment) -> Event {
        for (s, p) in self.smoothed.iter_mut().zip(&a.probs) {
            *s = (1.0 - ALPHA) * *s + ALPHA * p;
        }
        self.novelty = (1.0 - ALPHA) * self.novelty + ALPHA * a.novelty;
        let (fault, confidence) = self.top_fault();
        let fault_prob = 1.0 - self.smoothed[Condition::Healthy.index()];
        self.streak = if fault_prob > THRESHOLD {
            self.streak + 1
        } else {
            0
        };
        self.novelty_streak = if self.novelty > NOVELTY_THRESHOLD {
            self.novelty_streak + 1
        } else {
            0
        };
        let (next, conf) = if self.novelty_streak >= CONFIRM_HOURS {
            (Some(Diagnosis::Unrecognised), self.novelty)
        } else if self.streak >= CONFIRM_HOURS || self.active.is_some() {
            (Some(Diagnosis::Fault(fault)), confidence)
        } else {
            (None, 0.0)
        };
        match (self.active, next) {
            (None, Some(d)) => {
                self.active = Some(d);
                Event::Alarm {
                    diagnosis: d,
                    confidence: conf,
                }
            }
            (Some(current), Some(d))
                if current != d
                    && (d == Diagnosis::Unrecognised || confidence > THRESHOLD)
                    && current != Diagnosis::Unrecognised =>
            {
                self.active = Some(d);
                Event::Reclassified {
                    diagnosis: d,
                    confidence: conf,
                }
            }
            _ => Event::None,
        }
    }

    /// Smoothed novelty score (above 1 = unusual).
    pub fn novelty(&self) -> f32 {
        self.novelty
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
