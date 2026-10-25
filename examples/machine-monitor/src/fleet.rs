//! A simulated plant: machines that wear over time, measured every hour, and
//! alarm logic on top of the classifier.

use std::fmt;

use qanten::rng::SplitMix64;

use crate::model::Assessment;
use crate::sim::{Condition, Machine, Measurement};

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
    pub fn measure(&self, hour: usize, seed: u64) -> Measurement {
        let (c, s) = self.state_at(hour);
        self.machine
            .measure(c, s, seed ^ (hour as u64).wrapping_mul(0x9E37))
    }
}

/// A plant: the monitored assets and how long the simulation runs.
#[derive(Debug, Clone)]
pub struct Plant {
    /// The machines, in a fixed order.
    pub assets: Vec<Asset>,
    /// Hours simulated.
    pub hours: usize,
    seed: u64,
}

impl Plant {
    /// `n` assets; about a third develop a fault during `hours`. Every fourth
    /// fault is cavitation, which the classifier has never seen.
    pub fn simulate(n: usize, hours: usize, seed: u64) -> Self {
        let kinds = ["pump", "fan", "compressor", "conveyor", "mixer"];
        let mut rng = SplitMix64::new(seed);
        let mut count = [0usize; 5];
        let mut faults = 0;
        let assets = (0..n)
            .map(|i| {
                let k = rng.below(kinds.len());
                count[k] += 1;
                let degradation = (i % 3 == 1).then(|| {
                    faults += 1;
                    let condition = if faults % 4 == 2 {
                        Condition::Cavitation
                    } else {
                        Condition::ALL[1 + rng.below(Condition::ALL.len() - 1)]
                    };
                    Degradation {
                        condition,
                        onset_hour: (hours as f32 * rng.uniform(0.1, 0.5)) as usize,
                        ramp_hours: (hours as f32 * rng.uniform(0.3, 0.6)) as usize,
                    }
                });
                Asset {
                    name: format!("{}-{:02}", kinds[k], count[k]),
                    machine: Machine::random(rng.next_u64()),
                    degradation,
                }
            })
            .collect();
        Self {
            assets,
            hours,
            seed,
        }
    }

    /// Every asset's measurement at `hour`.
    pub fn measure(&self, hour: usize) -> Vec<Measurement> {
        self.assets
            .iter()
            .map(|a| a.measure(hour, self.seed))
            .collect()
    }
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
    /// A different fault that currently leads, and for how many hours.
    challenger: Option<(Condition, usize)>,
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
            challenger: None,
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
/// How much more likely a different known fault must be, for
/// `CONFIRM_HOURS` hours in a row, before an active alarm switches to it
/// (hysteresis: two similar faults don't flip-flop).
const SWITCH_MARGIN: f32 = 0.2;

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
        // Track a known fault that leads the active one by a clear margin.
        self.challenger = match self.active {
            Some(Diagnosis::Fault(current))
                if fault != current
                    && confidence > THRESHOLD
                    && confidence > self.smoothed[current.index()] + SWITCH_MARGIN =>
            {
                match self.challenger {
                    Some((c, hours)) if c == fault => Some((c, hours + 1)),
                    _ => Some((fault, 1)),
                }
            }
            _ => None,
        };
        let clearly_better = self
            .challenger
            .is_some_and(|(_, hours)| hours >= CONFIRM_HOURS);
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
                    && current != Diagnosis::Unrecognised
                    && (d == Diagnosis::Unrecognised || clearly_better) =>
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

/// An alarm event worth printing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Notice {
    /// Hour it happened.
    pub hour: usize,
    /// Index of the asset in [`Plant::assets`].
    pub asset: usize,
    /// `true` for a new alarm, `false` for a changed diagnosis.
    pub new_alarm: bool,
    /// What the alarm says.
    pub diagnosis: Diagnosis,
    /// Smoothed fault probability, or smoothed novelty score for `Unrecognised`.
    pub confidence: f32,
}

/// How an asset's alarm history compares with the ground truth.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Outcome {
    /// Healthy and no alarm.
    Quiet,
    /// The fault was found and finally diagnosed correctly; `severity` is how
    /// far it had developed at the first alarm.
    Detected {
        /// Severity at the first alarm.
        severity: f32,
        /// `true` if the fault was one the classifier never saw.
        new_fault_type: bool,
    },
    /// An alarm fired, but the final diagnosis is wrong.
    WrongDiagnosis {
        /// Severity at the first alarm.
        severity: f32,
    },
    /// The fault grew past 30% without an alarm.
    Missed,
    /// The fault started but stayed too small to expect an alarm.
    NotDevelopedYet,
    /// An alarm on a healthy machine.
    FalseAlarm,
}

impl Outcome {
    /// Short label for reports.
    pub fn label(&self) -> &'static str {
        match self {
            Outcome::Quiet => "ok",
            Outcome::Detected {
                new_fault_type: true,
                ..
            } => "detected (new fault type)",
            Outcome::Detected { .. } => "detected",
            Outcome::WrongDiagnosis { .. } => "detected, wrong diagnosis",
            Outcome::Missed => "MISSED",
            Outcome::NotDevelopedYet => "not developed yet",
            Outcome::FalseAlarm => "FALSE ALARM",
        }
    }

    /// Severity at the first alarm, for detected faults.
    pub fn severity(&self) -> Option<f32> {
        match *self {
            Outcome::Detected { severity, .. } | Outcome::WrongDiagnosis { severity } => {
                Some(severity)
            }
            _ => None,
        }
    }
}

/// Runs the alarm logic over a plant, hour by hour, and keeps the history.
#[derive(Debug)]
pub struct Monitor {
    trackers: Vec<AlarmTracker>,
    first_alarm: Vec<Option<(usize, Diagnosis)>>,
    final_call: Vec<Option<Diagnosis>>,
}

impl Monitor {
    /// A monitor for `assets` machines.
    pub fn new(assets: usize) -> Self {
        Self {
            trackers: vec![AlarmTracker::default(); assets],
            first_alarm: vec![None; assets],
            final_call: vec![None; assets],
        }
    }

    /// Feeds one hour of assessments (in asset order); returns what changed.
    pub fn step(&mut self, hour: usize, assessments: &[Assessment]) -> Vec<Notice> {
        let mut notices = Vec::new();
        for (asset, (tracker, a)) in self.trackers.iter_mut().zip(assessments).enumerate() {
            let (new_alarm, diagnosis, confidence) = match tracker.update(a) {
                Event::Alarm {
                    diagnosis,
                    confidence,
                } => (true, diagnosis, confidence),
                Event::Reclassified {
                    diagnosis,
                    confidence,
                } => (false, diagnosis, confidence),
                Event::None => continue,
            };
            if new_alarm {
                self.first_alarm[asset] = Some((hour, diagnosis));
            }
            self.final_call[asset] = Some(diagnosis);
            notices.push(Notice {
                hour,
                asset,
                new_alarm,
                diagnosis,
                confidence,
            });
        }
        notices
    }

    /// First alarm of asset `i`: hour and diagnosis.
    pub fn first_alarm(&self, i: usize) -> Option<(usize, Diagnosis)> {
        self.first_alarm[i]
    }

    /// Current smoothed health of asset `i`.
    pub fn health(&self, i: usize) -> f32 {
        self.trackers[i].health()
    }

    /// Compares asset `i`'s alarms with its ground truth after the run.
    pub fn outcome(&self, plant: &Plant, i: usize) -> Outcome {
        let asset = &plant.assets[i];
        match (asset.degradation, self.first_alarm[i]) {
            (None, None) => Outcome::Quiet,
            (None, Some(_)) => Outcome::FalseAlarm,
            (Some(_), None) if asset.state_at(plant.hours.saturating_sub(1)).1 > 0.3 => {
                Outcome::Missed
            }
            (Some(_), None) => Outcome::NotDevelopedYet,
            (Some(d), Some((hour, _))) => {
                let expected = match d.condition.class() {
                    Some(_) => Diagnosis::Fault(d.condition),
                    None => Diagnosis::Unrecognised,
                };
                let severity = asset.state_at(hour).1;
                if self.final_call[i] == Some(expected) {
                    Outcome::Detected {
                        severity,
                        new_fault_type: expected == Diagnosis::Unrecognised,
                    }
                } else {
                    Outcome::WrongDiagnosis { severity }
                }
            }
        }
    }
}
