//! Vibration features computed on the NPU.
//!
//! The classic vibration-analysis pipeline, with every filter and transform as
//! one batched NPU call:
//!
//! 1. A two-filter FIR bank (one NPU call per window): a low-pass for the
//!    running-speed band (0–450 Hz) and a band-pass around the bearing
//!    resonances (2.4–4.4 kHz).
//! 2. Both bands are decimated by 8. The band-pass is rectified first, which
//!    gives its envelope (demodulation).
//! 3. One DFT call transforms both bands of every window in the batch.
//! 4. Each spectrum is resampled from Hz to *shaft orders*, so the same fault
//!    looks the same at 900 rpm and at 2700 rpm.
//!
//! Plus five statistics: RMS, crest factor, kurtosis and energy of the
//! resonance band, and the envelope spectrum's noise floor (its median).

use qanten::compute::Matrix;
use qanten::signal::{Dft, FirBank, Spectrum, design, hann};
use qanten::{Error, Npu, Result};

use crate::sim::{Measurement, SAMPLE_RATE, WINDOW};

const TAPS: usize = 127;
const DECIMATE: usize = 8;
/// Samples per band after dropping the filter start-up and decimating.
const BAND_LEN: usize = (WINDOW - (TAPS - 1)) / DECIMATE;
const ORDER_STEP: f32 = 0.25;
/// Spectrum orders 0.25×…10× of the low band.
const SPEED_ORDERS: usize = 40;
/// Envelope orders 0.5×…12× of the resonance band.
const ENVELOPE_ORDERS: usize = 47;
/// Hann window coherent gain is n/2, so amplitude = 2·|X| / (n/2).
const TO_AMPLITUDE: f32 = 4.0 / BAND_LEN as f32;
const BIN_HZ: f32 = SAMPLE_RATE / DECIMATE as f32 / BAND_LEN as f32;

/// Length of the feature vector.
pub const NUM_FEATURES: usize = SPEED_ORDERS + ENVELOPE_ORDERS + STATS;
const STATS: usize = 5;

/// Turns measurements into feature vectors on one NPU.
#[derive(Debug)]
pub struct FeatureExtractor {
    filters: FirBank,
    dft: Dft,
    window: Vec<f32>,
}

/// The two decimated bands of one measurement plus its time-domain statistics.
#[derive(Debug)]
struct Bands {
    low: Vec<f32>,
    envelope: Vec<f32>,
    /// `[log RMS, log crest factor, log kurtosis, log band RMS]`.
    stats: [f32; 4],
}

impl Bands {
    /// From the filter bank output (row 0 low-pass, row 1 band-pass).
    fn new(filtered: &Matrix, signal: &[f32]) -> Self {
        let (low, resonance) = (&filtered.row(0)[TAPS - 1..], &filtered.row(1)[TAPS - 1..]);
        // Low band: plain decimation (it is already band-limited).
        let low = low
            .iter()
            .step_by(DECIMATE)
            .take(BAND_LEN)
            .copied()
            .collect();
        // Resonance band: rectify, average blocks of 8, remove DC.
        let mut envelope: Vec<f32> = resonance
            .chunks(DECIMATE)
            .take(BAND_LEN)
            .map(|c| c.iter().map(|v| v.abs()).sum::<f32>() / c.len() as f32)
            .collect();
        let mean = envelope.iter().sum::<f32>() / envelope.len() as f32;
        envelope.iter_mut().for_each(|v| *v -= mean);
        Self {
            low,
            envelope,
            stats: time_stats(signal, resonance),
        }
    }
}

impl FeatureExtractor {
    /// Uploads the filter bank and DFT matrix to `npu`.
    pub fn new(npu: &Npu) -> Result<Self> {
        let fraction = |hz: f32| hz / SAMPLE_RATE;
        let filters = FirBank::new(
            npu,
            &[
                design::lowpass(fraction(450.0), TAPS),
                design::bandpass(fraction(2400.0), fraction(4400.0), TAPS),
            ],
        )?;
        Ok(Self {
            filters,
            dft: Dft::new(npu, BAND_LEN)?,
            window: hann(BAND_LEN),
        })
    }

    /// Features of one measurement.
    pub fn extract(&self, m: &Measurement) -> Result<Vec<f32>> {
        Ok(self.extract_batch(std::slice::from_ref(m))?.into_vec())
    }

    /// Features of several measurements, one row each (`len × NUM_FEATURES`).
    /// The DFTs of the whole batch run as one NPU call.
    pub fn extract_batch(&self, batch: &[Measurement]) -> Result<Matrix> {
        let bands = batch
            .iter()
            .map(|m| {
                if m.samples.len() != WINDOW {
                    return Err(Error::Shape(format!(
                        "a measurement has {WINDOW} samples, got {}",
                        m.samples.len()
                    )));
                }
                Ok(Bands::new(&self.filters.apply(&m.samples)?, &m.samples))
            })
            .collect::<Result<Vec<_>>>()?;
        // Frames 2i (low band) and 2i + 1 (envelope), windowed, in one DFT call.
        let frames = Matrix::from_fn(2 * bands.len(), BAND_LEN, |r, t| {
            let b = &bands[r / 2];
            (if r % 2 == 0 { b.low[t] } else { b.envelope[t] }) * self.window[t]
        });
        let magnitude = Spectrum::magnitude(&self.dft.forward(&frames)?);
        let rows: Vec<Vec<f32>> = batch
            .iter()
            .zip(&bands)
            .enumerate()
            .map(|(i, (m, b))| {
                let (low, env) = (magnitude.row(2 * i), magnitude.row(2 * i + 1));
                order_features(low, m.shaft_hz, 1, SPEED_ORDERS)
                    .chain(order_features(env, m.shaft_hz, 2, ENVELOPE_ORDERS))
                    .chain(b.stats)
                    .chain([noise_floor(env)])
                    .collect()
            })
            .collect();
        Matrix::from_rows(&rows)
    }
}

/// `count` features: log amplitude at orders `first·step, (first+1)·step, …`.
fn order_features(
    spec: &[f32],
    shaft_hz: f32,
    first: usize,
    count: usize,
) -> impl Iterator<Item = f32> + '_ {
    (first..first + count).map(move |k| {
        let peak = order_peak(spec, k as f32 * ORDER_STEP, shaft_hz) * TO_AMPLITUDE;
        (peak + 1e-4).log10()
    })
}

/// Envelope noise floor: the median bin (bearing faults add peaks; broadband
/// faults raise the floor).
fn noise_floor(env: &[f32]) -> f32 {
    let mut sorted = env[1..].to_vec();
    sorted.sort_by(f32::total_cmp);
    (sorted[sorted.len() / 2] * TO_AMPLITUDE + 1e-5).log10()
}

/// Largest magnitude within ±half an order step of `order`.
fn order_peak(spec: &[f32], order: f32, shaft_hz: f32) -> f32 {
    let bin = |o: f32| (o * shaft_hz / BIN_HZ).round() as usize;
    let hi = bin(order + ORDER_STEP / 2.0).min(spec.len() - 1);
    let lo = bin(order - ORDER_STEP / 2.0).min(hi);
    spec[lo..=hi].iter().copied().fold(0.0, f32::max)
}

/// `[log10 RMS, log10 crest factor, log10 kurtosis and log10 RMS of the resonance band]`.
fn time_stats(signal: &[f32], resonance: &[f32]) -> [f32; 4] {
    let n = signal.len() as f32;
    let mean = signal.iter().sum::<f32>() / n;
    let rms = (signal.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n).sqrt();
    let peak = signal.iter().map(|v| (v - mean).abs()).fold(0.0, f32::max);
    let m = resonance.len() as f32;
    let var = resonance.iter().map(|v| v * v).sum::<f32>() / m;
    let kurtosis = resonance.iter().map(|v| v.powi(4)).sum::<f32>() / m / (var * var).max(1e-12);
    [
        (rms + 1e-6).log10(),
        (peak / rms.max(1e-6)).log10(),
        kurtosis.max(1e-3).log10(),
        (var.sqrt() + 1e-6).log10(),
    ]
}

/// Readable names of the features, for debugging.
pub fn feature_names() -> Vec<String> {
    let mut names = Vec::with_capacity(NUM_FEATURES);
    for k in 0..SPEED_ORDERS {
        names.push(format!("spectrum {:.2}x", (1 + k) as f32 * ORDER_STEP));
    }
    for k in 0..ENVELOPE_ORDERS {
        names.push(format!("envelope {:.2}x", (2 + k) as f32 * ORDER_STEP));
    }
    names.extend(
        [
            "log rms",
            "log crest",
            "log kurtosis",
            "log band rms",
            "log envelope floor",
        ]
        .map(String::from),
    );
    names
}
