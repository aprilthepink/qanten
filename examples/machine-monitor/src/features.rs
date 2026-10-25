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

use qanten::signal::{Dft, FirBank, design, hann};
use qanten::{Npu, Result};

use crate::sim::{SAMPLE_RATE, WINDOW};

const TAPS: usize = 127;
const DECIMATE: usize = 8;
/// Samples per band after dropping the filter start-up and decimating.
const BAND_LEN: usize = (WINDOW - (TAPS - 1)) / DECIMATE;
const ORDER_STEP: f32 = 0.25;
/// Spectrum orders 0.25×…10× of the low band.
const SPEED_ORDERS: usize = 40;
/// Envelope orders 0.5×…12× of the resonance band.
const ENVELOPE_ORDERS: usize = 47;

/// Length of the feature vector.
pub const NUM_FEATURES: usize = SPEED_ORDERS + ENVELOPE_ORDERS + STATS;
const STATS: usize = 5;

/// Turns raw windows into feature vectors on one NPU.
#[derive(Debug)]
pub struct FeatureExtractor {
    filters: FirBank,
    dft: Dft,
    window: Vec<f32>,
}

impl FeatureExtractor {
    /// Uploads the filter bank and DFT matrix to `npu`.
    pub fn new(npu: &Npu) -> Result<Self> {
        let nyquist_frac = |hz: f32| hz / SAMPLE_RATE;
        let filters = FirBank::new(
            npu,
            &[
                design::lowpass(nyquist_frac(450.0), TAPS),
                design::bandpass(nyquist_frac(2400.0), nyquist_frac(4400.0), TAPS),
            ],
        )?;
        Ok(Self {
            filters,
            dft: Dft::new(npu, BAND_LEN)?,
            window: hann(BAND_LEN),
        })
    }

    /// Features of one window measured at `shaft_hz`.
    pub fn extract(&self, shaft_hz: f32, signal: &[f32]) -> Result<Vec<f32>> {
        self.extract_batch(&[(shaft_hz, signal)])
    }

    /// Features of several windows, `rows × NUM_FEATURES`. The DFTs of the
    /// whole batch run as one NPU call.
    pub fn extract_batch(&self, windows: &[(f32, &[f32])]) -> Result<Vec<f32>> {
        let mut frames = Vec::with_capacity(windows.len() * 2 * BAND_LEN);
        let mut stats = Vec::with_capacity(windows.len());
        for &(_, signal) in windows {
            if signal.len() != WINDOW {
                return Err(qanten::Error::Shape(format!(
                    "a window has {WINDOW} samples, got {}",
                    signal.len()
                )));
            }
            let bands = self.filters.apply(signal)?;
            let (low, resonance) = (&bands[0][TAPS - 1..], &bands[1][TAPS - 1..]);
            // Low band: plain decimation (it is already band-limited).
            let low: Vec<f32> = low
                .iter()
                .step_by(DECIMATE)
                .take(BAND_LEN)
                .copied()
                .collect();
            // Resonance band: rectify, average blocks of 8, remove DC.
            let mut env: Vec<f32> = resonance
                .chunks(DECIMATE)
                .take(BAND_LEN)
                .map(|c| c.iter().map(|v| v.abs()).sum::<f32>() / c.len() as f32)
                .collect();
            let mean = env.iter().sum::<f32>() / env.len() as f32;
            env.iter_mut().for_each(|v| *v -= mean);
            for band in [&low, &env] {
                frames.extend(band.iter().zip(&self.window).map(|(x, w)| x * w));
            }
            stats.push(time_stats(signal, resonance));
        }
        let spectrum = self.dft.forward(&frames)?;
        let magnitude = spectrum.magnitude();
        let bins = spectrum.bins;
        // Hann window coherent gain is n/2; amplitude = 2·|X| / (n/2).
        let to_amplitude = 4.0 / BAND_LEN as f32;
        let bin_hz = SAMPLE_RATE / DECIMATE as f32 / BAND_LEN as f32;

        let mut out = Vec::with_capacity(windows.len() * NUM_FEATURES);
        for (i, &(shaft_hz, _)) in windows.iter().enumerate() {
            let low = &magnitude[(2 * i) * bins..(2 * i + 1) * bins];
            let env = &magnitude[(2 * i + 1) * bins..(2 * i + 2) * bins];
            for (spec, first, count) in [(low, 1, SPEED_ORDERS), (env, 2, ENVELOPE_ORDERS)] {
                for k in 0..count {
                    let order = (first + k) as f32 * ORDER_STEP;
                    let peak = order_peak(spec, order, shaft_hz, bin_hz) * to_amplitude;
                    out.push((peak + 1e-4).log10());
                }
            }
            out.extend(stats[i]);
            // Envelope noise floor: median bin (bearing faults add peaks; broadband faults raise it).
            let mut sorted: Vec<f32> = env[1..].to_vec();
            sorted.sort_by(f32::total_cmp);
            out.push((sorted[sorted.len() / 2] * to_amplitude + 1e-5).log10());
        }
        Ok(out)
    }
}

/// Largest magnitude within ±half an order step of `order`.
fn order_peak(spec: &[f32], order: f32, shaft_hz: f32, bin_hz: f32) -> f32 {
    let lo = ((order - ORDER_STEP / 2.0) * shaft_hz / bin_hz).round() as usize;
    let hi = ((order + ORDER_STEP / 2.0) * shaft_hz / bin_hz).round() as usize;
    let hi = hi.max(lo).min(spec.len() - 1);
    spec[lo.min(hi)..=hi].iter().copied().fold(0.0, f32::max)
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
