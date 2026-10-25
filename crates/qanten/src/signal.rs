//! Signal processing as matrix products on the NPU.
//!
//! Each tool keeps one constant matrix on the NPU as a
//! [`compute::NpuMatrix`](crate::compute::NpuMatrix) and pushes every frame
//! or block of the signal through it in a single batched call:
//!
//! | Type | Matrix on the NPU | Vectors streamed |
//! |---|---|---|
//! | [`Dft`] | `cos` / `sin` rows of the DFT | frames |
//! | [`Stft`] | windowed DFT rows | overlapping frames → spectrogram |
//! | [`FirBank`] | every filter at every shift of a block | signal blocks |
//! | [`TemplateMatcher`] | every template at every shift of a block | signal blocks → matches |
//!
//! **Precision.** bfloat16 gives roughly 40–50 dB of dynamic range: plenty
//! for spectrograms, detection, feature extraction and monitoring, not for
//! producing hi-fi audio.
//!
//! ```
//! # fn main() -> qanten::Result<()> {
//! use qanten::compute::Matrix;
//! use qanten::signal::Dft;
//! let npu = qanten::Npu::simulator();
//! let n = 64;
//! let tone = Matrix::from_fn(1, n, |_, t| (std::f32::consts::TAU * 5.0 * t as f32 / n as f32).cos());
//! let spectrum = Dft::new(&npu, n)?.forward(&tone)?;
//! assert_eq!(spectrum.peak_bin(0), 5);
//! # Ok(()) }
//! ```

use std::f64::consts::TAU;

use crate::compute::{Matrix, NpuMatrix};
use crate::device::Npu;
use crate::error::{Error, Result};

/// Complex spectra of one or more frames.
#[derive(Debug, Clone, PartialEq)]
pub struct Spectrum {
    /// Real parts, `frames × bins`.
    pub re: Matrix,
    /// Imaginary parts, `frames × bins`.
    pub im: Matrix,
}

impl Spectrum {
    /// Splits DFT output rows `[re | im]` (`frames × 2·bins`).
    fn from_stacked(y: &Matrix, bins: usize) -> Self {
        Self {
            re: y.columns(0..bins),
            im: y.columns(bins..2 * bins),
        }
    }

    /// Frequency bins per frame.
    pub fn bins(&self) -> usize {
        self.re.cols()
    }

    /// Number of frames.
    pub fn frames(&self) -> usize {
        self.re.rows()
    }

    /// Magnitudes `|X|`, `frames × bins`.
    pub fn magnitude(&self) -> Matrix {
        let mut m = self.re.clone();
        m.as_mut_slice()
            .iter_mut()
            .zip(self.im.as_slice())
            .for_each(|(r, i)| *r = r.hypot(*i));
        m
    }

    /// Power in decibels `20·log10(|X| + 1e-12)`, `frames × bins`.
    pub fn db(&self) -> Matrix {
        self.magnitude().map(|m| 20.0 * (m + 1e-12).log10())
    }

    /// Bin with the largest magnitude in `frame`.
    pub fn peak_bin(&self, frame: usize) -> usize {
        self.magnitude()
            .row(frame)
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map_or(0, |(i, _)| i)
    }
}

/// Periodic Hann window of length `n`.
pub fn hann(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (0.5 - 0.5 * (TAU * i as f64 / n as f64).cos()) as f32)
        .collect()
}

/// The `[cos(2πkt/n) ; −sin(2πkt/n)]` DFT rows (`2·bins × n`), times an
/// optional window, kept on the NPU.
#[derive(Debug)]
struct FourierMatrix {
    n: usize,
    bins: usize,
    resident: NpuMatrix,
}

impl FourierMatrix {
    fn new(npu: &Npu, n: usize, window: Option<&[f32]>) -> Result<Self> {
        let bins = n / 2 + 1;
        // The angle uses (k·t) mod n in integers, so it is exact for any n.
        let rows = Matrix::from_fn(2 * bins, n, |r, t| {
            let (k, sine) = (r % bins, r >= bins);
            let a = TAU * ((k * t) % n) as f64 / n as f64;
            let w = window.map_or(1.0, |w| w[t] as f64);
            (if sine { -a.sin() } else { a.cos() } * w) as f32
        });
        Ok(Self {
            n,
            bins,
            resident: NpuMatrix::upload(npu, &rows)?,
        })
    }

    fn transform(&self, frames: &Matrix) -> Result<Spectrum> {
        Ok(Spectrum::from_stacked(
            &self.resident.apply_batch(frames)?,
            self.bins,
        ))
    }
}

/// Discrete Fourier transform of real frames of length `n` (bins `0..=n/2`).
#[derive(Debug)]
pub struct Dft(FourierMatrix);

impl Dft {
    /// A DFT for frames of length `n`, uploaded to `npu`.
    pub fn new(npu: &Npu, n: usize) -> Result<Self> {
        if n < 2 {
            return Err(Error::InvalidArgument(
                "DFT length must be at least 2".into(),
            ));
        }
        Ok(Self(FourierMatrix::new(npu, n, None)?))
    }

    /// Frame length.
    pub fn len(&self) -> usize {
        self.0.n
    }

    /// Always `false` (length ≥ 2).
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Transforms every row of `frames` (`frames × n`) in one NPU call.
    pub fn forward(&self, frames: &Matrix) -> Result<Spectrum> {
        if frames.cols() != self.0.n || frames.rows() == 0 {
            return Err(Error::Shape(format!(
                "frames must be rows of {} samples, got {}x{}",
                self.0.n,
                frames.rows(),
                frames.cols()
            )));
        }
        self.0.transform(frames)
    }
}

/// Short-time Fourier transform: spectrogram of a long signal.
#[derive(Debug)]
pub struct Stft {
    hop: usize,
    fourier: FourierMatrix,
}

impl Stft {
    /// Frames of `frame` samples every `hop` samples, Hann-windowed.
    pub fn new(npu: &Npu, frame: usize, hop: usize) -> Result<Self> {
        if frame < 2 || hop == 0 {
            return Err(Error::InvalidArgument(
                "frame must be ≥ 2 and hop ≥ 1".into(),
            ));
        }
        Ok(Self {
            hop,
            fourier: FourierMatrix::new(npu, frame, Some(&hann(frame)))?,
        })
    }

    /// Frequency of `bin` in Hz for a given sample rate.
    pub fn bin_frequency(&self, bin: usize, sample_rate: f32) -> f32 {
        bin as f32 * sample_rate / self.fourier.n as f32
    }

    /// Spectrogram of `signal`: every full frame, all in one NPU call.
    pub fn spectrogram(&self, signal: &[f32]) -> Result<Spectrum> {
        let n = self.fourier.n;
        if signal.len() < n {
            return Err(Error::Shape(format!(
                "signal of {} samples is shorter than one frame ({n})",
                signal.len()
            )));
        }
        let count = (signal.len() - n) / self.hop + 1;
        let frames = Matrix::from_fn(count, n, |f, t| signal[f * self.hop + t]);
        self.fourier.transform(&frames)
    }
}

/// Sliding dot products of several kernels against a signal, as block
/// convolution.
///
/// Computes `y[t][k] = Σ_j kernel_k[j] · x[t + j]` for every `t` in
/// `0..=n − L` (all "valid" positions). Instead of sending every length-`L`
/// window to the NPU (`L` values per output), the signal is cut into blocks of
/// `B` outputs. Each block needs one input segment of `B + L − 1` samples, and
/// the resident matrix (`K·B` rows, `B + L − 1` columns) holds every kernel at
/// every shift inside the block. That sends about `(B + L − 1) / B ≈ 2` values
/// per output instead of `L`, for about twice the multiply-accumulates
/// (the matrix is banded). On the NPU, moving data costs more than
/// multiplying, so this is the better trade.
#[derive(Debug)]
struct BlockCorrelator {
    len: usize,
    kernels: usize,
    block: usize,
    resident: NpuMatrix,
}

impl BlockCorrelator {
    /// `kernels` must be non-empty and of equal length (checked by callers).
    fn new(npu: &Npu, kernels: &[Vec<f32>]) -> Result<Self> {
        let len = kernels[0].len();
        let k = kernels.len();
        let block = len.clamp(32, 256);
        // Row (i·K + kk): kernel kk shifted to output i of the block.
        let rows = Matrix::from_fn(block * k, block + len - 1, |row, p| {
            let (i, kk) = (row / k, row % k);
            p.checked_sub(i)
                .and_then(|j| kernels[kk].get(j))
                .copied()
                .unwrap_or(0.0)
        });
        Ok(Self {
            len,
            kernels: k,
            block,
            resident: NpuMatrix::upload(npu, &rows)?,
        })
    }

    /// `count × K` outputs, for `count = signal.len() − L + 1`.
    fn apply(&self, signal: &[f32]) -> Result<Matrix> {
        let count = signal.len() + 1 - self.len;
        let width = self.block + self.len - 1;
        let segments = Matrix::from_fn(count.div_ceil(self.block), width, |b, p| {
            signal.get(b * self.block + p).copied().unwrap_or(0.0)
        });
        let mut y = self.resident.apply_batch(&segments)?.into_vec();
        y.truncate(count * self.kernels);
        Matrix::new(count, self.kernels, y)
    }
}

fn equal_lengths(kernels: &[Vec<f32>], min: usize) -> Option<usize> {
    let len = kernels.first()?.len();
    (len >= min && kernels.iter().all(|k| k.len() == len)).then_some(len)
}

/// Several FIR filters applied to one signal in a single NPU call.
#[derive(Debug)]
pub struct FirBank {
    taps: usize,
    conv: BlockCorrelator,
}

impl FirBank {
    /// `filters[k]` are the taps `h_k[0..L]` of filter `k` (all the same length).
    pub fn new(npu: &Npu, filters: &[Vec<f32>]) -> Result<Self> {
        let taps = equal_lengths(filters, 1).ok_or_else(|| {
            Error::InvalidArgument("filters must be non-empty and of equal length".into())
        })?;
        // Convolution y[t] = Σ_j h[j]·x[t−j] is a correlation with h reversed.
        let reversed: Vec<Vec<f32>> = filters
            .iter()
            .map(|f| f.iter().rev().copied().collect())
            .collect();
        Ok(Self {
            taps,
            conv: BlockCorrelator::new(npu, &reversed)?,
        })
    }

    /// Number of filters.
    pub fn filters(&self) -> usize {
        self.conv.kernels
    }

    /// Filters `signal` with every filter. Row `k` of the result is filter
    /// `k`'s output, `signal.len()` samples (causal; the first `taps − 1`
    /// samples see zero history).
    pub fn apply(&self, signal: &[f32]) -> Result<Matrix> {
        if signal.is_empty() {
            return Err(Error::Shape("empty signal".into()));
        }
        let padded: Vec<f32> = std::iter::repeat_n(0.0, self.taps - 1)
            .chain(signal.iter().copied())
            .collect();
        Ok(self.conv.apply(&padded)?.transpose())
    }
}

/// Windowed-sinc FIR design (Hamming window).
pub mod design {
    use std::f64::consts::PI;

    fn hamming(n: usize, i: usize) -> f64 {
        0.54 - 0.46 * (2.0 * PI * i as f64 / (n - 1) as f64).cos()
    }

    fn sinc(x: f64) -> f64 {
        if x.abs() < 1e-12 {
            1.0
        } else {
            (PI * x).sin() / (PI * x)
        }
    }

    /// Low-pass filter with cut-off `cutoff` as a fraction of the sample rate (0–0.5).
    pub fn lowpass(cutoff: f32, taps: usize) -> Vec<f32> {
        let c = cutoff as f64;
        let mid = (taps as f64 - 1.0) / 2.0;
        let h: Vec<f64> = (0..taps)
            .map(|i| 2.0 * c * sinc(2.0 * c * (i as f64 - mid)) * hamming(taps, i))
            .collect();
        let sum: f64 = h.iter().sum();
        h.iter().map(|v| (v / sum) as f32).collect()
    }

    /// High-pass filter (spectral inversion of the low-pass; use an odd `taps`).
    pub fn highpass(cutoff: f32, taps: usize) -> Vec<f32> {
        let mut h = lowpass(cutoff, taps);
        h.iter_mut().for_each(|v| *v = -*v);
        h[taps / 2] += 1.0;
        h
    }

    /// Band-pass filter between `low` and `high` (fractions of the sample rate).
    pub fn bandpass(low: f32, high: f32, taps: usize) -> Vec<f32> {
        let lo = lowpass(low, taps);
        lowpass(high, taps)
            .iter()
            .zip(&lo)
            .map(|(a, b)| a - b)
            .collect()
    }
}

/// A template match found by [`Correlation::matches`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Match {
    /// Index of the template.
    pub template: usize,
    /// Sample offset where the template starts.
    pub offset: usize,
    /// Normalised correlation in `[−1, 1]` (1 = identical shape).
    pub score: f32,
}

/// Finds known shapes in signals by normalised cross-correlation. The
/// templates stay on the NPU, so the same matcher scans any number of signals.
#[derive(Debug)]
pub struct TemplateMatcher {
    len: usize,
    conv: BlockCorrelator,
}

/// Zero-mean, unit-norm copy (all zeros for a constant input).
fn normalised(v: &[f32]) -> Vec<f32> {
    let mean = v.iter().sum::<f32>() / v.len() as f32;
    let centred: Vec<f32> = v.iter().map(|x| x - mean).collect();
    let norm = centred.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        centred.iter().map(|x| x / norm).collect()
    } else {
        centred
    }
}

impl TemplateMatcher {
    /// Uploads `templates` (equal length ≥ 2).
    pub fn new(npu: &Npu, templates: &[Vec<f32>]) -> Result<Self> {
        let len = equal_lengths(templates, 2)
            .ok_or_else(|| Error::InvalidArgument("templates must have equal length ≥ 2".into()))?;
        let kernels: Vec<Vec<f32>> = templates.iter().map(|t| normalised(t)).collect();
        Ok(Self {
            len,
            conv: BlockCorrelator::new(npu, &kernels)?,
        })
    }

    /// Template length.
    pub fn template_len(&self) -> usize {
        self.len
    }

    /// Pearson correlation of every template with every window of `signal`,
    /// in one NPU call.
    pub fn scores(&self, signal: &[f32]) -> Result<Correlation> {
        if signal.len() < self.len {
            return Err(Error::InvalidArgument(format!(
                "signal of {} samples is shorter than the templates ({})",
                signal.len(),
                self.len
            )));
        }
        // Templates are zero-mean, so Σ t·(x − mean) = Σ t·x: the NPU computes
        // the raw sliding products and the window norms come from prefix sums.
        let raw = self.conv.apply(signal)?;
        let (mut s1, mut s2) = (vec![0.0f64], vec![0.0f64]);
        for &x in signal {
            s1.push(s1[s1.len() - 1] + x as f64);
            s2.push(s2[s2.len() - 1] + (x as f64).powi(2));
        }
        let n = self.len as f64;
        let mut scores = raw.transpose(); // templates × positions
        for row in scores.iter_rows_mut() {
            for (t, v) in row.iter_mut().enumerate() {
                let sum = s1[t + self.len] - s1[t];
                let var = (s2[t + self.len] - s2[t] - sum * sum / n).max(0.0);
                *v = if var > 1e-24 {
                    (*v as f64 / var.sqrt()).clamp(-1.0, 1.0) as f32
                } else {
                    0.0
                };
            }
        }
        Ok(Correlation {
            scores,
            template_len: self.len,
        })
    }
}

/// Correlation scores of a [`TemplateMatcher`] against one signal.
#[derive(Debug, Clone, PartialEq)]
pub struct Correlation {
    /// `templates × positions` scores in `[−1, 1]`.
    pub scores: Matrix,
    template_len: usize,
}

impl Correlation {
    /// Positions where a template's correlation is above `threshold` and a
    /// local maximum (peaks closer than the template length are merged),
    /// best first.
    pub fn matches(&self, threshold: f32) -> Vec<Match> {
        let mut found = Vec::new();
        for (template, s) in self.scores.iter_rows().enumerate() {
            let mut i = 0;
            while i < s.len() {
                if s[i] < threshold {
                    i += 1;
                    continue;
                }
                let end = (i + self.template_len).min(s.len());
                let (best, &score) = s[i..end]
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .unwrap_or((0, &s[i]));
                found.push(Match {
                    template,
                    offset: i + best,
                    score,
                });
                i += best + self.template_len;
            }
        }
        found.sort_by(|a, b| b.score.total_cmp(&a.score));
        found
    }
}
