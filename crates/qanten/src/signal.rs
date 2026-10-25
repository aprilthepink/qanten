//! Signal processing as matrix products on the NPU.
//!
//! Each tool keeps one constant matrix resident on the NPU and pushes every
//! frame or window of the signal through it in a single batched call:
//!
//! | Type | Matrix on the NPU | Vectors streamed |
//! |---|---|---|
//! | [`Dft`] | `cos` / `sin` rows of the DFT | frames |
//! | [`Stft`] | windowed DFT rows | overlapping frames → spectrogram |
//! | [`FirBank`] | every filter at every shift of a block | signal blocks |
//! | [`correlate`] | every template at every shift of a block | signal blocks → matches |
//!
//! **Precision.** bfloat16 gives roughly 40–50 dB of dynamic range: plenty
//! for spectrograms, detection, feature extraction and monitoring, not for
//! producing hi-fi audio.
//!
//! ```
//! # fn main() -> qanten::Result<()> {
//! use qanten::signal::Dft;
//! let npu = qanten::Npu::simulator();
//! let n = 64;
//! let tone: Vec<f32> = (0..n).map(|t| (std::f32::consts::TAU * 5.0 * t as f32 / n as f32).cos()).collect();
//! let spectrum = Dft::new(&npu, n)?.forward(&tone)?;
//! assert_eq!(spectrum.peak_bin(0), 5);
//! # Ok(()) }
//! ```

use std::f64::consts::TAU;

use crate::device::Npu;
use crate::error::{Error, Result};
use crate::ops::LinearWeights;
use crate::tensor::Tensor;

/// Runs `rows` (row-major, `len` each) through `w`, normalising each row to
/// unit peak first so bf16 sees O(1) values, then undoing the scale.
fn apply_scaled(w: &LinearWeights, rows: &[f32], len: usize) -> Result<Vec<f32>> {
    let k = rows.len() / len.max(1);
    let mut scales = Vec::with_capacity(k);
    let mut scaled = Vec::with_capacity(rows.len());
    for r in rows.chunks(len) {
        let m = r.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        let m = if m > 0.0 { m } else { 1.0 };
        scales.push(m);
        scaled.extend(r.iter().map(|v| v / m));
    }
    let y = w
        .forward(&Tensor::from_f32(&scaled, [k, len])?)?
        .to_f32_vec();
    let out = y.len() / k.max(1);
    Ok(y.chunks(out.max(1))
        .zip(&scales)
        .flat_map(|(r, &m)| r.iter().map(move |v| v * m))
        .collect())
}

/// Complex spectra of one or more frames: `bins` values per frame.
#[derive(Debug, Clone, PartialEq)]
pub struct Spectrum {
    /// Frequency bins per frame.
    pub bins: usize,
    /// Real parts, `frames × bins`.
    pub re: Vec<f32>,
    /// Imaginary parts, `frames × bins`.
    pub im: Vec<f32>,
}

impl Spectrum {
    /// Number of frames.
    pub fn frames(&self) -> usize {
        self.re.len() / self.bins.max(1)
    }

    /// Magnitudes `|X|`, `frames × bins`.
    pub fn magnitude(&self) -> Vec<f32> {
        self.re
            .iter()
            .zip(&self.im)
            .map(|(r, i)| (r * r + i * i).sqrt())
            .collect()
    }

    /// Power in decibels `20·log10(|X| + 1e-12)`, `frames × bins`.
    pub fn db(&self) -> Vec<f32> {
        self.magnitude()
            .iter()
            .map(|m| 20.0 * (m + 1e-12).log10())
            .collect()
    }

    /// Bin with the largest magnitude in `frame`.
    pub fn peak_bin(&self, frame: usize) -> usize {
        let m = self.magnitude();
        m[frame * self.bins..(frame + 1) * self.bins]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map_or(0, |(i, _)| i)
    }
}

/// Discrete Fourier transform of real frames of length `n` (bins `0..=n/2`).
#[derive(Debug)]
pub struct Dft {
    n: usize,
    bins: usize,
    /// Rows: `bins` cosine rows then `bins` (negated) sine rows, `n` columns.
    matrix: LinearWeights,
}

/// `[cos(2πkt/n) ; −sin(2πkt/n)]` rows times an optional window.
/// The angle uses `(k·t) mod n` in integers, so it is exact for any `n`.
fn dft_rows(n: usize, bins: usize, window: Option<&[f32]>) -> Vec<f32> {
    let mut m = vec![0.0f32; 2 * bins * n];
    for k in 0..bins {
        for t in 0..n {
            let w = window.map_or(1.0, |w| w[t] as f64);
            let a = TAU * ((k * t) % n) as f64 / n as f64;
            m[k * n + t] = (a.cos() * w) as f32;
            m[(bins + k) * n + t] = (-a.sin() * w) as f32;
        }
    }
    m
}

impl Dft {
    /// A DFT for frames of length `n`, uploaded to `npu`.
    pub fn new(npu: &Npu, n: usize) -> Result<Self> {
        if n < 2 {
            return Err(Error::InvalidArgument(
                "DFT length must be at least 2".into(),
            ));
        }
        let bins = n / 2 + 1;
        let matrix = LinearWeights::upload(
            npu,
            &Tensor::from_f32(&dft_rows(n, bins, None), [2 * bins, n])?,
        )?;
        Ok(Self { n, bins, matrix })
    }

    /// Frame length.
    pub fn len(&self) -> usize {
        self.n
    }

    /// Always `false` (length ≥ 2).
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Transforms `frames × n` samples (one or more frames, row-major).
    pub fn forward(&self, frames: &[f32]) -> Result<Spectrum> {
        if frames.is_empty() || !frames.len().is_multiple_of(self.n) {
            return Err(Error::Shape(format!(
                "{} samples is not a whole number of {}-sample frames",
                frames.len(),
                self.n
            )));
        }
        let y = apply_scaled(&self.matrix, frames, self.n)?;
        let mut re = Vec::with_capacity(y.len() / 2);
        let mut im = Vec::with_capacity(y.len() / 2);
        for row in y.chunks(2 * self.bins) {
            re.extend_from_slice(&row[..self.bins]);
            im.extend_from_slice(&row[self.bins..]);
        }
        Ok(Spectrum {
            bins: self.bins,
            re,
            im,
        })
    }
}

/// Short-time Fourier transform: spectrogram of a long signal.
#[derive(Debug)]
pub struct Stft {
    frame: usize,
    hop: usize,
    bins: usize,
    matrix: LinearWeights,
}

/// Periodic Hann window of length `n`.
pub fn hann(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (0.5 - 0.5 * (TAU * i as f64 / n as f64).cos()) as f32)
        .collect()
}

impl Stft {
    /// Frames of `frame` samples every `hop` samples, Hann-windowed.
    pub fn new(npu: &Npu, frame: usize, hop: usize) -> Result<Self> {
        if frame < 2 || hop == 0 {
            return Err(Error::InvalidArgument(
                "frame must be ≥ 2 and hop ≥ 1".into(),
            ));
        }
        let bins = frame / 2 + 1;
        let w = hann(frame);
        let matrix = LinearWeights::upload(
            npu,
            &Tensor::from_f32(&dft_rows(frame, bins, Some(&w)), [2 * bins, frame])?,
        )?;
        Ok(Self {
            frame,
            hop,
            bins,
            matrix,
        })
    }

    /// Frequency of `bin` in Hz for a given sample rate.
    pub fn bin_frequency(&self, bin: usize, sample_rate: f32) -> f32 {
        bin as f32 * sample_rate / self.frame as f32
    }

    /// Spectrogram of `signal`: every full frame, all in one NPU call.
    pub fn spectrogram(&self, signal: &[f32]) -> Result<Spectrum> {
        if signal.len() < self.frame {
            return Err(Error::Shape(format!(
                "signal of {} samples is shorter than one frame ({})",
                signal.len(),
                self.frame
            )));
        }
        let frames = (signal.len() - self.frame) / self.hop + 1;
        let mut rows = Vec::with_capacity(frames * self.frame);
        for f in 0..frames {
            rows.extend_from_slice(&signal[f * self.hop..f * self.hop + self.frame]);
        }
        let y = apply_scaled(&self.matrix, &rows, self.frame)?;
        let mut re = Vec::with_capacity(frames * self.bins);
        let mut im = Vec::with_capacity(frames * self.bins);
        for row in y.chunks(2 * self.bins) {
            re.extend_from_slice(&row[..self.bins]);
            im.extend_from_slice(&row[self.bins..]);
        }
        Ok(Spectrum {
            bins: self.bins,
            re,
            im,
        })
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
    matrix: LinearWeights,
}

impl BlockCorrelator {
    fn new(npu: &Npu, kernels: &[Vec<f32>]) -> Result<Self> {
        let len = kernels.first().map_or(0, Vec::len);
        let k = kernels.len();
        let block = len.clamp(32, 256);
        let width = block + len - 1;
        // Row (i·K + k): kernel k shifted to output i of the block.
        let mut m = vec![0.0f32; block * k * width];
        for i in 0..block {
            for (kk, g) in kernels.iter().enumerate() {
                let row = &mut m[(i * k + kk) * width..(i * k + kk + 1) * width];
                row[i..i + len].copy_from_slice(g);
            }
        }
        let matrix = LinearWeights::upload(npu, &Tensor::from_f32(&m, [block * k, width])?)?;
        Ok(Self {
            len,
            kernels: k,
            block,
            matrix,
        })
    }

    /// `count × K` outputs, row-major, for `count = signal.len() − L + 1`.
    fn apply(&self, signal: &[f32]) -> Result<Vec<f32>> {
        let count = signal.len() + 1 - self.len;
        let width = self.block + self.len - 1;
        let blocks = count.div_ceil(self.block);
        let mut segments = vec![0.0f32; blocks * width];
        for (b, seg) in segments.chunks_mut(width).enumerate() {
            let start = b * self.block;
            let end = (start + width).min(signal.len());
            seg[..end - start].copy_from_slice(&signal[start..end]);
        }
        let mut y = apply_scaled(&self.matrix, &segments, width)?;
        y.truncate(count * self.kernels);
        Ok(y)
    }
}

/// Several FIR filters applied to one signal in a single NPU call.
#[derive(Debug)]
pub struct FirBank {
    taps: usize,
    filters: usize,
    conv: BlockCorrelator,
}

impl FirBank {
    /// `filters[k]` are the taps `h_k[0..L]` of filter `k` (all the same length).
    pub fn new(npu: &Npu, filters: &[Vec<f32>]) -> Result<Self> {
        let taps = filters.first().map_or(0, Vec::len);
        if taps == 0 || filters.iter().any(|f| f.len() != taps) {
            return Err(Error::InvalidArgument(
                "filters must be non-empty and of equal length".into(),
            ));
        }
        // Convolution y[t] = Σ_j h[j]·x[t−j] is a correlation with h reversed.
        let reversed: Vec<Vec<f32>> = filters
            .iter()
            .map(|f| f.iter().rev().copied().collect())
            .collect();
        Ok(Self {
            taps,
            filters: filters.len(),
            conv: BlockCorrelator::new(npu, &reversed)?,
        })
    }

    /// Number of filters.
    pub fn filters(&self) -> usize {
        self.filters
    }

    /// Filters `signal` with every filter. Returns `filters × signal.len()`
    /// outputs (causal; the first `taps − 1` samples see zero history).
    pub fn apply(&self, signal: &[f32]) -> Result<Vec<Vec<f32>>> {
        if signal.is_empty() {
            return Err(Error::Shape("empty signal".into()));
        }
        let mut padded = vec![0.0f32; self.taps - 1];
        padded.extend_from_slice(signal);
        let y = self.conv.apply(&padded)?; // [T, K]
        let mut out = vec![Vec::with_capacity(signal.len()); self.filters];
        for row in y.chunks(self.filters) {
            for (k, v) in row.iter().enumerate() {
                out[k].push(*v);
            }
        }
        Ok(out)
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
        let hi = lowpass(high, taps);
        hi.iter().zip(&lo).map(|(a, b)| a - b).collect()
    }
}

/// A template match found by [`correlate`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Match {
    /// Index of the template.
    pub template: usize,
    /// Sample offset where the template starts.
    pub offset: usize,
    /// Normalised correlation in `[−1, 1]` (1 = identical shape).
    pub score: f32,
}

/// Normalised cross-correlation of every template against every position of
/// `signal`, in one NPU call. Returns `templates × positions` scores in
/// `[−1, 1]` (Pearson correlation of the template with each window).
pub fn correlate(npu: &Npu, signal: &[f32], templates: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
    let len = templates.first().map_or(0, Vec::len);
    if len < 2 || templates.iter().any(|t| t.len() != len) || signal.len() < len {
        return Err(Error::InvalidArgument(
            "templates must have equal length ≥ 2, no longer than the signal".into(),
        ));
    }
    let normalise = |v: &[f32]| -> Vec<f32> {
        let m = v.iter().sum::<f32>() / v.len() as f32;
        let c: Vec<f32> = v.iter().map(|x| x - m).collect();
        let n = c.iter().map(|x| x * x).sum::<f32>().sqrt();
        if n > 0.0 {
            c.iter().map(|x| x / n).collect()
        } else {
            c
        }
    };
    // Templates are zero-mean, so Σ t·(x − mean) = Σ t·x: the NPU computes the
    // raw sliding products and the window norms come from prefix sums.
    let kernels: Vec<Vec<f32>> = templates.iter().map(|t| normalise(t)).collect();
    let raw = BlockCorrelator::new(npu, &kernels)?.apply(signal)?;
    let (mut s1, mut s2) = (
        vec![0.0f64; signal.len() + 1],
        vec![0.0f64; signal.len() + 1],
    );
    for (i, &x) in signal.iter().enumerate() {
        s1[i + 1] = s1[i] + x as f64;
        s2[i + 1] = s2[i] + (x as f64) * (x as f64);
    }
    let k = templates.len();
    let y: Vec<f32> = raw
        .chunks(k)
        .enumerate()
        .flat_map(|(t, row)| {
            let sum = s1[t + len] - s1[t];
            let var = (s2[t + len] - s2[t] - sum * sum / len as f64).max(0.0);
            let norm = var.sqrt();
            row.iter().map(move |&v| {
                if norm > 1e-12 {
                    (v as f64 / norm).clamp(-1.0, 1.0) as f32
                } else {
                    0.0
                }
            })
        })
        .collect();
    let mut out = vec![Vec::with_capacity(y.len() / k); k];
    for row in y.chunks(k) {
        for (t, v) in row.iter().enumerate() {
            out[t].push(*v);
        }
    }
    Ok(out)
}

/// Positions where a template's correlation is above `threshold` and a local
/// maximum (closer peaks than the template length are merged), best first.
pub fn find_matches(scores: &[Vec<f32>], template_len: usize, threshold: f32) -> Vec<Match> {
    let mut found = Vec::new();
    for (t, s) in scores.iter().enumerate() {
        let mut i = 0;
        while i < s.len() {
            if s[i] >= threshold {
                let end = (i + template_len).min(s.len());
                let (best, &score) = s[i..end]
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .unwrap_or((0, &s[i]));
                found.push(Match {
                    template: t,
                    offset: i + best,
                    score,
                });
                i = i + best + template_len;
            } else {
                i += 1;
            }
        }
    }
    found.sort_by(|a, b| b.score.total_cmp(&a.score));
    found
}
