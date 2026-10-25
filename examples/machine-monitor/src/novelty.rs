//! Novelty detection: "this doesn't look like anything I was trained on".
//!
//! The classifier can only name the six conditions it knows. A new kind of
//! fault (here: pump cavitation) gets forced into one of them. This module
//! catches it with the two standard statistics of PCA process monitoring:
//!
//! - **SPE** (squared prediction error, "Q"): how much of a feature vector
//!   lies *outside* the directions the training data varies in. Catches new
//!   kinds of pattern.
//! - **Hotelling's T²**: how far *along* those directions it is, in units of
//!   their normal spread. Catches familiar patterns at unfamiliar strength.
//!
//! How it uses `qanten::compute`:
//!
//! 1. **Fit (once, on the training features):** the covariance `ZᵀZ / n`
//!    with [`compute::gemm`] (`Placement::Auto` keeps `Zᵀ` on the NPU and
//!    streams the 90 columns of `Z`), verified on the CPU with
//!    [`compute::check_product`]. The main directions `U` and their variances
//!    `λ` (90% of the variance) come from an exact eigendecomposition on the
//!    CPU: a 92 × 92 matrix used once has no reuse, so by the module's own
//!    rule it is CPU work.
//! 2. **One resident matrix for both statistics:** `P = I − U·Uᵀ` (residual)
//!    stacked on `W = Λ^(−1/2)·Uᵀ` (whitened scores), uploaded once as an
//!    [`NpuMatrix`]. Its per-row scaling copes with the very different row
//!    magnitudes of the two blocks.
//! 3. **Score (every batch):** one batched NPU call gives `P·z` and `W·z` for
//!    every feature vector; `SPE = ‖P·z‖²`, `T² = ‖W·z‖²`. Each is divided by
//!    its 99.5th percentile on known conditions, and the score is the larger
//!    of the two: above 1 means "unusual".
//!
//! `P·z` is used rather than `z − U·(Uᵀ·z)` because the subtraction would
//! lose most of bf16's three digits when the residual is small.

use qanten::compute::{self, AutoPolicy, Matrix, NpuMatrix, Placement};
use qanten::nn::StateDict;
use qanten::{Npu, Result, Tensor};

use crate::features::NUM_FEATURES;

/// Share of the variance the PCA model keeps.
const EXPLAINED: f64 = 0.90;
/// Quantile of known-condition statistics that becomes each threshold.
const QUANTILE: f64 = 0.995;

/// What fitting found, for logging.
#[derive(Debug, Clone)]
pub struct FitReport {
    /// Principal components kept.
    pub components: usize,
    /// Share of variance they explain.
    pub explained: f64,
    /// Whether `Placement::Auto` put the covariance product on the NPU.
    pub covariance_on_npu: bool,
    /// Relative error of that product, from `check_product`.
    pub covariance_error: f64,
    /// SPE and T² thresholds (99.5th percentiles on the validation set).
    pub thresholds: [f32; 2],
}

/// Fits the detector on normalised training features (`rows × NUM_FEATURES`)
/// and sets its thresholds on a validation set. Adds `novelty.matrix` and
/// `novelty.thresholds` to `sd`.
pub fn fit(npu: &Npu, train: &[f32], validation: &[f32], sd: &mut StateDict) -> Result<FitReport> {
    let n = train.len() / NUM_FEATURES;
    let z = Matrix::new(n, NUM_FEATURES, train.to_vec())?;
    let zt = z.transpose();

    // Covariance: 90×n · n×90. Large enough and reused enough for the NPU.
    let covariance_on_npu = AutoPolicy::default().use_npu(NUM_FEATURES, n, NUM_FEATURES);
    let gram = compute::gemm(npu, &zt, &z, Placement::Auto)?;
    let covariance_error = compute::check_product(&zt, &z, &gram, 4);
    let cov: Vec<f64> = gram
        .as_slice()
        .iter()
        .map(|v| *v as f64 / n as f64)
        .collect();

    // Principal directions: exact symmetric eigendecomposition on the CPU.
    let total: f64 = (0..NUM_FEATURES).map(|i| cov[i * NUM_FEATURES + i]).sum();
    let (values, vectors) = symmetric_eigen(&cov, NUM_FEATURES);
    let mut components = 0;
    let mut explained = 0.0;
    for v in &values {
        if explained >= EXPLAINED {
            break;
        }
        explained += v.max(0.0) / total;
        components += 1;
    }

    // Rows 0..F: P = I − Σ uᵢ·uᵢᵀ.  Rows F..F+k: uᵢᵀ / √λᵢ.
    let mut m = Matrix::zeros(NUM_FEATURES + components, NUM_FEATURES);
    for i in 0..NUM_FEATURES {
        m[(i, i)] = 1.0;
    }
    for (c, (u, lambda)) in vectors.iter().zip(&values).take(components).enumerate() {
        let inv = 1.0 / lambda.max(1e-9).sqrt();
        for i in 0..NUM_FEATURES {
            for j in 0..NUM_FEATURES {
                m[(i, j)] -= (u[i] * u[j]) as f32;
            }
            m[(NUM_FEATURES + c, i)] = (u[i] * inv) as f32;
        }
    }

    // Thresholds from known conditions the model was not fitted on.
    let mut detector = NoveltyDetector::new(npu, m.clone(), [1.0, 1.0])?;
    let stats = detector.statistics(validation)?;
    let mut thresholds = [0.0f32; 2];
    for (k, t) in thresholds.iter_mut().enumerate() {
        let mut v: Vec<f32> = stats.iter().map(|s| s[k]).collect();
        v.sort_by(f32::total_cmp);
        let idx = ((v.len() as f64 * QUANTILE) as usize).min(v.len().saturating_sub(1));
        *t = v.get(idx).copied().unwrap_or(1.0).max(1e-6);
    }
    detector.thresholds = thresholds;

    sd.insert("novelty.matrix", m.to_tensor()?);
    sd.insert("novelty.thresholds", Tensor::from_f32(&thresholds, [2])?);
    Ok(FitReport {
        components,
        explained,
        covariance_on_npu,
        covariance_error,
        thresholds,
    })
}

/// Eigenvalues (largest first) and unit eigenvectors of a symmetric `n × n`
/// matrix, by cyclic Jacobi rotations in f64.
fn symmetric_eigen(a: &[f64], n: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    let mut a = a.to_vec();
    let mut v = vec![0.0f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    for _sweep in 0..100 {
        let off: f64 = (0..n)
            .flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j)))
            .map(|(i, j)| a[i * n + j].powi(2))
            .sum();
        let diag: f64 = (0..n).map(|i| a[i * n + i].powi(2)).sum();
        if off <= 1e-24 * diag.max(f64::MIN_POSITIVE) {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = a[p * n + q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (akp, akq) = (a[k * n + p], a[k * n + q]);
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let (apk, aqk) = (a[p * n + k], a[q * n + k]);
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let (vkp, vkq) = (v[k * n + p], v[k * n + q]);
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| a[j * n + j].total_cmp(&a[i * n + i]));
    let values = order.iter().map(|&i| a[i * n + i]).collect();
    let vectors = order
        .iter()
        .map(|&j| (0..n).map(|k| v[k * n + j]).collect())
        .collect();
    (values, vectors)
}

/// The deployed detector: its matrix resident on one NPU.
#[derive(Debug)]
pub struct NoveltyDetector {
    resident: NpuMatrix,
    host: Matrix,
    thresholds: [f32; 2],
}

impl NoveltyDetector {
    fn new(npu: &Npu, host: Matrix, thresholds: [f32; 2]) -> Result<Self> {
        Ok(Self {
            resident: NpuMatrix::upload(npu, &host)?,
            host,
            thresholds,
        })
    }

    /// From a state dict written by [`fit`]; `None` for models trained without it.
    pub fn load(npu: &Npu, sd: &StateDict) -> Result<Option<Self>> {
        let (Some(m), Some(t)) = (
            sd.get_opt("novelty.matrix"),
            sd.get_opt("novelty.thresholds"),
        ) else {
            return Ok(None);
        };
        let t = t.to_f32_vec();
        Ok(Some(Self::new(npu, Matrix::from_tensor(m)?, [t[0], t[1]])?))
    }

    /// `[SPE, T²]` per row of normalised features, from one batched NPU call.
    pub fn statistics(&self, normalized: &[f32]) -> Result<Vec<[f32; 2]>> {
        let z = Matrix::new(
            normalized.len() / NUM_FEATURES,
            NUM_FEATURES,
            normalized.to_vec(),
        )?;
        let out = self.resident.apply_batch(&z)?;
        Ok((0..out.rows())
            .map(|i| {
                let (residual, scores) = out.row(i).split_at(NUM_FEATURES);
                [
                    residual.iter().map(|v| v * v).sum(),
                    scores.iter().map(|v| v * v).sum(),
                ]
            })
            .collect())
    }

    /// Novelty score per row: `max(SPE / δ, T² / τ)` (above 1 = unusual for
    /// every known condition).
    pub fn score(&self, normalized: &[f32]) -> Result<Vec<f32>> {
        Ok(self
            .statistics(normalized)?
            .into_iter()
            .map(|[q, t]| (q / self.thresholds[0]).max(t / self.thresholds[1]))
            .collect())
    }

    /// NPU self-test: runs `vectors` random vectors through the resident
    /// matrix and returns the relative error measured by
    /// [`compute::check_product`]. On analog hardware, a rising value means
    /// the device needs recalibration.
    pub fn self_check(&self, vectors: usize, seed: u64) -> Result<f64> {
        let x = Matrix::random_normal(vectors, NUM_FEATURES, seed);
        let y = self.resident.apply_batch(&x)?;
        Ok(compute::check_product(&x, &self.host.transpose(), &y, 4))
    }
}
