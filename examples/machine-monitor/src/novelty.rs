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
//!    with [`Matrix::matmul_on`] (`Placement::auto` keeps `Zᵀ` on the NPU and
//!    streams the 90 columns of `Z`), verified on the CPU with
//!    [`Matrix::product_error`]. The main directions `U` and their variances
//!    `λ` (90% of the variance) come from `Matrix64::symmetric_eigen`, exact
//!    on the CPU: a 92 × 92 matrix used once has no reuse, so by the compute
//!    module's own rule it is CPU work.
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

use qanten::compute::{AutoPolicy, Matrix, Matrix64, NpuMatrix, Placement};
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
    /// Whether `Placement::auto` put the covariance product on the NPU.
    pub covariance_on_npu: bool,
    /// Relative error of that product, from `Matrix::product_error`.
    pub covariance_error: f64,
    /// SPE and T² thresholds (99.5th percentiles on the validation set).
    pub thresholds: [f32; 2],
}

/// The `q`-quantile of `values`.
fn quantile(mut values: Vec<f32>, q: f64) -> f32 {
    values.sort_by(f32::total_cmp);
    let idx = ((values.len() as f64 * q) as usize).min(values.len().saturating_sub(1));
    values.get(idx).copied().unwrap_or(1.0).max(1e-6)
}

/// Fits the detector on normalised training features (one row per window)
/// and sets its thresholds on a validation set. Adds `novelty.matrix` and
/// `novelty.thresholds` to `sd`.
pub fn fit(
    npu: &Npu,
    train: &Matrix,
    validation: &Matrix,
    sd: &mut StateDict,
) -> Result<FitReport> {
    let n = train.rows();
    let zt = train.transpose();

    // Covariance: 92×n · n×92. Large enough and reused enough for the NPU.
    let covariance_on_npu = AutoPolicy::default().use_npu(NUM_FEATURES, n, NUM_FEATURES);
    let gram = zt.matmul_on(train, Placement::auto(npu))?;
    let covariance_error = gram.product_error(&zt, train, 4);
    let cov: Matrix64 = gram.cast::<f64>().map(|v| v / n as f64);

    // Principal directions: exact symmetric eigendecomposition on the CPU.
    let total: f64 = (0..NUM_FEATURES).map(|i| cov[(i, i)]).sum();
    let (values, vectors) = cov.symmetric_eigen()?;
    let mut components = 0;
    let mut explained = 0.0;
    for v in &values {
        if explained >= EXPLAINED {
            break;
        }
        explained += v.max(0.0) / total;
        components += 1;
    }

    // Rows 0..F: P = I − U·Uᵀ (residual).  Rows F..F+k: uᵢᵀ / √λᵢ (whitened scores).
    let u = vectors.slice_rows(0..components);
    let mut stacked = Matrix64::identity(NUM_FEATURES);
    let projection = u.transpose().matmul(&u)?;
    stacked
        .as_mut_slice()
        .iter_mut()
        .zip(projection.as_slice())
        .for_each(|(p, uu)| *p -= uu);
    for (row, lambda) in u.iter_rows().zip(&values) {
        let inv = 1.0 / lambda.max(1e-9).sqrt();
        stacked.push_row(&row.iter().map(|v| v * inv).collect::<Vec<_>>())?;
    }
    let matrix: Matrix = stacked.cast();

    // Thresholds from known conditions the model was not fitted on.
    let stats = NoveltyDetector::new(npu, matrix.clone(), [1.0, 1.0])?.statistics(validation)?;
    let thresholds = [0, 1].map(|k| quantile(stats.iter().map(|s| s[k]).collect(), QUANTILE));

    sd.insert("novelty.matrix", matrix.to_tensor()?);
    sd.insert("novelty.thresholds", Tensor::vector(&thresholds));
    Ok(FitReport {
        components,
        explained,
        covariance_on_npu,
        covariance_error,
        thresholds,
    })
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
    pub fn statistics(&self, normalized: &Matrix) -> Result<Vec<[f32; 2]>> {
        let out = self.resident.apply_batch(normalized)?;
        let sum_sq = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>();
        Ok(out
            .iter_rows()
            .map(|row| {
                let (residual, scores) = row.split_at(NUM_FEATURES);
                [sum_sq(residual), sum_sq(scores)]
            })
            .collect())
    }

    /// Novelty score per row: `max(SPE / δ, T² / τ)` (above 1 = unusual for
    /// every known condition).
    pub fn score(&self, normalized: &Matrix) -> Result<Vec<f32>> {
        let [d, t] = self.thresholds;
        Ok(self
            .statistics(normalized)?
            .into_iter()
            .map(|[q, t2]| (q / d).max(t2 / t))
            .collect())
    }

    /// NPU self-test: runs `vectors` random vectors through the resident
    /// matrix and returns the relative error measured by
    /// [`Matrix::product_error`]. On analog hardware, a rising value means
    /// the device needs recalibration.
    pub fn self_check(&self, vectors: usize, seed: u64) -> Result<f64> {
        let x = Matrix::random_normal(vectors, NUM_FEATURES, seed);
        let y = self.resident.apply_batch(&x)?;
        Ok(y.product_error(&x, &self.host.transpose(), 4))
    }
}
