//! Periodic layers for neural fields: Fourier features and SIREN's sine,
//! both evaluated on the NPU's native periodic nonlinearity.
//!
//! The NPU computes `tcos(u)·v`, a cosine, directly in optics. Its input is
//! bf16, which cannot hold a large phase accurately, so phases are reduced
//! modulo 2π on the host first (see [`Npu::cos`]).

use std::f64::consts::TAU;

use super::{Module, Sequential, StateDict};
use crate::compute::Matrix;
use crate::device::Npu;
use crate::error::{Error, Result, ensure_shape};
use crate::tensor::Tensor;

/// Random Fourier features (Tancik et al. 2020):
/// `γ(x) = [cos(2π B x), sin(2π B x)]` for a fixed `B` (`[features, dims]`).
///
/// They let a plain MLP learn high-frequency detail from low-dimensional
/// coordinates (pixels, 3-D points). The phases `2π B x` are computed on the
/// host in f64 (the input has only 2 or 3 dimensions, and a bf16 product
/// would lose accuracy as the phase grows) and the cosines and sines run on
/// the NPU.
#[derive(Debug, Clone)]
pub struct FourierFeatures {
    npu: Npu,
    b: Matrix,
}

impl FourierFeatures {
    /// Features for a given frequency matrix `b` (`[features, dims]`, cycles per unit).
    pub fn new(npu: &Npu, b: Matrix) -> Self {
        Self {
            npu: npu.clone(),
            b,
        }
    }

    /// `features` frequencies drawn from `N(0, σ²)` per dimension. Larger
    /// `sigma` lets the network fit finer detail (and overfit noise sooner).
    pub fn gaussian(npu: &Npu, features: usize, dims: usize, sigma: f32, seed: u64) -> Self {
        Self::new(
            npu,
            Matrix::random_normal(features, dims, seed).map(|v| v * sigma),
        )
    }

    /// Loads `{prefix}.b` (as saved by [`train::FourierFeatures`](crate::train::FourierFeatures)).
    pub fn load(npu: &Npu, sd: &StateDict, prefix: &str) -> Result<Self> {
        let key = if prefix.is_empty() {
            "b".to_owned()
        } else {
            format!("{prefix}.b")
        };
        Ok(Self::new(npu, Matrix::from_tensor(sd.get(&key)?)?))
    }

    /// The frequency matrix `B`.
    pub fn matrix(&self) -> &Matrix {
        &self.b
    }

    /// Input dimensions `d`.
    pub fn in_features(&self) -> usize {
        self.b.cols()
    }

    /// Output size `2 · features`: all cosines, then all sines.
    pub fn out_features(&self) -> usize {
        2 * self.b.rows()
    }

    /// `2π B x` per row of `coords`, reduced to `[0, 2π)` in f64.
    pub(crate) fn phases(&self, coords: &Matrix) -> Result<Vec<f32>> {
        ensure_shape!(
            coords.cols() == self.in_features(),
            "FourierFeatures: expected {} input dimensions, got {}",
            self.in_features(),
            coords.cols()
        );
        Ok(coords
            .iter_rows()
            .flat_map(|x| {
                self.b.iter_rows().map(move |b| {
                    let p: f64 = b.iter().zip(x).map(|(b, x)| *b as f64 * *x as f64).sum();
                    (TAU * p).rem_euclid(TAU) as f32
                })
            })
            .collect())
    }

    /// Encodes f32 coordinates `[n, dims]` into `[n, 2·features]`. Prefer
    /// this over [`forward`](Module::forward) for raw coordinates: a
    /// [`Tensor`] is bf16, which rounds coordinates before the phase is
    /// formed. The features are fixed, so a dataset can be encoded once.
    pub fn encode(&self, coords: &Matrix) -> Result<Matrix> {
        let phases = self.phases(coords)?;
        let m = self.b.rows();
        let cos = self.npu.cos(&phases)?;
        let sin = self.npu.sin(&phases)?;
        Ok(Matrix::from_fn(coords.rows(), 2 * m, |r, c| {
            if c < m {
                cos[r * m + c]
            } else {
                sin[r * m + c - m]
            }
        }))
    }
}

impl Module for FourierFeatures {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.encode(&Matrix::from_tensor(x)?)?.to_tensor()
    }

    fn name(&self) -> &'static str {
        "FourierFeatures"
    }
}

/// SIREN's activation `sin(ω₀ x)` on the NPU's periodic nonlinearity
/// (`sin(t) = cos(t − π/2)`, phase reduced on the host).
#[derive(Debug, Clone)]
pub struct Sine {
    npu: Npu,
    /// Frequency `ω₀` (SIREN uses 30).
    pub omega0: f32,
}

impl Sine {
    /// `sin(omega0 · x)` on `npu`.
    pub fn new(npu: &Npu, omega0: f32) -> Self {
        Self {
            npu: npu.clone(),
            omega0,
        }
    }
}

impl Module for Sine {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let scaled: Vec<f32> = x.to_f32_vec().iter().map(|v| self.omega0 * v).collect();
        Tensor::from_f32(&self.npu.sin(&scaled)?, x.shape().to_vec())
    }

    fn name(&self) -> &'static str {
        "Sine"
    }
}

impl Sequential {
    /// Loads a SIREN (optionally behind Fourier features) saved with
    /// [`train::Model::state_dict`](crate::train::Model::state_dict): layer
    /// `i` is a [`Linear`](super::Linear) if `{i}.weight` exists,
    /// [`FourierFeatures`] if `{i}.b` exists, and a [`Sine`] with `omega0`
    /// otherwise (activations have no parameters).
    pub fn load_siren(npu: &Npu, sd: &StateDict, omega0: f32) -> Result<Self> {
        let last = sd
            .keys()
            .filter_map(|k| k.split_once('.')?.0.parse::<usize>().ok())
            .max()
            .ok_or_else(|| Error::Format("state dict has no numbered layers".into()))?;
        let mut model = Self::new();
        for i in 0..=last {
            let prefix = i.to_string();
            if sd.contains(&format!("{i}.weight")) {
                model.push(super::Linear::load(npu, sd, &prefix)?);
            } else if sd.contains(&format!("{i}.b")) {
                model.push(FourierFeatures::load(npu, sd, &prefix)?);
            } else {
                model.push(Sine::new(npu, omega0));
            }
        }
        Ok(model)
    }
}
