//! Elementwise float functions on the NPU.

use std::f64::consts::{FRAC_PI_2, PI, TAU};

use crate::device::Npu;
use crate::error::{Error, Result};
use crate::ops;
use crate::tensor::Tensor;

impl Npu {
    /// `a[i] · b[i]` on the NPU.
    pub fn mul(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        if a.len() != b.len() {
            return Err(Error::Shape(format!(
                "mul: lengths {} and {} differ",
                a.len(),
                b.len()
            )));
        }
        if a.is_empty() {
            return Ok(Vec::new());
        }
        let n = [a.len()];
        Ok(ops::mul(self, &Tensor::from_f32(a, n)?, &Tensor::from_f32(b, n)?)?.to_f32_vec())
    }

    /// `cos(x[i])` on the NPU's native periodic nonlinearity (exact cos on
    /// the simulator; on hardware the optical transfer curve, so accuracy
    /// depends on calibration). Arguments are reduced to `[−π, π)` on the
    /// host in f64 first, so large ones keep their accuracy; about `±0.006`
    /// absolute error remains from the bf16 phase.
    pub fn cos(&self, x: &[f32]) -> Result<Vec<f32>> {
        self.shifted_cos(x, 0.0)
    }

    /// `sin(x[i]) = cos(x[i] − π/2)` on the NPU; see [`cos`](Self::cos).
    pub fn sin(&self, x: &[f32]) -> Result<Vec<f32>> {
        self.shifted_cos(x, -FRAC_PI_2)
    }

    fn shifted_cos(&self, x: &[f32], phase: f64) -> Result<Vec<f32>> {
        if x.is_empty() {
            return Ok(Vec::new());
        }
        let reduced: Vec<f32> = x
            .iter()
            .map(|&v| {
                let t = v as f64 + phase;
                (t - TAU * ((t + PI) / TAU).floor()) as f32
            })
            .collect();
        let n = [x.len()];
        Ok(ops::periodic_nl(self, &Tensor::from_f32(&reduced, n)?, &Tensor::ones(n))?.to_f32_vec())
    }
}
