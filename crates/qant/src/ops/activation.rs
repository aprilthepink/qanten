use half::bf16;
use rayon::prelude::*;

use crate::error::Result;
use crate::tensor::Tensor;

fn par_map(x: &Tensor, f: impl Fn(f32) -> f32 + Sync) -> Tensor {
    let data: Vec<bf16> = x
        .data()
        .par_iter()
        .map(|v| bf16::from_f32(f(v.to_f32())))
        .collect();
    #[allow(clippy::expect_used)]
    Tensor::new(data, x.shape()).expect("same length as input")
}

/// `max(x, 0)`.
pub fn relu(x: &Tensor) -> Tensor {
    par_map(x, |v| v.max(0.0))
}

/// `max(x, 0)` in place.
pub fn relu_inplace(x: &mut Tensor) {
    x.data_mut()
        .par_iter_mut()
        .for_each(|v| *v = v.max(bf16::ZERO));
}

/// `x` if positive, `slope * x` otherwise.
pub fn leaky_relu(x: &Tensor, slope: f32) -> Tensor {
    par_map(x, |v| if v > 0.0 { v } else { slope * v })
}

/// `1 / (1 + e^-x)`.
pub fn sigmoid(x: &Tensor) -> Tensor {
    par_map(x, |v| 1.0 / (1.0 + (-v).exp()))
}

/// Hyperbolic tangent.
pub fn tanh(x: &Tensor) -> Tensor {
    par_map(x, f32::tanh)
}

/// `x * sigmoid(x)` (a.k.a. swish).
pub fn silu(x: &Tensor) -> Tensor {
    par_map(x, |v| v / (1.0 + (-v).exp()))
}

/// GELU, tanh approximation.
pub fn gelu(x: &Tensor) -> Tensor {
    const C: f32 = 0.797_884_6; // sqrt(2/pi)
    par_map(x, |v| {
        0.5 * v * (1.0 + (C * (v + 0.044_715 * v * v * v)).tanh())
    })
}

/// Softmax over the last dimension, computed in `f32` and numerically stable.
/// Works for any batch shape (`[.., n]`).
pub fn softmax(x: &Tensor) -> Result<Tensor> {
    let n = (*x.shape().last().unwrap_or(&1)).max(1);
    let mut data = x.data().to_vec();
    data.par_chunks_mut(n).for_each(|row| {
        let max = row
            .iter()
            .map(|v| v.to_f32())
            .fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = row.iter().map(|v| (v.to_f32() - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        for (o, e) in row.iter_mut().zip(exps) {
            *o = bf16::from_f32(e / sum);
        }
    });
    Tensor::new(data, x.shape())
}
