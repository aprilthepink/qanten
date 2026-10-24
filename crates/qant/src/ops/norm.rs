use half::bf16;

use crate::device::Npu;
use crate::error::{Result, ensure_shape};
use crate::tensor::Tensor;

/// Inference-mode batch normalisation of an `NCHW` tensor
/// (PyTorch's `F.batch_norm` with running statistics).
///
/// `y = (x - mean) / sqrt(var + eps) * weight + bias`. The scaling runs on the NPU.
/// `weight` and `bias` default to ones and zeros.
pub fn batch_norm2d(
    npu: &Npu,
    x: &Tensor,
    running_mean: &Tensor,
    running_var: &Tensor,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    eps: f32,
) -> Result<Tensor> {
    let [_, c, h, w] = x.dims::<4>()?;
    let check = |t: &Tensor, name: &str| -> Result<Vec<f32>> {
        ensure_shape!(
            t.len() == c,
            "batch_norm2d: {name} has {} elements, input has {c} channels",
            t.len()
        );
        Ok(t.to_f32_vec())
    };
    let mean = check(running_mean, "running_mean")?;
    let var = check(running_var, "running_var")?;
    let gamma = weight
        .map(|t| check(t, "weight"))
        .transpose()?
        .unwrap_or_else(|| vec![1.0; c]);
    let beta = bias
        .map(|t| check(t, "bias"))
        .transpose()?
        .unwrap_or_else(|| vec![0.0; c]);
    let plane = h * w;

    let mut centered = Vec::with_capacity(x.len());
    let mut scale = Vec::with_capacity(x.len());
    for (i, v) in x.data().iter().enumerate() {
        let ch = (i / plane.max(1)) % c;
        centered.push(bf16::from_f32(v.to_f32() - mean[ch]));
        scale.push(bf16::from_f32(gamma[ch] / (var[ch] + eps).sqrt()));
    }
    let centered = Tensor::new(centered, x.shape())?;
    let scale = Tensor::new(scale, x.shape())?;
    let mut y = super::mul(npu, &centered, &scale)?;
    for (i, v) in y.data_mut().iter_mut().enumerate() {
        let ch = (i / plane.max(1)) % c;
        *v = bf16::from_f32(v.to_f32() + beta[ch]);
    }
    Ok(y)
}

/// Layer normalisation over the last dimension (host CPU).
pub fn layer_norm(
    x: &Tensor,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    eps: f32,
) -> Result<Tensor> {
    let n = (*x.shape().last().unwrap_or(&1)).max(1);
    for (t, name) in [(weight, "weight"), (bias, "bias")] {
        if let Some(t) = t {
            ensure_shape!(
                t.len() == n,
                "layer_norm: {name} has {} elements, expected {n}",
                t.len()
            );
        }
    }
    let gamma = weight.map(Tensor::to_f32_vec);
    let beta = bias.map(Tensor::to_f32_vec);
    let mut data = x.data().to_vec();
    for row in data.chunks_mut(n) {
        let vals: Vec<f32> = row.iter().map(|v| v.to_f32()).collect();
        let mean = vals.iter().sum::<f32>() / n as f32;
        let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for (j, (o, v)) in row.iter_mut().zip(vals).enumerate() {
            let mut y = (v - mean) * inv;
            if let Some(g) = &gamma {
                y *= g[j];
            }
            if let Some(b) = &beta {
                y += b[j];
            }
            *o = bf16::from_f32(y);
        }
    }
    Tensor::new(data, x.shape())
}
