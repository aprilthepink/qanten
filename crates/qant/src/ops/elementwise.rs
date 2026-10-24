use half::bf16;
use rayon::prelude::*;

use crate::device::Npu;
use crate::error::{Result, ensure_shape};
use crate::tensor::Tensor;

/// Elementwise product `a * b` on the NPU. Shapes must match.
pub fn mul(npu: &Npu, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    ensure_shape!(
        a.shape() == b.shape(),
        "mul: shapes differ: {:?} vs {:?}",
        a.shape(),
        b.shape()
    );
    let mut out = vec![bf16::ZERO; a.len()];
    npu.backend().mul(npu.id(), a.data(), b.data(), &mut out)?;
    Tensor::new(out, a.shape())
}

/// The photonic periodic nonlinearity on the NPU: `tcos(u) * v` elementwise,
/// where `tcos` is the cosine-shaped optical transfer function
/// (exactly `cos` on the simulator).
pub fn periodic_nl(npu: &Npu, u: &Tensor, v: &Tensor) -> Result<Tensor> {
    ensure_shape!(
        u.shape() == v.shape(),
        "periodic_nl: shapes differ: {:?} vs {:?}",
        u.shape(),
        v.shape()
    );
    let mut out = vec![bf16::ZERO; u.len()];
    npu.backend()
        .periodic_nl(npu.id(), u.data(), v.data(), &mut out)?;
    Tensor::new(out, u.shape())
}

/// Elementwise `a + b` on the host. Shapes must match.
pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    ensure_shape!(
        a.shape() == b.shape(),
        "add: shapes differ: {:?} vs {:?}",
        a.shape(),
        b.shape()
    );
    let data = a
        .data()
        .iter()
        .zip(b.data())
        .map(|(x, y)| bf16::from_f32(x.to_f32() + y.to_f32()))
        .collect();
    Tensor::new(data, a.shape())
}

/// Multiplies every element by `factor` on the host.
pub fn scale(x: &Tensor, factor: f32) -> Tensor {
    x.map(|v| v * factor)
}

/// Adds a 1-D `bias` along the last dimension (`[.., n] + [n]`), on the host.
pub fn add_bias(x: &Tensor, bias: &Tensor) -> Result<Tensor> {
    let n = *x.shape().last().unwrap_or(&1);
    ensure_shape!(
        bias.rank() == 1 && bias.len() == n,
        "add_bias: bias shape {:?} does not match last dimension of {:?}",
        bias.shape(),
        x.shape()
    );
    let b = bias.to_f32_vec();
    let mut data = x.data().to_vec();
    data.par_chunks_mut(n.max(1)).for_each(|row| {
        for (v, bb) in row.iter_mut().zip(&b) {
            *v = bf16::from_f32(v.to_f32() + bb);
        }
    });
    Tensor::new(data, x.shape())
}

/// Adds a per-channel bias to an `NCHW` tensor (`bias` has `C` elements), on the host.
pub fn add_channel_bias(x: &Tensor, bias: &Tensor) -> Result<Tensor> {
    let [_, c, h, w] = x.dims::<4>()?;
    ensure_shape!(
        bias.rank() == 1 && bias.len() == c,
        "add_channel_bias: bias shape {:?} does not match {c} channels",
        bias.shape()
    );
    let b = bias.to_f32_vec();
    let plane = (h * w).max(1);
    let mut data = x.data().to_vec();
    data.par_chunks_mut(plane).enumerate().for_each(|(i, p)| {
        let bb = b[i % c];
        for v in p {
            *v = bf16::from_f32(v.to_f32() + bb);
        }
    });
    Tensor::new(data, x.shape())
}
