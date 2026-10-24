use half::bf16;

use crate::device::Npu;
use crate::error::{Result, ensure_shape};
use crate::tensor::Tensor;

/// Q.ANT's photonic variant of a Kolmogorov-Arnold Network layer
/// (<https://arxiv.org/abs/2404.19756>), built on the periodic nonlinearity:
///
/// `y[b, j] = Σ_{i, k} tcos(ks[k] · x[b, i] + phis[j, i, k]) · ampls[j, i, k]`
///
/// Shapes: `x` `[batch, in]`, `phis` and `ampls` `[out, in, n_k]`, `ks` `[n_k]`.
/// Returns `[batch, out]`. Both the `k · x` products and the nonlinearity run on the NPU.
pub fn kan_layer(
    npu: &Npu,
    x: &Tensor,
    phis: &Tensor,
    ampls: &Tensor,
    ks: &Tensor,
) -> Result<Tensor> {
    let [nb, ni] = x.dims::<2>()?;
    let [no, ni2, nk] = phis.dims::<3>()?;
    ensure_shape!(ni == ni2, "kan_layer: x has {ni} inputs, phis expect {ni2}");
    ensure_shape!(
        phis.shape() == ampls.shape(),
        "kan_layer: phis {:?} and ampls {:?} must have the same shape",
        phis.shape(),
        ampls.shape()
    );
    ensure_shape!(
        ks.len() == nk,
        "kan_layer: ks has {} values, phis expect {nk}",
        ks.len()
    );

    // Lay everything out as [in, k, batch] so the NPU sees one flat elementwise op.
    let xd = x.data();
    let kd = ks.data();
    let mut xs_ikb = Vec::with_capacity(ni * nk * nb);
    let mut ks_ikb = Vec::with_capacity(ni * nk * nb);
    for i in 0..ni {
        for &k in kd {
            for b in 0..nb {
                xs_ikb.push(xd[b * ni + i]);
                ks_ikb.push(k);
            }
        }
    }
    let mut kx_ikb = vec![bf16::ZERO; xs_ikb.len()];
    npu.backend().mul(npu.id(), &xs_ikb, &ks_ikb, &mut kx_ikb)?;

    // [out, in, k, batch]: argument = k·x + phi, amplitude = ampl.
    let per_out = ni * nk * nb;
    let mut args = Vec::with_capacity(no * per_out);
    let mut amps = Vec::with_capacity(no * per_out);
    for j in 0..no {
        for ik in 0..ni * nk {
            let phi = phis.data()[j * ni * nk + ik].to_f32();
            let amp = ampls.data()[j * ni * nk + ik];
            for b in 0..nb {
                args.push(bf16::from_f32(kx_ikb[ik * nb + b].to_f32() + phi));
                amps.push(amp);
            }
        }
    }
    let mut res = vec![bf16::ZERO; args.len()];
    npu.backend()
        .periodic_nl(npu.id(), &args, &amps, &mut res)?;

    // Sum over (in, k), accumulated in f32, and return [batch, out].
    let mut out = vec![bf16::ZERO; nb * no];
    for j in 0..no {
        let chunk = &res[j * per_out..(j + 1) * per_out];
        for b in 0..nb {
            let s: f32 = (0..ni * nk).map(|ik| chunk[ik * nb + b].to_f32()).sum();
            out[b * no + j] = bf16::from_f32(s);
        }
    }
    Tensor::new(out, [nb, no])
}
