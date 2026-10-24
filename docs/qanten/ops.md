# Operations (`qanten::ops`)

Functions that take `&Npu` run their heavy part on the NPU. Everything else runs on
the host CPU, in parallel. This is the same split as Q.ANT's own toolkit: the NPU
multiplies, the host shuffles data and does cheap elementwise work.

All ops batch, and layouts follow PyTorch (`NCHW`, `[out, in]` weights).

## NPU primitives

| Function | Shapes | Computes |
|---|---|---|
| `mul(npu, a, b)` | any, equal shapes | `a · b` elementwise |
| `periodic_nl(npu, u, v)` | any, equal shapes | `tcos(u) · v`, the photonic nonlinearity (`tcos` is the optical transfer function, exactly `cos` in the simulator) |
| `matvec(npu, m, x)` | `m [r, c]`, `x [..., c]` | `out[.., r] = Σ_c m[r, c] · x[.., c]` → `[..., r]` |
| `matmul(npu, a, b)` | `[m, k] @ [k, n]` | `[m, n]` |

```rust
use qanten::{Npu, Tensor, ops};

fn main() -> qanten::Result<()> {
    let npu = Npu::simulator();
    let a = Tensor::vector(&[0.0, 1.0, 2.0]);
    let b = Tensor::vector(&[2.0, 2.0, 2.0]);
    println!("{:?}", ops::mul(&npu, &a, &b)?.to_f32_vec());          // [0, 2, 4]
    println!("{:?}", ops::periodic_nl(&npu, &a, &b)?.to_f32_vec());  // 2·cos(a)
    let m = Tensor::from_f32(&[1., 2., 3., 4.], [2, 2])?;
    println!("{:?}", ops::matmul(&npu, &m, &m)?.to_f32_vec());       // [7, 10, 15, 22]
    Ok(())
}
```

## Linear layers

| Function | Shapes |
|---|---|
| `linear(npu, x, weight, bias)` | `x [..., in]` (any leading dims), `weight [out, in]`, `bias [out]` → `[..., out]` |
| `LinearWeights::upload(npu, weight)` then `.forward(x)` | same, but the weights stay on the NPU between calls |

Use `LinearWeights` (or `nn::Linear`) whenever the same weights are used more than
once: each call then only transfers the inputs and outputs.

## Convolutions

| Function | Shapes |
|---|---|
| `conv2d(npu, x, w, bias, Conv2dParams)` | `x [N, C, H, W]`, `w [OC, C/groups, KH, KW]`, `bias [OC]` |
| `conv_transpose2d(npu, x, w, bias, ConvTranspose2dParams)` | `x [N, C, H, W]`, `w [C, OC/groups, KH, KW]` |
| `ConvWeights::upload(npu, w, groups)` → `.forward(x, &params)` | resident-weights version of `conv2d` |
| `ConvTransposeWeights::upload(npu, w, groups)` → `.forward(x, &params)` | resident-weights version of `conv_transpose2d` |

Parameters use a builder. Every size accepts a `usize` (square) or `(h, w)`:

```rust
use qanten::ops::{Conv2dParams, ConvTranspose2dParams, Hw};

fn main() -> qanten::Result<()> {
    let p = Conv2dParams::new().stride(2).padding((1, 2)).dilation(1).groups(1);
    assert_eq!(p.output_size(Hw::new(32, 32), Hw::new(3, 3))?, Hw::new(16, 17));
    let t = ConvTranspose2dParams::new().stride(2).padding(1).output_padding(1);
    assert_eq!(t.output_size(Hw::new(16, 16), Hw::new(3, 3))?, Hw::new(32, 32));
    Ok(())
}
```

Supported: any stride, zero padding, dilation, groups (including depthwise), output
padding (must be smaller than stride or dilation, as in PyTorch).

How it works: the input is unfolded into one row per output position (im2row), the
weights are the matrix, and one batched matrix-vector product per group runs on the NPU.

## Normalisation

| Function | Where | Notes |
|---|---|---|
| `batch_norm2d(npu, x, mean, var, weight, bias, eps)` | NPU (scaling) + host | inference mode, running statistics; `weight`/`bias` optional |
| `layer_norm(x, weight, bias, eps)` | host | over the last dimension |

## Pooling (host)

| Function | Notes |
|---|---|
| `max_pool2d(x, Pool2dParams)` | padding is ignored when taking the max |
| `avg_pool2d(x, Pool2dParams)` | `count_include_pad` like PyTorch (default `true`) |
| `adaptive_avg_pool2d(x, out)`, `adaptive_max_pool2d(x, out)` | any input size → `out`, PyTorch's window rule |
| `global_avg_pool2d(x)` | `[N, C, H, W]` → `[N, C]` |

```rust
use qanten::{Tensor, ops::{self, Pool2dParams}};

fn main() -> qanten::Result<()> {
    let x = Tensor::rand_uniform([1, 3, 7, 7], 0., 1., 1);
    let a = ops::max_pool2d(&x, Pool2dParams::new(3).stride(2).padding(1))?;
    let b = ops::adaptive_avg_pool2d(&x, (2, 3))?;
    assert_eq!((a.shape(), b.shape()), (&[1, 3, 4, 4][..], &[1, 3, 2, 3][..]));
    Ok(())
}
```

## Activations and elementwise (host)

`relu`, `relu_inplace`, `leaky_relu(x, slope)`, `sigmoid`, `tanh`, `silu`,
`gelu` (tanh approximation), `softmax` (last dimension, numerically stable, batched),
`add`, `scale(x, factor)`, `add_bias(x, bias)` (last dimension, every row),
`add_channel_bias(x, bias)` (per channel of `NCHW`).

## KAN layer

`kan_layer(npu, x, phis, ampls, ks)` is Q.ANT's photonic Kolmogorov-Arnold layer:

```text
y[b, j] = Σ_{i, k} tcos(ks[k] · x[b, i] + phis[j, i, k]) · ampls[j, i, k]
```

with `x [batch, in]`, `phis`/`ampls` `[out, in, n_k]`, `ks [n_k]`. Both the `k·x`
products and the nonlinearity run on the NPU. `cargo run -p qanten --example kan`
shows a layer that represents `sin(x) + 0.5·cos(2x)` exactly.

## Errors

Shape mismatches return `Error::Shape` with both shapes in the message; bad
parameters (zero stride, too-large output padding) return `Error::InvalidArgument`.
Nothing panics on bad input.
