# Training (`qanten::train`)

Train models with the NPU in the loop: forward and backward matrix products run on the
NPU, the optimizer runs on the host.

```rust
use qanten::train::{self, Adam, Dense, Model, Tanh};
use qanten::{Npu, Tensor};

fn main() -> qanten::Result<()> {
    let npu = Npu::auto(0)?;
    let mut model = Model::new()
        .add(Dense::new(&npu, 2, 32, 1)?)   // in, out, seed
        .add(Tanh::default())
        .add(Dense::new(&npu, 32, 2, 2)?);
    let mut opt = Adam::new(0.02);

    let x = Tensor::rand_uniform([64, 2], -1., 1., 3);
    let labels: Vec<usize> = x.to_f32_vec().chunks(2).map(|p| usize::from(p[0] * p[1] > 0.0)).collect();
    for _ in 0..100 {
        let logits = model.forward(&x)?;
        let (_loss, grad) = train::cross_entropy(&logits, &labels)?;
        model.backward(&grad)?;
        model.clip_grad_norm(1.0);
        model.step(&mut opt)?;
    }
    println!("accuracy {:.2}", train::accuracy(&model.forward(&x)?, &labels));

    // Deploy: parameters are named "{layer}.{param}", like PyTorch's nn.Sequential.
    let sd = model.state_dict()?;
    let _layer = qanten::nn::Linear::load(&npu, &sd, "0")?;
    Ok(())
}
```

## What's available

| Item | Notes |
|---|---|
| `Dense::new(npu, in, out, seed)` / `from_weights` | fully connected layer, PyTorch initialisation |
| `Kan::new(npu, in, out, &ks, seed)` | Q.ANT's photonic KAN layer; phases and amplitudes are learned, frequencies `ks` fixed |
| `Relu`, `Tanh`, `Sigmoid` | activations |
| `Sine::on_npu(npu, ω₀)` / `Sine::new(ω₀)` | SIREN's `sin(ω₀·x)` on the NPU's periodic nonlinearity / on the host |
| `Dense::siren(npu, in, out, ω₀, first, seed)` | SIREN initialisation: `±1/in` for the first layer, `±√(6/in)/ω₀` after |
| `siren(npu, &[in, hidden…, out], ω₀, seed)` | a whole SIREN; load it for inference with `nn::Sequential::load_siren` |
| `FourierFeatures::new(nn::FourierFeatures)` | fixed `[cos(2πBx), sin(2πBx)]` with an input gradient; `B` is saved as buffer `b` |
| `Model` | `add`, `forward`, `backward`, `step(&mut opt)`, `clip_grad_norm`, `num_params`, `state_dict` |
| `Sgd::new(lr, momentum)`, `Adam::new(lr)` | optimizers; `Adam` has `weight_decay` for AdamW |
| `mse`, `cross_entropy`, `accuracy` | losses return `(loss, ∂loss/∂prediction)` |
| `Layer`, `Optimizer` traits | implement your own |

## Where the work runs

| Layer | Forward | Input gradient | Weight gradient |
|---|---|---|---|
| `Dense` | `x·Wᵀ` on the NPU | `g·W` on the NPU | `gᵀ·x` on the NPU |
| `Kan` | `tcos(k·x+φ)·a` on the NPU | `−a·k·sin(…)` on the NPU | `cos(…)`, `−a·sin(…)` on the NPU |
| `Sine::on_npu` | `sin(ω₀x)` on the NPU | `ω₀·cos(ω₀x)` on the NPU | – |
| `FourierFeatures` | `cos`, `sin` on the NPU (phases in f64 on the host) | `2π·Bᵀ(g_sin·cos − g_cos·sin)` on the host | – |
| activations, losses, optimizer | host | host | host |

The sines in the KAN gradients come from the same photonic nonlinearity, using
`−sin θ = cos(θ + π/2)`.

## Precision: what makes this work on analog hardware

- **f32 master weights.** Parameters and optimizer state live in f32 on the host;
  only a bf16 copy goes to the NPU after each step. Small updates aren't lost to bf16
  rounding.
- **Per-tensor gradient scaling.** Gradients are often tiny (≈ 1/batch size). Before
  a gradient goes to the NPU it is scaled to unit range, and the result is scaled
  back on the host. Without it, absolute noise on the NPU outputs drowns the gradient:
  in the `train_spirals` example on a simulated NPU with noise σ = 0.05, test accuracy
  goes from 55.7 % (unscaled) to 100 % (scaled).
- **Gradient clipping.** `model.clip_grad_norm(max)` rescales all gradients when their
  combined norm exceeds `max`. On a noisy NPU, SGD with learning rate 0.1 and momentum
  0.9 trains to a low loss, then diverges around step 130; with clipping it stays stable
  (see `tests/train.rs`).

## Hardware-aware training

Train against `Npu::simulator_with(SimConfig { noise_std, .. })` to make a model
robust to analog imprecision before deploying it. On a real card, train on the card
itself: the model then adapts to that device's actual behaviour.

## Example

```sh
cargo run --release -p qanten --example train_spirals
```

Two-spirals classification (2,594 parameters), trained once on an exact NPU and once
on a noisy one; both reach 100 % test accuracy, and the trained weights are exported into
`nn` inference layers for the test.

## Neural fields (SIREN, Fourier features)

Networks that map coordinates to values (images, signed distance fields) need
high-frequency functions of a 2- or 3-D input, which a plain ReLU MLP learns very
slowly. Two standard fixes, both built on the NPU's periodic nonlinearity:

- **SIREN** (`train::siren`): sine activations with a special initialisation.
- **Fourier features** (`nn::FourierFeatures`): a fixed random encoding
  `[cos(2πBx), sin(2πBx)]` in front of an ordinary ReLU MLP. The features have no
  parameters, so encode the dataset once with `encode` (in f32) and train on the result.

The NPU's input is bf16, which cannot hold a large phase, so every phase is reduced
modulo 2π on the host before the cosine runs on the NPU. For Fourier features the
phase `2πBx` itself is computed on the host in f64: with 2–3 input dimensions it is
trivial, and a bf16 product would lose accuracy as the phase grows.

```rust
use qanten::train::{self, Adam};
use qanten::{Npu, Tensor};

fn main() -> qanten::Result<()> {
    let npu = Npu::simulator();
    // Fit y = sin(12x) on [-1, 1] with a small SIREN.
    let xs: Vec<f32> = (0..128).map(|i| -1.0 + i as f32 / 63.5).collect();
    let ys: Vec<f32> = xs.iter().map(|x| (12.0 * x).sin()).collect();
    let (x, y) = (Tensor::from_f32(&xs, [128, 1])?, Tensor::from_f32(&ys, [128, 1])?);
    let mut model = train::siren(&npu, &[1, 32, 32, 1], 30.0, 1)?;
    let mut opt = Adam::new(2e-3);
    for _ in 0..100 {
        let (_, grad) = train::mse(&model.forward(&x)?, &y)?;
        model.backward(&grad)?;
        model.step(&mut opt)?;
    }
    let (loss, _) = train::mse(&model.forward(&x)?, &y)?;
    assert!(loss < 0.05);
    // Inference: Linear + nn::Sine layers from the state dict.
    let _net = qanten::nn::Sequential::load_siren(&npu, &model.state_dict()?, 30.0)?;
    Ok(())
}
```

`qanten-graphics` has an example that fits an image with all three approaches and a
2-D signed distance field with a SIREN, all trained on the NPU (see
[neural fields](../graphics/neural-fields.md)).
