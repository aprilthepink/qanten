# Neural networks (`qanten::nn`)

PyTorch-style layers for inference. Layers with weights upload them to the NPU once
when constructed, so `forward` only moves activations. Names and tensor layouts match
`torch.nn`, so PyTorch `state_dict`s load directly.

## Building a model

```rust
use qanten::prelude::*;

fn main() -> qanten::Result<()> {
    let npu = Npu::simulator();
    let mut sd = nn::StateDict::new();
    sd.insert("0.weight", Tensor::rand_uniform([16, 3, 3, 3], -0.3, 0.3, 1));
    sd.insert("0.bias", Tensor::zeros([16]));
    sd.insert("4.weight", Tensor::rand_uniform([10, 16], -0.3, 0.3, 2));

    let model = nn::Sequential::new()
        .add(nn::Conv2d::load(&npu, &sd, "0", Conv2dParams::new().padding(1))?)
        .add(nn::ReLU)
        .add(nn::AdaptiveAvgPool2d::new(1))
        .add(nn::Flatten::default())
        .add(nn::Linear::load(&npu, &sd, "4")?)
        .add(nn::Softmax);

    let probs = model.forward(&Tensor::rand_uniform([4, 3, 32, 32], 0., 1., 3))?;
    println!("predicted classes: {:?}", probs.argmax_last());

    // Debugging: the output of every layer.
    for (layer, out) in model.layers().iter().zip(model.forward_trace(&Tensor::zeros([1, 3, 8, 8]))?) {
        println!("{:<20} {:?}", layer.name(), out.shape());
    }
    Ok(())
}
```

## Layers

| Layer | Construct | Loads from state dict |
|---|---|---|
| `Linear` | `new(npu, &weight, Option<bias>)` | `load(npu, sd, "fc")` → `fc.weight`, `fc.bias`? |
| `Conv2d` | `new(npu, &weight, bias, Conv2dParams)` | `load(npu, sd, prefix, params)` |
| `ConvTranspose2d` | `new(npu, &weight, bias, ConvTranspose2dParams)` | `load(npu, sd, prefix, params)` |
| `BatchNorm2d` | `new(npu, mean, var, weight, bias, eps)` | `load(npu, sd, prefix, eps)` → `running_mean`, `running_var`, `weight`?, `bias`? |
| `LayerNorm` | struct literal `{ weight, bias, eps }` | `load(sd, prefix, eps)` |
| `Kan` | `new(npu, phis, ampls, ks)` | `load(npu, sd, prefix)` → `phis`, `ampls`, `ks` |
| `MaxPool2d`, `AvgPool2d` | `new(kernel)` or `MaxPool2d(Pool2dParams)` | – |
| `AdaptiveAvgPool2d` | `new(output)` | – |
| `Flatten` | `Flatten::default()` (from dim 1) or `{ start_dim }` | – |
| `ReLU`, `Sigmoid`, `Tanh`, `Gelu`, `Silu`, `Softmax`, `LeakyReLU(slope)` | unit structs | – |
| `Lambda` | `Lambda::new("name", \|x\| ...)` | custom glue code |

`?` marks optional tensors. A missing required tensor gives `Error::MissingTensor("fc.weight")`.

`Module` is implemented for `Box<M>` and `Arc<M>`, and `Sequential` is itself a
`Module`, so models nest. All layers are `Send + Sync`: share one model across threads
with an `Arc`.

## Loading PyTorch weights (feature `safetensors`)

In Python:

```python
from safetensors.torch import save_file
save_file(model.state_dict(), "model.safetensors")
```

In Rust:

<!-- doccheck: ignore -->
```rust
let sd = qanten::nn::StateDict::load_safetensors("model.safetensors")?;
for name in sd.keys() { println!("{name}: {:?}", sd.get(name)?.shape()); }
```

Any float or integer dtype (`f32`, `f16`, `bf16`, `f64`, `i64`, …) converts to `bf16`.
`sd.sub("encoder")` returns the tensors under `encoder.` with the prefix stripped,
which helps when building sub-models. `save_safetensors` writes `bf16` files.

## Worked example: Q.ANT's digit recogniser

`crates/qanten/examples/digit_recognition.rs` runs Q.ANT's pretrained MNIST network
(784 → 133 → ReLU → 10 → softmax):

```sh
cargo run --release -p qanten --example digit_recognition --features safetensors
cargo run --release -p qanten --example digit_recognition --features safetensors -- --noise 0.2
```

On the full 10,000-image test set it scores 94.6 %, the same as an f32 numpy
reference. With simulated noise σ = 0.2 it drops to 92.5 %.
