# qanten

**A batteries-included Rust SDK for Q.ANT photonic NPUs.**

Q.ANT's [open-source toolkit](https://github.com/Q-ANT-GmbH/qant_native_computing_toolkit)
is written in Rust, but only ships C and Python bindings and is not on crates.io.
This workspace gives Rust its own first-class interface, built on the same driver:

| Crate | What it is |
|---|---|
| [`qanten`](crates/qanten) | Safe high-level library: device handles, tensors, ops, NN layers, model loading, simulator |
| [`qanten-sys`](crates/qanten-sys) | Raw FFI to `libqant_native_computing_driver`, loaded at runtime |
| [`qanten-graphics`](crates/qanten-graphics) | CPU-side 3D workloads on the NPU: ocean waves, skinning, blendshapes, light-probe baking, DCT |
| [`qanten-cli`](crates/qanten-cli) | `qanten` command: `doctor`, `devices`, `info`, `sensors`, `perf`, `bench`, `selftest` |

> Unofficial community project, not affiliated with or supported by Q.ANT GmbH.

**Documentation:** guides for every module are in [`docs/`](docs/README.md), API reference on docs.rs.

![Gerstner ocean computed on the NPU](docs/images/ocean.png)

## Why use this instead of the C or Python bindings

- **Builds anywhere.** The driver is `dlopen`ed at runtime, so `cargo build`
  needs no driver, no git submodules, no bindgen and no Python.
- **One binary for hardware and laptops.** `Npu::auto(0)` uses the NPU when
  the driver is installed and the built-in simulator otherwise.
- **Safe resource handling.** NPUs are initialised on open and released on
  drop. Device memory is freed on drop. Every pointer handed to the driver is
  length-checked first.
- **Mistakes caught at compile time.** `DeviceBuffer<Weights>` has no
  `download()`, and operand roles cannot be swapped.
- **PyTorch weights load directly.** `StateDict::load_safetensors` plus
  layers whose names and layouts match `torch.nn`.
- **More of PyTorch covered.** Batching for every op, conv dilation and
  groups (depthwise), transposed-conv output padding, adaptive pooling of any
  size, numerically stable batched softmax, layer norm, GELU/SiLU/tanh.
- **A simulator for testing.** Exact by default, with optional Gaussian
  noise, per-NPU memory limits and several NPUs.

## Quick start

```toml
[dependencies]
qanten = { version = "0.1", features = ["safetensors"] }
```

```rust
use qanten::prelude::*;

fn main() -> qanten::Result<()> {
    let npu = Npu::auto(0)?; // hardware if available, simulator otherwise

    // Single ops
    let x = Tensor::from_f32(&[1., 2., 3., 4., 5., 6.], [2, 3])?;
    let w = Tensor::rand_uniform([4, 3], -1., 1., 42);
    let y = ops::linear(&npu, &x, &w, None)?; // [2, 4], computed on the NPU

    // A model from a PyTorch state_dict (weights stay resident on the NPU)
    let sd = nn::StateDict::load_safetensors("model.safetensors")?;
    let model = nn::Sequential::new()
        .add(nn::Linear::load(&npu, &sd, "fc1")?)
        .add(nn::ReLU)
        .add(nn::Linear::load(&npu, &sd, "fc2")?)
        .add(nn::Softmax);
    let probs = model.forward(&Tensor::zeros([8, 784]))?;
    println!("{:?}", probs.argmax_last());
    Ok(())
}
```

Export from PyTorch with:

```python
from safetensors.torch import save_file
save_file(model.state_dict(), "model.safetensors")
```

## Examples

```sh
cargo run --example quickstart
cargo run --example cnn --release
cargo run --example kan
cargo run --example device_memory
cargo run --example digit_recognition --features safetensors
```

`digit_recognition` runs Q.ANT's own pretrained MNIST network
(784 → 133 → 10). Results on the full 10,000-image test set:

| Backend | Accuracy |
|---|---|
| numpy f32 reference | 94.6 % |
| `qanten` simulator (bf16) | 94.6 % |
| simulator, noise σ = 0.05 | 94.5 % |
| simulator, noise σ = 0.2 | 92.5 % |
| simulator, noise σ = 0.5 | 80.3 % |

## Choosing a backend

| Call | Behaviour |
|---|---|
| `Npu::open(id)` | Hardware only. Errors if the driver is missing. |
| `Npu::simulator()` | Shared, exact simulator. |
| `Npu::simulator_with(SimConfig { noise_std, memory_limit_bytes, num_npus, .. })` | Configured simulator. |
| `Npu::auto(id)` | Hardware if available, else simulator (logs a warning). Check `npu.is_simulated()`. |
| `Npu::from_env(id)` | Chosen by `QANTEN_BACKEND=driver\|sim\|auto`. |
| `Npu::with_backend(Arc<dyn Backend>, id)` | Your own `Backend` implementation. |

Environment variables:

| Variable | Effect |
|---|---|
| `QANTEN_DRIVER_LIB` | Path to `libqant_native_computing_driver.so` if it is not on the loader path |
| `QANTEN_BACKEND` | Backend for `Npu::from_env` |

## Cargo features of `qanten`

| Feature | Default | Adds |
|---|---|---|
| `driver` | yes | Hardware backend (runtime-loaded, nothing needed at build time) |
| `safetensors` | | `StateDict::{load,save}_safetensors` (f32/f16/bf16/f64/int → bf16) |
| `ndarray` | | `Tensor` ⇄ `ndarray::ArrayD` |
| `serde` | | `Serialize` for device info, sensors, perf counters |
| `full` | | All of the above |

## What runs where

As in Q.ANT's toolkit, the NPU does the multiplications. The host CPU does
data reshuffling and cheap elementwise work:

| On the NPU | On the host |
|---|---|
| `mul`, `periodic_nl` (tcos · v), `matvec`, `matmul`, `linear`, `conv2d`, `conv_transpose2d` (im2row → batched matvec), `batch_norm2d` scaling, `kan_layer` | bias, activations, softmax, pooling, layer norm, im2row and layout changes |

All tensors are `bf16`, the NPU's native format. Constructors and
`to_f32_vec()` convert for you.

## Coming from the Python or C API

| Q.ANT Python / C | `qanten` |
|---|---|
| `generic.init_npu(0)` / `release_npu(0)` | `let npu = Npu::open(0)?;` (released on drop) |
| `generic.get_available_npus()` | `qanten::devices()` |
| `native.mul_elementwise` | `ops::mul` |
| `native.calc_scaled_periodic_nl_fprop` | `ops::periodic_nl` |
| `ai.linear_fprop` + `ai.add_bias_fprop` | `ops::linear(&npu, &x, &w, Some(&b))` |
| `ai.linear_fprop_preloaded` | `ops::LinearWeights` / `nn::Linear` |
| `ai.conv_fprop` / `upload_conv_weights` / `conv_fprop_preloaded_weights` | `ops::conv2d` / `ops::ConvWeights` / `nn::Conv2d` |
| `ai.conv_transpose_fprop` (+ preloaded) | `ops::conv_transpose2d` / `nn::ConvTranspose2d` |
| `ai.batchnorm2d_fprop` | `ops::batch_norm2d` / `nn::BatchNorm2d` |
| `ai.maxpool2d_fprop`, `avgpool2d_fprop`, `adaptive_*` | `ops::max_pool2d`, `avg_pool2d`, `adaptive_*_pool2d` |
| `ai.relu_fprop`, `sigmoid_fprop`, `softmax_fprop` | `ops::relu`, `sigmoid`, `softmax` |
| `ai.calc_kan_layer_fprop` | `ops::kan_layer` / `nn::Kan` |
| `generic.allocate` / `upload` / `upload_into` / `download` | `DeviceBuffer::<Weights \| Features>::{alloc, from_tensor, upload, download}` |
| `info.get_driver_info`, `get_sensor_info`, `get_version_info` | `npu.driver_version()`, `npu.device_info()` |
| `info.get_perf_counter` / `reset_perf_counter` | `npu.perf_counters()` / `npu.reset_perf_counters()` |
| `info.setup_logging` | `qanten::setup_driver_logging` |

Behaviour that differs from upstream v2.4.0:

- In-place `add_bias` only added the bias to the first row of a batch.
  `qanten` adds it to every row.
- A failed device free in `Drop` built an error but never reported it.
  `qanten` logs it through `tracing`.
- Upstream rejects batched input for softmax, pooling, batch norm and
  transposed conv, and does not support dilation, groups or output padding.
  `qanten` supports all of these.

## Testing without hardware

```sh
cargo test --workspace --all-features   # simulator-backed tests

# The real dlopen + FFI path, against a mock driver that exports the driver's C symbols:
cargo build -p qanten-mock-driver
QANTEN_DRIVER_LIB=$PWD/target/debug/libqant_native_computing_driver.so \
  cargo test -p qanten --test driver_ffi
```

On a machine with a Q.ANT card, run `qanten doctor` and then `qanten selftest`,
which runs every op on the NPU and compares the result with the simulator.

## Status

The FFI layer follows the driver ABI used by Q.ANT's toolkit v2.4.0. It has
been tested against the mock driver, **not yet on Q.ANT hardware**. Reports
from people with hardware are very welcome, especially `qanten selftest` and
`qanten bench` output.

## License

Apache-2.0, the same as Q.ANT's toolkit. See [LICENSE](LICENSE) and
[NOTICE](NOTICE) for attribution.
