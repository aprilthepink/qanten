# Getting started

## Install

```toml
[dependencies]
qanten = "0.1"                                     # core library
qanten-graphics = "0.1"                            # optional: 3D workloads
# qanten = { version = "0.1", features = ["full"] } # + safetensors, ndarray, serde
```

Building needs **no** Q.ANT driver, no git submodules and no Python. The driver
library is loaded when the program runs. Minimum Rust version: 1.88.

## Your first program

```rust
use qanten::prelude::*;

fn main() -> qanten::Result<()> {
    // The NPU if Q.ANT's driver is installed, otherwise the built-in simulator.
    let npu = Npu::auto(0)?;
    println!("running on {}", npu.backend().name());

    let x = Tensor::from_f32(&[1., 2., 3., 4., 5., 6.], [2, 3])?; // 2 inputs of size 3
    let w = Tensor::from_f32(&[1., 0., 0., 0., 1., 1.], [2, 3])?; // [out, in]
    let y = ops::linear(&npu, &x, &w, None)?;                     // computed on the NPU
    assert_eq!(y.to_f32_vec(), vec![1.0, 5.0, 4.0, 11.0]);
    Ok(())
}
```

## Hardware or simulator?

| You want | Call |
|---|---|
| Real hardware only (fail if missing) | `Npu::open(0)` |
| Hardware if present, else simulator | `Npu::auto(0)` |
| Always the simulator | `Npu::simulator()` |
| Decided at run time by an env var | `Npu::from_env(0)` with `QANTEN_BACKEND=driver\|sim\|auto` |

Details: [NPUs and backends](qanten/npu.md).

## Environment variables

| Variable | Effect |
|---|---|
| `QANTEN_DRIVER_LIB` | Path to `libqant_native_computing_driver.so` if it isn't on the loader path |
| `QANTEN_BACKEND` | Backend picked by `Npu::from_env` |
| `QANTEN_SIM` | Makes the `qanten` CLI use the simulator (same as `--sim`) |

## Cargo features of `qanten`

| Feature | Default | Adds |
|---|---|---|
| `driver` | yes | Hardware backend (loaded at run time) |
| `safetensors` | | `StateDict::load_safetensors` / `save_safetensors` |
| `numpy` | | `.npy` / `.npz` load and save |
| `ndarray` | | `Tensor` ⇄ `ndarray::ArrayD` |
| `serde` | | `Serialize` for device info, sensors, perf counters |
| `full` | | all of the above |

## Precision in one paragraph

The NPU computes in **bfloat16**: 8 bits of mantissa, about 3 significant digits.
Results are typically within ~0.5 % of the largest value in an output. That is plenty
for neural networks and for the graphics workloads in `qanten-graphics`, which are
arranged to keep values small. Don't use it where you need exact arithmetic.

## Next steps

- Run a real model: [Neural networks](qanten/nn.md) and `cargo run -p qanten --example digit_recognition --features safetensors`.
- 3D work: [qanten-graphics](graphics/README.md).
- Check a machine: `qanten doctor`, then `qanten selftest` ([CLI](cli.md)).
