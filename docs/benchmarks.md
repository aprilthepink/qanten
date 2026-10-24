# Benchmarks

Two [criterion](https://docs.rs/criterion) suites. Both pick the backend from
`QANTEN_BACKEND` (`driver`, `sim`, `auto`), so the same code measures the simulator
today and a real card later.

```sh
cargo bench -p qanten --features onnx        # ops, layers, models
cargo bench -p qanten-graphics               # 3D workloads: NPU path vs CPU path
cargo bench -p qanten --features onnx -- matvec   # only one group
```

## Comparing hardware with the simulator

criterion stores results and can compare runs:

```sh
QANTEN_BACKEND=sim    cargo bench -p qanten-graphics -- --save-baseline sim
QANTEN_BACKEND=driver cargo bench -p qanten-graphics -- --baseline sim
```

The second command prints, for every benchmark, how much faster or slower the card
is than the simulator. HTML reports go to `target/criterion/`.

## What's measured

**`qanten` (`benches/core.rs`)**

| Group | Benchmarks |
|---|---|
| `matvec` | 128², 512², 1024² matrices × batch 32–64, weights streamed per call vs resident on the NPU |
| `elementwise` | `mul` and `periodic_nl` on 64 K values |
| `conv2d` | 16 → 32 channels, 3×3, 32×32 image, resident weights |
| `models` | Q.ANT's 784–133–10 MLP (batch 64), TinyGPT prefill (32 tokens) and one cached decoding step, one training step, the ONNX CNN fixture |

**`qanten-graphics` (`benches/graphics.rs`)**: each workload twice, once through the
NPU formulation and once through the plain CPU path.

## Simulator baseline

Medians on the built-in simulator (4 CPU threads, release build). These show what the
host side costs and give a baseline to compare hardware against; they say nothing about
NPU speed.

| Benchmark | Simulator |
|---|---|
| matvec 1024×1024 × 64, streamed / resident | 20.0 / 19.2 ms (≈ 3.4 G MAC/s) |
| matvec 128×128 × 32 | 0.27 ms |
| mul / periodic_nl, 64 K values | 0.10 / 0.19 ms |
| conv2d 16→32 ch, 3×3, 32×32 | 1.7 ms |
| MLP 784–133–10, batch 64 | 2.2 ms |
| TinyGPT prefill 32 tokens / decode step | 3.9 ms / 0.34 ms |
| training step 64–128–10, batch 64 | 1.2 ms |
| ONNX CNN fixture | 0.76 ms |

| Graphics workload | CPU path | NPU path (simulated) |
|---|---|---|
| ocean 256², 64 waves | 42 ms | 30 ms |
| skinning 20 × 1280 vertices, 12 bones | 0.11 ms | 1.7 ms |
| blendshapes 50 × 4000 vertices × 52 shapes | 4.8 ms | 13.5 ms |
| SH bake 256 probes × 1024 samples | 0.61 ms | 4.2 ms |
| DCT 512 × 512 | 2.4 ms | 6.8 ms |

Reading the graphics table: the simulated NPU path does the same multiplications as the
NPU plus format conversions, so it's slower than the CPU path for everything except
the ocean, whose matrix formulation removes most of the trigonometry. On hardware the
"NPU path" column is the one that changes.
