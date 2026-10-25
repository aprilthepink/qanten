# Matrix and float computation (`qanten::compute`)

A general-purpose interface for numeric work on the NPU: host matrices,
matrices kept on the device, products that choose NPU or CPU by themselves,
a cheap accuracy check, and the NPU's native elementwise functions.

## What is actually faster on the NPU

The NPU computes matrix–vector products: a matrix sits on the device and
vectors stream through it. What it costs is moving data. What it does well
is the multiply-accumulates (MACs) on data that is already there. So what
matters is **work per value moved**:

| Operation | MACs | Values moved | Work per value | Good fit? |
|---|---|---|---|---|
| `n × n` matrix kept on the device, `k` vectors | `n²·k` | `2n·k` (+ `n²` once) | about `n / 2` | yes, the bigger the better |
| `n × n · n × n` product (`gemm`) | `n³` | `3n²` | about `n / 3` | yes for large `n` |
| one matrix–vector product, matrix uploaded for it | `n²` | `n² + 2n` | about 1 | no: use the CPU |
| elementwise `mul` / `cos` | `n` | `3n` / `2n` | ≤ 1 | only inside pipelines |

Rules of thumb:

- **Keep matrices on the device and batch vectors.** Upload with
  `NpuMatrix::upload` once, then send hundreds or thousands of vectors per
  call. A 1024 × 1024 matrix does about 500 MACs per value moved.
- **Single products and tiny matrices belong on the CPU.**
  `Placement::Auto` does that for you.
- **Use several NPUs for one big matrix.** `NpuMatrix::upload_pool` gives
  each NPU a block of rows, and they all work on every batch in parallel.
- **Precision is about 3 digits** (bfloat16 values, wider accumulation).
  Use it where that is enough, or wrap it in exact correction like
  [`linalg`](linalg.md) does. `check_product` tells you the actual error.

## Host matrices: `Matrix`

A row-major `f32` matrix with the usual constructors, plus an exact,
multithreaded CPU product used as the reference and for small work:

```rust
use qanten::compute::Matrix;

let a = Matrix::from_fn(2, 3, |i, j| (i * 3 + j) as f32);   // [[0,1,2],[3,4,5]]
let b = Matrix::from_rows(&[vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]])?;
let c = a.matmul_cpu(&b)?;
assert_eq!(c.as_slice(), &[2.0, 3.0, 8.0, 9.0]);
assert_eq!(c[(1, 0)], 8.0);
let _ = (Matrix::identity(4), Matrix::random_normal(8, 8, 42), a.transpose());
```

## Matrices on the device: `NpuMatrix`

```rust
use qanten::compute::{Matrix, NpuMatrix};
use qanten::Npu;

let npu = Npu::auto(0)?;
let a = Matrix::random_normal(512, 256, 1);
let resident = NpuMatrix::upload(&npu, &a)?;         // the only time A moves

let x = Matrix::random_normal(2000, 256, 2);         // 2000 vectors of length 256
let y = resident.apply_batch(&x)?;                   // y[i] = A · x[i], shape 2000 × 512
let exact = x.matmul_cpu(&a.transpose())?;
println!("relative error {:.1e}", y.relative_error(&exact));   // ~3e-3

let b = Matrix::random_normal(256, 64, 3);
let ab = resident.matmul(&b)?;                       // A · B, streams B's 64 columns
assert_eq!((ab.rows(), ab.cols()), (512, 64));
```

| Method | Computes |
|---|---|
| `apply_batch(&X)` | every row: `Y = X · Aᵀ` (`k × cols → k × rows`) |
| `apply(&x)` | one vector `A · x` |
| `matmul(&B)` | `A · B` |
| `apply_repeated(&x, steps)` | `A^steps · x`, matrix stays on the device (Markov chains, power iteration, time-stepping) |

**Scaling.** Each matrix row and each input vector is scaled to unit range
before it reaches the NPU, and the scale is undone afterwards. A row of values
around `1e-6` and one around `1e6` get the same relative accuracy. Analog
hardware works best with values in a fixed range, so this matters more there
than on the simulator.

### Tiles and several NPUs

```rust
use qanten::compute::{Matrix, NpuMatrix, Tiling};
use qanten::pool::NpuPool;

let pool = NpuPool::simulated(4)?;            // or NpuPool::open_all()
let a = Matrix::random_normal(4096, 1024, 1);
let sharded = NpuMatrix::upload_pool(&pool, &a)?;   // 1024 rows per NPU
assert_eq!(sharded.tiles(), 4);

// Or cut tiles explicitly, e.g. to fit device memory:
let tiled = NpuMatrix::upload_tiled(pool.npus(), &a, Tiling {
    max_rows: Some(1024),
    max_cols: Some(512),       // column tiles' partial sums are added on the host in f32
})?;
assert_eq!(tiled.tiles(), 8);
let y = tiled.apply_batch(&Matrix::random_normal(16, 1024, 2))?;
assert_eq!(y.cols(), 4096);
```

Row blocks go to the NPUs round-robin. Every batch runs on all of them at
once, one thread per NPU.

## Products that pick their place: `gemm`, `gemv`

```rust
use qanten::compute::{Matrix, Placement, gemm, gemv};
use qanten::Npu;

let npu = Npu::auto(0)?;
let a = Matrix::random_normal(600, 400, 1);
let b = Matrix::random_normal(400, 300, 2);
let c = gemm(&npu, &a, &b, Placement::Auto)?;    // big enough: NPU
let tiny = gemm(&npu, &Matrix::identity(4), &Matrix::identity(4), Placement::Auto)?;  // CPU, exact
assert_eq!(tiny, Matrix::identity(4));
let y = gemv(&npu, &a, &vec![1.0; 400], Placement::Auto)?;   // one vector: CPU
assert_eq!((c.rows(), y.len()), (600, 600));
```

On the NPU, `gemm` keeps the larger operand on the device and streams the
other through it. `Placement::Auto` follows an `AutoPolicy`: use the NPU when
the product has at least `min_macs` MACs (default 2²⁰) **and** the device
matrix is reused for at least `min_reuse` vectors (default 16). These
defaults are guesses. On hardware, measure with `qanten bench` or
`FallbackPolicy::calibrate` and pass your own policy to `gemm_with`.
`Placement::Npu` and `Placement::Cpu` force a side.

## Checking results: `check_product`

How wrong is a product computed on the NPU? Recomputing `A · B` on the CPU
costs `O(n³)`. Freivalds' method costs `O(n²)`: for random vectors `v` it
compares `A · (B · v)` with `C · v`.

```rust
use qanten::compute::{Matrix, Placement, check_product, gemm};
use qanten::{Npu, SimConfig};

let noisy = Npu::simulator_with(SimConfig { noise_std: 0.02, ..SimConfig::default() });
let a = Matrix::random_normal(128, 128, 1);
let b = Matrix::random_normal(128, 128, 2);
let c = gemm(&noisy, &a, &b, Placement::Npu)?;
let err = check_product(&a, &b, &c, 8);          // relative error estimate
println!("relative error ≈ {err:.1e}");
assert!(err < 0.2);
```

Use it in production to watch hardware accuracy, for example to fall back to
the CPU or recalibrate when the error drifts.

## Elementwise: `cos`, `sin`, `mul`

The photonic periodic nonlinearity is native `cos`. `compute::cos` and
`compute::sin` reduce the argument to `[−π, π)` on the host in f64 first,
so large arguments keep their accuracy, then evaluate it on the NPU.
Absolute error is about `±0.006`, from the bf16 phase. On hardware the
function is the device's cosine-shaped optical transfer curve, so accuracy
also depends on calibration.

```rust
use qanten::compute;
use qanten::Npu;

let npu = Npu::auto(0)?;
let x: Vec<f32> = (0..1000).map(|i| i as f32 * 0.01).collect();
let c = compute::cos(&npu, &x)?;
let s = compute::sin(&npu, &x)?;
let p = compute::mul(&npu, &c, &s)?;     // = sin(2x) / 2
assert!((p[300] - (6.0f32).sin() / 2.0).abs() < 0.01);
let _ = compute::cos(&npu, &[12345.0])?;  // range-reduced on the host
```

Elementwise work moves more data than it computes, so on its own a CPU is
as fast. Use these where data is already headed to the NPU.
