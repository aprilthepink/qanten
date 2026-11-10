# Changelog

## 0.1.0 (unreleased)

First release. Targets the Q.ANT native computing driver ABI used by the
official toolkit v2.4.0.

- `qanten-sys`: raw driver FFI, loaded at runtime via `libloading`.
- `qanten`: `Npu` handle, `Tensor`, typed `DeviceBuffer<Weights|Features>`,
  ops (mul, periodic nonlinearity, matvec, matmul, linear, conv2d,
  conv_transpose2d, pooling, batch/layer norm, activations, KAN), `nn` layers,
  `Sequential`, `StateDict` with safetensors loading, a CPU simulator with
  noise / memory limit / multi-NPU support, `ndarray` and `serde` integration.
- `qanten-cli`: `qanten devices | info | sensors | perf | bench | selftest | doctor`.
- `qanten-graphics`: CPU-side 3D workloads as NPU matrix products: Gerstner
  ocean (one matmul per frame), linear blend skinning and blendshapes for
  crowds, SH9 light-probe baking, 8×8 DCT. CPU fallbacks and rendered examples.
- `examples/machine-monitor`: example application, vibration-based predictive
  maintenance (NPU signal processing, training, multi-NPU monitoring, HTTP serving).
- `qanten::compute`: general matrix and float computation: host `Matrix`,
  device-resident `NpuMatrix` (row/vector scaling, tiling, sharding over an
  `NpuPool`), `gemm`/`gemv` with automatic NPU/CPU placement, Freivalds
  accuracy check `check_product`, native `cos`/`sin`/`mul` on slices.
- `signal::FirBank` and `signal::correlate` use block convolution: about 2
  values of NPU traffic per output sample instead of one per tap.
- `examples/machine-monitor`: PCA novelty detector (SPE + Hotelling T²) built
  on `qanten::compute` (covariance via `gemm`, resident `NpuMatrix` scoring,
  daily `check_product` NPU self-check); a cavitation fault the classifier
  never sees; two new features (band energy, envelope floor).
- Security hardening (from a dedicated security review):
  - `qanten::limits` caps tensor size (2²⁸ elements) and rank (32). Every
    loader uses it: ONNX, npy/npz (decompression is capped too, so zip bombs
    fail) and safetensors. All size arithmetic is checked.
  - ONNX: hostile attributes are rejected with errors instead of panicking:
    zero strides, negative pads, bad permutations, out-of-range slices and
    splits, and `Clip` with min > max. Integer ops wrap instead of
    overflowing. A mutation fuzz test checks that corrupted models never
    panic.
  - Driver backend: a handle can no longer be freed while another thread
    uses it in an FFI call. `matvec_resident` checks NPU, roles and shapes
    before calling the driver.
  - `qanten-sys`: function-pointer fields are private and are called through
    `unsafe fn` methods. A relative `QANTEN_DRIVER_LIB` path with several
    components is refused.
  - `qanten-serve`: request body limit (`--max-body-mb`), inference timeout
    (`--timeout-s`), restricted model names, and escaped Prometheus labels.
  - `qanten run`: output file names are sanitised and deduplicated, and it
    refuses to write through symlinks.
- Second security review:
  - `qanten-serve` batches an ONNX model only when the new
    `OnnxModel::rows_are_independent` shows it never mixes rows. Before this,
    a model such as `Flatten(axis=0)` returned other callers' inputs. If an
    output does not line up with the batch, each request runs alone.
  - `qanten-serve`: `--max-concurrent` (default 256) answers 503 when the
    server is busy, before the body is read. A timed-out request now stops
    its model run (`OnnxModel::run_cancellable`, `Error::Cancelled`). Request
    bodies are parsed without the untagged-enum buffering, which used about
    20 times the body size in memory.
  - ONNX evaluator:
    - Every node output is checked against `limits` and for data that
      matches its shape. Values are dropped after their last use, with a
      4 GiB budget for what stays alive (`limits::MAX_LIVE_BYTES`), and
      constants are no longer copied per run.
    - Conv and ConvTranspose check their output and patch buffers before
      allocating; huge pads used to abort the process.
    - Pooling has a work budget (`limits::MAX_WORK`).
    - Repeated axes in Unsqueeze, Squeeze and Slice are errors, and global
      pools over empty planes no longer return too few values.
    - INT64 constants keep full precision.
    - Negative or huge input dims are treated as unknown.
    - Protobuf element counts are bounded.
  - npz: entries that overlap or share data are rejected (zip bomb), and
    local header signatures are checked.
  - safetensors: shapes go through `limits`.
  - `compute::Matrix` sizes are overflow-checked.
  - Driver backend:
    - Allocation byte sizes are bounded.
    - `matvec_resident` rejects the vectors block as the result.
    - A malformed perf counter is an error instead of a panic.
    - A failed free is logged.
  - `qanten run -o` writes through a temporary file and a rename, so a
    symlink in the output directory is replaced, never followed.
- `NpuExecutor`: a job's in-flight count drops before its result is
  published, so `queue_depths()` is never stale right after `wait()`.
- `compute::NpuMatrix::upload_replicated`: a full copy per NPU with batches
  split between them (vs `upload_pool`, which shards rows).
- `qanten-graphics` is built on `compute::NpuMatrix`: per-row/per-vector
  scaling (up to 3.4x lower error), `with_pool(&pool)` on every type,
  `self_check()` via `check_product`, and `sh::irradiance_batch` through
  `compute::gemm` with `Placement::Auto`.
- Code review pass, less C-like and without duplicated code:
  - `compute` split into modules; `Matrix` (f32) / `Matrix64` (f64) aliases
    of a generic `Mat<T>` with row iterators, `matmul_transposed`,
    `symmetric_eigen`, `orthonormalize_rows`; `gemm`/`gemv`/`check_product`
    became `Matrix::matmul_on` / `matvec_on` / `product_error` with a
    `Placement` that carries the NPU; `cos`/`sin`/`mul` are `Npu` methods.
    One scaling helper serves every NPU product.
  - `linalg` built on `NpuMatrix`: solvers are `LinearOperator` methods,
    `Pca::fit(npu, &Matrix64, k)`, one shared CG inner loop.
  - `signal` built on `NpuMatrix`; `Spectrum` holds matrices;
    `TemplateMatcher` + `Correlation::matches` replace `correlate` /
    `find_matches`; `FirBank::apply` returns a `Matrix`.
  - `train::Dense` stores `Matrix` weights and uses `matmul_on`.
  - `qanten::rng::SplitMix64` is public (used by the example, graphics and
    tests instead of private copies).
  - `nn::Sequential::load_mlp`, `ActivationKind: FromStr + Display`,
    `StateDict::load_file`, `Tensor::min_max` / `mean`, shared by CLI and server.
  - `qanten-graphics`: one `MatrixOp` per constant matrix with exact CPU
    path; per-item inputs/outputs (`skin_batch(&[poses])` → one `Skinned`
    each, `evaluate_batch(&Matrix)`, `project_batch(&[samples])`,
    `Dct8x8::forward(&Matrix)`); `IrradianceBaker` replaces `irradiance_batch`.
  - machine-monitor: `Measurement`, feature `Matrix`, `Confusion`, `Plant`,
    `Monitor`/`Outcome`; reclassification hysteresis.
  - CI shows failing doc snippets (they were hidden).
