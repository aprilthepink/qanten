# Changelog

## 0.1.0 (unreleased)

First release. Targets the Q.ANT native computing driver ABI used by the
official toolkit v2.4.0.

- `qant-sys`: raw driver FFI, loaded at runtime via `libloading`.
- `qant`: `Npu` handle, `Tensor`, typed `DeviceBuffer<Weights|Features>`,
  ops (mul, periodic nonlinearity, matvec, matmul, linear, conv2d,
  conv_transpose2d, pooling, batch/layer norm, activations, KAN), `nn` layers,
  `Sequential`, `StateDict` with safetensors loading, a CPU simulator with
  noise / memory limit / multi-NPU support, `ndarray` and `serde` integration.
- `qant-cli`: `qant devices | info | sensors | perf | bench | selftest | doctor`.
