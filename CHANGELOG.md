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
