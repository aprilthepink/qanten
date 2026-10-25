//! General matrix and float computation on the NPU.
//!
//! # What the NPU is good at
//!
//! The NPU computes matrix–vector products. A matrix of `r × c` values sits
//! on the device and every vector streamed through it costs `c` values in and
//! `r` out, for `r · c` multiply-accumulates. So the work per byte moved grows
//! with the matrix size, and it pays off when **one matrix is reused for many
//! vectors**: a big batch through a matrix kept on the device ([`NpuMatrix`]),
//! or a large product ([`Matrix::matmul_on`]) where one operand stays put while the other
//! streams through.
//!
//! Elementwise operations ([`Npu::mul`], [`Npu::cos`], [`Npu::sin`]) do one operation per
//! value moved, so a CPU is usually as fast for those on their own. They are
//! here because `cos` is native to the photonic hardware, and for pipelines
//! that keep data near the device.
//!
//! | Use | Type / function |
//! |---|---|
//! | Host matrix (`f32` or `f64`), exact CPU products, eigenvectors | [`Matrix`] |
//! | Matrix kept on one or more NPUs, tiled and scaled | [`NpuMatrix`] |
//! | `C = A·B` / `y = A·x` that decides NPU vs CPU | [`Matrix::matmul_on`], [`Placement`] |
//! | Cheap accuracy check of a product | [`Matrix::product_error`] |
//! | Elementwise `cos`, `sin`, `a·b` on slices | [`Npu::cos`], [`Npu::sin`], [`Npu::mul`] |
//!
//! # Precision
//!
//! Inputs and results on the NPU are bfloat16, about 3 significant digits;
//! sums are accumulated at higher precision. [`NpuMatrix`] scales every
//! matrix row and every input vector to unit range before sending it (which
//! matters on analog hardware), and adds tile results on the host in f32.
//! Expect relative errors around `1e-2`–`1e-3` per product, so this is for
//! work where that is enough (simulation, statistics, search, signal
//! processing, Monte Carlo, iterative methods with exact correction; see
//! [`linalg`](crate::linalg) for solvers that get f64 accuracy back).
//! [`Matrix::product_error`] measures the error of a result cheaply.
//!
//! ```
//! # fn main() -> qanten::Result<()> {
//! use qanten::compute::{Matrix, NpuMatrix};
//! let npu = qanten::Npu::simulator();
//! let a = Matrix::from_fn(256, 128, |i, j| ((i * 7 + j * 3) % 11) as f32 - 5.0);
//! let x = Matrix::from_fn(1000, 128, |i, j| ((i + j) % 5) as f32 * 0.1);
//!
//! let resident = NpuMatrix::upload(&npu, &a)?; // upload once
//! let y = resident.apply_batch(&x)?;            // 1000 vectors: y[i] = A · x[i]
//! assert_eq!((y.rows(), y.cols()), (1000, 256));
//!
//! // Freivalds-style check on the CPU in O(n²): y must equal x·Aᵀ.
//! let err = y.product_error(&x, &a.transpose(), 4);
//! assert!(err < 1e-2, "relative error {err}");
//! # Ok(()) }
//! ```

mod dispatch;
mod elementwise;
mod matrix;
mod resident;

pub use dispatch::{AutoPolicy, Placement};
pub use matrix::{Mat, Matrix, Matrix64, Scalar};
pub use resident::{NpuMatrix, Tiling};

#[cfg(doc)]
use crate::device::Npu;
