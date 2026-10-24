//! # qant — Rust SDK for Q.ANT photonic NPUs
//!
//! A safe, batteries-included Rust interface to Q.ANT native processing
//! units, built on the same driver as Q.ANT's official
//! [native computing toolkit](https://github.com/Q-ANT-GmbH/qant_native_computing_toolkit).
//!
//! - **[`Npu`]**: RAII device handle. Init on open, release on drop. Works
//!   on real hardware ([`Npu::open`]) or the built-in simulator
//!   ([`Npu::simulator`]); [`Npu::auto`] picks whichever is available.
//! - **[`Tensor`]**: owned row-major `bf16` tensor with `f32` conversions.
//! - **[`ops`]**: matmul, linear, conv2d, transposed conv, pooling,
//!   batch/layer norm, activations, the photonic periodic nonlinearity and
//!   KAN layers.
//! - **[`nn`]**: PyTorch-style layers with weights resident on the NPU,
//!   [`nn::Sequential`], and [`nn::StateDict`] to load `.safetensors` weights.
//! - **[`DeviceBuffer`]**: typed device memory; the type system stops you
//!   from downloading weights or mixing up operand roles.
//! - **Simulator** with optional noise, memory limits and multiple NPUs, for
//!   development and CI without hardware.
//!
//! The driver library is loaded at runtime, so nothing needs to be installed
//! to build. See [`backend`] for details.
//!
//! ```
//! use qant::{Npu, Tensor, ops};
//!
//! # fn main() -> qant::Result<()> {
//! let npu = Npu::auto(0)?;
//! let x = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], [2, 2])?;
//! let w = Tensor::from_f32(&[1.0, 0.0, 0.0, 1.0, 1.0, 1.0], [3, 2])?;
//! let y = ops::linear(&npu, &x, &w, None)?;
//! assert_eq!(y.to_f32_vec(), vec![1.0, 2.0, 3.0, 3.0, 4.0, 7.0]);
//! # Ok(()) }
//! ```
//!
//! This is an unofficial community crate and is not affiliated with Q.ANT GmbH.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(not(test), warn(clippy::unwrap_used, clippy::expect_used))]

pub mod backend;
mod device;
mod error;
pub mod info;
mod memory;
pub mod nn;
pub mod ops;
mod rng;
mod tensor;

pub use backend::{BackendKind, DataRole, NpuEntry, SimConfig};
#[cfg(feature = "driver")]
pub use device::setup_driver_logging;
pub use device::{BACKEND_ENV, Npu, devices, driver_available};
pub use error::{Error, Result};
pub use half::bf16;
pub use memory::{DeviceBuffer, Features, Role, Weights, matvec_resident};
pub use tensor::Tensor;

/// Re-export of the raw FFI crate.
#[cfg(feature = "driver")]
pub use qant_sys as sys;

/// Common imports: `use qant::prelude::*;`
pub mod prelude {
    pub use crate::nn::{self, Module};
    pub use crate::ops::{self, Conv2dParams, ConvTranspose2dParams, Hw, Pool2dParams};
    pub use crate::{DeviceBuffer, Error, Features, Npu, Result, Tensor, Weights};
}
