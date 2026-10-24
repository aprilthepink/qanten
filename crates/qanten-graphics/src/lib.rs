//! # qanten-graphics
//!
//! CPU-side graphics and 3D workloads, reformulated as large matrix products
//! so they can run on a Q.ANT photonic NPU through [`qanten`].
//!
//! These are jobs engines usually keep on the CPU because the result is needed
//! there (physics, gameplay queries, asset baking), not on the GPU:
//!
//! | Module | Workload | NPU formulation |
//! |---|---|---|
//! | [`ocean`] | Gerstner wave fields: heights, displacement, normals, foam | separable phases → one matmul per frame |
//! | [`blendshapes`] | morph targets for crowds | resident delta matrix × per-character weights |
//! | [`skinning`] | linear blend skinning on the CPU (collision, picking, cloth) | resident `[vertices, 4·bones]` matrix × bone matrices |
//! | [`sh`] | light-probe baking into spherical harmonics, irradiance | resident basis × radiance samples |
//! | [`dct`] | 8×8 block DCT for texture / image pipelines | resident 64×64 Kronecker matrix × blocks |
//!
//! Every type works without an NPU (plain CPU path) and switches to the NPU
//! with `.with_npu(&npu)`. The CPU path doubles as the accuracy reference.
//!
//! ## Precision
//!
//! The NPU computes in bfloat16 (~3 significant digits). Each module is
//! arranged so only bounded quantities go through it: phases are reduced on
//! the host in f64, skinning and blendshapes compute *offsets* from the rest
//! pose rather than absolute positions. Expect errors around 0.5 % of the
//! magnitude of each output: fine for physics proxies, culling, baking and
//! previews, not a replacement for exact final-frame geometry.

#![warn(missing_docs)]
#![cfg_attr(not(test), warn(clippy::unwrap_used, clippy::expect_used))]

mod accel;
pub mod blendshapes;
pub mod dct;
pub mod ocean;
pub mod sh;
pub mod skinning;

pub use glam;
pub use qanten::{Error, Npu, Result};
