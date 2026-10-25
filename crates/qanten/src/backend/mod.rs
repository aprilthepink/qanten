//! Execution backends.
//!
//! A [`Backend`] executes the primitive operations the Q.ANT driver offers.
//! Everything else in this crate (convolutions, pooling, layers, ...) is built
//! on top of these primitives, exactly like Q.ANT's official toolkit.
//!
//! Two backends ship with the crate:
//! - [`DriverBackend`] (feature `driver`): the real hardware via
//!   `libqant_native_computing_driver`, loaded at runtime.
//! - [`Simulator`]: a pure-Rust CPU reference with optional noise.
//!
//! The trait is public, so you can add your own (a mock, a remote NPU, ...).

use half::bf16;

use crate::error::Result;
use crate::info::{DeviceInfo, PerfCounters};

#[cfg(feature = "driver")]
mod driver;
mod sim;

#[cfg(feature = "driver")]
pub use driver::DriverBackend;
pub use sim::{SimConfig, Simulator};

/// Which kind of backend an [`Npu`](crate::Npu) runs on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "lowercase"))]
pub enum BackendKind {
    /// Q.ANT hardware through the native driver.
    Driver,
    /// The built-in CPU simulator.
    Simulator,
    /// A user-provided backend.
    Custom,
}

impl std::fmt::Display for BackendKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            BackendKind::Driver => "driver",
            BackendKind::Simulator => "simulator",
            BackendKind::Custom => "custom",
        })
    }
}

/// Role of a device memory block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataRole {
    /// The matrix operand of a matrix-vector product. Cannot be downloaded.
    Weights,
    /// Vector operands and results.
    Features,
}

/// Backend-specific identifier for a device memory block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RawHandle {
    /// NPU the memory lives on.
    pub npu: u32,
    /// Backend-assigned identifier of the block.
    pub key: u64,
}

/// An NPU listed by [`Backend::available_npus`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct NpuEntry {
    /// Index to pass to [`Npu::open`](crate::Npu::open).
    pub id: u32,
    /// Board serial number.
    pub serial: String,
}

/// Primitive operations of a Q.ANT NPU.
///
/// Implementations must check slice lengths themselves: callers may pass
/// any slices. All methods are safe to call from several threads at once.
pub trait Backend: Send + Sync + std::fmt::Debug {
    /// Which kind of backend this is.
    fn kind(&self) -> BackendKind;

    /// Human readable backend name, e.g. the driver library path.
    fn name(&self) -> String;

    /// Prepares NPU `npu` for use. Called once by [`Npu::with_backend`](crate::Npu::with_backend).
    fn init_npu(&self, npu: u32) -> Result<()>;
    /// Releases NPU `npu`. Called when the last [`Npu`](crate::Npu) handle drops.
    fn release_npu(&self, npu: u32) -> Result<()>;
    /// Lists the NPUs this backend can open.
    fn available_npus(&self) -> Result<Vec<NpuEntry>>;

    /// `out[i] = a[i] * b[i]`. All three slices have the same length.
    fn mul(&self, npu: u32, a: &[bf16], b: &[bf16], out: &mut [bf16]) -> Result<()>;

    /// The photonic periodic nonlinearity: `out[i] = tcos(u[i]) * v[i]`,
    /// where `tcos` is the cosine-shaped optical transfer function.
    fn periodic_nl(&self, npu: u32, u: &[bf16], v: &[bf16], out: &mut [bf16]) -> Result<()>;

    /// Batched matrix-vector product: `out[v, r] = sum_c matrix[r, c] * vectors[v, c]`.
    /// `matrix` is `rows x cols`, `vectors` is `n_vecs x cols`, `out` is `n_vecs x rows`.
    fn matvec_batched(
        &self,
        npu: u32,
        matrix: &[bf16],
        vectors: &[bf16],
        out: &mut [bf16],
        rows: usize,
        cols: usize,
    ) -> Result<()>;

    /// Reserves a `rows × cols` block of device memory.
    fn allocate(&self, npu: u32, rows: usize, cols: usize, role: DataRole) -> Result<RawHandle>;
    /// Releases a block. Freeing twice must return an error, not crash.
    fn free(&self, handle: RawHandle) -> Result<()>;
    /// Uploads exactly `rows * cols` values into `handle`.
    fn upload(&self, handle: RawHandle, data: &[bf16]) -> Result<()>;
    /// Downloads exactly `rows * cols` values from a `Features` block.
    fn download(&self, handle: RawHandle, out: &mut [bf16]) -> Result<()>;
    /// `result = vectors × matrixᵀ` with all operands resident on the device.
    fn matvec_resident(
        &self,
        matrix: RawHandle,
        vectors: RawHandle,
        result: RawHandle,
    ) -> Result<()>;

    /// Driver version string.
    fn driver_version(&self, npu: u32) -> Result<String>;
    /// Sensor readings and firmware info.
    fn device_info(&self, npu: u32) -> Result<DeviceInfo>;
    /// Performance counters since the last reset.
    fn perf_counters(&self, npu: u32) -> Result<PerfCounters>;
    /// Restarts the performance counters.
    fn reset_perf_counters(&self, npu: u32) -> Result<()>;
}

/// `a · b` for buffer sizes; an overflow is a shape error, never a wrapped
/// length that could slip past [`check_len`] and reach the driver.
pub(crate) fn checked_size(npu: u32, what: &str, a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| crate::Error::Shape(format!("NPU {npu}: {what}: {a} × {b} overflows usize")))
}

pub(crate) fn check_len(npu: u32, what: &str, got: usize, want: usize) -> Result<()> {
    if got != want {
        return Err(crate::Error::Shape(format!(
            "NPU {npu}: {what}: expected {want} values, got {got}"
        )));
    }
    Ok(())
}
