//! Raw FFI bindings to the Q.ANT native computing driver.
//!
//! The driver ships as a closed-source shared library
//! (`libqant_native_computing_driver.so`) in Q.ANT's Debian package. This crate
//! does **not** link against it at build time. It loads the library at runtime
//! with [`libloading`], so crates depending on `qanten-sys` build and test
//! anywhere, including machines without Q.ANT hardware (CI, docs.rs, laptops).
//!
//! The struct layouts and function signatures mirror the ones used by Q.ANT's
//! official toolkit (`qant_native_computing_toolkit` v2.4, Apache-2.0).
//!
//! Most users want the safe, high-level [`qanten`](https://docs.rs/qanten) crate
//! instead of this one.
//!
//! # Library lookup
//!
//! [`DriverLibrary::load`] tries, in order:
//! 1. the path in the `QANTEN_DRIVER_LIB` environment variable, if set;
//! 2. the platform default names in [`DEFAULT_LIBRARY_NAMES`], resolved by the
//!    system loader (`LD_LIBRARY_PATH`, `/etc/ld.so.conf`, ...);
//! 3. the absolute paths in [`DEFAULT_LIBRARY_PATHS`].

#![allow(non_camel_case_types)]
#![warn(missing_docs)]
#![cfg_attr(not(test), warn(clippy::unwrap_used, clippy::expect_used))]

use std::ffi::{CStr, OsStr, OsString, c_char, c_int};
use std::fmt;
use std::path::{Path, PathBuf};

pub use half::bf16;

/// Length of every fixed-size C string buffer used by the driver.
pub const C_STRING_DEFAULT_LENGTH: usize = 1024;

/// Environment variable that overrides the driver library path.
pub const DRIVER_LIB_ENV: &str = "QANTEN_DRIVER_LIB";

/// Library names handed to the system loader.
pub const DEFAULT_LIBRARY_NAMES: &[&str] = &[
    "libqant_native_computing_driver.so",
    "libqant_native_computing_driver.so.2",
];

/// Absolute fallback locations (where Q.ANT's Debian package installs the driver).
pub const DEFAULT_LIBRARY_PATHS: &[&str] = &[
    "/usr/lib/libqant_native_computing_driver.so",
    "/usr/local/lib/libqant_native_computing_driver.so",
    "/usr/lib/x86_64-linux-gnu/libqant_native_computing_driver.so",
    "/usr/lib/aarch64-linux-gnu/libqant_native_computing_driver.so",
];

/// Error struct returned by value from every driver call. `code == 0` means success.
#[repr(C)]
#[derive(Clone)]
pub struct QantDriverError {
    /// Status code: `0` on success, anything else is an error.
    pub code: c_int,
    /// NUL-terminated error message (empty on success).
    pub message: [c_char; C_STRING_DEFAULT_LENGTH],
}

impl QantDriverError {
    /// A success value (`code == 0`, empty message).
    pub const fn ok() -> Self {
        Self {
            code: 0,
            message: [0; C_STRING_DEFAULT_LENGTH],
        }
    }

    /// Builds an error value with `code` and `message` (truncated to fit).
    pub fn new(code: c_int, message: &str) -> Self {
        let mut buf = [0 as c_char; C_STRING_DEFAULT_LENGTH];
        for (dst, &src) in buf
            .iter_mut()
            .zip(message.as_bytes().iter().take(C_STRING_DEFAULT_LENGTH - 1))
        {
            *dst = src as c_char;
        }
        Self { code, message: buf }
    }

    /// `true` if `code == 0`.
    pub fn is_ok(&self) -> bool {
        self.code == 0
    }

    /// The message as a Rust string (lossy UTF-8).
    pub fn message(&self) -> String {
        c_buf_to_string(&self.message)
    }
}

impl fmt::Debug for QantDriverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QantDriverError")
            .field("code", &self.code)
            .field("message", &self.message())
            .finish()
    }
}

/// Board sensor readings (temperatures in °C, voltages in V, currents in A).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SensorInfo {
    /// Temperature sensor `pd1`, in °C.
    pub temp_pd1: f64,
    /// Temperature sensor `pd2`, in °C.
    pub temp_pd2: f64,
    /// Temperature sensor `pd3`, in °C.
    pub temp_pd3: f64,
    /// Temperature sensor `dac_1`, in °C.
    pub temp_dac_1: f64,
    /// Temperature sensor `dac_2`, in °C.
    pub temp_dac_2: f64,
    /// Measured voltage of the 1.8 V rail, in V.
    pub voltage_1v8: f64,
    /// Measured voltage of the 3.3 V rail, in V.
    pub voltage_3v3: f64,
    /// Measured voltage of the first 6.0 V rail, in V.
    pub voltage_6v0_1: f64,
    /// Measured voltage of the second 6.0 V rail, in V.
    pub voltage_6v0_2: f64,
    /// Measured voltage of the 6.2 V rail, in V.
    pub voltage_6v2: f64,
    /// Measured voltage of the 6.5 V rail, in V.
    pub voltage_6v5: f64,
    /// Measured voltage of the first 8.0 V rail, in V.
    pub voltage_8v0_1: f64,
    /// Measured voltage of the second 8.0 V rail, in V.
    pub voltage_8v0_2: f64,
    /// Measured voltage of the 12 V rail, in V.
    pub voltage_12v: f64,
    /// Measured voltage of the 12.5 V rail, in V.
    pub voltage_12v5: f64,
    /// Measured voltage of the −12.5 V rail, in V.
    pub voltage_n12v5: f64,
    /// Current drawn on the 12 V rail, in A.
    pub current_12v: f64,
}

#[repr(C)]
#[derive(Clone)]
/// Firmware and board identification, as fixed-size C strings.
pub struct QantVersionInfo {
    /// Firmware version string.
    pub fw_version: [c_char; C_STRING_DEFAULT_LENGTH],
    /// Version of the Zephyr RTOS running on the board.
    pub zephyr_version: [c_char; C_STRING_DEFAULT_LENGTH],
    /// Board serial number.
    pub board_serial_no: [c_char; C_STRING_DEFAULT_LENGTH],
}

impl fmt::Debug for QantVersionInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QantVersionInfo")
            .field("fw_version", &c_buf_to_string(&self.fw_version))
            .field("zephyr_version", &c_buf_to_string(&self.zephyr_version))
            .field("board_serial_no", &c_buf_to_string(&self.board_serial_no))
            .finish()
    }
}

#[repr(C)]
#[derive(Debug, Clone)]
/// Everything `qant_driver_get_device_info` returns, by value.
pub struct QantDeviceInfo {
    /// Board sensor readings.
    pub sensor_info: SensorInfo,
    /// Firmware and board identification.
    pub version_info: QantVersionInfo,
    /// Status of the call; the other fields are only valid if this is ok.
    pub error: QantDriverError,
}

/// Role of a device memory block. Weights are the matrix side of a
/// matrix-vector product, features the vector side and the result.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QantDataRole {
    /// Matrix operand of a matrix-vector product (upload only).
    Weights = 0,
    /// Vector operands and results (upload and download).
    Features = 1,
}

/// Opaque handle to a block of device memory.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QantDriverHandle {
    /// NPU the memory lives on.
    pub npu_id: u32,
    /// Driver-assigned identifier of the memory block.
    pub key: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
/// Performance counters returned by `qant_driver_get_perf_counter`.
pub struct PerformanceCounterInfo {
    /// Seconds since the last counter reset.
    pub timebased_counter: f64,
    /// Seconds spent stalled (idle) since the last reset.
    pub stalling_counter: f64,
}

/// `qant_driver_init_npu`: initialise an NPU before use.
pub type InitNpuFn = unsafe extern "C" fn(npu_id: u32) -> QantDriverError;
/// `qant_driver_release_npu`: release an NPU after use.
pub type ReleaseNpuFn = unsafe extern "C" fn(npu_id: u32) -> QantDriverError;
/// `qant_driver_setup_logging`: write driver logs to a folder (level 0 = trace … 4 = error).
pub type SetupLoggingFn =
    unsafe extern "C" fn(folder_name: *const c_char, loglevel: i32) -> QantDriverError;
/// `qant_driver_mul_npu`: elementwise `output[i] = us[i] · vs[i]` for `len` values.
pub type MulNpuFn = unsafe extern "C" fn(
    npu_id: u32,
    us: *const bf16,
    vs: *const bf16,
    output: *mut bf16,
    len: usize,
) -> QantDriverError;
/// `qant_driver_mul_matrix_vec_batched`: `output[v, r] = Σ_c matrix[r, c] · vector_batch[v, c]`.
pub type MulMatrixVecBatchedFn = unsafe extern "C" fn(
    npu_id: u32,
    matrix: *const bf16,
    vector_batch: *const bf16,
    output: *mut bf16,
    n_rows: usize,
    n_cols: usize,
    n_vecs: usize,
) -> QantDriverError;
/// `qant_driver_scaled_periodic_nl_npu`: the photonic nonlinearity `output[i] = tcos(us[i]) · vs[i]`.
pub type ScaledPeriodicNlFn = MulNpuFn;
/// `qant_driver_allocate`: reserve a `rows × cols` bf16 block in device memory.
pub type AllocateFn = unsafe extern "C" fn(
    npu_id: u32,
    rows: usize,
    cols: usize,
    data_role: QantDataRole,
    out_handle: *mut QantDriverHandle,
) -> QantDriverError;
/// `qant_driver_free`: release a block from `AllocateFn`.
pub type FreeFn = unsafe extern "C" fn(handle: QantDriverHandle) -> QantDriverError;
/// `qant_driver_upload`: copy `rows · cols` values from host memory into a block.
pub type UploadFn =
    unsafe extern "C" fn(handle: QantDriverHandle, data: *const bf16) -> QantDriverError;
/// `qant_driver_download`: copy a `Features` block (`rows · cols` values) back to the host.
pub type DownloadFn =
    unsafe extern "C" fn(handle: QantDriverHandle, output: *mut bf16) -> QantDriverError;
/// `qant_driver_mul_matrix_vec_batched_mem`: matrix-vector product with all operands resident on the device.
pub type MulMatrixVecBatchedMemFn = unsafe extern "C" fn(
    matrix: QantDriverHandle,
    vectors: QantDriverHandle,
    result: QantDriverHandle,
) -> QantDriverError;
/// `qant_driver_get_version`: write the driver version string (NUL-terminated) into `output`.
pub type GetVersionFn =
    unsafe extern "C" fn(npu_id: u32, output: *mut c_char, len: usize) -> QantDriverError;
/// `qant_driver_get_device_info`: sensor readings and version info, returned by value.
pub type GetDeviceInfoFn = unsafe extern "C" fn(npu_id: u32) -> QantDeviceInfo;
/// `qant_driver_get_available_npus`: fill up to `capacity` NPU ids and serial numbers.
pub type GetAvailableNpusFn = unsafe extern "C" fn(
    idxs_out: *mut u32,
    serials_out: *mut [c_char; C_STRING_DEFAULT_LENGTH],
    capacity: usize,
    out_count: *mut usize,
) -> QantDriverError;
/// `qant_driver_reset_perf_counter`: restart the performance counters.
pub type ResetPerfCounterFn = unsafe extern "C" fn(npu_id: u32) -> QantDriverError;
/// `qant_driver_get_perf_counter`: read the performance counters.
pub type GetPerfCounterFn =
    unsafe extern "C" fn(npu_id: u32, out: *mut PerformanceCounterInfo) -> QantDriverError;

/// Error while loading the driver library or resolving its symbols.
#[derive(Debug)]
pub enum LoadError {
    /// No candidate path could be opened. Holds every attempt and its error.
    NotFound(Vec<(OsString, libloading::Error)>),
    /// `QANTEN_DRIVER_LIB` is a relative path with directories; use an
    /// absolute path or a bare library name.
    RelativePath(OsString),
    /// The library opened but a required symbol is missing (driver too old?).
    MissingSymbol {
        /// Library that was opened.
        path: OsString,
        /// Name of the missing symbol.
        symbol: &'static str,
        /// Loader error.
        source: libloading::Error,
    },
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoadError::NotFound(attempts) => {
                write!(
                    f,
                    "Q.ANT driver library not found (set {DRIVER_LIB_ENV} to its path or install the qant_native_computing_driver package). Tried:"
                )?;
                for (path, err) in attempts {
                    write!(f, "\n  {}: {err}", path.to_string_lossy())?;
                    if let Some(src) = std::error::Error::source(err) {
                        write!(f, " ({src})")?;
                    }
                }
                Ok(())
            }
            LoadError::RelativePath(path) => write!(
                f,
                "{DRIVER_LIB_ENV}={} is a relative path; use an absolute path (or a bare library name for the system search path)",
                path.to_string_lossy()
            ),
            LoadError::MissingSymbol {
                path,
                symbol,
                source,
            } => write!(
                f,
                "Q.ANT driver library {} is missing symbol `{symbol}` ({source}); the driver may be incompatible",
                path.to_string_lossy()
            ),
        }
    }
}

impl std::error::Error for LoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            LoadError::NotFound(_) | LoadError::RelativePath(_) => None,
            LoadError::MissingSymbol { source, .. } => Some(source),
        }
    }
}

/// The loaded driver library and its resolved entry points.
///
/// The function pointers stay valid for as long as this struct is alive,
/// because it owns the [`libloading::Library`].
pub struct DriverLibrary {
    path: OsString,
    init_npu: InitNpuFn,
    release_npu: ReleaseNpuFn,
    setup_logging: SetupLoggingFn,
    mul_npu: MulNpuFn,
    mul_matrix_vec_batched: MulMatrixVecBatchedFn,
    scaled_periodic_nl_npu: ScaledPeriodicNlFn,
    allocate: AllocateFn,
    free: FreeFn,
    upload: UploadFn,
    download: DownloadFn,
    mul_matrix_vec_batched_mem: MulMatrixVecBatchedMemFn,
    get_version: GetVersionFn,
    get_device_info: GetDeviceInfoFn,
    get_available_npus: GetAvailableNpusFn,
    reset_perf_counter: ResetPerfCounterFn,
    get_perf_counter: GetPerfCounterFn,
    // Must be dropped last; keeps the function pointers above valid.
    _lib: libloading::Library,
}

/// Generates one `unsafe fn` per driver entry point. The function pointers
/// themselves stay private, so they can only be called through a live
/// `DriverLibrary` (never after the library was unloaded).
macro_rules! entry_points {
    ($( $(#[$doc:meta])* $name:ident ( $($arg:ident : $ty:ty),* ) -> $ret:ty; )*) => {
        impl DriverLibrary {
            $(
                $(#[$doc])*
                ///
                /// # Safety
                /// The arguments must satisfy the driver's contract for this
                /// call (valid pointers with the lengths the driver derives
                /// from the other arguments or the allocation, live handles).
                #[allow(clippy::too_many_arguments)] // mirrors the C signature
                pub unsafe fn $name(&self, $($arg: $ty),*) -> $ret {
                    // SAFETY: forwarded to the caller; `self` keeps the library loaded.
                    unsafe { (self.$name)($($arg),*) }
                }
            )*
        }
    };
}

entry_points! {
    /// See [`InitNpuFn`].
    init_npu(npu_id: u32) -> QantDriverError;
    /// See [`ReleaseNpuFn`].
    release_npu(npu_id: u32) -> QantDriverError;
    /// See [`SetupLoggingFn`].
    setup_logging(folder_name: *const c_char, loglevel: i32) -> QantDriverError;
    /// See [`MulNpuFn`].
    mul_npu(npu_id: u32, us: *const bf16, vs: *const bf16, output: *mut bf16, len: usize) -> QantDriverError;
    /// See [`MulMatrixVecBatchedFn`].
    mul_matrix_vec_batched(
        npu_id: u32,
        matrix: *const bf16,
        vector_batch: *const bf16,
        output: *mut bf16,
        n_rows: usize,
        n_cols: usize,
        n_vecs: usize
    ) -> QantDriverError;
    /// See [`ScaledPeriodicNlFn`].
    scaled_periodic_nl_npu(npu_id: u32, us: *const bf16, vs: *const bf16, output: *mut bf16, len: usize) -> QantDriverError;
    /// See [`AllocateFn`].
    allocate(npu_id: u32, rows: usize, cols: usize, role: QantDataRole, out: *mut QantDriverHandle) -> QantDriverError;
    /// See [`FreeFn`].
    free(handle: QantDriverHandle) -> QantDriverError;
    /// See [`UploadFn`].
    upload(handle: QantDriverHandle, data: *const bf16) -> QantDriverError;
    /// See [`DownloadFn`].
    download(handle: QantDriverHandle, output: *mut bf16) -> QantDriverError;
    /// See [`MulMatrixVecBatchedMemFn`].
    mul_matrix_vec_batched_mem(matrix: QantDriverHandle, vectors: QantDriverHandle, result: QantDriverHandle) -> QantDriverError;
    /// See [`GetVersionFn`].
    get_version(npu_id: u32, output: *mut c_char, len: usize) -> QantDriverError;
    /// See [`GetDeviceInfoFn`].
    get_device_info(npu_id: u32) -> QantDeviceInfo;
    /// See [`GetAvailableNpusFn`].
    get_available_npus(
        idxs_out: *mut u32,
        serials_out: *mut [c_char; C_STRING_DEFAULT_LENGTH],
        capacity: usize,
        out_count: *mut usize
    ) -> QantDriverError;
    /// See [`ResetPerfCounterFn`].
    reset_perf_counter(npu_id: u32) -> QantDriverError;
    /// See [`GetPerfCounterFn`].
    get_perf_counter(npu_id: u32, out: *mut PerformanceCounterInfo) -> QantDriverError;
}

impl fmt::Debug for DriverLibrary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DriverLibrary")
            .field("path", &self.path)
            .finish_non_exhaustive()
    }
}

impl DriverLibrary {
    /// Loads the driver from the default locations (see the crate docs).
    ///
    /// # Safety
    /// Loading a shared library runs its initialisers. The caller must trust
    /// the library found at the resolved location to be the Q.ANT driver with
    /// the ABI described in this crate.
    pub unsafe fn load() -> Result<Self, LoadError> {
        let mut candidates: Vec<OsString> = Vec::new();
        if let Some(path) = std::env::var_os(DRIVER_LIB_ENV).filter(|p| !p.is_empty()) {
            // A bare file name goes through the system loader's search path
            // (as intended); a relative path with directories would resolve
            // against whatever the current directory happens to be.
            let p = Path::new(&path);
            if p.is_relative() && p.components().count() > 1 {
                return Err(LoadError::RelativePath(path));
            }
            candidates.push(path);
        }
        candidates.extend(DEFAULT_LIBRARY_NAMES.iter().map(OsString::from));
        candidates.extend(DEFAULT_LIBRARY_PATHS.iter().map(OsString::from));

        let mut attempts = Vec::new();
        for candidate in candidates {
            // SAFETY: forwarded to the caller of `load`.
            match unsafe { libloading::Library::new(&candidate) } {
                Ok(lib) => return Self::from_library(lib, candidate),
                Err(err) => attempts.push((candidate, err)),
            }
        }
        Err(LoadError::NotFound(attempts))
    }

    /// Loads the driver from an explicit path.
    ///
    /// # Safety
    /// Same as [`DriverLibrary::load`].
    pub unsafe fn load_from(path: impl AsRef<Path>) -> Result<Self, LoadError> {
        let path = path.as_ref().as_os_str().to_owned();
        // SAFETY: forwarded to the caller.
        match unsafe { libloading::Library::new(&path) } {
            Ok(lib) => Self::from_library(lib, path),
            Err(err) => Err(LoadError::NotFound(vec![(path, err)])),
        }
    }

    fn from_library(lib: libloading::Library, path: OsString) -> Result<Self, LoadError> {
        macro_rules! sym {
            ($name:literal, $ty:ty) => {{
                // SAFETY: the symbol type matches the driver ABI documented by Q.ANT's toolkit.
                match unsafe { lib.get::<$ty>(concat!($name, "\0").as_bytes()) } {
                    Ok(s) => *s,
                    Err(source) => {
                        return Err(LoadError::MissingSymbol {
                            path,
                            symbol: $name,
                            source,
                        });
                    }
                }
            }};
        }
        let init_npu = sym!("qant_driver_init_npu", InitNpuFn);
        let release_npu = sym!("qant_driver_release_npu", ReleaseNpuFn);
        let setup_logging = sym!("qant_driver_setup_logging", SetupLoggingFn);
        let mul_npu = sym!("qant_driver_mul_npu", MulNpuFn);
        let mul_matrix_vec_batched =
            sym!("qant_driver_mul_matrix_vec_batched", MulMatrixVecBatchedFn);
        let scaled_periodic_nl_npu = sym!("qant_driver_scaled_periodic_nl_npu", ScaledPeriodicNlFn);
        let allocate = sym!("qant_driver_allocate", AllocateFn);
        let free = sym!("qant_driver_free", FreeFn);
        let upload = sym!("qant_driver_upload", UploadFn);
        let download = sym!("qant_driver_download", DownloadFn);
        let mul_matrix_vec_batched_mem = sym!(
            "qant_driver_mul_matrix_vec_batched_mem",
            MulMatrixVecBatchedMemFn
        );
        let get_version = sym!("qant_driver_get_version", GetVersionFn);
        let get_device_info = sym!("qant_driver_get_device_info", GetDeviceInfoFn);
        let get_available_npus = sym!("qant_driver_get_available_npus", GetAvailableNpusFn);
        let reset_perf_counter = sym!("qant_driver_reset_perf_counter", ResetPerfCounterFn);
        let get_perf_counter = sym!("qant_driver_get_perf_counter", GetPerfCounterFn);
        Ok(Self {
            path,
            init_npu,
            release_npu,
            setup_logging,
            mul_npu,
            mul_matrix_vec_batched,
            scaled_periodic_nl_npu,
            allocate,
            free,
            upload,
            download,
            mul_matrix_vec_batched_mem,
            get_version,
            get_device_info,
            get_available_npus,
            reset_perf_counter,
            get_perf_counter,
            _lib: lib,
        })
    }

    /// The path or name the library was loaded from.
    pub fn path(&self) -> &OsStr {
        &self.path
    }

    /// Returns the path as a `PathBuf` (convenience for display).
    pub fn path_buf(&self) -> PathBuf {
        PathBuf::from(&self.path)
    }
}

/// Converts a NUL-terminated fixed-size C buffer to a `String` (lossy).
/// Reads at most the buffer length, even if no NUL byte is present.
pub fn c_buf_to_string(buf: &[c_char]) -> String {
    // SAFETY: c_char and u8 have the same size and alignment.
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(buf.as_ptr().cast::<u8>(), buf.len()) };
    match CStr::from_bytes_until_nul(bytes) {
        Ok(s) => s.to_string_lossy().into_owned(),
        Err(_) => String::from_utf8_lossy(bytes).into_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_sizes() {
        assert_eq!(std::mem::size_of::<SensorInfo>(), 17 * 8);
        assert_eq!(std::mem::size_of::<QantDriverHandle>(), 16);
        assert_eq!(std::mem::size_of::<QantDataRole>(), 4);
        assert_eq!(std::mem::size_of::<PerformanceCounterInfo>(), 16);
    }

    #[test]
    fn error_roundtrip() {
        let e = QantDriverError::new(-3, "boom");
        assert!(!e.is_ok());
        assert_eq!(e.message(), "boom");
        assert!(QantDriverError::ok().is_ok());
        let long = "x".repeat(5000);
        assert_eq!(
            QantDriverError::new(1, &long).message().len(),
            C_STRING_DEFAULT_LENGTH - 1
        );
    }

    #[test]
    fn missing_library_reports_attempts() {
        // SAFETY: the path does not exist, nothing gets loaded.
        let err = unsafe { DriverLibrary::load_from("/nonexistent/libqant.so") }.unwrap_err();
        assert!(err.to_string().contains("/nonexistent/libqant.so"));
    }
}
