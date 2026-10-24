//! Hardware backend: the Q.ANT native computing driver, loaded at runtime.

use std::collections::HashMap;
use std::ffi::{CString, c_char};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use half::bf16;
use qanten_sys::{
    C_STRING_DEFAULT_LENGTH, DriverLibrary, PerformanceCounterInfo, QantDataRole, QantDriverError,
    QantDriverHandle, c_buf_to_string,
};

use super::{Backend, BackendKind, DataRole, NpuEntry, RawHandle, check_len};
use crate::error::{Error, Result};
use crate::info::{DeviceInfo, PerfCounters, SensorReadings, VersionInfo};

#[derive(Debug, Clone, Copy)]
struct Alloc {
    rows: usize,
    cols: usize,
    role: DataRole,
}

/// Backend that talks to Q.ANT hardware through `libqant_native_computing_driver`.
///
/// The driver reads and writes raw pointers with lengths it derives from the
/// allocation size. To keep that sound behind a safe API, this backend
/// remembers every allocation it made and checks slice lengths and handle
/// validity before calling into the driver.
#[derive(Debug)]
pub struct DriverBackend {
    lib: DriverLibrary,
    allocs: Mutex<HashMap<RawHandle, Alloc>>,
}

static GLOBAL: Mutex<Option<Arc<DriverBackend>>> = Mutex::new(None);

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn check(npu: u32, err: QantDriverError) -> Result<()> {
    if err.is_ok() {
        Ok(())
    } else {
        Err(Error::Driver {
            npu,
            code: err.code,
            message: err.message(),
        })
    }
}

fn to_sys_role(role: DataRole) -> QantDataRole {
    match role {
        DataRole::Weights => QantDataRole::Weights,
        DataRole::Features => QantDataRole::Features,
    }
}

fn to_sys_handle(h: RawHandle) -> QantDriverHandle {
    QantDriverHandle {
        npu_id: h.npu,
        key: h.key,
    }
}

impl DriverBackend {
    /// Loads the driver library (see [`qanten_sys::DriverLibrary::load`] for the
    /// lookup order). Prefer [`DriverBackend::global`], which loads it once.
    pub fn load() -> Result<Self> {
        // SAFETY: we load the library named by the user (env var) or the
        // standard Q.ANT install location and trust it to be the driver.
        let lib = unsafe { DriverLibrary::load() }
            .map_err(|e| Error::DriverUnavailable(e.to_string()))?;
        tracing::debug!(path = ?lib.path(), "loaded Q.ANT driver");
        Ok(Self {
            lib,
            allocs: Mutex::new(HashMap::new()),
        })
    }

    /// Loads the driver from an explicit path.
    pub fn load_from(path: impl AsRef<std::path::Path>) -> Result<Self> {
        // SAFETY: see `load`.
        let lib = unsafe { DriverLibrary::load_from(path) }
            .map_err(|e| Error::DriverUnavailable(e.to_string()))?;
        Ok(Self {
            lib,
            allocs: Mutex::new(HashMap::new()),
        })
    }

    /// The process-wide driver instance, loaded on first use.
    pub fn global() -> Result<Arc<Self>> {
        let mut slot = lock(&GLOBAL);
        if let Some(b) = slot.as_ref() {
            return Ok(Arc::clone(b));
        }
        let b = Arc::new(Self::load()?);
        *slot = Some(Arc::clone(&b));
        Ok(b)
    }

    /// The underlying raw library.
    pub fn library(&self) -> &DriverLibrary {
        &self.lib
    }

    fn alloc_of(&self, h: RawHandle) -> Result<Alloc> {
        lock(&self.allocs)
            .get(&h)
            .copied()
            .ok_or_else(|| Error::Driver {
                npu: h.npu,
                code: -1,
                message: "invalid memory handle".into(),
            })
    }

    /// Configures the driver's own log files.
    pub fn setup_logging(&self, folder: &std::path::Path, level: tracing::Level) -> Result<()> {
        let folder = CString::new(folder.to_string_lossy().as_bytes())
            .map_err(|_| Error::InvalidArgument("log folder contains a NUL byte".into()))?;
        let level = match level {
            tracing::Level::TRACE => 0,
            tracing::Level::DEBUG => 1,
            tracing::Level::INFO => 2,
            tracing::Level::WARN => 3,
            tracing::Level::ERROR => 4,
        };
        // SAFETY: valid NUL-terminated string that outlives the call.
        check(0, unsafe {
            (self.lib.setup_logging)(folder.as_ptr(), level)
        })
    }
}

impl Backend for DriverBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Driver
    }

    fn name(&self) -> String {
        format!("qanten driver ({})", self.lib.path().to_string_lossy())
    }

    fn init_npu(&self, npu: u32) -> Result<()> {
        // SAFETY: plain value call.
        check(npu, unsafe { (self.lib.init_npu)(npu) })
    }

    fn release_npu(&self, npu: u32) -> Result<()> {
        // SAFETY: plain value call.
        check(npu, unsafe { (self.lib.release_npu)(npu) })
    }

    fn available_npus(&self) -> Result<Vec<NpuEntry>> {
        const CAP: usize = 64;
        let mut ids = vec![0u32; CAP];
        let mut serials = vec![[0 as c_char; C_STRING_DEFAULT_LENGTH]; CAP];
        let mut count = 0usize;
        // SAFETY: both buffers hold CAP entries, as announced by `capacity`.
        let err = unsafe {
            (self.lib.get_available_npus)(ids.as_mut_ptr(), serials.as_mut_ptr(), CAP, &mut count)
        };
        check(0, err)?;
        Ok((0..count.min(CAP))
            .map(|i| NpuEntry {
                id: ids[i],
                serial: c_buf_to_string(&serials[i]),
            })
            .collect())
    }

    fn mul(&self, npu: u32, a: &[bf16], b: &[bf16], out: &mut [bf16]) -> Result<()> {
        check_len(npu, "mul rhs", b.len(), a.len())?;
        check_len(npu, "mul output", out.len(), a.len())?;
        if a.is_empty() {
            return Ok(());
        }
        // SAFETY: all three buffers hold `a.len()` elements.
        check(npu, unsafe {
            (self.lib.mul_npu)(npu, a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), a.len())
        })
    }

    fn periodic_nl(&self, npu: u32, u: &[bf16], v: &[bf16], out: &mut [bf16]) -> Result<()> {
        check_len(npu, "periodic_nl weights", v.len(), u.len())?;
        check_len(npu, "periodic_nl output", out.len(), u.len())?;
        if u.is_empty() {
            return Ok(());
        }
        // SAFETY: all three buffers hold `u.len()` elements.
        check(npu, unsafe {
            (self.lib.scaled_periodic_nl_npu)(
                npu,
                u.as_ptr(),
                v.as_ptr(),
                out.as_mut_ptr(),
                u.len(),
            )
        })
    }

    fn matvec_batched(
        &self,
        npu: u32,
        matrix: &[bf16],
        vectors: &[bf16],
        out: &mut [bf16],
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        check_len(npu, "matrix", matrix.len(), rows * cols)?;
        if cols == 0 || !vectors.len().is_multiple_of(cols) {
            return Err(Error::Shape(format!(
                "vector batch of {} values is not a multiple of {cols} columns",
                vectors.len()
            )));
        }
        let n_vecs = vectors.len() / cols;
        check_len(npu, "matvec output", out.len(), n_vecs * rows)?;
        if out.is_empty() {
            return Ok(());
        }
        // SAFETY: lengths checked above match (rows, cols, n_vecs).
        check(npu, unsafe {
            (self.lib.mul_matrix_vec_batched)(
                npu,
                matrix.as_ptr(),
                vectors.as_ptr(),
                out.as_mut_ptr(),
                rows,
                cols,
                n_vecs,
            )
        })
    }

    fn allocate(&self, npu: u32, rows: usize, cols: usize, role: DataRole) -> Result<RawHandle> {
        if rows.checked_mul(cols).is_none() {
            return Err(Error::InvalidArgument("allocation size overflows".into()));
        }
        let mut out = QantDriverHandle {
            npu_id: npu,
            key: 0,
        };
        // SAFETY: `out` is a valid destination for one handle.
        check(npu, unsafe {
            (self.lib.allocate)(npu, rows, cols, to_sys_role(role), &mut out)
        })?;
        let h = RawHandle {
            npu: out.npu_id,
            key: out.key,
        };
        lock(&self.allocs).insert(h, Alloc { rows, cols, role });
        Ok(h)
    }

    fn free(&self, handle: RawHandle) -> Result<()> {
        if lock(&self.allocs).remove(&handle).is_none() {
            return Err(Error::Driver {
                npu: handle.npu,
                code: -1,
                message: "invalid memory handle (double free?)".into(),
            });
        }
        // SAFETY: the handle came from `allocate` and was not freed before.
        check(handle.npu, unsafe {
            (self.lib.free)(to_sys_handle(handle))
        })
    }

    fn upload(&self, handle: RawHandle, data: &[bf16]) -> Result<()> {
        let a = self.alloc_of(handle)?;
        check_len(handle.npu, "upload", data.len(), a.rows * a.cols)?;
        // SAFETY: the driver reads rows*cols values, which `data` holds.
        check(handle.npu, unsafe {
            (self.lib.upload)(to_sys_handle(handle), data.as_ptr())
        })
    }

    fn download(&self, handle: RawHandle, out: &mut [bf16]) -> Result<()> {
        let a = self.alloc_of(handle)?;
        if a.role != DataRole::Features {
            return Err(Error::Unsupported(
                "weights memory cannot be downloaded".into(),
            ));
        }
        check_len(handle.npu, "download", out.len(), a.rows * a.cols)?;
        // SAFETY: the driver writes rows*cols values, which fit into `out`.
        check(handle.npu, unsafe {
            (self.lib.download)(to_sys_handle(handle), out.as_mut_ptr())
        })
    }

    fn matvec_resident(
        &self,
        matrix: RawHandle,
        vectors: RawHandle,
        result: RawHandle,
    ) -> Result<()> {
        // Validate handles so the driver never sees a stale one.
        self.alloc_of(matrix)?;
        self.alloc_of(vectors)?;
        self.alloc_of(result)?;
        // SAFETY: all handles are live allocations of this driver.
        check(matrix.npu, unsafe {
            (self.lib.mul_matrix_vec_batched_mem)(
                to_sys_handle(matrix),
                to_sys_handle(vectors),
                to_sys_handle(result),
            )
        })
    }

    fn driver_version(&self, npu: u32) -> Result<String> {
        let mut buf = vec![0 as c_char; C_STRING_DEFAULT_LENGTH];
        // SAFETY: `buf` holds `len` bytes.
        check(npu, unsafe {
            (self.lib.get_version)(npu, buf.as_mut_ptr(), buf.len())
        })?;
        Ok(c_buf_to_string(&buf))
    }

    fn device_info(&self, npu: u32) -> Result<DeviceInfo> {
        // SAFETY: plain value call returning a struct by value.
        let info = unsafe { (self.lib.get_device_info)(npu) };
        check(npu, info.error.clone())?;
        let s = info.sensor_info;
        Ok(DeviceInfo {
            sensors: SensorReadings {
                temp_pd1: s.temp_pd1,
                temp_pd2: s.temp_pd2,
                temp_pd3: s.temp_pd3,
                temp_dac_1: s.temp_dac_1,
                temp_dac_2: s.temp_dac_2,
                voltage_1v8: s.voltage_1v8,
                voltage_3v3: s.voltage_3v3,
                voltage_6v0_1: s.voltage_6v0_1,
                voltage_6v0_2: s.voltage_6v0_2,
                voltage_6v2: s.voltage_6v2,
                voltage_6v5: s.voltage_6v5,
                voltage_8v0_1: s.voltage_8v0_1,
                voltage_8v0_2: s.voltage_8v0_2,
                voltage_12v: s.voltage_12v,
                voltage_12v5: s.voltage_12v5,
                voltage_n12v5: s.voltage_n12v5,
                current_12v: s.current_12v,
            },
            version: VersionInfo {
                firmware: c_buf_to_string(&info.version_info.fw_version),
                zephyr: c_buf_to_string(&info.version_info.zephyr_version),
                board_serial: c_buf_to_string(&info.version_info.board_serial_no),
            },
        })
    }

    fn perf_counters(&self, npu: u32) -> Result<PerfCounters> {
        let mut out = PerformanceCounterInfo::default();
        // SAFETY: `out` is a valid destination.
        check(npu, unsafe { (self.lib.get_perf_counter)(npu, &mut out) })?;
        Ok(PerfCounters {
            elapsed: Duration::from_secs_f64(out.timebased_counter.max(0.0)),
            stalled: Duration::from_secs_f64(out.stalling_counter.max(0.0)),
        })
    }

    fn reset_perf_counters(&self, npu: u32) -> Result<()> {
        // SAFETY: plain value call.
        check(npu, unsafe { (self.lib.reset_perf_counter)(npu) })
    }
}
