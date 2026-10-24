//! A fake `libqant_native_computing_driver.so` exporting the driver's C ABI.
//!
//! Every entry point forwards to the qanten simulator. It exists so CI can
//! exercise the real `dlopen` + FFI path of `qanten::backend::DriverBackend`
//! without Q.ANT hardware. Not published.
//!
//! The simulator is configured with 2 NPUs. Set `QANTEN_MOCK_NOISE` to add noise.

#![allow(clippy::missing_safety_doc)]

use std::ffi::{CStr, c_char};
use std::sync::OnceLock;

use qanten::backend::{Backend, DataRole, RawHandle, SimConfig, Simulator};
use qanten_sys::{
    C_STRING_DEFAULT_LENGTH, PerformanceCounterInfo, QantDataRole, QantDeviceInfo, QantDriverError,
    QantDriverHandle, QantVersionInfo, SensorInfo, bf16,
};

fn sim() -> &'static Simulator {
    static SIM: OnceLock<Simulator> = OnceLock::new();
    SIM.get_or_init(|| {
        Simulator::new(SimConfig {
            num_npus: 2,
            noise_std: std::env::var("QANTEN_MOCK_NOISE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.0),
            ..SimConfig::default()
        })
    })
}

fn status(r: qanten::Result<()>) -> QantDriverError {
    match r {
        Ok(()) => QantDriverError::ok(),
        Err(e) => QantDriverError::new(-1, &e.to_string()),
    }
}

fn raw(h: QantDriverHandle) -> RawHandle {
    RawHandle {
        npu: h.npu_id,
        key: h.key,
    }
}

/// Elements stored behind a handle, needed to build slices from raw pointers.
/// The real driver knows this from its own bookkeeping; we keep a side table.
fn len_of(h: QantDriverHandle) -> Option<usize> {
    lens().lock().ok()?.get(&h.key).copied()
}

fn lens() -> &'static std::sync::Mutex<std::collections::HashMap<u64, usize>> {
    static L: OnceLock<std::sync::Mutex<std::collections::HashMap<u64, usize>>> = OnceLock::new();
    L.get_or_init(Default::default)
}

unsafe fn slice<'a>(p: *const bf16, n: usize) -> &'a [bf16] {
    if n == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(p, n) }
    }
}

unsafe fn slice_mut<'a>(p: *mut bf16, n: usize) -> &'a mut [bf16] {
    if n == 0 {
        &mut []
    } else {
        unsafe { std::slice::from_raw_parts_mut(p, n) }
    }
}

fn write_c_str(dst: &mut [c_char], s: &str) {
    let n = s.len().min(dst.len().saturating_sub(1));
    for (d, b) in dst.iter_mut().zip(s.bytes().take(n)) {
        *d = b as c_char;
    }
    if let Some(d) = dst.get_mut(n) {
        *d = 0;
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn qant_driver_init_npu(id: u32) -> QantDriverError {
    status(sim().init_npu(id))
}

#[unsafe(no_mangle)]
pub extern "C" fn qant_driver_release_npu(id: u32) -> QantDriverError {
    status(sim().release_npu(id))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn qant_driver_setup_logging(
    folder: *const c_char,
    _level: i32,
) -> QantDriverError {
    if folder.is_null() || unsafe { CStr::from_ptr(folder) }.to_bytes().is_empty() {
        return QantDriverError::new(-1, "empty log folder");
    }
    QantDriverError::ok()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn qant_driver_mul_npu(
    id: u32,
    us: *const bf16,
    vs: *const bf16,
    out: *mut bf16,
    len: usize,
) -> QantDriverError {
    unsafe { status(sim().mul(id, slice(us, len), slice(vs, len), slice_mut(out, len))) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn qant_driver_scaled_periodic_nl_npu(
    id: u32,
    us: *const bf16,
    vs: *const bf16,
    out: *mut bf16,
    len: usize,
) -> QantDriverError {
    unsafe { status(sim().periodic_nl(id, slice(us, len), slice(vs, len), slice_mut(out, len))) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn qant_driver_mul_matrix_vec_batched(
    id: u32,
    matrix: *const bf16,
    vectors: *const bf16,
    out: *mut bf16,
    n_rows: usize,
    n_cols: usize,
    n_vecs: usize,
) -> QantDriverError {
    unsafe {
        status(sim().matvec_batched(
            id,
            slice(matrix, n_rows * n_cols),
            slice(vectors, n_vecs * n_cols),
            slice_mut(out, n_vecs * n_rows),
            n_rows,
            n_cols,
        ))
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn qant_driver_allocate(
    id: u32,
    rows: usize,
    cols: usize,
    role: QantDataRole,
    out: *mut QantDriverHandle,
) -> QantDriverError {
    let role = match role {
        QantDataRole::Weights => DataRole::Weights,
        QantDataRole::Features => DataRole::Features,
    };
    match sim().allocate(id, rows, cols, role) {
        Ok(h) => {
            if let Ok(mut l) = lens().lock() {
                l.insert(h.key, rows * cols);
            }
            unsafe {
                *out = QantDriverHandle {
                    npu_id: h.npu,
                    key: h.key,
                };
            }
            QantDriverError::ok()
        }
        Err(e) => QantDriverError::new(-1, &e.to_string()),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn qant_driver_free(h: QantDriverHandle) -> QantDriverError {
    if let Ok(mut l) = lens().lock() {
        l.remove(&h.key);
    }
    status(sim().free(raw(h)))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn qant_driver_upload(
    h: QantDriverHandle,
    data: *const bf16,
) -> QantDriverError {
    let Some(n) = len_of(h) else {
        return QantDriverError::new(-1, "invalid memory handle");
    };
    unsafe { status(sim().upload(raw(h), slice(data, n))) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn qant_driver_download(
    h: QantDriverHandle,
    out: *mut bf16,
) -> QantDriverError {
    let Some(n) = len_of(h) else {
        return QantDriverError::new(-1, "invalid memory handle");
    };
    unsafe { status(sim().download(raw(h), slice_mut(out, n))) }
}

#[unsafe(no_mangle)]
pub extern "C" fn qant_driver_mul_matrix_vec_batched_mem(
    m: QantDriverHandle,
    v: QantDriverHandle,
    r: QantDriverHandle,
) -> QantDriverError {
    status(sim().matvec_resident(raw(m), raw(v), raw(r)))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn qant_driver_get_version(
    id: u32,
    out: *mut c_char,
    len: usize,
) -> QantDriverError {
    match sim().driver_version(id) {
        Ok(v) => {
            let dst = unsafe { std::slice::from_raw_parts_mut(out, len) };
            write_c_str(dst, &format!("mock-driver;{v}"));
            QantDriverError::ok()
        }
        Err(e) => QantDriverError::new(-1, &e.to_string()),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn qant_driver_get_device_info(id: u32) -> QantDeviceInfo {
    let mut version = QantVersionInfo {
        fw_version: [0; C_STRING_DEFAULT_LENGTH],
        zephyr_version: [0; C_STRING_DEFAULT_LENGTH],
        board_serial_no: [0; C_STRING_DEFAULT_LENGTH],
    };
    match sim().device_info(id) {
        Ok(info) => {
            let s = info.sensors;
            write_c_str(&mut version.fw_version, "mock-fw-1.0");
            write_c_str(&mut version.zephyr_version, "mock-zephyr");
            write_c_str(&mut version.board_serial_no, &format!("MOCK-{id:04}"));
            QantDeviceInfo {
                sensor_info: SensorInfo {
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
                    current_12v: 7.5,
                },
                version_info: version,
                error: QantDriverError::ok(),
            }
        }
        Err(e) => QantDeviceInfo {
            sensor_info: SensorInfo::default(),
            version_info: version,
            error: QantDriverError::new(-1, &e.to_string()),
        },
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn qant_driver_get_available_npus(
    ids: *mut u32,
    serials: *mut [c_char; C_STRING_DEFAULT_LENGTH],
    capacity: usize,
    count: *mut usize,
) -> QantDriverError {
    let Ok(list) = sim().available_npus() else {
        return QantDriverError::new(-1, "enumeration failed");
    };
    let n = list.len().min(capacity);
    unsafe {
        let ids = std::slice::from_raw_parts_mut(ids, capacity);
        let serials = std::slice::from_raw_parts_mut(serials, capacity);
        for (i, e) in list.iter().take(n).enumerate() {
            ids[i] = e.id;
            write_c_str(&mut serials[i], &e.serial.replace("SIM", "MOCK"));
        }
        *count = n;
    }
    QantDriverError::ok()
}

#[unsafe(no_mangle)]
pub extern "C" fn qant_driver_reset_perf_counter(id: u32) -> QantDriverError {
    status(sim().reset_perf_counters(id))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn qant_driver_get_perf_counter(
    id: u32,
    out: *mut PerformanceCounterInfo,
) -> QantDriverError {
    match sim().perf_counters(id) {
        Ok(p) => {
            unsafe {
                *out = PerformanceCounterInfo {
                    timebased_counter: p.elapsed.as_secs_f64(),
                    stalling_counter: p.stalled.as_secs_f64(),
                };
            }
            QantDriverError::ok()
        }
        Err(e) => QantDriverError::new(-1, &e.to_string()),
    }
}
