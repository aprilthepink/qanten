//! A fake `libqant_native_computing_driver.so` exporting the driver's C ABI.
//!
//! Every entry point forwards to the qanten simulator. It exists so CI can
//! exercise the real `dlopen` + FFI path of `qanten::backend::DriverBackend`
//! without Q.ANT hardware. Not published.
//!
//! The simulator is configured with 2 NPUs. Set `QANTEN_MOCK_NOISE` to add noise.

#![allow(clippy::missing_safety_doc)]

use std::ffi::{CStr, c_char, c_int};
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

/// An entry point's result before it becomes a C status; the error is the message.
type Checked<T> = Result<T, String>;

fn error(msg: &str) -> QantDriverError {
    QantDriverError::new(-1, msg)
}

/// Runs an entry point body, turning its `Err` into the returned status.
fn run(body: impl FnOnce() -> Checked<QantDriverError>) -> QantDriverError {
    body().unwrap_or_else(|msg| error(&msg))
}

/// `a · b` elements, or an error instead of a wrapped-around size.
fn count(a: usize, b: usize) -> Checked<usize> {
    a.checked_mul(b).ok_or_else(|| "size overflow".to_string())
}

/// Rejects pointers `from_raw_parts` may not be given: null or misaligned.
/// A zero-length buffer needs no pointer at all.
fn check_ptr<T>(p: *const T, n: usize, what: &str) -> Checked<()> {
    if n > 0 && (p.is_null() || !p.is_aligned()) {
        return Err(format!("{what}: null or misaligned pointer"));
    }
    if n.checked_mul(size_of::<T>())
        .is_none_or(|b| b > isize::MAX as usize)
    {
        return Err(format!("{what}: buffer too large"));
    }
    Ok(())
}

unsafe fn slice<'a, T>(p: *const T, n: usize, what: &str) -> Checked<&'a [T]> {
    check_ptr(p, n, what)?;
    Ok(if n == 0 {
        &[]
    } else {
        // SAFETY: non-null and aligned (checked); length is the caller's contract.
        unsafe { std::slice::from_raw_parts(p, n) }
    })
}

unsafe fn slice_mut<'a, T>(p: *mut T, n: usize, what: &str) -> Checked<&'a mut [T]> {
    check_ptr(p, n, what)?;
    Ok(if n == 0 {
        &mut []
    } else {
        // SAFETY: as for `slice`.
        unsafe { std::slice::from_raw_parts_mut(p, n) }
    })
}

/// Writes one value through an out-pointer, rejecting null and misalignment.
unsafe fn write_out<T>(p: *mut T, value: T, what: &str) -> Checked<()> {
    check_ptr(p, 1, what)?;
    // SAFETY: non-null and aligned (checked); validity is the caller's contract.
    unsafe { p.write(value) };
    Ok(())
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
        return error("empty log folder");
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
    run(|| unsafe {
        Ok(status(sim().mul(
            id,
            slice(us, len, "us")?,
            slice(vs, len, "vs")?,
            slice_mut(out, len, "out")?,
        )))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn qant_driver_scaled_periodic_nl_npu(
    id: u32,
    us: *const bf16,
    vs: *const bf16,
    out: *mut bf16,
    len: usize,
) -> QantDriverError {
    run(|| unsafe {
        Ok(status(sim().periodic_nl(
            id,
            slice(us, len, "us")?,
            slice(vs, len, "vs")?,
            slice_mut(out, len, "out")?,
        )))
    })
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
    run(|| unsafe {
        Ok(status(sim().matvec_batched(
            id,
            slice(matrix, count(n_rows, n_cols)?, "matrix")?,
            slice(vectors, count(n_vecs, n_cols)?, "vectors")?,
            slice_mut(out, count(n_vecs, n_rows)?, "out")?,
            n_rows,
            n_cols,
        )))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn qant_driver_allocate(
    id: u32,
    rows: usize,
    cols: usize,
    // Taken as a plain integer: a C caller can pass any value, and turning
    // an out-of-range one into a Rust enum would be undefined behaviour.
    role: c_int,
    out: *mut QantDriverHandle,
) -> QantDriverError {
    let role = match role {
        r if r == QantDataRole::Weights as c_int => DataRole::Weights,
        r if r == QantDataRole::Features as c_int => DataRole::Features,
        _ => return error("invalid data role"),
    };
    run(|| {
        // Validate everything before allocating, so a bad call leaks nothing.
        check_ptr(out, 1, "out")?;
        let len = count(rows, cols)?;
        let h = sim()
            .allocate(id, rows, cols, role)
            .map_err(|e| e.to_string())?;
        if let Ok(mut l) = lens().lock() {
            l.insert(h.key, len);
        }
        let handle = QantDriverHandle {
            npu_id: h.npu,
            key: h.key,
        };
        unsafe { write_out(out, handle, "out")? };
        Ok(QantDriverError::ok())
    })
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
        return error("invalid memory handle");
    };
    run(|| unsafe { Ok(status(sim().upload(raw(h), slice(data, n, "data")?))) })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn qant_driver_download(
    h: QantDriverHandle,
    out: *mut bf16,
) -> QantDriverError {
    let Some(n) = len_of(h) else {
        return error("invalid memory handle");
    };
    run(|| unsafe { Ok(status(sim().download(raw(h), slice_mut(out, n, "out")?))) })
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
    run(|| {
        let dst = unsafe { slice_mut(out, len, "out")? };
        if dst.is_empty() {
            return Err("version buffer is empty".into());
        }
        let v = sim().driver_version(id).map_err(|e| e.to_string())?;
        write_c_str(dst, &format!("mock-driver;{v}"));
        Ok(QantDriverError::ok())
    })
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
    run(|| {
        let ids = unsafe { slice_mut(ids, capacity, "ids")? };
        let serials = unsafe { slice_mut(serials, capacity, "serials")? };
        check_ptr(count, 1, "count")?;
        let list = sim()
            .available_npus()
            .map_err(|_| "enumeration failed".to_string())?;
        let n = list.len().min(capacity);
        for ((e, id), serial) in list.iter().zip(ids).zip(serials) {
            *id = e.id;
            write_c_str(serial, &e.serial.replace("SIM", "MOCK"));
        }
        unsafe { write_out(count, n, "count")? };
        Ok(QantDriverError::ok())
    })
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
    run(|| {
        check_ptr(out, 1, "out")?;
        let p = sim().perf_counters(id).map_err(|e| e.to_string())?;
        let info = PerformanceCounterInfo {
            timebased_counter: p.elapsed.as_secs_f64(),
            stalling_counter: p.stalled.as_secs_f64(),
        };
        unsafe { write_out(out, info, "out")? };
        Ok(QantDriverError::ok())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr::{null, null_mut};

    fn fails_with(status: QantDriverError, text: &str) {
        assert!(!status.is_ok(), "expected an error mentioning {text:?}");
        assert!(status.message().contains(text), "{}", status.message());
    }

    #[test]
    fn null_and_misaligned_pointers_are_errors_not_crashes() {
        let buf = [bf16::from_f32(1.0); 4];
        let mut out = [bf16::ZERO; 4];
        unsafe {
            fails_with(
                qant_driver_mul_npu(0, null(), buf.as_ptr(), out.as_mut_ptr(), 4),
                "us",
            );
            fails_with(
                qant_driver_mul_npu(0, buf.as_ptr(), buf.as_ptr(), null_mut(), 4),
                "out",
            );
            let odd = buf.as_ptr().cast::<u8>().wrapping_add(1).cast::<bf16>();
            fails_with(
                qant_driver_mul_npu(0, odd, buf.as_ptr(), out.as_mut_ptr(), 2),
                "misaligned",
            );
            fails_with(
                qant_driver_mul_matrix_vec_batched(
                    0,
                    buf.as_ptr(),
                    null(),
                    out.as_mut_ptr(),
                    2,
                    2,
                    1,
                ),
                "vectors",
            );
            fails_with(
                qant_driver_mul_matrix_vec_batched(
                    0,
                    buf.as_ptr(),
                    buf.as_ptr(),
                    out.as_mut_ptr(),
                    usize::MAX,
                    2,
                    1,
                ),
                "overflow",
            );
            fails_with(qant_driver_allocate(0, 2, 2, 0, null_mut()), "out");
            fails_with(qant_driver_allocate(0, 2, 2, 7, null_mut()), "role");
            fails_with(qant_driver_get_version(0, null_mut(), 64), "out");
            fails_with(qant_driver_get_version(0, null_mut(), 0), "empty");
            fails_with(qant_driver_get_perf_counter(0, null_mut()), "out");
            fails_with(qant_driver_setup_logging(null(), 0), "log folder");
            let mut n = 0;
            fails_with(
                qant_driver_get_available_npus(null_mut(), null_mut(), 4, &mut n),
                "ids",
            );
            fails_with(
                qant_driver_get_available_npus(null_mut(), null_mut(), 0, null_mut()),
                "count",
            );
        }
    }

    #[test]
    fn handle_calls_reject_null_data() {
        assert!(qant_driver_init_npu(1).is_ok());
        let mut h = QantDriverHandle { npu_id: 0, key: 0 };
        unsafe {
            assert!(qant_driver_allocate(1, 2, 2, QantDataRole::Features as c_int, &mut h).is_ok());
            fails_with(qant_driver_upload(h, null()), "data");
            fails_with(qant_driver_download(h, null_mut()), "out");
            let data = [bf16::from_f32(2.0); 4];
            let mut back = [bf16::ZERO; 4];
            assert!(qant_driver_upload(h, data.as_ptr()).is_ok());
            assert!(qant_driver_download(h, back.as_mut_ptr()).is_ok());
            assert_eq!(back, data);
        }
        assert!(qant_driver_free(h).is_ok());
        // Zero-length calls need no pointers at all.
        unsafe { assert!(qant_driver_mul_npu(1, null(), null(), null_mut(), 0).is_ok()) };
        assert!(qant_driver_release_npu(1).is_ok());
    }
}
