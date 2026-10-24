//! End-to-end test of the hardware backend through a real `dlopen`.
//!
//! Runs against the mock driver in CI:
//! ```sh
//! cargo build -p qant-mock-driver
//! QANT_DRIVER_LIB=$PWD/target/debug/libqant_native_computing_driver.so cargo test -p qant --test driver_ffi
//! ```
//! On a machine with Q.ANT hardware, run it with the real driver installed
//! (and `QANT_DRIVER_LIB` unset) to check the actual device.
#![cfg(feature = "driver")]

use qant::nn::{self, Module};
use qant::ops::{self, Conv2dParams};
use qant::{BackendKind, DeviceBuffer, Features, Npu, Tensor, Weights};

fn driver_npu() -> Option<Npu> {
    if std::env::var_os(qant::sys::DRIVER_LIB_ENV).is_some() {
        // Explicitly requested: a load failure is a test failure, not a skip.
        if let Err(e) = qant::backend::DriverBackend::load() {
            panic!("{e}");
        }
    }
    if !qant::driver_available() {
        eprintln!("skipping: no Q.ANT driver (set QANT_DRIVER_LIB to the mock driver to run)");
        return None;
    }
    Some(Npu::open(0).expect("driver loads but NPU 0 does not open"))
}

#[test]
fn driver_end_to_end() {
    let Some(npu) = driver_npu() else { return };
    let sim = Npu::simulator();
    assert_eq!(npu.backend_kind(), BackendKind::Driver);

    // Opening the same id again shares one handle (one init/release pair).
    let again = Npu::open(0).unwrap();
    assert!(again.same_device(&npu));

    let devs = qant::devices().unwrap();
    assert!(!devs.is_empty());
    assert!(!devs[0].serial.is_empty());

    let info = npu.device_info().unwrap();
    assert!(info.sensors.voltage_12v > 10.0, "{info:?}");
    assert!(!info.version.board_serial.is_empty());
    assert!(!npu.driver_version().unwrap().is_empty());
    npu.reset_perf_counters().unwrap();
    let p = npu.perf_counters().unwrap();
    assert!(p.stalled <= p.elapsed + std::time::Duration::from_millis(1));

    let tol = 0.05;
    let a = Tensor::rand_uniform([3, 33], -1., 1., 1);
    let b = Tensor::rand_uniform([3, 33], -1., 1., 2);
    let w = Tensor::rand_uniform([17, 33], -0.5, 0.5, 3);
    for (name, got, want) in [
        (
            "mul",
            ops::mul(&npu, &a, &b).unwrap(),
            ops::mul(&sim, &a, &b).unwrap(),
        ),
        (
            "nl",
            ops::periodic_nl(&npu, &a, &b).unwrap(),
            ops::periodic_nl(&sim, &a, &b).unwrap(),
        ),
        (
            "linear",
            ops::linear(&npu, &a, &w, None).unwrap(),
            ops::linear(&sim, &a, &w, None).unwrap(),
        ),
    ] {
        let err = got.max_abs_diff(&want).unwrap();
        assert!(err < tol, "{name}: error {err}");
    }

    let img = Tensor::rand_uniform([2, 3, 9, 9], 0., 1., 4);
    let k = Tensor::rand_uniform([4, 3, 3, 3], -0.5, 0.5, 5);
    let p = Conv2dParams::new().padding(1);
    let conv = nn::Conv2d::new(&npu, &k, None, p).unwrap();
    let err = conv
        .forward(&img)
        .unwrap()
        .max_abs_diff(&ops::conv2d(&sim, &img, &k, None, p).unwrap())
        .unwrap();
    assert!(err < tol, "conv: error {err}");

    // Typed buffers through the FFI.
    let xb = DeviceBuffer::<Features>::from_tensor(&npu, &a).unwrap();
    assert_eq!(xb.download().unwrap(), a);
    let wb = DeviceBuffer::<Weights>::from_tensor(&npu, &w).unwrap();
    let mut out = DeviceBuffer::<Features>::alloc(&npu, 3, 17).unwrap();
    qant::matvec_resident(&wb, &xb, &mut out).unwrap();
    assert_eq!(out.download().unwrap().shape(), &[3, 17]);

    // Errors from the driver are propagated, not panics.
    assert!(Npu::open(99).is_err());
}
