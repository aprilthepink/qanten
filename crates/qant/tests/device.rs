use std::sync::Arc;

use qant::backend::{Backend, Simulator};
use qant::nn::{self, Module};
use qant::{DeviceBuffer, Error, Features, Npu, SimConfig, Tensor, Weights, matvec_resident};

#[test]
fn device_buffers_roundtrip_and_resident_matvec() {
    let npu = Npu::simulator();
    let w = Tensor::from_f32(&[1., 2., 3., 4., 5., 6.], [2, 3]).unwrap();
    let x = Tensor::from_f32(&[1., 0., 0., 0., 1., 1.], [2, 3]).unwrap();
    let wb = DeviceBuffer::<Weights>::from_tensor(&npu, &w).unwrap();
    let xb = DeviceBuffer::<Features>::from_tensor(&npu, &x).unwrap();
    assert_eq!(xb.download().unwrap(), x);
    let mut out = DeviceBuffer::<Features>::alloc(&npu, 2, 2).unwrap();
    matvec_resident(&wb, &xb, &mut out).unwrap();
    assert_eq!(out.download().unwrap().to_f32_vec(), vec![1., 4., 5., 11.]);
    // Wrong result shape is rejected before reaching the backend.
    let mut bad = DeviceBuffer::<Features>::alloc(&npu, 3, 2).unwrap();
    assert!(matches!(
        matvec_resident(&wb, &xb, &mut bad),
        Err(Error::Shape(_))
    ));
    assert!(xb.upload(&Tensor::zeros([3, 2])).is_err());
}

#[test]
fn buffers_and_layers_free_device_memory() {
    let sim = Arc::new(Simulator::default());
    let npu = Npu::simulated(&sim, 0).unwrap();
    {
        let _a = DeviceBuffer::<Weights>::alloc(&npu, 4, 4).unwrap();
        let _l = nn::Linear::new(&npu, &Tensor::ones([3, 4]), None).unwrap();
        let _c = nn::Conv2d::new(
            &npu,
            &Tensor::ones([4, 1, 3, 3]),
            None,
            qant::ops::Conv2dParams::new().groups(1),
        )
        .unwrap();
        let y = _l.forward(&Tensor::ones([2, 4])).unwrap();
        assert_eq!(y.to_f32_vec(), vec![4.0; 6]);
        assert_eq!(sim.live_allocations(), 3);
    }
    assert_eq!(sim.live_allocations(), 0, "device memory leaked");
}

#[test]
fn simulator_memory_limit_and_multi_npu() {
    let sim = Arc::new(Simulator::new(SimConfig {
        num_npus: 2,
        memory_limit_bytes: Some(64),
        ..SimConfig::default()
    }));
    let a = Npu::simulated(&sim, 0).unwrap();
    let b = Npu::simulated(&sim, 1).unwrap();
    assert!(Npu::simulated(&sim, 2).is_err());
    assert_eq!(sim.available_npus().unwrap().len(), 2);
    let _ok = DeviceBuffer::<Features>::alloc(&a, 4, 8).unwrap(); // 64 bytes
    assert!(
        DeviceBuffer::<Features>::alloc(&a, 1, 1).is_err(),
        "limit exceeded"
    );
    let _other = DeviceBuffer::<Features>::alloc(&b, 4, 8).unwrap(); // other NPU has its own budget

    let w = DeviceBuffer::<Weights>::alloc(&b, 1, 1);
    assert!(w.is_err());
    let wa = DeviceBuffer::<Weights>::alloc(&Npu::simulator(), 1, 1).unwrap();
    let xa = DeviceBuffer::<Features>::alloc(&Npu::simulator(), 1, 1).unwrap();
    let mut rb = DeviceBuffer::<Features>::alloc(
        &Npu::simulated(&Arc::new(Simulator::default()), 0).unwrap(),
        1,
        1,
    )
    .unwrap();
    assert!(matches!(
        matvec_resident(&wa, &xa, &mut rb),
        Err(Error::DeviceMismatch(_))
    ));
}

#[test]
fn noise_model_perturbs_results() {
    let exact = Npu::simulator();
    let noisy = Npu::simulator_with(SimConfig {
        noise_std: 0.05,
        ..SimConfig::default()
    });
    let a = Tensor::rand_uniform([1000], -1., 1., 1);
    let b = Tensor::ones([1000]);
    let e = qant::ops::mul(&exact, &a, &b).unwrap();
    let n = qant::ops::mul(&noisy, &a, &b).unwrap();
    let diff = e.max_abs_diff(&n).unwrap();
    assert!(diff > 0.05 && diff < 0.5, "unexpected noise level {diff}");
}

#[test]
fn many_threads_share_one_npu() {
    let npu = Npu::simulator();
    let w =
        Arc::new(nn::Linear::new(&npu, &Tensor::rand_uniform([16, 32], -1., 1., 1), None).unwrap());
    let handles: Vec<_> = (0..16)
        .map(|i| {
            let w = Arc::clone(&w);
            std::thread::spawn(move || {
                let x = Tensor::rand_uniform([4, 32], -1., 1., i);
                w.forward(&x).unwrap().shape().to_vec()
            })
        })
        .collect();
    for h in handles {
        assert_eq!(h.join().unwrap(), vec![4, 16]);
    }
}

#[test]
fn info_and_perf_counters() {
    let npu = Npu::simulator();
    let info = npu.device_info().unwrap();
    assert!(info.sensors.voltage_12v > 10.0);
    assert!(npu.driver_version().unwrap().contains("simulator"));
    npu.reset_perf_counters().unwrap();
    qant::ops::mul(&npu, &Tensor::ones([10_000]), &Tensor::ones([10_000])).unwrap();
    let p = npu.perf_counters().unwrap();
    assert!(p.stalled <= p.elapsed);
    assert!((0.0..=1.0).contains(&p.utilization()));
}

#[test]
fn backend_selection() {
    // With no driver installed (CI), `auto` must fall back to the simulator.
    if !qant::driver_available() {
        assert!(Npu::auto(0).unwrap().is_simulated());
        assert!(matches!(
            Npu::open(0),
            Err(Error::DriverUnavailable(_) | Error::DriverFeatureDisabled)
        ));
        assert!(qant::devices().is_err());
    }
}

#[test]
fn sequential_model_and_state_dict() {
    let npu = Npu::simulator();
    let mut sd = nn::StateDict::new();
    sd.insert(
        "features.0.weight",
        Tensor::rand_uniform([4, 1, 3, 3], -1., 1., 1),
    );
    sd.insert("features.0.bias", Tensor::zeros([4]));
    sd.insert("features.1.running_mean", Tensor::zeros([4]));
    sd.insert("features.1.running_var", Tensor::ones([4]));
    sd.insert("features.1.num_batches_tracked", Tensor::zeros([]));
    sd.insert(
        "classifier.weight",
        Tensor::rand_uniform([3, 16], -1., 1., 2),
    );
    let p = qant::ops::Conv2dParams::new().padding(1);
    let model = nn::Sequential::new()
        .add(nn::Conv2d::load(&npu, &sd, "features.0", p).unwrap())
        .add(nn::BatchNorm2d::load(&npu, &sd, "features.1", 1e-5).unwrap())
        .add(nn::ReLU)
        .add(nn::MaxPool2d::new(2))
        .add(nn::AdaptiveAvgPool2d::new(2))
        .add(nn::Flatten::default())
        .add(nn::Linear::load(&npu, &sd, "classifier").unwrap())
        .add(nn::Softmax);
    let y = model
        .forward(&Tensor::rand_uniform([2, 1, 8, 8], 0., 1., 3))
        .unwrap();
    assert_eq!(y.shape(), &[2, 3]);
    for row in y.to_f32_vec().chunks(3) {
        assert!((row.iter().sum::<f32>() - 1.0).abs() < 0.02);
    }
    assert_eq!(
        model
            .forward_trace(&Tensor::zeros([1, 1, 8, 8]))
            .unwrap()
            .len(),
        8
    );
    assert!(matches!(
        nn::Linear::load(&npu, &sd, "missing"),
        Err(Error::MissingTensor(_))
    ));
    assert_eq!(sd.sub("features.1").len(), 3);
}

#[cfg(feature = "safetensors")]
#[test]
fn safetensors_roundtrip() {
    let mut sd = nn::StateDict::new();
    sd.insert("a.weight", Tensor::rand_uniform([3, 4], -1., 1., 1));
    sd.insert("a.bias", Tensor::vector(&[1., 2., 3.]));
    let bytes = sd.to_safetensors_bytes().unwrap();
    let back = nn::StateDict::from_safetensors_bytes(&bytes).unwrap();
    assert_eq!(back.get("a.weight").unwrap(), sd.get("a.weight").unwrap());
    assert_eq!(back.get("a.bias").unwrap().to_f32_vec(), vec![1., 2., 3.]);
}

#[cfg(feature = "ndarray")]
#[test]
fn ndarray_conversion() {
    let a =
        ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 2]), vec![1.0f32, 2., 3., 4.]).unwrap();
    let t = Tensor::from(&a);
    assert_eq!(t.shape(), &[2, 2]);
    assert_eq!(t.to_ndarray(), a);
}
