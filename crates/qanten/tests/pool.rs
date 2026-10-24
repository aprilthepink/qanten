use std::sync::Arc;

use qanten::backend::Simulator;
use qanten::fallback::FallbackPolicy;
use qanten::nn::{self, Module};
use qanten::pool::NpuPool;
use qanten::profile::OpKind;
use qanten::{Npu, SimConfig, Tensor, ops};

#[test]
fn split_and_concat() {
    let t = Tensor::rand_uniform([7, 3], -1., 1., 1);
    let parts = t.split_batch(3);
    assert_eq!(
        parts.iter().map(|p| p.shape()[0]).collect::<Vec<_>>(),
        vec![3, 2, 2]
    );
    assert_eq!(Tensor::concat_batch(&parts).unwrap(), t);
    assert_eq!(t.split_batch(10).len(), 7, "never more parts than rows");
    assert!(Tensor::concat_batch(&[Tensor::zeros([1, 2]), Tensor::zeros([1, 3])]).is_err());
}

#[test]
fn replicated_model_matches_single_npu_and_uses_every_npu() {
    let sim = Arc::new(Simulator::new(SimConfig {
        num_npus: 3,
        ..SimConfig::default()
    }));
    let handles: Vec<(Npu, _)> = (0..3)
        .map(|i| Npu::simulated(&sim, i).unwrap().profiled())
        .collect();
    let pool = NpuPool::new(handles.iter().map(|(n, _)| n.clone()).collect()).unwrap();

    let w1 = Tensor::rand_uniform([32, 16], -0.5, 0.5, 1);
    let w2 = Tensor::rand_uniform([4, 32], -0.5, 0.5, 2);
    let build = |npu: &Npu| -> qanten::Result<nn::Sequential> {
        Ok(nn::Sequential::new()
            .add(nn::Linear::new(npu, &w1, None)?)
            .add(nn::ReLU)
            .add(nn::Linear::new(npu, &w2, None)?))
    };
    let replicated = pool.replicate(build).unwrap();
    let single = build(&Npu::simulator()).unwrap();

    let x = Tensor::rand_uniform([10, 16], -1., 1., 3);
    assert_eq!(replicated.forward(&x).unwrap(), single.forward(&x).unwrap());
    for (_, prof) in &handles {
        assert_eq!(
            prof.report().op(OpKind::MatVecResident).calls,
            2,
            "every NPU did its share"
        );
    }
    // A batch smaller than the pool still works.
    assert_eq!(
        replicated.forward(&Tensor::ones([1, 16])).unwrap().shape(),
        &[1, 4]
    );
    assert!(NpuPool::new(vec![]).is_err());
}

#[test]
fn pool_split_batch_with_ops() {
    let pool = NpuPool::simulated(4).unwrap();
    let w = Tensor::rand_uniform([8, 5], -1., 1., 4);
    let x = Tensor::rand_uniform([9, 5], -1., 1., 5);
    let y = pool
        .split_batch(&x, |_, npu, part| ops::linear(npu, part, &w, None))
        .unwrap();
    assert_eq!(y, ops::linear(&Npu::simulator(), &x, &w, None).unwrap());
}

#[test]
fn cpu_fallback_routes_by_size() {
    let (npu, prof) = Npu::simulator().profiled();
    let (fb, monitor) = npu.with_cpu_fallback(FallbackPolicy::min_macs(1000));
    let small = Tensor::ones([10]);
    let big = Tensor::ones([5000]);
    assert_eq!(
        ops::mul(&fb, &small, &small).unwrap().to_f32_vec(),
        vec![1.0; 10]
    );
    ops::mul(&fb, &big, &big).unwrap();
    ops::linear(&fb, &Tensor::ones([2, 4]), &Tensor::ones([3, 4]), None).unwrap(); // 24 MACs → CPU
    let s = monitor.stats();
    assert_eq!((s.cpu_ops, s.npu_ops), (2, 1));
    // Only the big op reached the (profiled) NPU backend.
    assert_eq!(prof.report().op(OpKind::Mul).calls, 1);
    assert_eq!(prof.report().op(OpKind::MatVec).calls, 0);
    // Resident layers always use the NPU.
    let layer = nn::Linear::new(&fb, &Tensor::ones([3, 4]), None).unwrap();
    layer.forward(&Tensor::ones([1, 4])).unwrap();
    assert_eq!(prof.report().op(OpKind::MatVecResident).calls, 1);
    assert!(fb.backend().name().contains("cpu fallback"));
    let p = FallbackPolicy::calibrate(&Npu::simulator()).unwrap();
    assert!(p.min_macs > 0);
}
