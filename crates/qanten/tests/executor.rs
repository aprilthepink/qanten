use std::sync::Arc;

use qanten::executor::NpuExecutor;
use qanten::nn::{self, Module};
use qanten::pool::NpuPool;
use qanten::{Npu, SimConfig, Tensor, ops};

#[tokio::test(flavor = "multi_thread")]
async fn many_concurrent_jobs_in_tokio() {
    let pool = NpuPool::simulated(3).unwrap();
    let exec = NpuExecutor::for_pool(&pool);
    assert_eq!(exec.workers(), 3);
    let jobs: Vec<_> = (0..30)
        .map(|i| {
            exec.run(move |npu| ops::mul(npu, &Tensor::full([4], i as f32), &Tensor::ones([4])))
        })
        .collect();
    for (i, job) in jobs.into_iter().enumerate() {
        assert_eq!(job.await.unwrap().to_f32_vec(), vec![i as f32; 4]);
    }
}

#[tokio::test]
async fn async_model_forward() {
    let npu = Npu::simulator();
    let model: Arc<dyn Module> = Arc::new(
        nn::Sequential::new()
            .add(nn::Linear::new(&npu, &Tensor::ones([2, 3]), None).unwrap())
            .add(nn::ReLU),
    );
    let exec = NpuExecutor::new(npu);
    let (a, b) = tokio::join!(
        exec.forward(&model, Tensor::ones([1, 3])),
        exec.forward(&model, Tensor::full([1, 3], -1.0))
    );
    assert_eq!(a.unwrap().to_f32_vec(), vec![3.0, 3.0]);
    assert_eq!(b.unwrap().to_f32_vec(), vec![0.0, 0.0]);
}

#[test]
fn work_is_spread_over_pool_workers() {
    let pool = NpuPool::simulated(4).unwrap();
    let exec = NpuExecutor::for_pool(&pool);
    let ids: Vec<u32> = (0..64)
        .map(|_| {
            exec.run(|npu| {
                std::thread::sleep(std::time::Duration::from_millis(2));
                Ok(npu.id())
            })
        })
        .collect::<Vec<_>>()
        .into_iter()
        .map(|j| j.wait().unwrap())
        .collect();
    let mut used: Vec<u32> = ids.clone();
    used.sort();
    used.dedup();
    assert!(used.len() >= 2, "jobs ran on only {used:?}");
}

#[test]
fn errors_and_panics_come_back_as_errors() {
    let exec = NpuExecutor::new(Npu::simulator_with(SimConfig::default()));
    let bad = exec.run(|npu| ops::mul(npu, &Tensor::ones([2]), &Tensor::ones([3])));
    assert!(matches!(bad.wait(), Err(qanten::Error::Shape(_))));
    let boom = exec.run(|_| -> qanten::Result<()> { panic!("boom") });
    assert!(boom.wait().is_err());
    // The worker survived the panic.
    assert_eq!(exec.run(|npu| Ok(npu.id())).wait().unwrap(), 0);
}

#[test]
fn drop_finishes_queued_jobs() {
    let exec = NpuExecutor::with_workers(Npu::simulator(), 2);
    let jobs: Vec<_> = (0..10).map(|i| exec.run(move |_| Ok(i))).collect();
    drop(exec);
    let sum: i32 = jobs.into_iter().map(|j| j.wait().unwrap()).sum();
    assert_eq!(sum, 45);
}
