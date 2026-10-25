//! Core benchmarks. The backend comes from `QANTEN_BACKEND` (driver, sim, auto):
//!
//! ```sh
//! cargo bench -p qanten --features onnx                       # simulator on machines without a card
//! QANTEN_BACKEND=driver cargo bench -p qanten --features onnx # real hardware
//! ```

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use qanten::nn::transformer::{Gpt, GptConfig};
use qanten::nn::{self, Module};
use qanten::ops::{self, Conv2dParams, ConvWeights, LinearWeights};
use qanten::train::{self, Adam, Dense, Model, Relu};
use qanten::{Npu, Tensor};

fn npu() -> Npu {
    let npu = Npu::from_env(0).unwrap_or_else(|e| panic!("cannot open NPU: {e}"));
    eprintln!("benchmarking on: {}", npu.backend().name());
    npu
}

fn matvec(c: &mut Criterion) {
    let npu = npu();
    let mut g = c.benchmark_group("matvec");
    g.sample_size(10).measurement_time(Duration::from_secs(2));
    for &(rows, cols, batch) in &[
        (128usize, 128usize, 32usize),
        (512, 512, 64),
        (1024, 1024, 64),
    ] {
        let w = Tensor::rand_uniform([rows, cols], -1., 1., 1);
        let x = Tensor::rand_uniform([batch, cols], -1., 1., 2);
        g.throughput(Throughput::Elements((rows * cols * batch) as u64));
        let id = format!("{rows}x{cols}x{batch}");
        g.bench_with_input(BenchmarkId::new("streamed", &id), &(), |b, _| {
            b.iter(|| ops::matvec(&npu, black_box(&w), black_box(&x)).unwrap())
        });
        let resident = LinearWeights::upload(&npu, &w).unwrap();
        g.bench_with_input(BenchmarkId::new("resident", &id), &(), |b, _| {
            b.iter(|| resident.forward(black_box(&x)).unwrap())
        });
    }
    g.finish();
}

fn elementwise(c: &mut Criterion) {
    let npu = npu();
    let mut g = c.benchmark_group("elementwise");
    g.sample_size(10).measurement_time(Duration::from_secs(2));
    let n = 1 << 16;
    let (a, b) = (
        Tensor::rand_uniform([n], -1., 1., 1),
        Tensor::rand_uniform([n], -1., 1., 2),
    );
    g.throughput(Throughput::Elements(n as u64));
    g.bench_function("mul_64k", |bch| {
        bch.iter(|| ops::mul(&npu, &a, &b).unwrap())
    });
    g.bench_function("periodic_nl_64k", |bch| {
        bch.iter(|| ops::periodic_nl(&npu, &a, &b).unwrap())
    });
    g.finish();
}

fn conv(c: &mut Criterion) {
    let npu = npu();
    let mut g = c.benchmark_group("conv2d");
    g.sample_size(10).measurement_time(Duration::from_secs(2));
    let x = Tensor::rand_uniform([1, 16, 32, 32], 0., 1., 1);
    let w =
        ConvWeights::upload(&npu, &Tensor::rand_uniform([32, 16, 3, 3], -0.2, 0.2, 2), 1).unwrap();
    let p = Conv2dParams::new().padding(1);
    g.throughput(Throughput::Elements((32 * 16 * 9 * 32 * 32) as u64));
    g.bench_function("16->32ch_3x3_32x32", |b| {
        b.iter(|| w.forward(black_box(&x), &p).unwrap())
    });
    g.finish();
}

fn models(c: &mut Criterion) {
    let npu = npu();
    let mut g = c.benchmark_group("models");
    g.sample_size(10).measurement_time(Duration::from_secs(3));

    // Q.ANT's digit-recognition shape: 784 -> 133 -> 10, batch 64.
    let mlp = nn::Sequential::new()
        .add(
            nn::Linear::new(
                &npu,
                &Tensor::rand_uniform([133, 784], -0.05, 0.05, 1),
                None,
            )
            .unwrap(),
        )
        .add(nn::ReLU)
        .add(nn::Linear::new(&npu, &Tensor::rand_uniform([10, 133], -0.1, 0.1, 2), None).unwrap())
        .add(nn::Softmax);
    let x = Tensor::rand_uniform([64, 784], 0., 1., 3);
    g.bench_function("mlp_784_133_10_batch64", |b| {
        b.iter(|| mlp.forward(black_box(&x)).unwrap())
    });

    // TinyGPT: one cached decoding step and a 32-token prefill.
    let gpt = Gpt::random(&npu, GptConfig::tiny(65), 1).unwrap();
    let prompt: Vec<usize> = (0..32).map(|i| i % 65).collect();
    g.bench_function("tiny_gpt_prefill_32", |b| {
        b.iter(|| gpt.logits(black_box(&prompt), 1).unwrap())
    });
    g.bench_function("tiny_gpt_decode_step", |b| {
        b.iter_batched(
            || {
                let mut cache = gpt.new_cache();
                gpt.step(&prompt[..31], &mut cache).unwrap();
                cache
            },
            |mut cache| gpt.step(&[7], &mut cache).unwrap(),
            criterion::BatchSize::LargeInput,
        )
    });

    // One training step of a small MLP.
    let mut model = Model::new()
        .add(Dense::new(&npu, 64, 128, 1).unwrap())
        .add(Relu::default())
        .add(Dense::new(&npu, 128, 10, 2).unwrap());
    let mut opt = Adam::new(1e-3);
    let xt = Tensor::rand_uniform([64, 64], -1., 1., 4);
    let labels: Vec<usize> = (0..64).map(|i| i % 10).collect();
    g.bench_function("train_step_64_128_10_batch64", |b| {
        b.iter(|| {
            let (_, grad) = train::cross_entropy(&model.forward(&xt).unwrap(), &labels).unwrap();
            model.backward(&grad).unwrap();
            model.step(&mut opt).unwrap();
        })
    });

    // An ONNX model from the test fixtures.
    let onnx = qanten::onnx::OnnxModel::load(
        &npu,
        concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/onnx/cnn.onnx"),
    )
    .unwrap();
    let img = Tensor::rand_uniform([2, 3, 12, 12], 0., 1., 5);
    g.bench_function("onnx_cnn_fixture", |b| {
        b.iter(|| onnx.run_single(black_box(&img)).unwrap())
    });
    g.finish();
}

fn compute(c: &mut Criterion) {
    use qanten::compute::{Matrix, NpuMatrix};
    let npu = npu();
    let mut g = c.benchmark_group("compute");
    g.sample_size(10).measurement_time(Duration::from_secs(2));
    for &(n, k) in &[(256usize, 256usize), (1024, 256)] {
        let a = Matrix::random_normal(n, n, 1);
        let x = Matrix::random_normal(k, n, 2);
        g.throughput(Throughput::Elements((n * n * k) as u64));
        let id = format!("{n}x{n}x{k}");
        let resident = NpuMatrix::upload(&npu, &a).unwrap();
        g.bench_with_input(BenchmarkId::new("npu_matrix", &id), &(), |b, _| {
            b.iter(|| resident.apply_batch(black_box(&x)).unwrap())
        });
        let at = a.transpose();
        g.bench_with_input(BenchmarkId::new("cpu_f32", &id), &(), |b, _| {
            b.iter(|| black_box(&x).matmul_cpu(&at).unwrap())
        });
    }
    g.finish();
}

criterion_group!(benches, matvec, elementwise, conv, models, compute);
criterion_main!(benches);
