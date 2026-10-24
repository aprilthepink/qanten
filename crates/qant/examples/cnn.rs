//! A small convolutional network (conv → BN → ReLU → pool → conv → global pool → linear).
//!
//! `cargo run --release --example cnn`

use qant::prelude::*;

fn main() -> anyhow::Result<()> {
    let npu = Npu::auto(0)?;
    let r = |shape: &[usize], seed| Tensor::rand_uniform(shape.to_vec(), -0.3, 0.3, seed);

    let model = nn::Sequential::new()
        .add(nn::Conv2d::new(
            &npu,
            &r(&[16, 3, 3, 3], 1),
            Some(Tensor::zeros([16])),
            Conv2dParams::new().padding(1),
        )?)
        .add(nn::BatchNorm2d::new(
            &npu,
            Tensor::zeros([16]),
            Tensor::ones([16]),
            None,
            None,
            1e-5,
        ))
        .add(nn::ReLU)
        .add(nn::MaxPool2d::new(2))
        .add(nn::Conv2d::new(
            &npu,
            &r(&[32, 16, 3, 3], 2),
            None,
            Conv2dParams::new().padding(1).stride(2),
        )?)
        .add(nn::Silu)
        .add(nn::AdaptiveAvgPool2d::new(1))
        .add(nn::Flatten::default())
        .add(nn::Linear::new(&npu, &r(&[10, 32], 3), None)?)
        .add(nn::Softmax);

    let images = Tensor::rand_uniform([4, 3, 32, 32], 0., 1., 4);
    let t = std::time::Instant::now();
    for (i, out) in model.forward_trace(&images)?.iter().enumerate() {
        println!("{:<18} -> {:?}", model.layers()[i].name(), out.shape());
    }
    let probs = model.forward(&images)?;
    println!(
        "predictions: {:?}  ({:.1} ms for 2 passes)",
        probs.argmax_last(),
        t.elapsed().as_secs_f64() * 1e3
    );
    Ok(())
}
