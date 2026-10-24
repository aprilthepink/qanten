//! Trains a small network on the two-spirals problem with the NPU in the loop,
//! once on an exact NPU and once on a noisy one, and compares.
//!
//! ```sh
//! cargo run --release -p qanten --example train_spirals
//! ```

use std::f32::consts::PI;
use std::time::Instant;

use qanten::nn::{self, Module};
use qanten::train::{self, Adam, Dense, Model, Tanh};
use qanten::{Npu, SimConfig, Tensor};

fn spirals(n: usize, noise_seed: u64) -> (Tensor, Vec<usize>) {
    let jitter = Tensor::rand_uniform([4 * n], -0.02, 0.02, noise_seed).to_f32_vec();
    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..n {
        let t = i as f32 / n as f32 * 3.0 * PI;
        let r = 0.1 + t / (3.0 * PI);
        for (class, phase) in [(0usize, 0.0f32), (1, PI)] {
            let k = x.len();
            x.extend([
                r * (t + phase).cos() + jitter[k],
                r * (t + phase).sin() + jitter[k + 1],
            ]);
            y.push(class);
        }
    }
    (
        Tensor::from_f32(&x, [2 * n, 2]).unwrap_or_else(|_| Tensor::zeros([0, 2])),
        y,
    )
}

fn train_on(npu: &Npu, label: &str) -> qanten::Result<()> {
    let (x, y) = spirals(200, 1);
    let (x_test, y_test) = spirals(150, 2);
    let mut model = Model::new()
        .add(Dense::new(npu, 2, 48, 1)?)
        .add(Tanh::default())
        .add(Dense::new(npu, 48, 48, 2)?)
        .add(Tanh::default())
        .add(Dense::new(npu, 48, 2, 3)?);
    let mut opt = Adam::new(0.02);
    let start = Instant::now();
    for epoch in 0..=500 {
        let logits = model.forward(&x)?;
        let (loss, grad) = train::cross_entropy(&logits, &y)?;
        if epoch % 100 == 0 {
            println!(
                "  [{label}] epoch {epoch:>3}: loss {loss:.3}, train accuracy {:.1}%",
                100.0 * train::accuracy(&logits, &y)
            );
        }
        model.backward(&grad)?;
        model.clip_grad_norm(1.0);
        model.step(&mut opt)?;
    }
    // Deploy: export the trained weights into inference layers.
    let sd = model.state_dict()?;
    let deployed = nn::Sequential::new()
        .add(nn::Linear::load(npu, &sd, "0")?)
        .add(nn::Tanh)
        .add(nn::Linear::load(npu, &sd, "2")?)
        .add(nn::Tanh)
        .add(nn::Linear::load(npu, &sd, "4")?);
    let acc = train::accuracy(&deployed.forward(&x_test)?, &y_test);
    println!(
        "  [{label}] {} parameters, {:.1} s, test accuracy {:.1}%\n",
        model.num_params(),
        start.elapsed().as_secs_f64(),
        100.0 * acc
    );
    Ok(())
}

fn main() -> qanten::Result<()> {
    let npu = Npu::auto(0)?;
    println!("backend: {}\n", npu.backend().name());
    train_on(&npu, "exact")?;
    train_on(
        &Npu::simulator_with(SimConfig {
            noise_std: 0.05,
            ..SimConfig::default()
        }),
        "noise 0.05",
    )?;
    Ok(())
}
