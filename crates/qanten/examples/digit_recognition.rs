//! Handwritten digit recognition with Q.ANT's pretrained network.
//!
//! The weights are the ones from Q.ANT's official example
//! (`qant_native_computing_toolkit_examples/python/digit_recognition`),
//! converted to safetensors: 784 → 133 → ReLU → 10 → softmax.
//!
//! ```sh
//! cargo run --example digit_recognition --features safetensors
//! # full MNIST test set (download and gunzip t10k-*-ubyte.gz first):
//! cargo run --release --example digit_recognition --features safetensors -- \
//!     --images t10k-images-idx3-ubyte --labels t10k-labels-idx1-ubyte
//! # how robust is the model to analog noise? (simulator only)
//! cargo run --release --example digit_recognition --features safetensors -- --noise 0.2
//! ```

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use qanten::nn::{self, Module};
use qanten::{Npu, SimConfig, Tensor};

const DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/examples/data");

fn read_idx(path: &PathBuf, magic: u32) -> Result<(Vec<u32>, Vec<u8>)> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let be = |i: usize| u32::from_be_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]);
    if bytes.len() < 8 || be(0) != magic {
        bail!(
            "{} is not an uncompressed IDX file (gunzip it first)",
            path.display()
        );
    }
    let ndims = (magic & 0xff) as usize;
    let dims: Vec<u32> = (0..ndims).map(|i| be(4 + 4 * i)).collect();
    Ok((dims, bytes[4 + 4 * ndims..].to_vec()))
}

/// Same preprocessing as Q.ANT's Python example: scale every image to [0, 1].
fn preprocess(pixels: &[u8]) -> Vec<f32> {
    let (lo, hi) = pixels
        .iter()
        .fold((u8::MAX, 0u8), |(lo, hi), &p| (lo.min(p), hi.max(p)));
    let range = (hi - lo).max(1) as f32;
    pixels.iter().map(|&p| (p - lo) as f32 / range).collect()
}

fn main() -> Result<()> {
    let mut images = PathBuf::from(DATA).join("mnist_t10k_100-images.idx");
    let mut labels = PathBuf::from(DATA).join("mnist_t10k_100-labels.idx");
    let mut noise = 0.0f32;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--images" => images = args.next().context("--images needs a path")?.into(),
            "--labels" => labels = args.next().context("--labels needs a path")?.into(),
            "--noise" => noise = args.next().context("--noise needs a value")?.parse()?,
            other => bail!("unknown argument {other}"),
        }
    }

    let npu = if noise > 0.0 {
        Npu::simulator_with(SimConfig {
            noise_std: noise,
            ..SimConfig::default()
        })
    } else {
        Npu::from_env(0)? // QANTEN_BACKEND=driver|sim|auto
    };
    println!("running on: {}", npu.backend().name());

    let sd =
        nn::StateDict::load_safetensors(PathBuf::from(DATA).join("digit_recognition.safetensors"))?;
    let model = nn::Sequential::new()
        .add(nn::Linear::load(&npu, &sd, "fc1")?)
        .add(nn::ReLU)
        .add(nn::Linear::load(&npu, &sd, "fc2")?)
        .add(nn::Softmax);

    let (idims, pixels) = read_idx(&images, 0x0803)?;
    let (_, truth) = read_idx(&labels, 0x0801)?;
    let n = idims[0] as usize;
    let px = (idims[1] * idims[2]) as usize;

    let start = Instant::now();
    let mut correct = 0;
    let batch = 100;
    for b0 in (0..n).step_by(batch) {
        let b1 = (b0 + batch).min(n);
        let x: Vec<f32> = (b0..b1)
            .flat_map(|i| preprocess(&pixels[i * px..(i + 1) * px]))
            .collect();
        let probs = model.forward(&Tensor::from_f32(&x, [b1 - b0, px])?)?;
        for (i, pred) in probs.argmax_last().into_iter().enumerate() {
            correct += usize::from(pred == truth[b0 + i] as usize);
            if b0 == 0 && i < 5 {
                let p = probs.get(&[i, pred]).unwrap_or(0.0);
                println!(
                    "image {i}: predicted {pred} ({:.1}%), label {}",
                    100.0 * p,
                    truth[i]
                );
            }
        }
    }
    println!(
        "accuracy: {correct}/{n} = {:.1}%  ({:.1} ms)",
        100.0 * correct as f64 / n as f64,
        start.elapsed().as_secs_f64() * 1e3
    );
    Ok(())
}
