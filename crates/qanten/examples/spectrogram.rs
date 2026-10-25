//! Spectrogram of a synthetic recording (a chirp, a warbling tone, clicks and
//! noise), computed as one NPU matrix product, plus template detection.
//!
//! ```sh
//! cargo run --release -p qanten --example spectrogram -- [output.png]
//! ```

use std::f32::consts::TAU;
use std::time::Instant;

use qanten::Npu;
use qanten::rng::SplitMix64;
use qanten::signal::{Stft, TemplateMatcher};

fn main() -> anyhow::Result<()> {
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "spectrogram.png".into());
    let npu = Npu::auto(0)?;
    println!("backend: {}", npu.backend().name());

    let rate = 16_000.0f32;
    let n = (rate * 3.0) as usize;
    let mut rng = SplitMix64::new(1);
    let click: Vec<f32> = (0..96)
        .map(|i| (-(i as f32) / 12.0).exp() * (TAU * 7000.0 * i as f32 / rate).sin())
        .collect();
    let click_at = [4000usize, 15000, 26000, 37000];
    let sig: Vec<f32> = (0..n)
        .map(|t| {
            let s = t as f32 / rate;
            let chirp = 0.6 * (TAU * (300.0 * s + 0.5 * 1800.0 * s * s)).sin();
            let warble =
                0.4 * (TAU * (5000.0 * s + 400.0 / (TAU * 3.0) * (TAU * 3.0 * s).sin())).sin();
            let hum = 0.25 * (TAU * 120.0 * s).sin();
            let c = click_at
                .iter()
                .filter_map(|&p| t.checked_sub(p).and_then(|i| click.get(i)))
                .sum::<f32>();
            chirp + warble + hum + 1.5 * c + rng.uniform(-0.025, 0.025)
        })
        .collect();

    let stft = Stft::new(&npu, 512, 96)?;
    let start = Instant::now();
    let spec = stft.spectrogram(&sig)?;
    println!(
        "spectrogram: {} frames × {} bins in {:.1} ms (one NPU call)",
        spec.frames(),
        spec.bins(),
        start.elapsed().as_secs_f64() * 1e3
    );

    let matcher = TemplateMatcher::new(&npu, std::slice::from_ref(&click))?;
    let mut hits = matcher.scores(&sig)?.matches(0.35);
    hits.sort_by_key(|m| m.offset);
    println!(
        "clicks found at samples: {:?} (inserted at {click_at:?})",
        hits.iter().map(|m| m.offset).collect::<Vec<_>>()
    );

    // Render: time → x, frequency → y (low at the bottom), dB → colour.
    let (w, h) = (spec.frames(), spec.bins());
    let db = spec.db();
    let top = db.as_slice().iter().copied().fold(f32::MIN, f32::max);
    let colour = |v: f32| -> [u8; 3] {
        let t = ((v - top + 70.0) / 70.0).clamp(0.0, 1.0);
        let stops = [
            [0.0, 0.0, 0.02],
            [0.25, 0.05, 0.40],
            [0.80, 0.20, 0.35],
            [0.99, 0.65, 0.20],
            [1.0, 1.0, 0.85],
        ];
        let x = t * (stops.len() - 1) as f32;
        let i = (x as usize).min(stops.len() - 2);
        let f = x - i as f32;
        let c: Vec<u8> = (0..3)
            .map(|k| ((stops[i][k] * (1.0 - f) + stops[i + 1][k] * f) * 255.0) as u8)
            .collect();
        [c[0], c[1], c[2]]
    };
    let mut pixels = Vec::with_capacity(w * h * 3);
    for bin in (0..h).rev() {
        for frame in 0..w {
            pixels.extend(colour(db[(frame, bin)]));
        }
    }
    let file = std::io::BufWriter::new(std::fs::File::create(&out)?);
    let mut enc = png::Encoder::new(file, w as u32, h as u32);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header()?.write_image_data(&pixels)?;
    println!("wrote {out} ({w}×{h})");
    Ok(())
}
