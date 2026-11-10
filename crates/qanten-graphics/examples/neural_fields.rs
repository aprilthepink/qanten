//! Neural fields trained on the NPU.
//!
//! Top row: a 64×64 image and three networks fitted to it, all trained on
//! the NPU (matrix products in bf16, f32 master weights): a ReLU MLP on raw
//! coordinates (blurry), a ReLU MLP on Fourier features, and a SIREN.
//! Bottom: a SIREN fitted to a 2-D signed distance field; the learned zero
//! contour (white, marching squares) over the exact one (red).
//!
//! ```sh
//! cargo run --release -p qanten-graphics --example neural_fields -- [output.png] [steps]
//! ```

mod common;

use std::time::Instant;

use common::Canvas;
use glam::{Vec2, Vec3};
use qanten::compute::Matrix;
use qanten::nn::FourierFeatures;
use qanten::train::{self, Adam, Dense, Model, Relu};
use qanten::{Npu, Tensor};

const N: usize = 64;

/// The target picture: hard edges and fine stripes a plain MLP can't fit.
fn target(u: f32, v: f32) -> Vec3 {
    let p = Vec2::new(u, v) * 2.0 - Vec2::ONE;
    let mut c = Vec3::new(0.1, 0.15, 0.3).lerp(Vec3::new(0.9, 0.6, 0.3), (v * 1.2).min(1.0));
    if (p - Vec2::new(-0.35, -0.3)).length() < 0.35 {
        c = Vec3::new(0.95, 0.85, 0.2);
    }
    if (p.x - 0.4).abs() < 0.3 && (p.y - 0.25).abs() < 0.45 {
        let stripe = ((p.y * 18.0).sin() > 0.0) as u8 as f32;
        c = Vec3::new(0.2, 0.7, 0.9).lerp(Vec3::new(0.9, 0.2, 0.5), stripe);
    }
    if ((p.x + p.y) * 5.0).floor() as i32 % 2 == 0 && p.y > 0.55 {
        c *= 0.6;
    }
    c
}

/// Signed distance to a circle unioned with a rounded box.
fn sdf(p: Vec2) -> f32 {
    let circle = (p - Vec2::new(-0.3, 0.1)).length() - 0.38;
    let q = (p - Vec2::new(0.35, -0.2)).abs() - Vec2::new(0.3, 0.18);
    let boxed = q.max(Vec2::ZERO).length() + q.x.max(q.y).min(0.0) - 0.08;
    circle.min(boxed)
}

fn coords(n: usize) -> Matrix {
    Matrix::from_fn(n * n, 2, |r, c| {
        let k = if c == 0 { r % n } else { r / n };
        (k as f32 + 0.5) / n as f32 * 2.0 - 1.0
    })
}

fn fit(model: &mut Model, x: &Tensor, y: &Tensor, steps: usize, lr: f32) -> anyhow::Result<f32> {
    let mut opt = Adam::new(lr);
    let mut loss = f32::NAN;
    for _ in 0..steps {
        let pred = model.forward(x)?;
        let (l, grad) = train::mse(&pred, y)?;
        model.backward(&grad)?;
        model.step(&mut opt)?;
        loss = l;
    }
    Ok(loss)
}

fn psnr(mse: f32) -> f32 {
    -10.0 * mse.max(1e-12).log10()
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let out = args.next().unwrap_or_else(|| "neural_fields.png".into());
    let steps: usize = args.next().map_or(Ok(400), |s| s.parse())?;
    let npu = Npu::auto(0)?;
    println!("backend: {}", npu.backend().name());

    // --- Image fitting.
    let xy = coords(N);
    let rgb: Vec<f32> = (0..N * N)
        .flat_map(|r| {
            let c = target(
                (r % N) as f32 / (N - 1) as f32,
                (r / N) as f32 / (N - 1) as f32,
            );
            [c.x, c.y, c.z]
        })
        .collect();
    let y = Tensor::from_f32(&rgb, [N * N, 3])?;
    let raw = xy.to_tensor()?;
    // Fourier features are fixed: encode the coordinates once, in f32.
    let ff = FourierFeatures::gaussian(&npu, 128, 2, 6.0, 1);
    let encoded = ff.encode(&xy)?.to_tensor()?;

    let mut results = Vec::new();
    for (name, mut model, input, lr) in [
        (
            "ReLU MLP",
            Model::new()
                .add(Dense::new(&npu, 2, 128, 1)?)
                .add(Relu::default())
                .add(Dense::new(&npu, 128, 128, 2)?)
                .add(Relu::default())
                .add(Dense::new(&npu, 128, 3, 3)?),
            &raw,
            2e-3,
        ),
        (
            "Fourier features + ReLU",
            Model::new()
                .add(Dense::new(&npu, 256, 128, 4)?)
                .add(Relu::default())
                .add(Dense::new(&npu, 128, 128, 5)?)
                .add(Relu::default())
                .add(Dense::new(&npu, 128, 3, 6)?),
            &encoded,
            2e-3,
        ),
        (
            "SIREN",
            train::siren(&npu, &[2, 128, 128, 3], 30.0, 7)?,
            &raw,
            5e-4,
        ),
    ] {
        let t0 = Instant::now();
        let loss = fit(&mut model, input, &y, steps, lr)?;
        let image = model.forward(input)?.to_f32_vec();
        println!(
            "{name:>24}: PSNR {:5.1} dB after {steps} steps ({:.1} s)",
            psnr(loss),
            t0.elapsed().as_secs_f64()
        );
        results.push(image);
    }

    // --- SDF fitting.
    let s = 96;
    let grid = coords(s);
    let d: Vec<f32> = grid
        .iter_rows()
        .map(|p| sdf(Vec2::new(p[0], p[1])))
        .collect();
    let mut sdf_net = train::siren(&npu, &[2, 64, 64, 1], 30.0, 9)?;
    let t0 = Instant::now();
    let loss = fit(
        &mut sdf_net,
        &grid.to_tensor()?,
        &Tensor::from_f32(&d, [s * s, 1])?,
        steps,
        5e-4,
    )?;
    let learned = sdf_net.forward(&grid.to_tensor()?)?.to_f32_vec();
    println!(
        "{:>24}: RMS distance error {:.4} ({:.1} s)",
        "SDF SIREN",
        loss.sqrt(),
        t0.elapsed().as_secs_f64()
    );

    // --- Picture: 4 panels of 192 px on top, the SDF below.
    let panel = 192;
    let (w, h) = (4 * panel, panel + 2 * panel);
    let mut canvas = Canvas::new(w, h, |_, _| Vec3::splat(0.02));
    let mut put = |x: usize, y: usize, c: Vec3| {
        if x < w && y < h {
            canvas.color[y * w + x] = c;
        }
    };
    let images: Vec<&[f32]> = std::iter::once(rgb.as_slice())
        .chain(results.iter().map(Vec::as_slice))
        .collect();
    for (k, img) in images.iter().enumerate() {
        for py in 0..panel {
            for px in 0..panel {
                let (u, v) = (px * N / panel, py * N / panel);
                let i = 3 * (v * N + u);
                put(
                    k * panel + px,
                    py,
                    Vec3::new(img[i], img[i + 1], img[i + 2]).clamp(Vec3::ZERO, Vec3::ONE),
                );
            }
        }
    }
    // SDF panel: learned distance as colour bands, both zero contours on top.
    let size = 2 * panel;
    let x0 = (w - size) / 2;
    for py in 0..size {
        for px in 0..size {
            let (u, v) = (px * s / size, py * s / size);
            let dist = learned[v * s + u];
            let inside = Vec3::new(0.25, 0.45, 0.85);
            let outside = Vec3::new(0.85, 0.55, 0.25);
            let band = 0.75 + 0.25 * (dist * 60.0).cos();
            let c = if dist < 0.0 { inside } else { outside }
                * band
                * (1.0 - (-6.0 * dist.abs()).exp() * 0.5);
            put(x0 + px, panel + py, c);
        }
    }
    let to_px = |p: Vec2| {
        (
            x0 as f32 + (p.x + 1.0) * 0.5 * size as f32,
            panel as f32 + (p.y + 1.0) * 0.5 * size as f32,
        )
    };
    let exact: Vec<f32> = d.clone();
    for (field, colour) in [
        (&exact, Vec3::new(1.6, 0.15, 0.1)),
        (&learned, Vec3::splat(1.6)),
    ] {
        for (a, b) in marching_squares(field, s) {
            let steps = 24;
            for k in 0..=steps {
                let (fx, fy) = to_px(a.lerp(b, k as f32 / steps as f32));
                put(fx as usize, fy as usize, colour);
            }
        }
    }
    canvas.save(&out)?;
    println!("wrote {out}");
    Ok(())
}

/// Zero-contour segments of a field sampled at the centres of an `n × n` grid over `[-1, 1]²`.
fn marching_squares(f: &[f32], n: usize) -> Vec<(Vec2, Vec2)> {
    let at = |i: usize, j: usize| f[j * n + i];
    let pos = |i: f32, j: f32| {
        Vec2::new(
            (i + 0.5) / n as f32 * 2.0 - 1.0,
            (j + 0.5) / n as f32 * 2.0 - 1.0,
        )
    };
    let mut segments = Vec::new();
    for j in 0..n - 1 {
        for i in 0..n - 1 {
            // Corners counter-clockwise: (i,j), (i+1,j), (i+1,j+1), (i,j+1).
            let c = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)];
            let v = c.map(|(a, b)| at(a, b));
            let mut crossings = Vec::new();
            for e in 0..4 {
                let (a, b) = (e, (e + 1) % 4);
                if (v[a] < 0.0) != (v[b] < 0.0) {
                    let t = v[a] / (v[a] - v[b]);
                    let pa = pos(c[a].0 as f32, c[a].1 as f32);
                    let pb = pos(c[b].0 as f32, c[b].1 as f32);
                    crossings.push(pa.lerp(pb, t));
                }
            }
            for pair in crossings.chunks_exact(2) {
                segments.push((pair[0], pair[1]));
            }
        }
    }
    segments
}
