//! Light-probe baking: projects many environments to spherical harmonics in
//! one NPU call, then shades spheres with the baked irradiance.
//!
//! ```sh
//! cargo run --release -p qanten-graphics --example probes -- [output.png]
//! ```

mod common;

use std::time::Instant;

use common::{Canvas, ms};
use glam::Vec3;
use qanten::Npu;
use qanten_graphics::sh::{Sh9, ShProjector};

/// A family of procedural skies: sun direction, sun colour and sky tint vary with `k`.
fn environment(k: usize, d: Vec3) -> Vec3 {
    let a = k as f32 * 0.7;
    let sun = Vec3::new(a.cos(), 0.35 + 0.3 * (a * 1.3).sin(), a.sin()).normalize();
    let tint = Vec3::new(
        0.5 + 0.5 * (a).cos(),
        0.5 + 0.5 * (a + 2.0).cos(),
        0.5 + 0.5 * (a + 4.0).cos(),
    );
    let sky = tint * 0.3 * (0.5 + 0.5 * d.y.max(0.0));
    let ground = Vec3::new(0.12, 0.10, 0.08) * (-d.y).max(0.0);
    sky + ground + Vec3::new(1.0, 0.85, 0.6) * d.dot(sun).max(0.0).powf(32.0) * 6.0
}

fn main() -> anyhow::Result<()> {
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "probes.png".into());
    let npu = Npu::auto(0)?;
    println!("backend: {}", npu.backend().name());

    let samples = 2048;
    let probes = 1024;
    let proj = ShProjector::fibonacci(samples);
    let radiance: Vec<Vec3> = (0..probes)
        .flat_map(|p| {
            proj.directions()
                .iter()
                .map(move |&d| environment(p, d))
                .collect::<Vec<_>>()
        })
        .collect();
    let t0 = Instant::now();
    let want = proj.project_batch_cpu(&radiance);
    let cpu_ms = ms(t0.elapsed());
    let proj = proj.with_npu(&npu)?;
    let t0 = Instant::now();
    let baked = proj.project_batch(&radiance)?;
    let npu_ms = ms(t0.elapsed());
    let err = baked
        .iter()
        .zip(&want)
        .flat_map(|(a, b)| {
            a.coeffs
                .iter()
                .zip(&b.coeffs)
                .map(|(x, y)| (*x - *y).abs().max_element())
        })
        .fold(0.0f32, f32::max);
    let scale = want
        .iter()
        .map(|s| s.coeffs[0].max_element())
        .fold(0.0f32, f32::max);
    println!("bake: {probes} probes × {samples} samples × RGB → SH9");
    println!(
        "  CPU: {cpu_ms:7.1} ms   NPU path: {npu_ms:7.1} ms (1 call), max error {err:.4} (DC up to {scale:.2})"
    );

    // Shade a 4×2 grid of spheres, each lit by a different baked probe.
    let (w, h) = (1280, 640);
    let mut canvas = Canvas::new(w, h, |_, v| Vec3::splat(0.02 + 0.03 * v));
    let cell = 160.0;
    for (slot, probe) in [0usize, 3, 7, 12, 18, 25, 33, 40].iter().enumerate() {
        let sh: &Sh9 = &baked[*probe];
        let (cx, cy) = (
            cell * (slot % 4) as f32 * 2.0 + cell,
            cell * (slot / 4) as f32 * 2.0 + cell,
        );
        for py in (cy - cell) as usize..(cy + cell) as usize {
            for px in (cx - cell) as usize..(cx + cell) as usize {
                let (dx, dy) = (
                    (px as f32 - cx) / (cell * 0.85),
                    (cy - py as f32) / (cell * 0.85),
                );
                let r2 = dx * dx + dy * dy;
                let i = py * w + px;
                if r2 <= 1.0 {
                    let n = Vec3::new(dx, dy, (1.0 - r2).sqrt());
                    canvas.color[i] = Vec3::splat(0.8) * sh.irradiance(n) / std::f32::consts::PI;
                }
            }
        }
    }
    canvas.save(&out)?;
    println!("wrote {out}");
    Ok(())
}
