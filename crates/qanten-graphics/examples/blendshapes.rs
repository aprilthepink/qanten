//! Morph targets: one sphere, four targets, six characters evaluated in one NPU call.
//!
//! ```sh
//! cargo run --release -p qanten-graphics --example blendshapes -- [output.png]
//! ```

mod common;

use std::f32::consts::{PI, TAU};

use common::{Canvas, max_err_all, view_proj};
use glam::Vec3;
use qanten::Npu;
use qanten::compute::Matrix;
use qanten_graphics::blendshapes::Blendshapes;

const LAT: usize = 48;
const LON: usize = 96;

fn main() -> anyhow::Result<()> {
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "blendshapes.png".into());
    let npu = Npu::auto(0)?;

    // Base mesh: a unit UV sphere.
    let mut base = Vec::new();
    for i in 0..=LAT {
        let theta = PI * i as f32 / LAT as f32;
        for j in 0..LON {
            let phi = TAU * j as f32 / LON as f32;
            base.push(Vec3::new(
                theta.sin() * phi.cos(),
                theta.cos(),
                theta.sin() * phi.sin(),
            ));
        }
    }
    // Four targets, given as offsets from the base.
    let squash: Vec<Vec3> = base
        .iter()
        .map(|p| Vec3::new(p.x * 0.35, -p.y * 0.45, p.z * 0.35))
        .collect();
    let bulge: Vec<Vec3> = base
        .iter()
        .map(|p| *p * 0.9 * p.x.max(0.0).powi(3))
        .collect();
    let spikes: Vec<Vec3> = base
        .iter()
        .map(|p| {
            let (theta, phi) = (p.y.acos(), p.z.atan2(p.x));
            *p * 0.35 * ((6.0 * theta).sin() * (5.0 * phi).sin()).max(0.0).powi(2)
        })
        .collect();
    let ripple: Vec<Vec3> = base
        .iter()
        .map(|p| *p * 0.12 * (10.0 * p.y).sin())
        .collect();
    let shapes =
        Blendshapes::new(base.clone(), &[squash, bulge, spikes, ripple])?.with_npu(&npu)?;

    // Six characters' weights, evaluated together.
    let weights = Matrix::from_rows(&[
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.4, 0.6, 0.7, 0.5],
    ])?;
    let all = shapes.evaluate_batch(&weights)?;
    let reference = shapes.evaluate_batch_cpu(&weights)?;
    println!(
        "6 characters × {} vertices × 4 shapes in one call, max error {:.5}",
        base.len(),
        max_err_all(&all, &reference)
    );

    let (w, h) = (1500, 300);
    let vp = view_proj(
        Vec3::new(0.0, 1.8, 12.5),
        Vec3::ZERO,
        18f32.to_radians(),
        w as f32 / h as f32,
        0.1,
        100.0,
    );
    let mut canvas = Canvas::new(w, h, |_, v| {
        Vec3::new(0.03, 0.035, 0.05).lerp(Vec3::new(0.10, 0.08, 0.12), v)
    });
    let light = Vec3::new(-0.5, 0.8, 0.6).normalize();
    for (c, pos) in all.iter().enumerate() {
        let offset = Vec3::new((c as f32 - 2.5) * 2.85, 0.0, 0.0);
        let albedo = [
            Vec3::new(0.8, 0.8, 0.85),
            Vec3::new(0.9, 0.5, 0.3),
            Vec3::new(0.4, 0.8, 0.5),
            Vec3::new(0.85, 0.35, 0.6),
            Vec3::new(0.35, 0.6, 0.95),
            Vec3::new(0.95, 0.8, 0.35),
        ][c];
        let idx = |i: usize, j: usize| i * LON + j % LON;
        for i in 0..LAT {
            for j in 0..LON {
                let q = [idx(i, j), idx(i + 1, j), idx(i, j + 1), idx(i + 1, j + 1)]
                    .map(|k| pos[k] + offset);
                for t in [[q[0], q[2], q[1]], [q[1], q[2], q[3]]] {
                    let n = (t[1] - t[0]).cross(t[2] - t[0]).normalize_or(Vec3::Y);
                    let col = albedo * (0.12 + n.dot(light).max(0.0))
                        + Vec3::splat(0.25) * n.dot(light).max(0.0).powi(24);
                    canvas.triangle(&vp, t, [col; 3]);
                }
            }
        }
    }
    canvas.save(&out)?;
    println!("wrote {out}");
    Ok(())
}
