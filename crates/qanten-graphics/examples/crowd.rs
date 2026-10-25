//! A crowd of 100 animated tentacles skinned in one NPU call per frame,
//! plus a blendshape benchmark (200 faces × 52 ARKit-style shapes).
//!
//! ```sh
//! cargo run --release -p qanten-graphics --example crowd -- [output.png]
//! ```

mod common;

use std::f32::consts::TAU;
use std::time::Instant;

use common::{Canvas, max_err, max_err_all, ms, view_proj};
use glam::{Mat4, Quat, Vec3};
use qanten::Npu;
use qanten::compute::Matrix;
use qanten_graphics::blendshapes::Blendshapes;
use qanten_graphics::skinning::{Influences, SkinnedMesh};

const RINGS: usize = 64;
const SIDES: usize = 20;
const BONES: usize = 12;
const HEIGHT: f32 = 3.0;

/// A tapered tube along +y, weighted to a chain of bones.
fn tentacle() -> (Vec<Vec3>, Vec<Vec3>, Vec<Influences>, Vec<[u32; 3]>) {
    let (mut pos, mut nrm, mut inf, mut tris) = (vec![], vec![], vec![], vec![]);
    let seg = HEIGHT / BONES as f32;
    for r in 0..RINGS {
        let y = HEIGHT * r as f32 / (RINGS - 1) as f32;
        let radius = 0.28 * (1.0 - 0.85 * y / HEIGHT);
        // Smoothly blend between the two nearest bones.
        let fb = (y / seg - 0.5).clamp(0.0, (BONES - 1) as f32);
        let b0 = fb.floor() as usize;
        let b1 = (b0 + 1).min(BONES - 1);
        let t = fb - b0 as f32;
        for s in 0..SIDES {
            let a = TAU * s as f32 / SIDES as f32;
            let n = Vec3::new(a.cos(), 0.0, a.sin());
            pos.push(n * radius + Vec3::Y * y);
            nrm.push(n);
            inf.push([(b0 as u16, 1.0 - t), (b1 as u16, t), (0, 0.0), (0, 0.0)]);
        }
    }
    for r in 0..RINGS - 1 {
        for s in 0..SIDES {
            let a = (r * SIDES + s) as u32;
            let b = (r * SIDES + (s + 1) % SIDES) as u32;
            let (c, d) = (a + SIDES as u32, b + SIDES as u32);
            tris.push([a, c, b]);
            tris.push([b, c, d]);
        }
    }
    (pos, nrm, inf, tris)
}

/// Skinning matrices for one character: a swaying bone chain.
fn pose(character: usize, time: f32) -> Vec<Mat4> {
    let seg = HEIGHT / BONES as f32;
    let phase = character as f32 * 1.7;
    let mut world = Mat4::IDENTITY;
    (0..BONES)
        .map(|b| {
            let wave = (time * 1.6 + phase - b as f32 * 0.55).sin();
            let local = Mat4::from_translation(Vec3::Y * if b == 0 { 0.0 } else { seg })
                * Mat4::from_quat(
                    Quat::from_rotation_z(0.22 * wave)
                        * Quat::from_rotation_x(0.12 * (time + phase + b as f32).cos()),
                );
            world *= local;
            world * Mat4::from_translation(-Vec3::Y * (b as f32 * seg)) // × inverse bind
        })
        .collect()
}

fn main() -> anyhow::Result<()> {
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "crowd.png".into());
    let npu = Npu::auto(0)?;
    println!("backend: {}", npu.backend().name());

    // --- Skinning: 100 characters, one call ---
    let (rest, normals, influences, tris) = tentacle();
    let grid = 10;
    let chars = grid * grid;
    let poses: Vec<Vec<Mat4>> = (0..chars).map(|c| pose(c, 2.0)).collect();
    let mesh = SkinnedMesh::new(
        rest.clone(),
        Some(normals.clone()),
        influences.clone(),
        BONES,
    )?;
    let t0 = Instant::now();
    let want = mesh.skin_batch_cpu(&poses);
    let cpu_ms = ms(t0.elapsed());
    let mesh = mesh.with_npu(&npu)?;
    let t0 = Instant::now();
    let skinned = mesh.skin_batch(&poses)?;
    let npu_ms = ms(t0.elapsed());
    println!(
        "skinning: {chars} characters × {} vertices × {BONES} bones",
        rest.len()
    );
    println!("  CPU LBS:  {cpu_ms:7.1} ms   NPU path: {npu_ms:7.1} ms (1 call)");
    let worst = |f: fn(&qanten_graphics::skinning::Skinned) -> &Vec<Vec3>| {
        skinned
            .iter()
            .zip(&want)
            .map(|(a, b)| max_err(f(a), f(b)))
            .fold(0.0, f32::max)
    };
    println!(
        "  max error: position {:.4} m on a {HEIGHT} m mesh, normal {:.4}",
        worst(|s| &s.positions),
        worst(|s| &s.normals)
    );

    // --- Blendshapes: 200 faces × 52 shapes ---
    let face_v = 4000;
    let base: Vec<Vec3> = (0..face_v)
        .map(|i| Vec3::new((i % 80) as f32, (i / 80) as f32, 0.0) * 0.003)
        .collect();
    let deltas: Vec<Vec<Vec3>> = (0..52)
        .map(|s| {
            (0..face_v)
                .map(|i| Vec3::new(0.0, 0.0, 0.004 * ((i as f32 * 0.01 + s as f32).sin())))
                .collect()
        })
        .collect();
    let faces = 200;
    let weights = Matrix::from_fn(faces, 52, |f, s| {
        0.5 + 0.5 * ((f * 52 + s) as f32 * 0.37).sin()
    });
    let bs = Blendshapes::new(base, &deltas)?;
    let t0 = Instant::now();
    let bw = bs.evaluate_batch_cpu(&weights)?;
    let cpu_ms = ms(t0.elapsed());
    let bs = bs.with_npu(&npu)?;
    let t0 = Instant::now();
    let bn = bs.evaluate_batch(&weights)?;
    let npu_ms = ms(t0.elapsed());
    println!("blendshapes: {faces} faces × {face_v} vertices × 52 shapes");
    println!(
        "  CPU:      {cpu_ms:7.1} ms   NPU path: {npu_ms:7.1} ms (1 call), max error {:.5} m",
        max_err_all(&bn, &bw)
    );

    // --- Render the crowd ---
    let (w, h) = (1280, 720);
    let eye = Vec3::new(-9.0, 9.0, -14.0);
    let target = Vec3::new(9.0, 0.5, 9.0);
    let vp = view_proj(
        eye,
        target,
        45f32.to_radians(),
        w as f32 / h as f32,
        0.1,
        200.0,
    );
    let mut canvas = Canvas::new(w, h, |_, v| {
        Vec3::new(0.05, 0.07, 0.12).lerp(Vec3::new(0.30, 0.22, 0.30), v)
    });
    let sun = Vec3::new(-0.4, 0.8, -0.3).normalize();
    let ground = |x: f32, z: f32| {
        let check = ((x.floor() + z.floor()) as i32).rem_euclid(2) as f32;
        Vec3::splat(0.06 + 0.03 * check)
    };
    for gx in -2..22 {
        for gz in -2..22 {
            let (x, z) = (gx as f32, gz as f32);
            let c = ground(x, z);
            let q = [
                Vec3::new(x, 0.0, z),
                Vec3::new(x + 1.0, 0.0, z),
                Vec3::new(x, 0.0, z + 1.0),
                Vec3::new(x + 1.0, 0.0, z + 1.0),
            ];
            canvas.triangle(&vp, [q[0], q[1], q[2]], [c; 3]);
            canvas.triangle(&vp, [q[1], q[3], q[2]], [c; 3]);
        }
    }
    for (c, character) in skinned.iter().enumerate() {
        let offset = Vec3::new((c % grid) as f32 * 2.0, 0.0, (c / grid) as f32 * 2.0);
        let hue = c as f32 / chars as f32 * TAU;
        let albedo = Vec3::new(
            0.6 + 0.35 * hue.cos(),
            0.35 + 0.25 * (hue + 2.1).cos(),
            0.55 + 0.35 * (hue + 4.2).cos(),
        );
        let (pos, nrm) = (&character.positions, &character.normals);
        let shade = |k: usize| {
            let n = nrm[k];
            let rim = (1.0 - n.dot((eye - pos[k] - offset).normalize()).abs()).powi(3);
            albedo * (0.15 + 1.2 * n.dot(sun).max(0.0)) + Vec3::new(0.4, 0.5, 0.9) * rim * 0.6
        };
        for t in &tris {
            let [a, b, d] = t.map(|i| i as usize);
            canvas.triangle(
                &vp,
                [pos[a] + offset, pos[b] + offset, pos[d] + offset],
                [shade(a), shade(b), shade(d)],
            );
        }
    }
    canvas.save(&out)?;
    println!("wrote {out}");
    Ok(())
}
