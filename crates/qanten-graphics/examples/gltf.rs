//! Load a skinned, morphing glTF model and play its animations on the NPU:
//! eight frames of the `Bend` clip skinned in one call (top row) and eight
//! weights of its morph target in one call (bottom row).
//!
//! ```sh
//! cargo run --release -p qanten-graphics --features gltf --example gltf -- [model.glb] [output.png]
//! ```
//!
//! Without arguments it loads the test asset `tests/data/gltf/tentacle.glb`.

mod common;

use std::f32::consts::TAU;

use common::{Canvas, max_err_all, view_proj};
use glam::{Mat4, Vec3};
use qanten::Npu;
use qanten::compute::Matrix;
use qanten_graphics::gltf::GltfScene;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let model = args.next().unwrap_or_else(|| {
        concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/gltf/tentacle.glb").into()
    });
    let out = args.next().unwrap_or_else(|| "gltf.png".into());
    let npu = Npu::auto(0)?;
    println!("backend: {}", npu.backend().name());

    let scene = GltfScene::load(&model)?;
    println!(
        "{model}: {} mesh(es), {} nodes, animations {:?}",
        scene.mesh_count(),
        scene.node_count(),
        scene.animation_names()
    );

    // --- Skinning: eight frames of the first clip in one NPU call.
    let rigged = scene.skinned_mesh(0)?;
    let clip = scene
        .animation("Bend")
        .or_else(|| scene.animation_at(0))
        .ok_or_else(|| anyhow::anyhow!("the model has no animations"))?;
    let frames = 8;
    let times: Vec<f32> = (0..frames)
        .map(|k| clip.start() + clip.duration() * k as f32 / (frames - 1) as f32)
        .collect();
    let poses: Vec<Vec<Mat4>> = times
        .iter()
        .map(|&t| clip.skinning_matrices(&rigged.skin, t))
        .collect();
    let cpu = rigged.mesh.skin_batch_cpu(&poses);
    let mesh = rigged.mesh.with_npu(&npu)?;
    let skinned = mesh.skin_batch(&poses)?;
    let positions = |s: &[qanten_graphics::skinning::Skinned]| -> Vec<Vec<Vec3>> {
        s.iter().map(|x| x.positions.clone()).collect()
    };
    println!(
        "skinned {frames} frames of `{}` ({} vertices, {} joints): max error {:.4} vs CPU",
        clip.name().unwrap_or("?"),
        mesh.vertex_count(),
        rigged.skin.joints.len(),
        max_err_all(&positions(&skinned), &positions(&cpu))
    );

    // --- Morph targets: eight weights in one NPU call.
    let morph = scene.blendshapes(0).ok();
    let morphed = match &morph {
        Some(m) => {
            let shapes = m.shapes.shape_count();
            let weights = Matrix::from_fn(frames, shapes, |f, _| f as f32 / (frames - 1) as f32);
            let cpu = m.shapes.evaluate_batch_cpu(&weights)?;
            let on_npu = scene.blendshapes(0)?.shapes.with_npu(&npu)?;
            let got = on_npu.evaluate_batch(&weights)?;
            println!(
                "morphed {frames} weight settings of {shapes} target(s): max error {:.4} vs CPU",
                max_err_all(&got, &cpu)
            );
            Some(got)
        }
        None => None,
    };

    // --- Render both rows.
    let (w, h) = (1280, 640);
    let eye = Vec3::new(4.2, 2.6, 9.5);
    let target = Vec3::new(4.2, 0.9, 0.0);
    let vp = view_proj(
        eye,
        target,
        40f32.to_radians(),
        w as f32 / h as f32,
        0.1,
        100.0,
    );
    let mut canvas = Canvas::new(w, h, |_, v| {
        Vec3::new(0.05, 0.06, 0.10).lerp(Vec3::new(0.22, 0.18, 0.26), v)
    });
    let sun = Vec3::new(0.3, 0.8, 0.5).normalize();
    let mut draw = |offset: Vec3, pos: &[Vec3], hue: f32| {
        let albedo = Vec3::new(
            0.6 + 0.35 * hue.cos(),
            0.35 + 0.25 * (hue + 2.1).cos(),
            0.55 + 0.35 * (hue + 4.2).cos(),
        );
        for t in &rigged.triangles {
            let [a, b, c] = t.map(|i| pos[i as usize] + offset);
            let n = (b - a).cross(c - a).normalize_or_zero();
            let light = albedo * (0.18 + 1.1 * n.dot(sun).abs());
            canvas.triangle(&vp, [a, b, c], [light; 3]);
        }
    };
    let spacing = 1.2;
    for (k, s) in skinned.iter().enumerate() {
        let hue = k as f32 / frames as f32 * TAU * 0.5;
        draw(Vec3::new(k as f32 * spacing, 0.6, -1.0), &s.positions, hue);
    }
    if let Some(rows) = &morphed {
        for (k, pos) in rows.iter().enumerate() {
            let hue = 3.5 + k as f32 / frames as f32;
            draw(Vec3::new(k as f32 * spacing, -2.0, 0.5), pos, hue);
        }
    }
    canvas.save(&out)?;
    println!("wrote {out}");
    Ok(())
}
