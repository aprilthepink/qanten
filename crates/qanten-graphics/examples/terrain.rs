//! Procedural terrain: a 3×3 block of 129×129 tiles generated in one NPU
//! call, shaded by height and slope (grass, rock on steep slopes, snow).
//!
//! ```sh
//! cargo run --release -p qanten-graphics --example terrain -- [output.png]
//! ```

mod common;

use std::time::Instant;

use common::{Canvas, ms, smoothstep, view_proj};
use glam::{Vec2, Vec3};
use qanten::Npu;
use qanten_graphics::terrain::{Terrain, TerrainTile, fbm_spectrum};

fn main() -> anyhow::Result<()> {
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "terrain.png".into());
    let npu = Npu::auto(0)?;
    println!("backend: {}", npu.backend().name());

    let waves = fbm_spectrum(96, 1200.0, 4.0, 45.0, 1.6, 11);
    let tile = Vec2::splat(256.0);
    let n = 129;
    let origins: Vec<Vec2> = (0..9)
        .map(|k| Vec2::new((k % 3) as f32, (k / 3) as f32) * tile)
        .collect();

    let terrain = Terrain::new(waves.clone(), tile, n, n)?;
    let t0 = Instant::now();
    let want = terrain.tiles_cpu(&origins);
    let cpu_ms = ms(t0.elapsed());
    let terrain = terrain.with_npu(&npu)?;
    let t0 = Instant::now();
    let tiles = terrain.tiles(&origins)?;
    let npu_ms = ms(t0.elapsed());
    let err = tiles
        .iter()
        .zip(&want)
        .flat_map(|(a, b)| a.heights.iter().zip(&b.heights).map(|(x, y)| (x - y).abs()))
        .fold(0.0, f32::max);
    println!(
        "terrain: 9 tiles × {n}×{n} samples × {} waves",
        terrain.waves().len()
    );
    println!("  CPU f64:  {cpu_ms:7.1} ms   NPU path: {npu_ms:7.1} ms (1 call)");
    println!(
        "  max height error {err:.3} m (heights within ±{:.0} m), self-check {:.4}",
        terrain.max_height(),
        terrain.self_check()?.unwrap_or(0.0)
    );
    let steep: usize = tiles
        .iter()
        .map(|t| {
            t.slope_mask(35f32.to_radians())
                .iter()
                .filter(|&&m| m)
                .count()
        })
        .sum();
    println!(
        "  {:.1} % of samples steeper than 35° (rock)",
        100.0 * steep as f64 / (9 * n * n) as f64
    );
    let probe = Vec2::new(300.0, 420.0);
    println!(
        "  exact height at {probe}: {:.2} m",
        terrain.height_at(probe.x, probe.y)
    );

    // --- Render.
    let (w, h) = (1280, 720);
    let centre = Vec3::new(384.0, 0.0, 384.0);
    let eye = centre + Vec3::new(-760.0, 560.0, -820.0);
    let vp = view_proj(
        eye,
        centre + Vec3::Y * -40.0,
        30f32.to_radians(),
        w as f32 / h as f32,
        1.0,
        4000.0,
    );
    let mut canvas = Canvas::new(w, h, |_, v| {
        Vec3::new(0.35, 0.55, 0.85).lerp(Vec3::new(0.85, 0.80, 0.75), v)
    });
    let sun = Vec3::new(-0.5, 0.6, -0.3).normalize();
    let colour = |p: Vec3, n: Vec3| {
        let grass = Vec3::new(0.18, 0.32, 0.10);
        let rock = Vec3::new(0.32, 0.28, 0.25);
        let snow = Vec3::new(0.95, 0.95, 1.0);
        let slope = 1.0 - n.y;
        let base = grass.lerp(rock, smoothstep(0.12, 0.25, slope));
        let base = base.lerp(
            snow,
            smoothstep(45.0, 60.0, p.y) * (1.0 - smoothstep(0.2, 0.35, slope)),
        );
        let fog = smoothstep(900.0, 2200.0, (p - eye).length());
        (base * (0.2 + 0.9 * n.dot(sun).max(0.0))).lerp(Vec3::new(0.7, 0.75, 0.8), fog * 0.5)
    };
    for t in &tiles {
        draw_tile(&mut canvas, &vp, t, &colour);
    }
    canvas.save(&out)?;
    println!("wrote {out}");
    Ok(())
}

fn draw_tile(
    canvas: &mut Canvas,
    vp: &glam::Mat4,
    t: &TerrainTile,
    colour: &dyn Fn(Vec3, Vec3) -> Vec3,
) {
    for j in 0..t.nz - 1 {
        for i in 0..t.nx - 1 {
            let idx = [(i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1)];
            let p = idx.map(|(a, b)| t.position(a, b));
            let c = idx.map(|(a, b)| colour(t.position(a, b), t.normals[t.index(a, b)]));
            canvas.triangle(vp, [p[0], p[2], p[1]], [c[0], c[2], c[1]]);
            canvas.triangle(vp, [p[1], p[2], p[3]], [c[1], c[2], c[3]]);
        }
    }
}
