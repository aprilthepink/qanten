//! Image filters on the NPU: bloom on an HDR image (three blur radii, two NPU
//! calls), a mip chain, and Sobel edges plus a normal map from a terrain
//! height map.
//!
//! ```sh
//! cargo run --release -p qanten-graphics --example filters -- [output.png]
//! ```

mod common;

use std::time::Instant;

use common::{Canvas, ms};
use glam::{Vec2, Vec3};
use qanten::Npu;
use qanten::compute::Matrix;
use qanten_graphics::filters::{Bloom, Separable, Stencil3x3};
use qanten_graphics::terrain::{Terrain, fbm_spectrum};

const N: usize = 384;

/// A dark HDR scene with a few very bright lights.
fn hdr_scene() -> Vec<Matrix> {
    let lights = [
        (Vec2::new(90.0, 110.0), Vec3::new(12.0, 6.0, 2.0), 7.0),
        (Vec2::new(250.0, 80.0), Vec3::new(3.0, 8.0, 14.0), 5.0),
        (Vec2::new(300.0, 280.0), Vec3::new(14.0, 3.0, 9.0), 9.0),
        (Vec2::new(140.0, 300.0), Vec3::new(6.0, 12.0, 4.0), 4.0),
    ];
    let pixel = |y: usize, x: usize| {
        let p = Vec2::new(x as f32, y as f32);
        let mut c = Vec3::new(0.02, 0.025, 0.05) * (1.0 + (y as f32 / N as f32));
        // A faint grid, so the glow visibly spills over structure.
        if x % 48 < 2 || y % 48 < 2 {
            c += Vec3::splat(0.08);
        }
        for (centre, colour, radius) in lights {
            if (p - centre).length() < radius {
                c += colour;
            }
        }
        c
    };
    (0..3)
        .map(|ch| Matrix::from_fn(N, N, |y, x| pixel(y, x)[ch]))
        .collect()
}

fn main() -> anyhow::Result<()> {
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "filters.png".into());
    let npu = Npu::auto(0)?;
    println!("backend: {}", npu.backend().name());

    // --- Bloom: 3 radii × 3 channels in two NPU calls.
    let scene = hdr_scene();
    let bloom = Bloom::new(N, N, &[2.0, 6.0, 16.0], 1.0, 2.0)?;
    let t0 = Instant::now();
    let want = bloom.apply_cpu(&scene)?;
    let cpu_ms = ms(t0.elapsed());
    let bloom = bloom.with_npu(&npu)?;
    let t0 = Instant::now();
    let glowing = bloom.apply(&scene)?;
    let npu_ms = ms(t0.elapsed());
    let err = glowing
        .iter()
        .zip(&want)
        .map(|(a, b)| a.relative_error(b))
        .fold(0.0, f64::max);
    println!("bloom: {N}×{N} × 3 channels × 3 radii");
    println!(
        "  CPU:      {cpu_ms:7.1} ms   NPU path: {npu_ms:7.1} ms (2 calls), relative error {err:.4}"
    );

    // --- Mip chain: each level is two NPU calls for all channels.
    let mut level = scene.clone();
    let mut sizes = vec![N];
    while level[0].rows() > 8 {
        let n = level[0].rows();
        level = Separable::downsample(n, n).with_npu(&npu)?.apply(&level)?;
        sizes.push(level[0].rows());
    }
    println!("mip chain: {sizes:?}");

    // --- Edges and a normal map from a terrain height map. The height map is
    // the exact CPU tile: differentiating amplifies the NPU's bf16 noise, so
    // height maps meant for normal maps should be generated exactly.
    let terrain = Terrain::new(
        fbm_spectrum(64, 600.0, 8.0, 30.0, 1.4, 5),
        Vec2::splat(300.0),
        N,
        N,
    )?;
    let tile = terrain.tiles_cpu(&[Vec2::ZERO]).remove(0);
    let height = Matrix::new(N, N, tile.heights.clone())?;
    let sobel = Stencil3x3::sobel().with_npu(&npu)?;
    let t0 = Instant::now();
    let edges = sobel.edges(&height)?;
    let metres_per_pixel = tile.spacing.x;
    let normals = sobel.normal_map(&height, 1.0 / metres_per_pixel)?;
    // Compare with the terrain's exact normals (tangent space: x, z, up),
    // away from the clamped borders.
    let interior = (1..N - 1).flat_map(|y| (1..N - 1).map(move |x| y * N + x));
    let mean_err = interior
        .map(|p| {
            let b = tile.normals[p];
            (normals[p] - Vec3::new(b.x, b.z, b.y)).length()
        })
        .sum::<f32>()
        / ((N - 2) * (N - 2)) as f32;
    println!(
        "sobel + normal map: {N}×{N} in {:.1} ms; mean |n − exact| = {mean_err:.3}",
        ms(t0.elapsed())
    );

    // --- 2×2 panel: scene, bloom, edges, normal map.
    let mut canvas = Canvas::new(2 * N, 2 * N, |_, _| Vec3::ZERO);
    let edge_scale = edges
        .as_slice()
        .iter()
        .copied()
        .fold(0.0, f32::max)
        .max(1e-6);
    for y in 0..N {
        for x in 0..N {
            let rgb = |m: &[Matrix]| Vec3::new(m[0][(y, x)], m[1][(y, x)], m[2][(y, x)]);
            canvas.color[y * 2 * N + x] = rgb(&scene);
            canvas.color[y * 2 * N + x + N] = rgb(&glowing);
            let e = (edges[(y, x)] / edge_scale).sqrt();
            canvas.color[(y + N) * 2 * N + x] = Vec3::new(e, e * 0.9, e * 0.6) * 0.9;
            let n = normals[y * N + x];
            canvas.color[(y + N) * 2 * N + x + N] = (n * 0.5 + Vec3::splat(0.5)) * 0.9;
        }
    }
    canvas.save(&out)?;
    println!("wrote {out}");
    Ok(())
}
