//! Graphics workloads: the NPU formulation vs the plain CPU path, side by side.
//! Backend from `QANTEN_BACKEND` (driver, sim, auto).
//!
//! ```sh
//! cargo bench -p qanten-graphics
//! QANTEN_BACKEND=driver cargo bench -p qanten-graphics
//! ```

use std::f32::consts::TAU;
use std::hint::black_box;
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};
use qanten::Npu;
use qanten::compute::Matrix;
use qanten_graphics::blendshapes::Blendshapes;
use qanten_graphics::dct::Dct8x8;
use qanten_graphics::glam::{Mat4, Quat, Vec2, Vec3};
use qanten_graphics::ocean::{GerstnerOcean, Grid, wind_sea};
use qanten_graphics::sh::ShProjector;
use qanten_graphics::skinning::SkinnedMesh;

fn npu() -> Npu {
    let npu = Npu::from_env(0).unwrap_or_else(|e| panic!("cannot open NPU: {e}"));
    eprintln!("benchmarking on: {}", npu.backend().name());
    npu
}

fn group<'a>(
    c: &'a mut Criterion,
    name: &str,
) -> criterion::BenchmarkGroup<'a, criterion::measurement::WallTime> {
    let mut g = c.benchmark_group(name);
    g.sample_size(10).measurement_time(Duration::from_secs(3));
    g
}

fn ocean(c: &mut Criterion) {
    let npu = npu();
    let waves = wind_sea(64, Vec2::new(0.0, -1.0), 60.0, 0.8, 1);
    let grid = Grid::new(Vec2::ZERO, Vec2::splat(256.0), 256, 256);
    let cpu = GerstnerOcean::new(waves.clone(), grid).unwrap();
    let acc = GerstnerOcean::new(waves, grid)
        .unwrap()
        .with_npu(&npu)
        .unwrap();
    let mut g = group(c, "ocean_256x256_64waves");
    g.bench_function("cpu_direct", |b| {
        b.iter(|| cpu.evaluate_cpu(black_box(3.0)))
    });
    g.bench_function("npu_matmul", |b| {
        b.iter(|| acc.evaluate(black_box(3.0)).unwrap())
    });
    g.finish();
}

/// A strip of `verts` vertices, each blending two neighbouring bones.
fn strip_mesh(verts: usize, bones: usize) -> SkinnedMesh {
    let rest = (0..verts)
        .map(|i| Vec3::new((i % 20) as f32 * 0.01, i as f32 * 0.002, 0.0))
        .collect();
    let infl = (0..verts)
        .map(|i| {
            let b = (i * bones / verts) as u16;
            [
                (b, 0.7),
                ((b + 1).min(bones as u16 - 1), 0.3),
                (0, 0.0),
                (0, 0.0),
            ]
        })
        .collect();
    SkinnedMesh::new(rest, None, infl, bones).unwrap()
}

fn skinning(c: &mut Criterion) {
    let npu = npu();
    let (verts, bones, chars) = (1280usize, 12usize, 60usize);
    let poses: Vec<Vec<Mat4>> = (0..chars)
        .map(|c| {
            (0..bones)
                .map(|b| {
                    let k = (c * bones + b) as f32;
                    Mat4::from_rotation_translation(
                        Quat::from_rotation_z(k * 0.01),
                        Vec3::splat(0.1),
                    )
                })
                .collect()
        })
        .collect();
    let mut g = group(c, "skinning_60x1280v_12b");
    let cpu = strip_mesh(verts, bones);
    g.bench_function("cpu_lbs", |b| {
        b.iter(|| cpu.skin_batch_cpu(black_box(&poses)))
    });
    let mesh = strip_mesh(verts, bones).with_npu(&npu).unwrap();
    g.bench_function("npu_matmul", |b| {
        b.iter(|| mesh.skin_batch(black_box(&poses)).unwrap())
    });
    // A copy of the skinning matrix on each of 3 NPUs; characters split between them.
    let pool = qanten::pool::NpuPool::simulated(3).unwrap();
    let pooled = strip_mesh(verts, bones).with_pool(&pool).unwrap();
    g.bench_function("npu_pool3", |b| {
        b.iter(|| pooled.skin_batch(black_box(&poses)).unwrap())
    });
    g.finish();
}

fn blendshapes(c: &mut Criterion) {
    let npu = npu();
    let (verts, shapes, chars) = (4000usize, 52usize, 50usize);
    let base: Vec<Vec3> = (0..verts)
        .map(|i| Vec3::new(i as f32, 0.0, 0.0) * 0.001)
        .collect();
    let deltas: Vec<Vec<Vec3>> = (0..shapes)
        .map(|s| {
            (0..verts)
                .map(|i| Vec3::Z * ((i + s) as f32 * 0.01).sin() * 0.01)
                .collect()
        })
        .collect();
    let bs = Blendshapes::new(base, &deltas).unwrap();
    let w = Matrix::from_fn(chars, shapes, |c, s| {
        ((c * shapes + s) as f32 * 0.1).sin().abs()
    });
    let mut g = group(c, "blendshapes_50x4000v_52s");
    g.bench_function("cpu", |b| {
        b.iter(|| bs.evaluate_batch_cpu(black_box(&w)).unwrap())
    });
    let bs = bs.with_npu(&npu).unwrap();
    g.bench_function("npu_matmul", |b| {
        b.iter(|| bs.evaluate_batch(black_box(&w)).unwrap())
    });
    g.finish();
}

fn probes(c: &mut Criterion) {
    let npu = npu();
    let proj = ShProjector::fibonacci(1024);
    let radiance: Vec<Vec<Vec3>> = (0..256)
        .map(|p| {
            proj.directions()
                .iter()
                .map(|d| Vec3::splat(1.0 + d.y * (p as f32 / 256.0 * TAU).sin()))
                .collect()
        })
        .collect();
    let mut g = group(c, "sh_bake_256probes_1024samples");
    g.bench_function("cpu", |b| {
        b.iter(|| proj.project_batch_cpu(black_box(&radiance)).unwrap())
    });
    let proj = proj.with_npu(&npu).unwrap();
    g.bench_function("npu_matmul", |b| {
        b.iter(|| proj.project_batch(black_box(&radiance)).unwrap())
    });
    g.finish();
}

fn dct(c: &mut Criterion) {
    let npu = npu();
    let plane = Matrix::from_fn(512, 512, |y, x| ((y * 512 + x) % 255) as f32);
    let mut g = group(c, "dct_512x512");
    let cpu = Dct8x8::new();
    g.bench_function("cpu", |b| {
        b.iter(|| cpu.forward(black_box(&plane)).unwrap())
    });
    let acc = Dct8x8::new().with_npu(&npu).unwrap();
    g.bench_function("npu_matmul", |b| {
        b.iter(|| acc.forward(black_box(&plane)).unwrap())
    });
    g.finish();
}

criterion_group!(benches, ocean, skinning, blendshapes, probes, dct);
criterion_main!(benches);
