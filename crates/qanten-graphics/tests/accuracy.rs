//! Each NPU formulation (run on the simulator) against its plain CPU reference.

use glam::{Mat4, Quat, Vec2, Vec3};
use qanten::Npu;
use qanten::compute::Matrix;
use qanten::pool::NpuPool;
use qanten_graphics::blendshapes::Blendshapes;
use qanten_graphics::dct::Dct8x8;
use qanten_graphics::ocean::{GerstnerOcean, Grid, wind_sea};
use qanten_graphics::sh::{IrradianceBaker, ShProjector};
use qanten_graphics::skinning::SkinnedMesh;

fn max_err(a: &[Vec3], b: &[Vec3]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (*x - *y).abs().max_element())
        .fold(0.0, f32::max)
}

/// Largest error over per-character results.
fn max_err_all(a: &[Vec<Vec3>], b: &[Vec<Vec3>]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| max_err(x, y))
        .fold(0.0, f32::max)
}

/// A column of `n` vertices along y, blending from bone 0 to bone 1.
fn two_bone_column(n: usize, normals: bool) -> SkinnedMesh {
    let rest: Vec<Vec3> = (0..n)
        .map(|i| Vec3::new(0.1, i as f32 * 0.01, 0.05))
        .collect();
    let infl = (0..n)
        .map(|i| {
            let t = i as f32 / (n - 1) as f32;
            [(0, 1.0 - t), (1, t), (0, 0.0), (0, 0.0)]
        })
        .collect();
    let normals = normals.then(|| vec![Vec3::X; n]);
    SkinnedMesh::new(rest, normals, infl, 2).unwrap()
}

fn two_bone_poses(characters: usize) -> Vec<[Mat4; 2]> {
    (0..characters)
        .map(|c| {
            let a = c as f32 * 0.2;
            [
                Mat4::from_translation(Vec3::new(a, 0.0, 0.0)),
                Mat4::from_rotation_translation(Quat::from_rotation_z(a), Vec3::new(0.0, 0.0, a)),
            ]
        })
        .collect()
}

#[test]
fn ocean_npu_matches_direct_evaluation() {
    let waves = wind_sea(32, Vec2::new(1.0, 0.3), 60.0, 0.8, 7);
    let grid = Grid::new(Vec2::new(-500.0, 300.0), Vec2::new(200.0, 150.0), 64, 48);
    let cpu = GerstnerOcean::new(waves.clone(), grid).unwrap();
    let npu = GerstnerOcean::new(waves, grid)
        .unwrap()
        .with_npu(&Npu::simulator())
        .unwrap();
    assert!(npu.is_accelerated() && !cpu.is_accelerated());
    let scale = cpu.max_height();
    for t in [0.0, 13.7, 1000.0] {
        let want = cpu.evaluate(t).unwrap();
        let got = npu.evaluate(t).unwrap();
        let pos = max_err(&got.positions, &want.positions);
        let nrm = max_err(&got.normals, &want.normals);
        let jac = got
            .jacobian
            .iter()
            .zip(&want.jacobian)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        // Far from the origin (x = -500) and late in time, still bounded error.
        assert!(
            pos < 0.02 * scale,
            "t={t}: position error {pos} vs scale {scale}"
        );
        assert!(nrm < 0.02, "t={t}: normal error {nrm}");
        assert!(jac < 0.03, "t={t}: jacobian error {jac}");
    }
}

#[test]
fn ocean_point_sample_agrees_with_grid() {
    let waves = wind_sea(16, Vec2::X, 40.0, 0.7, 3);
    let grid = Grid::new(Vec2::ZERO, Vec2::new(50.0, 50.0), 51, 51);
    let ocean = GerstnerOcean::new(waves, grid).unwrap();
    let frame = ocean.evaluate_cpu(2.5);
    // A displaced grid vertex has height y; sampling at its displaced (x, z) must give y back.
    for &k in &[0usize, 777, 1300, 2600] {
        let p = frame.positions[k];
        let s = ocean.sample(p.x, p.z, 2.5);
        assert!((s.height - p.y).abs() < 1e-3, "{} vs {}", s.height, p.y);
    }
}

#[test]
fn blendshapes_npu_matches_cpu() {
    let v = 300;
    let base: Vec<Vec3> = (0..v)
        .map(|i| Vec3::new(i as f32 * 0.01, 1.0, -2.0))
        .collect();
    let deltas: Vec<Vec<Vec3>> = (0..12)
        .map(|s| {
            (0..v)
                .map(|i| Vec3::splat(((i * 7 + s * 13) % 17) as f32 / 170.0))
                .collect()
        })
        .collect();
    let bs = Blendshapes::new(base, &deltas).unwrap();
    let weights = Matrix::from_fn(5, 12, |c, s| ((c * 12 + s) % 9) as f32 / 9.0);
    let want = bs.evaluate_batch_cpu(&weights).unwrap();
    let bs = bs.with_npu(&Npu::simulator()).unwrap();
    let got = bs.evaluate_batch(&weights).unwrap();
    assert_eq!((got.len(), got[0].len()), (5, v));
    assert!(
        max_err_all(&got, &want) < 0.01,
        "{}",
        max_err_all(&got, &want)
    );
    assert_eq!(bs.evaluate(weights.row(2)).unwrap(), got[2]);
    assert!(bs.evaluate(&[1.0; 3]).is_err());
}

#[test]
fn skinning_npu_matches_cpu() {
    let mesh = two_bone_column(200, true);
    let poses = two_bone_poses(8);
    let want = mesh.skin_batch_cpu(&poses);
    let mesh = mesh.with_npu(&Npu::simulator()).unwrap();
    let got = mesh.skin_batch(&poses).unwrap();
    for (g, w) in got.iter().zip(&want) {
        assert!(max_err(&g.positions, &w.positions) < 0.02);
        assert!(max_err(&g.normals, &w.normals) < 0.02);
    }
    assert_eq!(mesh.skin(&poses[3]).unwrap(), got[3]);
    assert!(mesh.skin(&[Mat4::IDENTITY]).is_err(), "wrong bone count");
    assert!(SkinnedMesh::new(vec![Vec3::ZERO], None, vec![[(5, 1.0); 4]], 2).is_err());
}

#[test]
fn sh_projection_and_irradiance() {
    // Constant white sky: irradiance on any normal is π.
    let proj = ShProjector::fibonacci(2048)
        .with_npu(&Npu::simulator())
        .unwrap();
    let sky = proj.project_fn(|_| Vec3::ONE).unwrap();
    assert!((sky.irradiance(Vec3::Y).x - std::f32::consts::PI).abs() < 0.05);
    // A directional-ish environment: NPU vs CPU projection.
    let env = |d: Vec3| Vec3::new(1.0 + d.y, 0.5 + 0.5 * d.x, d.z.max(0.0) * 3.0);
    let probes: Vec<Vec<Vec3>> = (1..=4)
        .map(|p| {
            proj.directions()
                .iter()
                .map(|&d| env(d) * p as f32)
                .collect()
        })
        .collect();
    let got = proj.project_batch(&probes).unwrap();
    let want = proj.project_batch_cpu(&probes).unwrap();
    for (g, w) in got.iter().zip(&want) {
        assert!(max_err(&g.coeffs, &w.coeffs) < 0.05 * 4.0);
    }
    // Baking irradiance at a few normals: NPU vs exact per-normal evaluation.
    let normals = [Vec3::Y, Vec3::X, -Vec3::Y, Vec3::new(0.0, 0.6, 0.8)];
    let baker = IrradianceBaker::new(&normals)
        .with_npu(&Npu::simulator())
        .unwrap();
    let baked = baker.bake(&want).unwrap();
    let exact: Vec<Vec<Vec3>> = want
        .iter()
        .map(|p| normals.iter().map(|&n| p.irradiance(n)).collect())
        .collect();
    assert!(
        max_err_all(&baked, &exact) < 0.1,
        "{}",
        max_err_all(&baked, &exact)
    );
    assert!(max_err_all(&baker.bake_cpu(&want).unwrap(), &exact) < 1e-5);
}

#[test]
fn dct_roundtrip() {
    let img = Matrix::from_fn(16, 32, |r, c| ((r * 32 + c) * 37 % 255) as f32 / 255.0);
    let cpu = Dct8x8::new();
    let npu = Dct8x8::new().with_npu(&Npu::simulator()).unwrap();
    let c_cpu = cpu.forward(&img).unwrap();
    let c_npu = npu.forward(&img).unwrap();
    let close = |a: &Matrix, b: &Matrix, tol: f32| {
        a.as_slice()
            .iter()
            .zip(b.as_slice())
            .all(|(x, y)| (x - y).abs() < tol)
    };
    assert!(close(&cpu.inverse(&c_cpu).unwrap(), &img, 1e-4));
    assert!(close(&c_cpu, &c_npu, 0.05));
    assert!(close(&npu.inverse(&c_npu).unwrap(), &img, 0.02));
    assert!(cpu.forward(&Matrix::zeros(16, 30)).is_err());
}

#[test]
fn pools_match_one_npu_and_self_checks_pass() {
    let pool = NpuPool::simulated(3).unwrap();
    let one = Npu::simulator();

    // Ocean: each NPU gets a share of the rows; results identical to one NPU.
    let waves = wind_sea(16, Vec2::X, 40.0, 0.7, 3);
    let grid = Grid::new(Vec2::ZERO, Vec2::new(80.0, 60.0), 40, 31);
    let a = GerstnerOcean::new(waves.clone(), grid)
        .unwrap()
        .with_npu(&one)
        .unwrap();
    let b = GerstnerOcean::new(waves.clone(), grid)
        .unwrap()
        .with_pool(&pool)
        .unwrap();
    assert_eq!(
        a.evaluate(3.0).unwrap().positions,
        b.evaluate(3.0).unwrap().positions
    );
    assert_eq!(
        GerstnerOcean::new(waves, grid)
            .unwrap()
            .self_check()
            .unwrap(),
        None
    );

    // Skinning: characters split across the pool.
    let poses = two_bone_poses(10);
    let m1 = two_bone_column(100, false).with_npu(&one).unwrap();
    let m3 = two_bone_column(100, false).with_pool(&pool).unwrap();
    assert_eq!(
        m1.skin_batch(&poses).unwrap(),
        m3.skin_batch(&poses).unwrap()
    );

    // Every accelerated type reports a small self-check error.
    let checks = [
        b.self_check().unwrap(),
        m3.self_check().unwrap(),
        ShProjector::fibonacci(256)
            .with_pool(&pool)
            .unwrap()
            .self_check()
            .unwrap(),
        Dct8x8::new()
            .with_pool(&pool)
            .unwrap()
            .self_check()
            .unwrap(),
        IrradianceBaker::new(&[Vec3::Y, Vec3::X])
            .with_pool(&pool)
            .unwrap()
            .self_check()
            .unwrap(),
        Blendshapes::new(
            vec![Vec3::ZERO; 10],
            &[vec![Vec3::ONE; 10], vec![Vec3::X; 10]],
        )
        .unwrap()
        .with_npu(&one)
        .unwrap()
        .self_check()
        .unwrap(),
    ];
    for c in checks {
        let err = c.expect("accelerated");
        assert!(err < 1e-2, "self-check error {err}");
    }
}
