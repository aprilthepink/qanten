//! Each NPU formulation (run on the simulator) against its plain CPU reference.

use glam::{Mat4, Quat, Vec2, Vec3};
use qanten::Npu;
use qanten_graphics::blendshapes::Blendshapes;
use qanten_graphics::dct::Dct8x8;
use qanten_graphics::ocean::{GerstnerOcean, Grid, wind_sea};
use qanten_graphics::sh::{self, ShProjector};
use qanten_graphics::skinning::SkinnedMesh;

fn max_err(a: &[Vec3], b: &[Vec3]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (*x - *y).abs().max_element())
        .fold(0.0, f32::max)
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
    let weights: Vec<f32> = (0..5 * 12).map(|k| (k % 9) as f32 / 9.0).collect();
    let want = bs.evaluate_batch_cpu(&weights);
    let got = bs
        .with_npu(&Npu::simulator())
        .unwrap()
        .evaluate_batch(&weights)
        .unwrap();
    assert_eq!(got.len(), 5 * v);
    assert!(max_err(&got, &want) < 0.01, "{}", max_err(&got, &want));
}

#[test]
fn skinning_npu_matches_cpu() {
    // A column of vertices along y, two bones blending in the middle.
    let rest: Vec<Vec3> = (0..200)
        .map(|i| Vec3::new(0.1, i as f32 * 0.01, 0.05))
        .collect();
    let normals: Vec<Vec3> = rest.iter().map(|_| Vec3::X).collect();
    let infl = (0..200)
        .map(|i| {
            let t = i as f32 / 199.0;
            [(0, 1.0 - t), (1, t), (0, 0.0), (0, 0.0)]
        })
        .collect();
    let mesh = SkinnedMesh::new(rest, Some(normals), infl, 2).unwrap();
    let poses: Vec<Mat4> = (0..8)
        .flat_map(|c| {
            let a = c as f32 * 0.2;
            [
                Mat4::from_translation(Vec3::new(a, 0.0, 0.0)),
                Mat4::from_rotation_translation(Quat::from_rotation_z(a), Vec3::new(0.0, 0.0, a)),
            ]
        })
        .collect();
    let want = mesh.skin_batch_cpu(&poses);
    let got = mesh
        .with_npu(&Npu::simulator())
        .unwrap()
        .skin_batch(&poses)
        .unwrap();
    assert!(
        max_err(&got.positions, &want.positions) < 0.02,
        "{}",
        max_err(&got.positions, &want.positions)
    );
    assert!(max_err(&got.normals, &want.normals) < 0.02);
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
    let samples: Vec<Vec3> = (0..4)
        .flat_map(|p| {
            proj.directions()
                .iter()
                .map(move |&d| env(d) * (p + 1) as f32)
        })
        .collect();
    let got = proj.project_batch(&samples).unwrap();
    let want = proj.project_batch_cpu(&samples);
    for (g, w) in got.iter().zip(&want) {
        assert!(max_err(&g.coeffs, &w.coeffs) < 0.05 * 4.0);
    }
    let normals = [Vec3::Y, Vec3::X, -Vec3::Y, Vec3::new(0.0, 0.6, 0.8)];
    let a = sh::irradiance_batch(Some(&Npu::simulator()), &want, &normals).unwrap();
    let b = sh::irradiance_batch(None, &want, &normals).unwrap();
    assert!(max_err(&a, &b) < 0.1, "{}", max_err(&a, &b));
}

#[test]
fn dct_roundtrip() {
    let (w, h) = (32, 16);
    let img: Vec<f32> = (0..w * h)
        .map(|k| ((k * 37) % 255) as f32 / 255.0)
        .collect();
    let cpu = Dct8x8::new();
    let npu = Dct8x8::new().with_npu(&Npu::simulator()).unwrap();
    let c_cpu = cpu.forward(&img, w, h).unwrap();
    let c_npu = npu.forward(&img, w, h).unwrap();
    let back = cpu.inverse(&c_cpu, w, h).unwrap();
    assert!(img.iter().zip(&back).all(|(a, b)| (a - b).abs() < 1e-4));
    assert!(c_cpu.iter().zip(&c_npu).all(|(a, b)| (a - b).abs() < 0.05));
    let back_npu = npu.inverse(&c_npu, w, h).unwrap();
    assert!(img.iter().zip(&back_npu).all(|(a, b)| (a - b).abs() < 0.02));
    assert!(cpu.forward(&img, 30, 16).is_err());
}

#[test]
fn pools_match_one_npu_and_self_checks_pass() {
    use qanten::pool::NpuPool;
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
    let cpu = GerstnerOcean::new(waves, grid).unwrap();
    assert_eq!(cpu.self_check().unwrap(), None);

    // Skinning: many characters split across the pool.
    let rest: Vec<Vec3> = (0..100)
        .map(|i| Vec3::new(0.0, i as f32 * 0.02, 0.0))
        .collect();
    let infl: Vec<_> = (0..100)
        .map(|i| {
            [
                (0, 1.0 - i as f32 / 99.0),
                (1, i as f32 / 99.0),
                (0, 0.0),
                (0, 0.0),
            ]
        })
        .collect();
    let poses: Vec<Mat4> = (0..10)
        .flat_map(|c| [Mat4::IDENTITY, Mat4::from_rotation_z(c as f32 * 0.1)])
        .collect();
    let m1 = SkinnedMesh::new(rest.clone(), None, infl.clone(), 2)
        .unwrap()
        .with_npu(&one)
        .unwrap();
    let m3 = SkinnedMesh::new(rest, None, infl, 2)
        .unwrap()
        .with_pool(&pool)
        .unwrap();
    assert_eq!(
        m1.skin_batch(&poses).unwrap().positions,
        m3.skin_batch(&poses).unwrap().positions
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

#[test]
fn small_irradiance_bakes_stay_exact_on_the_cpu() {
    // A few probes × normals: Placement::Auto keeps it on the CPU, so the
    // "NPU" result is exactly the CPU result.
    let probes = [ShProjector::fibonacci(512)
        .project_fn(|d| Vec3::new(1.0 + d.y, 0.5, 0.2))
        .unwrap()];
    let normals = [Vec3::Y, Vec3::X, Vec3::Z];
    let a = sh::irradiance_batch(Some(&Npu::simulator()), &probes, &normals).unwrap();
    let b = sh::irradiance_batch(None, &probes, &normals).unwrap();
    assert!(max_err(&a, &b) < 1e-5, "{}", max_err(&a, &b));
}
