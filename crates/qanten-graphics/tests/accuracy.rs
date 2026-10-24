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
