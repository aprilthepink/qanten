//! Renders a Gerstner ocean whose wave field is computed as one NPU matrix
//! product per frame, lit by a sky baked to spherical harmonics on the NPU.
//!
//! ```sh
//! cargo run --release -p qanten-graphics --example ocean -- [grid_resolution] [output.png]
//! ```

mod common;

use std::time::Instant;

use common::{Canvas, camera_ray, max_err, ms, smoothstep, view_proj};
use glam::{Vec2, Vec3};
use qanten::Npu;
use qanten_graphics::ocean::{GerstnerOcean, Grid, wind_sea};
use qanten_graphics::sh::ShProjector;

fn sky(d: Vec3, sun: Vec3) -> Vec3 {
    let up = d.y.max(0.0);
    let horizon = Vec3::new(1.0, 0.62, 0.42);
    let zenith = Vec3::new(0.16, 0.30, 0.62);
    let base = horizon.lerp(zenith, up.powf(0.35));
    let mu = d.dot(sun).max(0.0);
    base * 0.55 + Vec3::new(1.0, 0.6, 0.3) * (mu.powf(6.0) * 0.5 + mu.powf(1200.0) * 60.0)
}

fn main() -> anyhow::Result<()> {
    let res: usize = std::env::args()
        .nth(1)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(400);
    let out = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "ocean.png".into());
    let npu = Npu::auto(0)?;
    println!("backend: {}", npu.backend().name());

    let sun = Vec3::new(0.15, 0.10, 1.0).normalize();
    let waves = wind_sea(64, Vec2::new(-0.3, -1.0), 60.0, 0.8, 42);
    let grid = Grid::new(Vec2::new(-120.0, 0.0), Vec2::new(240.0, 300.0), res, res);
    let t = 7.0;

    // Wave field: NPU formulation vs the usual per-point CPU loop.
    let cpu = GerstnerOcean::new(waves.clone(), grid)?;
    let t0 = Instant::now();
    let reference = cpu.evaluate_cpu(t);
    let cpu_ms = ms(t0.elapsed());
    let ocean = GerstnerOcean::new(waves, grid)?.with_npu(&npu)?;
    let t0 = Instant::now();
    let frame = ocean.evaluate(t)?;
    let npu_ms = ms(t0.elapsed());
    let macs = 9.0 * (res * res) as f64 * 2.0 * ocean.waves().len() as f64;
    println!(
        "grid {res}x{res}, {} waves, 1 matmul/frame = {:.0} M MAC",
        ocean.waves().len(),
        macs / 1e6
    );
    println!(
        "  CPU direct (f64, {} threads): {cpu_ms:8.1} ms",
        rayon::current_num_threads()
    );
    println!(
        "  NPU path ({}):        {npu_ms:8.1} ms",
        if npu.is_simulated() {
            "simulated"
        } else {
            "hardware"
        }
    );
    println!(
        "  max error: position {:.3} m of {:.2} m wave height, normal {:.4}",
        max_err(&frame.positions, &reference.positions),
        ocean.max_height(),
        max_err(&frame.normals, &reference.normals)
    );
    let buoy = ocean.sample(3.0, 40.0, t);
    println!(
        "  buoy at (3, 40): height {:.2} m, rising at {:.2} m/s",
        buoy.height, buoy.velocity_y
    );

    // Ambient sky light baked to SH on the NPU.
    let probe = ShProjector::fibonacci(4096)
        .with_npu(&npu)?
        .project_fn(|d| sky(d, sun))?;

    let (w, h) = (1280, 720);
    let eye = Vec3::new(0.0, 4.5, -2.0);
    let target = Vec3::new(0.0, 2.0, 60.0);
    let fov = 55f32.to_radians();
    let aspect = w as f32 / h as f32;
    let vp = view_proj(eye, target, fov, aspect, 0.5, 2000.0);
    let mut canvas = Canvas::new(w, h, |u, v| {
        sky(camera_ray(eye, target, fov, aspect, u, v), sun)
    });

    let deep = Vec3::new(0.004, 0.025, 0.045);
    let shallow = Vec3::new(0.02, 0.16, 0.17);
    let ambient = probe.irradiance(Vec3::Y) / std::f32::consts::PI;
    let pp = &frame.positions;
    let shade = |tri: [usize; 3], b: Vec3| -> Vec3 {
        let lerp3 = |v: [Vec3; 3]| v[0] * b.x + v[1] * b.y + v[2] * b.z;
        let p = lerp3(tri.map(|k| pp[k]));
        let n = lerp3(tri.map(|k| frame.normals[k])).normalize();
        let jac = tri.map(|k| frame.jacobian[k]);
        let jac = jac[0] * b.x + jac[1] * b.y + jac[2] * b.z;
        let v = (eye - p).normalize();
        let r = (-v).reflect(n);
        let fresnel = 0.02 + 0.98 * (1.0 - n.dot(v).max(0.0)).powi(5);
        // Subsurface-ish tint on wave crests facing the sun.
        let sss = smoothstep(0.0, 2.5, p.y) * (0.5 + 0.5 * sun.dot(-v).max(0.0));
        let body = deep * (probe.irradiance(n) / std::f32::consts::PI) * 4.0 + shallow * sss * 0.6;
        let mut c = body.lerp(sky(Vec3::new(r.x, r.y.abs(), r.z), sun), fresnel);
        let foam = smoothstep(0.35, 0.05, jac);
        c = c.lerp(ambient * 1.1, foam * 0.85);
        let fog = 1.0 - (-(p - eye).length() / 420.0).exp();
        c.lerp(
            sky(
                ((p - eye) * Vec3::new(1.0, 0.0, 1.0)).normalize() + Vec3::Y * 0.02,
                sun,
            ),
            fog,
        )
    };
    for j in 0..frame.nz - 1 {
        for i in 0..frame.nx - 1 {
            let [a, b, c, d] = [
                frame.index(i, j),
                frame.index(i + 1, j),
                frame.index(i, j + 1),
                frame.index(i + 1, j + 1),
            ];
            canvas.triangle_with(&vp, [pp[a], pp[b], pp[c]], |w| shade([a, b, c], w));
            canvas.triangle_with(&vp, [pp[b], pp[d], pp[c]], |w| shade([b, d, c], w));
        }
    }
    canvas.save(&out)?;
    println!("wrote {out}");
    Ok(())
}
