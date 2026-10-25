# Light probes (`qanten_graphics::sh`)

![Spheres lit by baked probes](../images/probes.png)

*Eight of 1024 procedural environments, each baked to 9 RGB spherical-harmonic
coefficients on the NPU (1024 probes × 2048 samples in one call), then used to shade
a diffuse sphere.*

Order-2 spherical harmonics (SH9) are the standard way to store diffuse lighting in
light probes. Baking them, i.e. projecting thousands of radiance samples per probe onto
nine basis functions, is classic CPU work in lightmappers and asset pipelines.

## Usage

```rust
use qanten::Npu;
use qanten_graphics::glam::Vec3;
use qanten_graphics::sh::{IrradianceBaker, ShProjector};

fn main() -> qanten::Result<()> {
    let npu = Npu::auto(0)?;
    let projector = ShProjector::fibonacci(2048).with_npu(&npu)?;

    // One environment given as a function of direction.
    let sky = |d: Vec3| Vec3::new(0.3, 0.5, 1.0) * d.y.max(0.0) + Vec3::splat(0.05);
    let probe = projector.project_fn(sky)?;
    println!("irradiance facing up: {:?}", probe.irradiance(Vec3::Y));

    // Many probes at once: one list of samples (one per direction) per probe.
    let samples: Vec<Vec<Vec3>> = (0..64)
        .map(|p| projector.directions().iter().map(|&d| sky(d) * (1.0 + p as f32 * 0.1)).collect())
        .collect();
    let probes = projector.project_batch(&samples)?;

    // Bake vertex lighting: every probe at every normal, on the NPU.
    let baker = IrradianceBaker::new(&[Vec3::Y, Vec3::X, -Vec3::Y]).with_npu(&npu)?;
    let lit = baker.bake(&probes)?;              // lit[probe][normal]
    assert_eq!((lit.len(), lit[0].len()), (64, 3));
    Ok(())
}
```

## API

| Item | Purpose |
|---|---|
| `basis(dir)` | the 9 real SH basis values for a unit direction (y up) |
| `Sh9 { coeffs: [Vec3; 9] }` | one RGB probe; `radiance(dir)` reconstructs incoming light, `irradiance(normal)` gives diffuse irradiance (multiply by `albedo / π` for outgoing radiance) |
| `ShProjector::fibonacci(n)` | `n` evenly spread directions with equal solid angles |
| `ShProjector::from_directions(dirs, &solid_angles)` | your own sampling, e.g. cube-map texels |
| `.with_npu(&npu)` / `.with_pool(&pool)` | uploads the `[9, n]` weighted basis |
| `directions()` | where to sample your environment |
| `project_batch(&probes)` / `project_fn(f)` / `project_batch_cpu(..)` | one sample list per probe → `Vec<Sh9>` |
| `IrradianceBaker::new(&normals)` + `.with_npu` | the `[normals, 9]` convolved basis, kept on the NPU |
| `bake(&probes)` / `bake_cpu(&probes)` | irradiance of every probe at every normal: `result[probe][normal]` |

`irradiance` uses the cosine-lobe factors from Ramamoorthi & Hanrahan (2001):
π, 2π/3 and π/4 per band.

## How it maps to the NPU

Projection is `c_i = Σ_n Y_i(d_n) · dω_n · L(d_n)`. The `[9, n]` matrix of basis values
times solid angles is the same for every probe, so it stays on the NPU. Each probe and
colour channel is one row of `n` samples: 1024 RGB probes are 3072 rows in one call.
`IrradianceBaker` is the same shape the other way round: `[normals, 9]` basis rows
stay on the NPU, `[3·probes, 9]` coefficients stream through, so the same normals can be
lit by any number of probe sets.

## Accuracy

The `probes` example measures a largest coefficient error of 0.005 against a DC term of
about 1.1 (0.5 %). A constant white sky reproduces the exact irradiance π within 0.05.
