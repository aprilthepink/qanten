# qanten-graphics

Offload CPU-side graphics and 3D work to Q.ANT photonic NPUs.

Engines keep some geometry work on the CPU because the result is needed
there, not on the GPU: wave heights for buoyancy, deformed meshes for
collision and picking, baked lighting in asset pipelines. Each module here
rewrites one of those jobs as a large matrix product with the big, constant
matrix kept resident on the NPU, so each frame only moves small inputs.

| Module | Workload | NPU formulation |
|---|---|---|
| `ocean` | Gerstner waves on a grid: positions, normals, foam mask; exact point queries for buoyancy | angle-sum split of every phase → **one** `[9·Nz, 2W] × [2W, Nx]` matmul per frame |
| `skinning` | linear blend skinning for crowds (positions + normals) | resident `[vertices, 4·bones]` × per-character `(M − I)` rows |
| `blendshapes` | morph targets for crowds | resident `[3·vertices, shapes]` × per-character weights |
| `sh` | light-probe baking to SH9, irradiance for many normals | resident `[9, samples]` basis × radiance rows |
| `dct` | 8×8 block DCT / inverse for image and texture pipelines | resident 64×64 Kronecker matrix × all blocks |

Every type works on the CPU out of the box and moves to the NPU with
`.with_npu(&npu)`. The CPU path is also the accuracy reference in tests.

```rust
use qanten::Npu;
use qanten_graphics::glam::Vec2;
use qanten_graphics::ocean::{GerstnerOcean, Grid, wind_sea};

let npu = Npu::auto(0)?;
let waves = wind_sea(64, Vec2::new(0.0, -1.0), 60.0, 0.8, 42);
let grid = Grid::new(Vec2::new(-120.0, 0.0), Vec2::new(240.0, 300.0), 400, 400);
let ocean = GerstnerOcean::new(waves, grid)?.with_npu(&npu)?;

let frame = ocean.evaluate(t)?;          // positions, normals, foam (Jacobian)
let buoy = ocean.sample(3.0, 40.0, t);   // exact height + vertical velocity
```

## Examples

```sh
cargo run --release -p qanten-graphics --example ocean    # renders ocean.png
cargo run --release -p qanten-graphics --example crowd    # 100 skinned tentacles + blendshape bench
cargo run --release -p qanten-graphics --example probes   # bakes 1024 light probes
```

Each example prints NPU-path vs CPU-reference timing and the maximum error.
Measured on the built-in simulator (4 CPU threads):

| Example | CPU reference | NPU path (simulated) | Max error |
|---|---|---|---|
| ocean 400², 64 waves | 103 ms | 63 ms | 2 cm of 11 m wave height |
| skinning 100 × 1280 verts | 2 ms | 19 ms | 8.5 mm on a 3 m mesh |
| blendshapes 200 × 4000 verts × 52 | 22 ms | 65 ms | 0.14 mm |
| SH bake 1024 probes × 2048 samples | 5 ms | 46 ms | 0.005 of DC ≈ 1.1 |

The simulator does the same multiplications as the NPU plus format
conversions, so it shows correctness, not hardware speed. The ocean is the
exception: its matrix formulation is already faster than the usual
per-point loop on a plain CPU. Real speedups for the other modules depend
on the NPU's throughput and transfer cost. Run the examples on hardware to
find out.

## Precision

The NPU computes in bfloat16 (about 3 significant digits). The modules keep
values bounded: phases are reduced on the host in f64; skinning and
blendshapes compute offsets from the rest pose, so the error scales with how
far the mesh moved, not with its size. That is fine for physics proxies,
culling, baking and previews, not for exact final-frame geometry.

Unofficial community project, not affiliated with Q.ANT GmbH.
