# Ocean waves (`qanten_graphics::ocean`)

![A Gerstner ocean at sunset](../images/ocean.png)

*400 × 400 grid, 64 waves. The whole wave field (positions, normals, foam) is one NPU
matrix product; the sky lighting is baked to spherical harmonics on the NPU too.*

Games run ocean waves on the GPU for rendering but also on the CPU for gameplay:
boats need the height, slope and vertical speed of the water under them. This module
evaluates Gerstner waves on a grid via the NPU, and answers exact point queries on the
host.

## Usage

```rust
use qanten::Npu;
use qanten_graphics::glam::Vec2;
use qanten_graphics::ocean::{GerstnerOcean, Grid, wind_sea};

fn main() -> qanten::Result<()> {
    let npu = Npu::auto(0)?;
    // 64 waves, wind blowing along -z, longest wavelength 60 m, choppiness 0.8.
    let waves = wind_sea(64, Vec2::new(0.0, -1.0), 60.0, 0.8, 42);
    // 240 m × 300 m starting at (-120, 0), sampled 400 × 400.
    let grid = Grid::new(Vec2::new(-120.0, 0.0), Vec2::new(240.0, 300.0), 400, 400);
    let ocean = GerstnerOcean::new(waves, grid)?.with_npu(&npu)?;

    let t = 7.0; // seconds
    let frame = ocean.evaluate(t)?;
    let k = frame.index(200, 100);              // x index 200, z index 100
    println!("vertex {:?}, normal {:?}", frame.positions[k], frame.normals[k]);
    let foam = frame.jacobian[k] < 0.3;         // compressed surface = breaking crest
    println!("foam: {foam}");

    // Exact query for a boat at world position (3, 40):
    let s = ocean.sample(3.0, 40.0, t);
    println!("height {:.2} m, rising {:.2} m/s, normal {:?}", s.height, s.velocity_y, s.normal);
    Ok(())
}
```

## API

| Item | Purpose |
|---|---|
| `Wave { direction, wavelength, amplitude, steepness, phase }` | one Gerstner wave; `wavenumber()` and `angular_frequency()` (deep water, `ω = √(g·k)`) |
| `wind_sea(count, wind_dir, longest, choppiness, seed)` | a plausible sea: wavelengths from `longest` down to `longest/32`, directions within ±60° of the wind, steepness scaled so crests never fold (`Σ Q·k·A = choppiness ≤ 1`) |
| `Grid::new(origin, size, nx, nz)` | sample points on the `y = 0` plane, edges included |
| `GerstnerOcean::new(waves, grid)` | CPU ocean; `.with_npu(&npu)` uploads the time-independent basis |
| `evaluate(t)` | whole grid → `OceanFrame { positions, normals, jacobian }`, row-major over `(z, x)`, index with `frame.index(i, j)` |
| `evaluate_cpu(t)` | the usual per-point f64 loop (reference) |
| `sample(x, z, t)` | exact `height`, `normal`, `velocity_y` at a world position |
| `max_height()` | sum of amplitudes, an upper bound for error scales |

`sample` accounts for the horizontal motion of Gerstner waves: it first finds which
grid point ends up at `(x, z)` (fixed-point iteration), then evaluates there. A
displaced grid vertex sampled at its own `(x, z)` gives its height back within 1 mm.

## How it maps to the NPU

A Gerstner surface is built from nine sums over the waves, all of the form
`Σ_w α_w · cos θ_w` or `Σ_w α_w · sin θ_w`, where `θ = k·(D·p) − ωt + φ`:
the displacement in x and z, the height, the three normal components, and three
Jacobian terms for foam.

The direct way evaluates `sin` and `cos` for every wave at every grid point:
`Nx · Nz · W` trig calls. Splitting each phase into an x part and a z part,

```text
θ = a·x + (b·z + c)       with  a = k·Dx,  b = k·Dz,  c = φ − ω·t
```

and applying the angle-sum formulas,

```text
Σ_w α_w cos θ_w = Σ_w [α_w cos(b_w z + c_w)] · cos(a_w x)  −  [α_w sin(b_w z + c_w)] · sin(a_w x)
```

turns each sum into a product of a **z-side table** and an **x-side table**. Stacking
all nine fields gives one matrix product per frame:

```text
[9·Nz, 2W]   ×   [2W, Nx]   =   [9·Nz, Nx]
 per frame       resident        all fields on the whole grid
 (has time)      (no time)
```

- The x-side matrix `[cos(a·x) | sin(a·x)]` doesn't depend on time, so it is uploaded
  once by `with_npu` and stays on the NPU.
- Each frame uploads `9·Nz·2W` values and downloads `9·Nz·Nx`.
- Only `(Nx + Nz)·W` trig calls remain, done on the host in f64 with exact reduction
  modulo 2π. The NPU never sees a large phase, which matters because bf16 can't
  represent a phase like 1000.37 rad precisely.

That is also why the NPU path is faster than the direct loop even on the simulator:
the reformulation replaces most of the trig work with multiply-adds.

## Accuracy

On a 64 × 48 grid placed 500 m from the origin, at t = 0, 13.7 s and 1000 s, the
NPU path stays within 2 % of the maximum wave height for positions, 0.02 for normals
and 0.03 for the Jacobian (see `tests/accuracy.rs`). The rendered example measures
2 cm of error on 11 m of combined amplitude.

## Tips

- Grid spacing should be below half the shortest wavelength (`longest / 32` for
  `wind_sea`), or short waves alias.
- `jacobian < ~0.3` makes a good foam mask; values near or below 0 mean the surface is
  about to fold.
- For floating objects, call `sample` at a few hull points per object per frame.
  It's f64 on the host and costs one pass over the waves per iteration.
