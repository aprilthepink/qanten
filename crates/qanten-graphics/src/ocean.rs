//! Gerstner ocean waves on a grid: displaced positions, normals and a foam
//! (Jacobian) mask, plus exact height queries for buoyancy.
//!
//! ![A 400×400 Gerstner ocean with 64 waves, evaluated as one NPU matrix product (example `ocean`)](https://raw.githubusercontent.com/aprilthepink/qanten/main/docs/images/ocean.png)
//!
//! # How it maps to the NPU
//!
//! Each wave's phase is `θ = a·x + (b·z + c)` with `a = k·Dx`, `b = k·Dz`,
//! `c = φ − ω·t`. The angle-sum formulas split every `Σ α·cos θ` and
//! `Σ α·sin θ` into a product of a z-side table and an x-side table:
//!
//! ```text
//! Σ_w α_w cos θ = Σ_w (α_w cos(b z + c)) cos(a x) − (α_w sin(b z + c)) sin(a x)
//! ```
//!
//! So all nine fields Gerstner needs are one matrix product per frame:
//! `[9·Nz, 2W] × [2W, Nx]`. The x-side matrix `[cos(a x) | sin(a x)]` does
//! not depend on time and stays resident on the NPU. Only `9·Nz·2W` values
//! go up per frame while `9·Nz·Nx` come back. The tables are computed on the
//! host in f64 (only `(Nx + Nz)·W` trig calls instead of `Nx·Nz·W`), so the
//! NPU only ever sees values in `[-1, 1] · α` and never a large phase.

use std::f64::consts::TAU;

use glam::{Vec2, Vec3};
use rayon::prelude::*;

use crate::accel::ResidentMatrix;
use crate::{Error, Npu, Result};

/// Gravity used for the deep-water dispersion relation `ω = √(g·k)`.
pub const GRAVITY: f64 = 9.81;

/// One Gerstner wave.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Wave {
    /// Travel direction (normalised on use).
    pub direction: Vec2,
    /// Crest-to-crest distance in metres.
    pub wavelength: f32,
    /// Crest height in metres.
    pub amplitude: f32,
    /// Gerstner steepness `Q` (0 = sine wave, larger = sharper crests).
    pub steepness: f32,
    /// Phase offset in radians.
    pub phase: f32,
}

impl Wave {
    /// Wavenumber `k = 2π / wavelength`, in rad/m.
    pub fn wavenumber(&self) -> f64 {
        TAU / self.wavelength as f64
    }

    /// Angular frequency `ω = √(g·k)` (deep water), in rad/s.
    pub fn angular_frequency(&self) -> f64 {
        (GRAVITY * self.wavenumber()).sqrt()
    }

    fn dir(&self) -> (f64, f64) {
        let d = self.direction.normalize_or(Vec2::X);
        (d.x as f64, d.y as f64)
    }
}

/// A plausible wind-driven sea: `count` waves with wavelengths spread
/// geometrically from `longest` down to `longest / 32`, travelling within
/// ±60° of `wind_direction`. Steepness is scaled so crests never loop.
pub fn wind_sea(
    count: usize,
    wind_direction: Vec2,
    longest: f32,
    choppiness: f32,
    seed: u64,
) -> Vec<Wave> {
    let mut rng = seed ^ 0x9E37_79B9_7F4A_7C15;
    let mut next = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng >> 11) as f32 / (1u64 << 53) as f32
    };
    let wind = wind_direction.normalize_or(Vec2::X);
    let base = wind.y.atan2(wind.x);
    let n = count.max(1);
    let mut waves: Vec<Wave> = (0..n)
        .map(|i| {
            let t = i as f32 / (n.max(2) - 1) as f32;
            let wavelength = longest * (1.0 / 32.0f32).powf(t) * (0.9 + 0.2 * next());
            let spread = (next() - 0.5) * 2.0 * std::f32::consts::FRAC_PI_3 * (0.4 + 0.6 * t);
            let angle = base + spread;
            Wave {
                direction: Vec2::new(angle.cos(), angle.sin()),
                wavelength,
                amplitude: wavelength * 0.011 * (0.6 + 0.8 * next()),
                steepness: 0.0,
                phase: next() * std::f32::consts::TAU,
            }
        })
        .collect();
    // Σ Q·k·A ≤ choppiness keeps the surface from folding over itself.
    let total: f64 = waves
        .iter()
        .map(|w| w.wavenumber() * w.amplitude as f64)
        .sum();
    for w in &mut waves {
        w.steepness = (choppiness.clamp(0.0, 1.0) as f64 / total.max(1e-9)) as f32;
    }
    waves
}

/// A regular grid of sample points on the `y = 0` plane.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Grid {
    /// World position of sample `(0, 0)`.
    pub origin: Vec2,
    /// Extent in metres along x and z.
    pub size: Vec2,
    /// Samples along x and z.
    pub resolution: (usize, usize),
}

impl Grid {
    /// `nx × nz` samples covering `size` metres from `origin` (edges included).
    pub fn new(origin: Vec2, size: Vec2, nx: usize, nz: usize) -> Self {
        Self {
            origin,
            size,
            resolution: (nx, nz),
        }
    }

    fn axis(origin: f32, size: f32, n: usize) -> Vec<f64> {
        let step = if n > 1 {
            size as f64 / (n - 1) as f64
        } else {
            0.0
        };
        (0..n).map(|i| origin as f64 + i as f64 * step).collect()
    }

    fn xs(&self) -> Vec<f64> {
        Self::axis(self.origin.x, self.size.x, self.resolution.0)
    }

    fn zs(&self) -> Vec<f64> {
        Self::axis(self.origin.y, self.size.y, self.resolution.1)
    }
}

/// One evaluated frame. Arrays are row-major over `(z, x)`: index `j * nx + i`.
#[derive(Debug, Clone, PartialEq)]
pub struct OceanFrame {
    /// Samples along x.
    pub nx: usize,
    /// Samples along z.
    pub nz: usize,
    /// Displaced world-space positions.
    pub positions: Vec<Vec3>,
    /// Unit surface normals.
    pub normals: Vec<Vec3>,
    /// Jacobian of the horizontal displacement. Values below ~0.3 mean the
    /// surface is compressed into a breaking crest: use it as a foam mask.
    pub jacobian: Vec<f32>,
}

impl OceanFrame {
    /// Array index of grid sample `(i, j)` (x index, z index).
    pub fn index(&self, i: usize, j: usize) -> usize {
        j * self.nx + i
    }
}

/// Exact single-point sample (f64 on the host), see [`GerstnerOcean::sample`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OceanSample {
    /// Surface height above `y = 0`, in metres.
    pub height: f32,
    /// Unit surface normal.
    pub normal: Vec3,
    /// Vertical velocity of the surface, in m/s (positive = rising).
    pub velocity_y: f32,
}

// The nine sums Gerstner needs, as (uses_sin, coefficient(wave)).
const FIELDS: usize = 9;

fn coefficients(w: &Wave) -> [f64; FIELDS] {
    let (dx, dz) = w.dir();
    let a = w.amplitude as f64;
    let q = w.steepness as f64;
    let k = w.wavenumber();
    [
        q * a * dx,          // 0 cos: x displacement
        q * a * dz,          // 1 cos: z displacement
        k * a * dx,          // 2 cos: normal x (negated)
        k * a * dz,          // 3 cos: normal z (negated)
        a,                   // 4 sin: height
        q * k * a,           // 5 sin: normal y (1 - ...)
        q * a * k * dx * dx, // 6 sin: dX/dx (1 - ...)
        q * a * k * dz * dz, // 7 sin: dZ/dz (1 - ...)
        q * a * k * dx * dz, // 8 sin: dX/dz (- ...)
    ]
}

const USES_SIN: [bool; FIELDS] = [false, false, false, false, true, true, true, true, true];

/// A set of Gerstner waves evaluated on a fixed grid.
#[derive(Debug)]
pub struct GerstnerOcean {
    waves: Vec<Wave>,
    grid: Grid,
    npu_basis: Option<ResidentMatrix>,
}

impl GerstnerOcean {
    /// A CPU-evaluated ocean. Fails on an empty wave list, non-positive wavelengths or an empty grid.
    pub fn new(waves: Vec<Wave>, grid: Grid) -> Result<Self> {
        if waves.is_empty() {
            return Err(Error::InvalidArgument(
                "ocean needs at least one wave".into(),
            ));
        }
        if waves
            .iter()
            .any(|w| !w.wavelength.is_finite() || w.wavelength <= 0.0)
        {
            return Err(Error::InvalidArgument("wavelengths must be > 0".into()));
        }
        if grid.resolution.0 == 0 || grid.resolution.1 == 0 {
            return Err(Error::InvalidArgument("grid resolution must be > 0".into()));
        }
        Ok(Self {
            waves,
            grid,
            npu_basis: None,
        })
    }

    /// Uploads the time-independent x-side basis `[cos(a·x) | sin(a·x)]`
    /// (`Nx × 2W`) to `npu`. Later [`evaluate`](Self::evaluate) calls run there.
    pub fn with_npu(mut self, npu: &Npu) -> Result<Self> {
        let xs = self.grid.xs();
        let w = self.waves.len();
        let mut basis = vec![0.0f32; xs.len() * 2 * w];
        for (i, &x) in xs.iter().enumerate() {
            for (n, wave) in self.waves.iter().enumerate() {
                let a = wave.wavenumber() * wave.dir().0;
                let phase = (a * x).rem_euclid(TAU);
                basis[i * 2 * w + n] = phase.cos() as f32;
                basis[i * 2 * w + w + n] = phase.sin() as f32;
            }
        }
        self.npu_basis = Some(ResidentMatrix::upload(npu, &basis, xs.len(), 2 * w)?);
        Ok(self)
    }

    /// The waves.
    pub fn waves(&self) -> &[Wave] {
        &self.waves
    }

    /// The evaluation grid.
    pub fn grid(&self) -> &Grid {
        &self.grid
    }

    /// `true` after [`with_npu`](Self::with_npu).
    pub fn is_accelerated(&self) -> bool {
        self.npu_basis.is_some()
    }

    /// Evaluates the whole grid at time `t` (seconds). Uses the NPU if
    /// attached, otherwise [`evaluate_cpu`](Self::evaluate_cpu).
    pub fn evaluate(&self, t: f64) -> Result<OceanFrame> {
        let Some(basis) = &self.npu_basis else {
            return Ok(self.evaluate_cpu(t));
        };
        let zs = self.grid.zs();
        let (nx, nz) = self.grid.resolution;
        let w = self.waves.len();
        let coeffs: Vec<[f64; FIELDS]> = self.waves.iter().map(coefficients).collect();

        // z-side rows: for field f and row j, [α·cos(bz+c) or α·sin(bz+c) | ±α·(other)].
        let mut rows = vec![0.0f32; FIELDS * nz * 2 * w];
        rows.par_chunks_mut(nz * 2 * w)
            .enumerate()
            .for_each(|(f, field_rows)| {
                for (j, &z) in zs.iter().enumerate() {
                    let row = &mut field_rows[j * 2 * w..(j + 1) * 2 * w];
                    for (n, wave) in self.waves.iter().enumerate() {
                        let (_, dz) = wave.dir();
                        let phase = (wave.wavenumber() * dz * z + wave.phase as f64
                            - wave.angular_frequency() * t)
                            .rem_euclid(TAU);
                        let (s, c) = phase.sin_cos();
                        let alpha = coeffs[n][f];
                        if USES_SIN[f] {
                            // Σ α sin θ = Σ (α sin) cos(ax) + (α cos) sin(ax)
                            row[n] = (alpha * s) as f32;
                            row[w + n] = (alpha * c) as f32;
                        } else {
                            // Σ α cos θ = Σ (α cos) cos(ax) − (α sin) sin(ax)
                            row[n] = (alpha * c) as f32;
                            row[w + n] = (-alpha * s) as f32;
                        }
                    }
                }
            });

        let out = basis.apply(&rows, FIELDS * nz)?; // [FIELDS * nz, nx]
        let field = |f: usize, j: usize, i: usize| out[(f * nz + j) * nx + i];
        Ok(self.assemble(|j, i| std::array::from_fn(|f| field(f, j, i))))
    }

    /// Plain CPU evaluation (f64 per point, parallel). The usual way to do
    /// this on the CPU, and the accuracy reference for the NPU path.
    pub fn evaluate_cpu(&self, t: f64) -> OceanFrame {
        let xs = self.grid.xs();
        let zs = self.grid.zs();
        let coeffs: Vec<[f64; FIELDS]> = self.waves.iter().map(coefficients).collect();
        let (nx, nz) = self.grid.resolution;
        let mut sums = vec![[0.0f32; FIELDS]; nx * nz];
        sums.par_chunks_mut(nx).enumerate().for_each(|(j, row)| {
            for (i, out) in row.iter_mut().enumerate() {
                let mut acc = [0.0f64; FIELDS];
                for (wave, alpha) in self.waves.iter().zip(&coeffs) {
                    let (s, c) = self.phase(wave, xs[i], zs[j], t).sin_cos();
                    for f in 0..FIELDS {
                        acc[f] += alpha[f] * if USES_SIN[f] { s } else { c };
                    }
                }
                *out = acc.map(|v| v as f32);
            }
        });
        self.assemble(|j, i| sums[j * nx + i])
    }

    fn phase(&self, wave: &Wave, x: f64, z: f64, t: f64) -> f64 {
        let (dx, dz) = wave.dir();
        wave.wavenumber() * (dx * x + dz * z) + wave.phase as f64 - wave.angular_frequency() * t
    }

    fn assemble(&self, fields: impl Fn(usize, usize) -> [f32; FIELDS] + Sync) -> OceanFrame {
        let xs = self.grid.xs();
        let zs = self.grid.zs();
        let (nx, nz) = self.grid.resolution;
        let mut positions = vec![Vec3::ZERO; nx * nz];
        let mut normals = vec![Vec3::Y; nx * nz];
        let mut jacobian = vec![1.0f32; nx * nz];
        positions
            .par_chunks_mut(nx)
            .zip(normals.par_chunks_mut(nx))
            .zip(jacobian.par_chunks_mut(nx))
            .enumerate()
            .for_each(|(j, ((p, n), jac))| {
                for i in 0..nx {
                    let f = fields(j, i);
                    p[i] = Vec3::new(xs[i] as f32 + f[0], f[4], zs[j] as f32 + f[1]);
                    n[i] = Vec3::new(-f[2], 1.0 - f[5], -f[3]).normalize_or(Vec3::Y);
                    jac[i] = (1.0 - f[6]) * (1.0 - f[7]) - f[8] * f[8];
                }
            });
        OceanFrame {
            nx,
            nz,
            positions,
            normals,
            jacobian,
        }
    }

    /// Exact surface sample at world position `(x, z)` (f64 on the host).
    ///
    /// Gerstner waves move points horizontally, so this first finds the grid
    /// parameter that lands on `(x, z)` (fixed-point iteration), then reads
    /// height, normal and vertical velocity there. Use it for buoyancy and
    /// other gameplay queries.
    pub fn sample(&self, x: f32, z: f32, t: f64) -> OceanSample {
        let target = (x as f64, z as f64);
        let mut p = target;
        for _ in 0..6 {
            let (mut ox, mut oz) = (0.0, 0.0);
            for w in &self.waves {
                let (dx, dz) = w.dir();
                let c = self.phase(w, p.0, p.1, t).cos();
                ox += w.steepness as f64 * w.amplitude as f64 * dx * c;
                oz += w.steepness as f64 * w.amplitude as f64 * dz * c;
            }
            p = (target.0 - ox, target.1 - oz);
        }
        let (mut h, mut n, mut vy) = (0.0f64, [0.0f64, 1.0, 0.0], 0.0f64);
        for (w, a) in self.waves.iter().zip(self.waves.iter().map(coefficients)) {
            let (s, c) = self.phase(w, p.0, p.1, t).sin_cos();
            h += a[4] * s;
            n[0] -= a[2] * c;
            n[1] -= a[5] * s;
            n[2] -= a[3] * c;
            vy -= w.amplitude as f64 * w.angular_frequency() * c;
        }
        OceanSample {
            height: h as f32,
            normal: Vec3::new(n[0] as f32, n[1] as f32, n[2] as f32).normalize_or(Vec3::Y),
            velocity_y: vy as f32,
        }
    }

    /// Sum of all amplitudes: an upper bound on `|height|`, handy for error scales.
    pub fn max_height(&self) -> f32 {
        self.waves.iter().map(|w| w.amplitude).sum()
    }
}
