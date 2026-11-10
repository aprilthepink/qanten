//! Procedural terrain: heightfields from a sum of plane waves with a
//! power-law spectrum (fBm-like), evaluated in tiles on the NPU, plus exact
//! point queries for gameplay.
//!
//! ![A 3×3 block of terrain tiles generated in one NPU call (example `terrain`)](https://raw.githubusercontent.com/aprilthepink/qanten/main/docs/images/terrain.png)
//!
//! # How it maps to the NPU
//!
//! The height is `h(x, z) = Σ_w A_w sin(θ_w)` with `θ_w = a_w x + b_w z + φ_w`.
//! Inside a tile with origin `(x₀, z₀)` write `x = x₀ + u`; the angle-sum
//! formulas then split every sum into an x-side table that only depends on the
//! local coordinate `u` and a z-side table that holds everything else:
//!
//! ```text
//! Σ A sin θ = Σ (A cos ψ) sin(a u) + (A sin ψ) cos(a u),   ψ = a x₀ + b z + φ
//! ```
//!
//! The x-side matrix `[cos(a u) | sin(a u)]` is the same for every tile of the
//! same size, so it is uploaded once and stays resident; each tile only sends
//! its z-side rows (height and both slopes: `3·Nz` rows of `2W` values).
//! Many tiles go through in a single call, which suits background streaming.
//! Phases are reduced modulo 2π on the host in f64, so the NPU only sees
//! bounded values, however far the tile is from the origin.
//!
//! The terrain is static, so tiles can be generated ahead of the camera in
//! batches. [`Terrain::height_at`] evaluates the same sum exactly in f64 for
//! physics and placement.

use std::f64::consts::TAU;

use glam::{Vec2, Vec3};
use rayon::prelude::*;

use qanten::compute::Matrix;
use qanten::rng::SplitMix64;

use crate::accel::{Accelerated, MatrixOp, accelerated_methods};
use crate::{Error, Result};

/// One plane wave of the height spectrum.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlaneWave {
    /// Direction of travel (normalised on use).
    pub direction: Vec2,
    /// Crest-to-crest distance in metres.
    pub wavelength: f32,
    /// Peak height in metres.
    pub amplitude: f32,
    /// Phase offset in radians.
    pub phase: f32,
}

impl PlaneWave {
    /// `(a, b)`: the wave vector `2π/λ · direction`, in rad/m.
    fn wave_vector(&self) -> (f64, f64) {
        let d = self.direction.normalize_or(Vec2::X);
        let k = TAU / self.wavelength as f64;
        (k * d.x as f64, k * d.y as f64)
    }
}

/// An fBm-like spectrum: `count` waves with wavelengths spread geometrically
/// from `longest` to `shortest`, random directions and phases, and
/// amplitudes falling as `λ^roughness` (`roughness` ≈ 1 gives classic
/// fractal hills, larger is smoother). The tallest wave is `height` high.
pub fn fbm_spectrum(
    count: usize,
    longest: f32,
    shortest: f32,
    height: f32,
    roughness: f32,
    seed: u64,
) -> Vec<PlaneWave> {
    let mut rng = SplitMix64::new(seed);
    let n = count.max(1);
    let ratio = (shortest / longest).max(1e-6);
    (0..n)
        .map(|i| {
            let t = i as f32 / (n.max(2) - 1) as f32;
            let wavelength = longest * ratio.powf(t) * (0.85 + 0.3 * rng.next_f32());
            let angle = rng.next_f32() * std::f32::consts::TAU;
            PlaneWave {
                direction: Vec2::new(angle.cos(), angle.sin()),
                wavelength,
                amplitude: height * (wavelength / longest).powf(roughness),
                phase: rng.next_f32() * std::f32::consts::TAU,
            }
        })
        .collect()
}

/// One generated tile. Arrays are row-major over `(z, x)`: index `j * nx + i`.
#[derive(Debug, Clone, PartialEq)]
pub struct TerrainTile {
    /// World position of sample `(0, 0)`.
    pub origin: Vec2,
    /// Samples along x.
    pub nx: usize,
    /// Samples along z.
    pub nz: usize,
    /// Distance between neighbouring samples, in metres (x, z).
    pub spacing: Vec2,
    /// Heights in metres.
    pub heights: Vec<f32>,
    /// Unit surface normals.
    pub normals: Vec<Vec3>,
}

impl TerrainTile {
    /// Array index of sample `(i, j)` (x index, z index).
    pub fn index(&self, i: usize, j: usize) -> usize {
        j * self.nx + i
    }

    /// World position of sample `(i, j)`.
    pub fn position(&self, i: usize, j: usize) -> Vec3 {
        Vec3::new(
            self.origin.x + i as f32 * self.spacing.x,
            self.heights[self.index(i, j)],
            self.origin.y + j as f32 * self.spacing.y,
        )
    }

    /// Slope angle at every sample, in radians (0 = flat).
    pub fn slopes(&self) -> Vec<f32> {
        self.normals
            .iter()
            .map(|n| n.y.clamp(-1.0, 1.0).acos())
            .collect()
    }

    /// `true` where the slope is steeper than `max_angle` radians: cliffs,
    /// rock texture, no trees.
    pub fn slope_mask(&self, max_angle: f32) -> Vec<bool> {
        let cos = max_angle.cos();
        self.normals.iter().map(|n| n.y < cos).collect()
    }
}

/// The three sums a tile needs: height, ∂h/∂x, ∂h/∂z.
const FIELDS: usize = 3;

/// A heightfield made of plane waves, evaluated in tiles of a fixed size.
#[derive(Debug)]
pub struct Terrain {
    waves: Vec<PlaneWave>,
    size: Vec2,
    resolution: (usize, usize),
    /// x-side table `[cos(a·u) | sin(a·u)]` (`Nx × 2W`) over local coordinates.
    basis: MatrixOp,
}

impl Accelerated for Terrain {
    fn ops(&self) -> Vec<&MatrixOp> {
        vec![&self.basis]
    }
    fn ops_mut(&mut self) -> Vec<&mut MatrixOp> {
        vec![&mut self.basis]
    }
}

accelerated_methods!(Terrain);

impl Terrain {
    /// Tiles of `size` metres with `nx × nz` samples each (edges included,
    /// so neighbouring tiles share their border samples).
    pub fn new(waves: Vec<PlaneWave>, size: Vec2, nx: usize, nz: usize) -> Result<Self> {
        if waves.is_empty() {
            return Err(Error::InvalidArgument(
                "terrain needs at least one wave".into(),
            ));
        }
        if waves
            .iter()
            .any(|w| !w.wavelength.is_finite() || w.wavelength <= 0.0 || !w.amplitude.is_finite())
        {
            return Err(Error::InvalidArgument(
                "wavelengths must be > 0 and amplitudes finite".into(),
            ));
        }
        if nx < 2 || nz < 2 || !(size.x > 0.0 && size.y > 0.0) {
            return Err(Error::InvalidArgument(
                "tiles need at least 2×2 samples and a positive size".into(),
            ));
        }
        let mut t = Self {
            waves,
            size,
            resolution: (nx, nz),
            basis: MatrixOp::new(Matrix::zeros(0, 0)),
        };
        let us = t.local_x();
        let w = t.waves.len();
        let vectors: Vec<(f64, f64)> = t.waves.iter().map(PlaneWave::wave_vector).collect();
        t.basis = MatrixOp::new(Matrix::from_fn(nx, 2 * w, |i, col| {
            let phase = (vectors[col % w].0 * us[i]).rem_euclid(TAU);
            (if col < w { phase.cos() } else { phase.sin() }) as f32
        }));
        Ok(t)
    }

    /// The waves.
    pub fn waves(&self) -> &[PlaneWave] {
        &self.waves
    }

    /// Tile size in metres.
    pub fn tile_size(&self) -> Vec2 {
        self.size
    }

    /// Samples per tile `(nx, nz)`.
    pub fn resolution(&self) -> (usize, usize) {
        self.resolution
    }

    /// Sum of all amplitudes: an upper bound on `|height|`.
    pub fn max_height(&self) -> f32 {
        self.waves.iter().map(|w| w.amplitude.abs()).sum()
    }

    fn spacing(&self) -> Vec2 {
        let (nx, nz) = self.resolution;
        Vec2::new(self.size.x / (nx - 1) as f32, self.size.y / (nz - 1) as f32)
    }

    fn local_x(&self) -> Vec<f64> {
        let dx = self.spacing().x as f64;
        (0..self.resolution.0).map(|i| i as f64 * dx).collect()
    }

    fn local_z(&self) -> Vec<f64> {
        let dz = self.spacing().y as f64;
        (0..self.resolution.1).map(|j| j as f64 * dz).collect()
    }

    /// The tile whose sample `(0, 0)` is at `origin`.
    pub fn tile(&self, origin: Vec2) -> Result<TerrainTile> {
        Ok(self.tiles(&[origin])?.remove(0))
    }

    /// Several tiles in one NPU call (or on the CPU without an NPU).
    pub fn tiles(&self, origins: &[Vec2]) -> Result<Vec<TerrainTile>> {
        if !self.basis.is_accelerated() {
            return Ok(self.tiles_cpu(origins));
        }
        let nz = self.resolution.1;
        let w = self.waves.len();
        let zs = self.local_z();
        let vectors: Vec<(f64, f64)> = self.waves.iter().map(PlaneWave::wave_vector).collect();
        // Rows (tile, field, z): the z-side tables with the tile origin folded in.
        let rows_per_tile = FIELDS * nz;
        let rows = Matrix::from_fn(origins.len() * rows_per_tile, 2 * w, |r, col| {
            let (tile, f, j) = (r / rows_per_tile, (r / nz) % FIELDS, r % nz);
            let n = col % w;
            let (wave, (a, b)) = (&self.waves[n], vectors[n]);
            let o = origins[tile];
            let psi =
                (a * o.x as f64 + b * (o.y as f64 + zs[j]) + wave.phase as f64).rem_euclid(TAU);
            let (s, c) = psi.sin_cos();
            let amp = wave.amplitude as f64;
            // Σ A sin(au + ψ) = Σ (A sin ψ) cos(au) + (A cos ψ) sin(au)
            // Σ A k cos(au + ψ) = Σ (A k cos ψ) cos(au) − (A k sin ψ) sin(au)
            let v = match (f, col < w) {
                (0, true) => amp * s,
                (0, false) => amp * c,
                (1, true) => amp * a * c,
                (1, false) => -amp * a * s,
                (_, true) => amp * b * c,
                (_, false) => -amp * b * s,
            };
            v as f32
        });
        let out = self.basis.apply(&rows)?; // [tiles · FIELDS · nz, nx]
        Ok(origins
            .iter()
            .enumerate()
            .map(|(tile, &origin)| {
                let at = |f: usize, j: usize, i: usize| out[(tile * rows_per_tile + f * nz + j, i)];
                self.assemble(origin, |j, i| [at(0, j, i), at(1, j, i), at(2, j, i)])
            })
            .collect())
    }

    /// Plain CPU evaluation (f64 per sample, parallel): the accuracy
    /// reference for [`tiles`](Self::tiles).
    pub fn tiles_cpu(&self, origins: &[Vec2]) -> Vec<TerrainTile> {
        origins
            .iter()
            .map(|&origin| {
                let (nx, nz) = self.resolution;
                let (xs, zs) = (self.local_x(), self.local_z());
                let mut fields = vec![[0.0f32; FIELDS]; nx * nz];
                fields.par_chunks_mut(nx).enumerate().for_each(|(j, row)| {
                    for (i, out) in row.iter_mut().enumerate() {
                        let x = origin.x as f64 + xs[i];
                        let z = origin.y as f64 + zs[j];
                        *out = self.fields_at(x, z).map(|v| v as f32);
                    }
                });
                self.assemble(origin, |j, i| fields[j * nx + i])
            })
            .collect()
    }

    /// Height, ∂h/∂x and ∂h/∂z at a world position, exactly in f64.
    fn fields_at(&self, x: f64, z: f64) -> [f64; FIELDS] {
        let mut acc = [0.0; FIELDS];
        for w in &self.waves {
            let (a, b) = w.wave_vector();
            let (s, c) = (a * x + b * z + w.phase as f64).rem_euclid(TAU).sin_cos();
            let amp = w.amplitude as f64;
            acc[0] += amp * s;
            acc[1] += amp * a * c;
            acc[2] += amp * b * c;
        }
        acc
    }

    /// Exact height at world position `(x, z)`, for physics and placement.
    pub fn height_at(&self, x: f32, z: f32) -> f32 {
        self.fields_at(x as f64, z as f64)[0] as f32
    }

    /// Exact unit normal at world position `(x, z)`.
    pub fn normal_at(&self, x: f32, z: f32) -> Vec3 {
        let f = self.fields_at(x as f64, z as f64);
        Vec3::new(-f[1] as f32, 1.0, -f[2] as f32).normalize()
    }

    fn assemble(
        &self,
        origin: Vec2,
        fields: impl Fn(usize, usize) -> [f32; FIELDS],
    ) -> TerrainTile {
        let (nx, nz) = self.resolution;
        let mut heights = Vec::with_capacity(nx * nz);
        let mut normals = Vec::with_capacity(nx * nz);
        for j in 0..nz {
            for i in 0..nx {
                let [h, dx, dz] = fields(j, i);
                heights.push(h);
                normals.push(Vec3::new(-dx, 1.0, -dz).normalize());
            }
        }
        TerrainTile {
            origin,
            nx,
            nz,
            spacing: self.spacing(),
            heights,
            normals,
        }
    }
}
