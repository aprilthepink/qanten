//! Order-2 spherical harmonics (9 coefficients) for light-probe baking and
//! diffuse irradiance.
//!
//! ![Spheres lit by SH9 probes baked on the NPU (example `probes`)](https://raw.githubusercontent.com/aprilthepink/qanten/main/docs/images/probes.png)
//!
//! Baking projects many radiance samples per probe onto the 9 basis
//! functions: a `[9, samples]` matrix (basis × solid angle) resident on the
//! NPU times one row of samples per probe and colour channel. Thousands of
//! probes go through in one call. Evaluating irradiance for many normals
//! (per-vertex or per-texel lighting, [`IrradianceBaker`]) is the same shape
//! the other way round.

use std::f32::consts::PI;

use glam::Vec3;
use qanten::compute::Matrix;
use rayon::prelude::*;

use crate::accel::{Accelerated, MatrixOp, accelerated_methods};
use crate::{Error, Result};

/// Real SH basis `Y_lm(d)` for `l ≤ 2`, `d` a unit vector (y up).
pub fn basis(d: Vec3) -> [f32; 9] {
    let (x, y, z) = (d.x, d.y, d.z);
    [
        0.282_095,
        0.488_603 * y,
        0.488_603 * z,
        0.488_603 * x,
        1.092_548 * x * y,
        1.092_548 * y * z,
        0.315_392 * (3.0 * z * z - 1.0),
        1.092_548 * x * z,
        0.546_274 * (x * x - y * y),
    ]
}

/// Cosine-lobe convolution factors per band (Ramamoorthi & Hanrahan 2001).
const BAND: [f32; 9] = [
    PI,
    2.0 * PI / 3.0,
    2.0 * PI / 3.0,
    2.0 * PI / 3.0,
    PI / 4.0,
    PI / 4.0,
    PI / 4.0,
    PI / 4.0,
    PI / 4.0,
];

/// RGB spherical-harmonic coefficients of one probe.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Sh9 {
    /// RGB coefficient per basis function, in the order of [`basis`].
    pub coeffs: [Vec3; 9],
}

impl Sh9 {
    /// Reconstructed radiance arriving from direction `d`.
    pub fn radiance(&self, d: Vec3) -> Vec3 {
        basis(d)
            .iter()
            .zip(&self.coeffs)
            .map(|(y, c)| *c * *y)
            .sum()
    }

    /// Irradiance on a surface with unit normal `n`. Multiply by `albedo / π`
    /// for Lambertian outgoing radiance.
    pub fn irradiance(&self, n: Vec3) -> Vec3 {
        basis(n)
            .iter()
            .zip(&self.coeffs)
            .zip(BAND)
            .map(|((y, c), a)| *c * (*y * a))
            .sum()
    }
}

impl Sh9 {
    /// The coefficients as three rows of 9 (R, G, B).
    fn channel_rows(&self) -> [[f32; 9]; 3] {
        std::array::from_fn(|ch| std::array::from_fn(|i| self.coeffs[i][ch]))
    }

    /// From three rows of 9 coefficients (R, G, B).
    fn from_channel_rows(r: &[f32], g: &[f32], b: &[f32]) -> Self {
        Self {
            coeffs: std::array::from_fn(|i| Vec3::new(r[i], g[i], b[i])),
        }
    }
}

/// Rows (item, channel) of a batch of RGB samples: `3·items × len`.
fn channel_matrix<S: AsRef<[Vec3]> + Sync>(items: &[S], len: usize) -> Result<Matrix> {
    // Filled in parallel: for big bakes this buffer is tens of MB.
    let mut data = vec![0.0f32; 3 * items.len() * len];
    data.par_chunks_mut(len.max(1))
        .enumerate()
        .for_each(|(r, row)| {
            for (v, s) in row.iter_mut().zip(items[r / 3].as_ref()) {
                *v = s[r % 3];
            }
        });
    Matrix::new(3 * items.len(), len, data)
}

/// Projects radiance samples onto SH. Holds the sample directions.
#[derive(Debug)]
pub struct ShProjector {
    dirs: Vec<Vec3>,
    /// `[9, N]`: `Y_i(d_n) · dω_n`.
    weighted: MatrixOp,
}

impl Accelerated for ShProjector {
    fn ops(&self) -> Vec<&MatrixOp> {
        vec![&self.weighted]
    }
    fn ops_mut(&mut self) -> Vec<&mut MatrixOp> {
        vec![&mut self.weighted]
    }
}

accelerated_methods!(ShProjector);

impl ShProjector {
    /// `n` directions spread evenly over the sphere (Fibonacci lattice).
    pub fn fibonacci(n: usize) -> Self {
        let golden = PI * (3.0 - 5.0f32.sqrt());
        let dirs = (0..n)
            .map(|i| {
                let y = 1.0 - 2.0 * (i as f32 + 0.5) / n as f32;
                let r = (1.0 - y * y).max(0.0).sqrt();
                let phi = golden * i as f32;
                Vec3::new(r * phi.cos(), y, r * phi.sin())
            })
            .collect();
        let w = 4.0 * PI / n.max(1) as f32;
        Self::build(dirs, |_| w)
    }

    /// Custom directions with their solid angles (e.g. cube-map texels).
    pub fn from_directions(dirs: Vec<Vec3>, solid_angles: &[f32]) -> Result<Self> {
        if dirs.len() != solid_angles.len() {
            return Err(Error::Shape(
                "one solid angle per direction required".into(),
            ));
        }
        Ok(Self::build(dirs, |k| solid_angles[k]))
    }

    fn build(dirs: Vec<Vec3>, solid_angle: impl Fn(usize) -> f32) -> Self {
        let bases: Vec<[f32; 9]> = dirs
            .iter()
            .map(|d| basis(d.normalize_or(Vec3::Y)))
            .collect();
        let weighted = Matrix::from_fn(9, dirs.len(), |i, k| bases[k][i] * solid_angle(k));
        Self {
            dirs,
            weighted: MatrixOp::new(weighted),
        }
    }

    /// The sample directions; evaluate your environment at these.
    pub fn directions(&self) -> &[Vec3] {
        &self.dirs
    }

    /// Projects many probes at once: `probes[p][k]` is the radiance probe `p`
    /// receives from direction `k` (one RGB value per [`directions`](Self::directions)).
    pub fn project_batch<S: AsRef<[Vec3]> + Sync>(&self, probes: &[S]) -> Result<Vec<Sh9>> {
        self.project_with(probes, |rows| self.weighted.apply(rows))
    }

    /// CPU reference / fallback (exact).
    pub fn project_batch_cpu<S: AsRef<[Vec3]> + Sync>(&self, probes: &[S]) -> Result<Vec<Sh9>> {
        self.project_with(probes, |rows| self.weighted.apply_cpu(rows))
    }

    fn project_with<S: AsRef<[Vec3]> + Sync>(
        &self,
        probes: &[S],
        apply: impl Fn(&Matrix) -> Result<Matrix>,
    ) -> Result<Vec<Sh9>> {
        let n = self.dirs.len();
        if n == 0 || probes.iter().any(|p| p.as_ref().len() != n) {
            return Err(Error::Shape(format!(
                "every probe needs one sample per direction ({n})"
            )));
        }
        let out = apply(&channel_matrix(probes, n)?)?; // [3·probes, 9]
        Ok(out
            .as_slice()
            .chunks_exact(27)
            .map(|c| Sh9::from_channel_rows(&c[..9], &c[9..18], &c[18..]))
            .collect())
    }

    /// Projects one environment given as a function of direction.
    pub fn project_fn(&self, env: impl Fn(Vec3) -> Vec3) -> Result<Sh9> {
        let samples: Vec<Vec3> = self.dirs.iter().map(|&d| env(d)).collect();
        Ok(self.project_batch(&[samples])?.remove(0))
    }
}

/// Evaluates the irradiance of probes at a fixed set of surface normals,
/// e.g. for baking per-vertex or per-texel lighting.
///
/// The `[normals, 9]` cosine-convolved basis is the resident matrix; each
/// [`bake`](Self::bake) streams the probes' coefficients through it, so the
/// same normals can be lit by any number of probe sets.
#[derive(Debug)]
pub struct IrradianceBaker {
    normals: usize,
    basis: MatrixOp,
}

impl Accelerated for IrradianceBaker {
    fn ops(&self) -> Vec<&MatrixOp> {
        vec![&self.basis]
    }
    fn ops_mut(&mut self) -> Vec<&mut MatrixOp> {
        vec![&mut self.basis]
    }
}

accelerated_methods!(IrradianceBaker);

impl IrradianceBaker {
    /// A baker for these surface normals (normalised for you).
    pub fn new(normals: &[Vec3]) -> Self {
        let bases: Vec<[f32; 9]> = normals
            .iter()
            .map(|n| basis(n.normalize_or(Vec3::Y)))
            .collect();
        Self {
            normals: normals.len(),
            basis: MatrixOp::new(Matrix::from_fn(normals.len(), 9, |k, i| {
                bases[k][i] * BAND[i]
            })),
        }
    }

    /// Irradiance of every probe at every normal: `result[p][k]` for probe
    /// `p` and normal `k`.
    pub fn bake(&self, probes: &[Sh9]) -> Result<Vec<Vec<Vec3>>> {
        self.bake_with(probes, |rows| self.basis.apply(rows))
    }

    /// CPU reference / fallback (exact).
    pub fn bake_cpu(&self, probes: &[Sh9]) -> Result<Vec<Vec<Vec3>>> {
        self.bake_with(probes, |rows| self.basis.apply_cpu(rows))
    }

    fn bake_with(
        &self,
        probes: &[Sh9],
        apply: impl Fn(&Matrix) -> Result<Matrix>,
    ) -> Result<Vec<Vec<Vec3>>> {
        if probes.is_empty() || self.normals == 0 {
            return Ok(vec![Vec::new(); probes.len()]);
        }
        let rows: Vec<[f32; 9]> = probes.iter().flat_map(|p| p.channel_rows()).collect();
        let out = apply(&Matrix::from_rows(&rows)?)?; // [3·probes, normals]
        Ok(out
            .as_slice()
            .chunks_exact(3 * self.normals)
            .map(|p| {
                let (r, rest) = p.split_at(self.normals);
                let (g, b) = rest.split_at(self.normals);
                r.iter()
                    .zip(g)
                    .zip(b)
                    .map(|((r, g), b)| Vec3::new(*r, *g, *b))
                    .collect()
            })
            .collect())
    }
}
