//! Order-2 spherical harmonics (9 coefficients) for light-probe baking and
//! diffuse irradiance.
//!
//! ![Spheres lit by SH9 probes baked on the NPU (example `probes`)](https://raw.githubusercontent.com/aprilthepink/qanten/main/docs/images/probes.png)
//!
//! Baking projects many radiance samples per probe onto the 9 basis
//! functions: a `[9, samples]` matrix (basis × solid angle) resident on the
//! NPU times one row of samples per probe and colour channel. Thousands of
//! probes go through in one call. Evaluating irradiance for many normals
//! (per-vertex or per-texel lighting) is the same shape the other way round.

use std::f32::consts::PI;

use glam::Vec3;
use rayon::prelude::*;

use crate::accel::{ResidentMatrix, Target};
use qanten::compute::{self, Matrix, Placement};
use qanten::pool::NpuPool;

use crate::{Error, Npu, Result};

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

/// Projects radiance samples onto SH. Holds the sample directions.
#[derive(Debug)]
pub struct ShProjector {
    dirs: Vec<Vec3>,
    /// `[9, N]`: `Y_i(d_n) · dω_n`.
    weighted: Vec<f32>,
    npu: Option<ResidentMatrix>,
}

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
            .collect::<Vec<_>>();
        let w = 4.0 * PI / n.max(1) as f32;
        Self::from_directions(dirs, &vec![w; n]).unwrap_or_else(|_| Self::empty())
    }

    fn empty() -> Self {
        Self {
            dirs: Vec::new(),
            weighted: Vec::new(),
            npu: None,
        }
    }

    /// Custom directions with their solid angles (e.g. cube-map texels).
    pub fn from_directions(dirs: Vec<Vec3>, solid_angles: &[f32]) -> Result<Self> {
        if dirs.len() != solid_angles.len() {
            return Err(Error::Shape(
                "one solid angle per direction required".into(),
            ));
        }
        let n = dirs.len();
        let mut weighted = vec![0.0f32; 9 * n];
        for (k, (d, w)) in dirs.iter().zip(solid_angles).enumerate() {
            for (i, y) in basis(d.normalize_or(Vec3::Y)).iter().enumerate() {
                weighted[i * n + k] = y * w;
            }
        }
        Ok(Self {
            dirs,
            weighted,
            npu: None,
        })
    }

    /// Uploads the weighted basis to `npu`; later projections run there.
    pub fn with_npu(self, npu: &Npu) -> Result<Self> {
        self.upload_to(Target::Npu(npu))
    }

    /// Like [`with_npu`](Self::with_npu), but with a full copy on every NPU of
    /// `pool`; each batch is split between them, so they all work at once.
    pub fn with_pool(self, pool: &NpuPool) -> Result<Self> {
        self.upload_to(Target::Pool(pool))
    }

    /// NPU self-test: relative error of the resident matrices on random
    /// vectors, checked on the CPU with
    /// [`qanten::compute::check_product`]. `None` on the CPU path. Run it now
    /// and then on hardware: a rising value means the NPU needs recalibration.
    pub fn self_check(&self) -> Result<Option<f64>> {
        self.npu.as_ref().map(|m| m.self_check(1)).transpose()
    }

    fn upload_to(mut self, target: Target<'_>) -> Result<Self> {
        self.npu = Some(ResidentMatrix::upload(
            target,
            &self.weighted,
            9,
            self.dirs.len(),
        )?);
        Ok(self)
    }

    /// The sample directions; evaluate your environment at these.
    pub fn directions(&self) -> &[Vec3] {
        &self.dirs
    }

    /// `radiance` is `probes × samples`, row-major, one RGB value per direction.
    pub fn project_batch(&self, radiance: &[Vec3]) -> Result<Vec<Sh9>> {
        let n = self.dirs.len();
        if n == 0 || !radiance.len().is_multiple_of(n) {
            return Err(Error::Shape(format!(
                "{} samples is not a multiple of {n} directions",
                radiance.len()
            )));
        }
        let probes = radiance.len() / n;
        let Some(m) = &self.npu else {
            return Ok(self.project_batch_cpu(radiance));
        };
        // Rows (probe, channel) of N samples.
        let mut rows = vec![0.0f32; probes * 3 * n];
        rows.par_chunks_mut(n).enumerate().for_each(|(r, row)| {
            let (p, ch) = (r / 3, r % 3);
            for (k, v) in row.iter_mut().enumerate() {
                *v = radiance[p * n + k][ch];
            }
        });
        let out = m.apply(&rows, probes * 3)?; // [probes*3, 9]
        Ok((0..probes)
            .map(|p| Sh9 {
                coeffs: std::array::from_fn(|i| {
                    Vec3::new(
                        out[(p * 3) * 9 + i],
                        out[(p * 3 + 1) * 9 + i],
                        out[(p * 3 + 2) * 9 + i],
                    )
                }),
            })
            .collect())
    }

    /// Projects one environment given as a function of direction.
    pub fn project_fn(&self, env: impl Fn(Vec3) -> Vec3) -> Result<Sh9> {
        let samples: Vec<Vec3> = self.dirs.iter().map(|&d| env(d)).collect();
        self.project_batch(&samples)?
            .into_iter()
            .next()
            .ok_or_else(|| Error::Shape("projector has no directions".into()))
    }

    /// CPU reference / fallback.
    pub fn project_batch_cpu(&self, radiance: &[Vec3]) -> Vec<Sh9> {
        let n = self.dirs.len().max(1);
        radiance
            .par_chunks(n)
            .map(|samples| Sh9 {
                coeffs: std::array::from_fn(|i| {
                    let w = &self.weighted[i * n..(i + 1) * n];
                    samples.iter().zip(w).map(|(s, w)| *s * *w).sum()
                }),
            })
            .collect()
    }
}

/// Irradiance of every probe at every normal: returns `probes × normals`.
///
/// With an NPU this is one product of the `[3·probes, 9]` coefficients with
/// the `[9, normals]` convolved basis, e.g. for baking vertex lighting. It
/// goes through [`compute::gemm`] with `Placement::Auto`, so small bakes run
/// exactly on the CPU and only large ones use the NPU.
pub fn irradiance_batch(npu: Option<&Npu>, probes: &[Sh9], normals: &[Vec3]) -> Result<Vec<Vec3>> {
    let nn = normals.len();
    let Some(npu) = npu else {
        return Ok(probes
            .iter()
            .flat_map(|p| normals.iter().map(|&n| p.irradiance(n)))
            .collect());
    };
    let mut basis_m = vec![0.0f32; nn * 9];
    for (k, n) in normals.iter().enumerate() {
        for (i, y) in basis(n.normalize_or(Vec3::Y)).iter().enumerate() {
            basis_m[k * 9 + i] = y * BAND[i];
        }
    }
    let rows: Vec<f32> = probes
        .iter()
        .flat_map(|p| (0..3).flat_map(move |ch| (0..9).map(move |i| p.coeffs[i][ch])))
        .collect();
    // [3P, 9] · [9, Nn]. Used once, so `Placement::Auto` decides: the NPU for
    // big bakes, the (exact) CPU for a handful of probes or normals.
    let a = Matrix::new(probes.len() * 3, 9, rows)?;
    let b = Matrix::new(nn, 9, basis_m)?.transpose();
    let out = compute::gemm(npu, &a, &b, Placement::Auto)?.into_vec(); // [3P, Nn]
    Ok((0..probes.len() * nn)
        .map(|k| {
            let (p, n) = (k / nn.max(1), k % nn.max(1));
            Vec3::new(
                out[(p * 3) * nn + n],
                out[(p * 3 + 1) * nn + n],
                out[(p * 3 + 2) * nn + n],
            )
        })
        .collect())
}
