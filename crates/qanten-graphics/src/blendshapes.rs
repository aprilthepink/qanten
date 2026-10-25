//! Morph targets (blendshapes) for many characters at once.
//!
//! ![One sphere, four morph targets, six characters in one NPU call (example `blendshapes`)](https://raw.githubusercontent.com/aprilthepink/qanten/main/docs/images/blendshapes.png)
//!
//! `vertex = base + Σ_s weight_s · delta_s`. The delta matrix
//! `[3·vertices, shapes]` is uploaded once and stays resident; each frame
//! only the weights of every character go up (`characters × shapes` values).
//! Only the offsets pass through the NPU, the base mesh is added in f32 on
//! the host, so the bf16 error scales with the deformation, not the mesh size.

use glam::Vec3;
use rayon::prelude::*;

use crate::accel::{ResidentMatrix, Target};
use qanten::pool::NpuPool;

use crate::{Error, Npu, Result};

/// A base mesh plus morph-target offsets.
#[derive(Debug)]
pub struct Blendshapes {
    base: Vec<Vec3>,
    /// Row-major `[3·V, S]`: row `3v + c`, column `s` = component `c` of target `s`'s offset at vertex `v`.
    deltas: Vec<f32>,
    shapes: usize,
    npu: Option<ResidentMatrix>,
}

impl Blendshapes {
    /// `deltas[s][v]` is the offset of target `s` at vertex `v`.
    pub fn new(base: Vec<Vec3>, deltas: &[Vec<Vec3>]) -> Result<Self> {
        let v = base.len();
        if deltas.iter().any(|d| d.len() != v) {
            return Err(Error::Shape(format!("every target needs {v} offsets")));
        }
        let shapes = deltas.len();
        let mut m = vec![0.0f32; 3 * v * shapes];
        for (s, target) in deltas.iter().enumerate() {
            for (vi, d) in target.iter().enumerate() {
                for c in 0..3 {
                    m[(3 * vi + c) * shapes + s] = d[c];
                }
            }
        }
        Ok(Self {
            base,
            deltas: m,
            shapes,
            npu: None,
        })
    }

    /// Builds from absolute target positions (offsets are computed for you).
    pub fn from_target_positions(base: Vec<Vec3>, targets: &[Vec<Vec3>]) -> Result<Self> {
        let deltas: Vec<Vec<Vec3>> = targets
            .iter()
            .map(|t| t.iter().zip(&base).map(|(p, b)| *p - *b).collect())
            .collect();
        Self::new(base, &deltas)
    }

    /// Uploads the delta matrix to `npu`.
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
            &self.deltas,
            3 * self.base.len(),
            self.shapes,
        )?);
        Ok(self)
    }

    /// Vertices in the base mesh.
    pub fn vertex_count(&self) -> usize {
        self.base.len()
    }

    /// Number of morph targets.
    pub fn shape_count(&self) -> usize {
        self.shapes
    }

    /// One character: `weights` has one value per shape.
    pub fn evaluate(&self, weights: &[f32]) -> Result<Vec<Vec3>> {
        self.evaluate_batch(weights)
    }

    /// Many characters: `weights` is `characters × shapes`, row-major.
    /// Returns `characters × vertices` positions.
    pub fn evaluate_batch(&self, weights: &[f32]) -> Result<Vec<Vec3>> {
        let chars = self.check(weights)?;
        let Some(m) = &self.npu else {
            return Ok(self.evaluate_batch_cpu(weights));
        };
        let offsets = m.apply(weights, chars)?; // [chars, 3V]
        Ok(self.add_base(&offsets, chars))
    }

    /// CPU reference / fallback.
    pub fn evaluate_batch_cpu(&self, weights: &[f32]) -> Vec<Vec3> {
        let chars = weights.len() / self.shapes.max(1);
        let s = self.shapes;
        let mut offsets = vec![0.0f32; chars * 3 * self.base.len()];
        offsets
            .par_chunks_mut(3 * self.base.len().max(1))
            .enumerate()
            .for_each(|(c, out)| {
                let w = &weights[c * s..(c + 1) * s];
                for (r, o) in out.iter_mut().enumerate() {
                    let row = &self.deltas[r * s..(r + 1) * s];
                    *o = row.iter().zip(w).map(|(a, b)| a * b).sum();
                }
            });
        self.add_base(&offsets, chars)
    }

    fn check(&self, weights: &[f32]) -> Result<usize> {
        if self.shapes == 0 || !weights.len().is_multiple_of(self.shapes) {
            return Err(Error::Shape(format!(
                "{} weights is not a multiple of {} shapes",
                weights.len(),
                self.shapes
            )));
        }
        Ok(weights.len() / self.shapes)
    }

    fn add_base(&self, offsets: &[f32], chars: usize) -> Vec<Vec3> {
        let v = self.base.len();
        (0..chars * v)
            .into_par_iter()
            .map(|k| {
                let (c, vi) = (k / v, k % v);
                let o = &offsets[c * 3 * v + 3 * vi..][..3];
                self.base[vi] + Vec3::new(o[0], o[1], o[2])
            })
            .collect()
    }
}
