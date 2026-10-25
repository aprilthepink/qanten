//! Morph targets (blendshapes) for many characters at once.
//!
//! ![One sphere, four morph targets, six characters in one NPU call (example `blendshapes`)](https://raw.githubusercontent.com/aprilthepink/qanten/main/docs/images/blendshapes.png)
//!
//! `vertex = base + Σ_s weight_s · delta_s`. The delta matrix
//! `[3·vertices, shapes]` is uploaded once and stays resident; each frame
//! only the weights of every character go up (one row of `shapes` values per
//! character).
//! Only the offsets pass through the NPU, the base mesh is added in f32 on
//! the host, so the bf16 error scales with the deformation, not the mesh size.

use glam::Vec3;
use qanten::compute::Matrix;

use crate::accel::{Accelerated, MatrixOp, accelerated_methods};
use crate::{Error, Result};

/// A base mesh plus morph-target offsets.
#[derive(Debug)]
pub struct Blendshapes {
    base: Vec<Vec3>,
    /// `[3·V, S]`: row `3v + c`, column `s` = component `c` of target `s`'s offset at vertex `v`.
    deltas: MatrixOp,
    shapes: usize,
}

impl Accelerated for Blendshapes {
    fn ops(&self) -> Vec<&MatrixOp> {
        vec![&self.deltas]
    }
    fn ops_mut(&mut self) -> Vec<&mut MatrixOp> {
        vec![&mut self.deltas]
    }
}

accelerated_methods!(Blendshapes);

impl Blendshapes {
    /// `deltas[s][v]` is the offset of target `s` at vertex `v`.
    pub fn new(base: Vec<Vec3>, deltas: &[Vec<Vec3>]) -> Result<Self> {
        let v = base.len();
        if deltas.iter().any(|d| d.len() != v) {
            return Err(Error::Shape(format!("every target needs {v} offsets")));
        }
        let matrix = Matrix::from_fn(3 * v, deltas.len(), |r, s| deltas[s][r / 3][r % 3]);
        Ok(Self {
            base,
            deltas: MatrixOp::new(matrix),
            shapes: deltas.len(),
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
        let one = Matrix::new(1, weights.len(), weights.to_vec())?;
        Ok(self.evaluate_batch(&one)?.remove(0))
    }

    /// Many characters at once: one row of weights per character
    /// (`characters × shapes`). Returns one mesh per character.
    pub fn evaluate_batch(&self, weights: &Matrix) -> Result<Vec<Vec<Vec3>>> {
        self.check(weights)?;
        Ok(self.add_base(&self.deltas.apply(weights)?))
    }

    /// CPU reference / fallback (exact).
    pub fn evaluate_batch_cpu(&self, weights: &Matrix) -> Result<Vec<Vec<Vec3>>> {
        self.check(weights)?;
        Ok(self.add_base(&self.deltas.apply_cpu(weights)?))
    }

    fn check(&self, weights: &Matrix) -> Result<()> {
        if weights.cols() != self.shapes || weights.rows() == 0 {
            return Err(Error::Shape(format!(
                "weights must be rows of {} values (one per shape), got {}x{}",
                self.shapes,
                weights.rows(),
                weights.cols()
            )));
        }
        Ok(())
    }

    /// Base mesh plus each character's offsets (`characters × 3V`).
    fn add_base(&self, offsets: &Matrix) -> Vec<Vec<Vec3>> {
        offsets
            .iter_rows()
            .map(|row| {
                self.base
                    .iter()
                    .zip(row.chunks_exact(3))
                    .map(|(b, o)| *b + Vec3::from_slice(o))
                    .collect()
            })
            .collect()
    }
}
