//! Small glue between f32 host data and resident NPU matrices.

use qanten::ops::LinearWeights;
use qanten::{Npu, Result, Tensor};

/// A dense `rows × cols` f32 matrix kept resident on an NPU.
///
/// `apply` computes `out[b, r] = Σ_c matrix[r, c] · vectors[b, c]`, i.e.
/// `vectors × matrixᵀ`, which is the one primitive every module here reduces to.
#[derive(Debug)]
pub(crate) struct ResidentMatrix {
    weights: LinearWeights,
}

impl ResidentMatrix {
    pub(crate) fn upload(npu: &Npu, data: &[f32], rows: usize, cols: usize) -> Result<Self> {
        Ok(Self {
            weights: LinearWeights::upload(npu, &Tensor::from_f32(data, [rows, cols])?)?,
        })
    }

    pub(crate) fn cols(&self) -> usize {
        self.weights.in_features()
    }

    /// `vectors` holds `batch` rows of `cols` values; returns `batch` rows of `rows` values.
    pub(crate) fn apply(&self, vectors: &[f32], batch: usize) -> Result<Vec<f32>> {
        let x = Tensor::from_f32(vectors, [batch, self.cols()])?;
        Ok(self.weights.forward(&x)?.to_f32_vec())
    }
}
