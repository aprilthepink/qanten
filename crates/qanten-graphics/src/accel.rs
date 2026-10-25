//! Glue between the graphics modules and [`qanten::compute`].

use qanten::compute::{Matrix, NpuMatrix, check_product};
use qanten::pool::NpuPool;
use qanten::{Npu, Result};

/// Where a module's constant matrices go.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Target<'a> {
    /// One NPU.
    Npu(&'a Npu),
    /// A full copy on every NPU of a pool; batches are split between them.
    Pool(&'a NpuPool),
}

/// A dense `rows × cols` f32 matrix kept on the NPU(s) as a
/// [`compute::NpuMatrix`](qanten::compute::NpuMatrix), plus its host copy for
/// self-checks.
///
/// `apply` computes `out[b, r] = Σ_c matrix[r, c] · vectors[b, c]`, i.e.
/// `vectors × matrixᵀ`, which is the one primitive every module here reduces
/// to. `NpuMatrix` scales every matrix row and input vector to unit range
/// before it reaches the NPU, so very large or small values (HDR radiance,
/// world-space offsets) keep their relative precision.
#[derive(Debug)]
pub(crate) struct ResidentMatrix {
    resident: NpuMatrix,
    host: Matrix,
}

impl ResidentMatrix {
    pub(crate) fn upload(
        target: Target<'_>,
        data: &[f32],
        rows: usize,
        cols: usize,
    ) -> Result<Self> {
        let host = Matrix::new(rows, cols, data.to_vec())?;
        let resident = match target {
            Target::Npu(npu) => NpuMatrix::upload(npu, &host)?,
            Target::Pool(pool) => NpuMatrix::upload_replicated(pool, &host)?,
        };
        Ok(Self { resident, host })
    }

    /// `vectors` holds `batch` rows of `cols` values; returns `batch` rows of `rows` values.
    pub(crate) fn apply(&self, vectors: &[f32], batch: usize) -> Result<Vec<f32>> {
        let x = Matrix::new(batch, self.host.cols(), vectors.to_vec())?;
        Ok(self.resident.apply_batch(&x)?.into_vec())
    }

    /// Relative error of the NPU product on random vectors, measured on the
    /// CPU with [`check_product`] in `O(rows · cols)`.
    pub(crate) fn self_check(&self, seed: u64) -> Result<f64> {
        let x = Matrix::random_normal(16, self.host.cols(), seed);
        let y = self.resident.apply_batch(&x)?;
        Ok(check_product(&x, &self.host.transpose(), &y, 4))
    }
}

/// Worst [`ResidentMatrix::self_check`] over several matrices.
pub(crate) fn worst_check<'a>(
    matrices: impl IntoIterator<Item = &'a ResidentMatrix>,
    seed: u64,
) -> Result<f64> {
    let mut worst = 0.0f64;
    for m in matrices {
        worst = worst.max(m.self_check(seed)?);
    }
    Ok(worst)
}
