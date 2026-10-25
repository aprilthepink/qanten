//! 8×8 block DCT-II (the transform behind JPEG and many texture and video
//! codecs) for image and asset pipelines.
//!
//! ![Original, JPEG quality 50 and quality 10 via the NPU DCT (example `dct`)](https://raw.githubusercontent.com/aprilthepink/qanten/main/docs/images/dct.png)
//!
//! Each 8×8 block is a 64-vector; the 2-D DCT is one 64×64 matrix (the
//! Kronecker product of two 1-D DCTs), resident on the NPU. All blocks of
//! an image go through in one call.

use qanten::compute::Matrix;

use crate::accel::{Accelerated, MatrixOp, accelerated_methods};
use crate::{Error, Result};

/// Forward and inverse 8×8 DCT-II (orthonormal, as in JPEG).
#[derive(Debug)]
pub struct Dct8x8 {
    forward: MatrixOp,
    inverse: MatrixOp,
}

impl Default for Dct8x8 {
    fn default() -> Self {
        Self::new()
    }
}

impl Accelerated for Dct8x8 {
    fn ops(&self) -> Vec<&MatrixOp> {
        vec![&self.forward, &self.inverse]
    }
    fn ops_mut(&mut self) -> Vec<&mut MatrixOp> {
        vec![&mut self.forward, &mut self.inverse]
    }
}

accelerated_methods!(Dct8x8);

/// 1-D orthonormal DCT-II basis value `c(u, x)` for 8 samples.
fn dct_1d(u: usize, x: usize) -> f64 {
    let a = if u == 0 {
        (1.0f64 / 8.0).sqrt()
    } else {
        (2.0f64 / 8.0).sqrt()
    };
    a * ((2 * x + 1) as f64 * u as f64 * std::f64::consts::PI / 16.0).cos()
}

impl Dct8x8 {
    /// Builds the forward and inverse 64×64 matrices (CPU only until
    /// [`with_npu`](Self::with_npu)): the Kronecker product of two 1-D DCTs.
    pub fn new() -> Self {
        // Row (u·8 + v), column (y·8 + x) of the forward transform.
        let forward = Matrix::from_fn(64, 64, |uv, yx| {
            (dct_1d(uv / 8, yx / 8) * dct_1d(uv % 8, yx % 8)) as f32
        });
        Self {
            inverse: MatrixOp::new(forward.transpose()), // orthonormal: inverse = transpose
            forward: MatrixOp::new(forward),
        }
    }

    /// Transforms a plane (`height × width`, both multiples of 8). The output
    /// has the same layout, every 8×8 block replaced by its coefficients (DC
    /// at the block's top-left).
    pub fn forward(&self, plane: &Matrix) -> Result<Matrix> {
        Self::blockwise(plane, &self.forward)
    }

    /// Inverse of [`forward`](Self::forward).
    pub fn inverse(&self, coeffs: &Matrix) -> Result<Matrix> {
        Self::blockwise(coeffs, &self.inverse)
    }

    /// Every 8×8 block of `plane` as a 64-vector through `op`, all in one call.
    fn blockwise(plane: &Matrix, op: &MatrixOp) -> Result<Matrix> {
        let (h, w) = plane.shape();
        if !w.is_multiple_of(8) || !h.is_multiple_of(8) || w == 0 || h == 0 {
            return Err(Error::Shape(format!(
                "plane must have both sides multiples of 8, got {h}x{w}"
            )));
        }
        let bw = w / 8;
        // Pixel (row, col) of block b, offset k within the block.
        let pixel = |b: usize, k: usize| ((b / bw) * 8 + k / 8, (b % bw) * 8 + k % 8);
        let blocks = Matrix::from_fn(bw * (h / 8), 64, |b, k| plane[pixel(b, k)]);
        let out = op.apply(&blocks)?;
        let mut result = Matrix::zeros(h, w);
        for (b, block) in out.iter_rows().enumerate() {
            for (k, &v) in block.iter().enumerate() {
                result[pixel(b, k)] = v;
            }
        }
        Ok(result)
    }
}
