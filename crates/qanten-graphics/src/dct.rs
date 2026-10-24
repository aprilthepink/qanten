//! 8×8 block DCT-II (the transform behind JPEG and many texture and video
//! codecs) for image and asset pipelines.
//!
//! ![Original, JPEG quality 50 and quality 10 via the NPU DCT (example `dct`)](https://raw.githubusercontent.com/aprilthepink/qanten/main/docs/images/dct.png)
//!
//! Each 8×8 block is a 64-vector; the 2-D DCT is one 64×64 matrix (the
//! Kronecker product of two 1-D DCTs), resident on the NPU. All blocks of
//! an image go through in one call.

use rayon::prelude::*;

use crate::accel::ResidentMatrix;
use crate::{Error, Npu, Result};

/// Orthonormal 8×8 DCT-II and its inverse.
#[derive(Debug)]
pub struct Dct8x8 {
    /// Row-major 64×64: `K[(u,v),(y,x)] = C[u][y]·C[v][x]`.
    forward: Vec<f32>,
    inverse: Vec<f32>,
    npu: Option<(ResidentMatrix, ResidentMatrix)>,
}

impl Default for Dct8x8 {
    fn default() -> Self {
        Self::new()
    }
}

impl Dct8x8 {
    /// Builds the forward and inverse 64×64 matrices (CPU only until [`with_npu`](Self::with_npu)).
    pub fn new() -> Self {
        let c = |u: usize, x: usize| -> f64 {
            let a = if u == 0 {
                (1.0f64 / 8.0).sqrt()
            } else {
                (2.0f64 / 8.0).sqrt()
            };
            a * ((2 * x + 1) as f64 * u as f64 * std::f64::consts::PI / 16.0).cos()
        };
        let mut forward = vec![0.0f32; 64 * 64];
        let mut inverse = vec![0.0f32; 64 * 64];
        for u in 0..8 {
            for v in 0..8 {
                for y in 0..8 {
                    for x in 0..8 {
                        let k = (c(u, y) * c(v, x)) as f32;
                        forward[(u * 8 + v) * 64 + y * 8 + x] = k;
                        inverse[(y * 8 + x) * 64 + u * 8 + v] = k;
                    }
                }
            }
        }
        Self {
            forward,
            inverse,
            npu: None,
        }
    }

    /// Uploads both matrices to `npu`; later calls run there.
    pub fn with_npu(mut self, npu: &Npu) -> Result<Self> {
        self.npu = Some((
            ResidentMatrix::upload(npu, &self.forward, 64, 64)?,
            ResidentMatrix::upload(npu, &self.inverse, 64, 64)?,
        ));
        Ok(self)
    }

    /// Transforms a `width × height` plane (both multiples of 8). The output
    /// has the same layout, every 8×8 block replaced by its coefficients
    /// (DC at the block's top-left).
    pub fn forward(&self, plane: &[f32], width: usize, height: usize) -> Result<Vec<f32>> {
        self.run(plane, width, height, true)
    }

    /// Inverse of [`forward`](Self::forward).
    pub fn inverse(&self, coeffs: &[f32], width: usize, height: usize) -> Result<Vec<f32>> {
        self.run(coeffs, width, height, false)
    }

    fn run(&self, plane: &[f32], w: usize, h: usize, fwd: bool) -> Result<Vec<f32>> {
        if !w.is_multiple_of(8) || !h.is_multiple_of(8) || plane.len() != w * h {
            return Err(Error::Shape(format!(
                "plane must be {w}x{h} with both sides multiples of 8, got {} values",
                plane.len()
            )));
        }
        let (bw, bh) = (w / 8, h / 8);
        let mut blocks = vec![0.0f32; bw * bh * 64];
        blocks.par_chunks_mut(64).enumerate().for_each(|(b, blk)| {
            let (by, bx) = (b / bw, b % bw);
            for y in 0..8 {
                blk[y * 8..y * 8 + 8].copy_from_slice(&plane[(by * 8 + y) * w + bx * 8..][..8]);
            }
        });
        let out = match &self.npu {
            Some((f, i)) => (if fwd { f } else { i }).apply(&blocks, bw * bh)?,
            None => {
                let m = if fwd { &self.forward } else { &self.inverse };
                let mut o = vec![0.0f32; blocks.len()];
                o.par_chunks_mut(64)
                    .zip(blocks.par_chunks(64))
                    .for_each(|(dst, src)| {
                        for (r, d) in dst.iter_mut().enumerate() {
                            *d = m[r * 64..(r + 1) * 64]
                                .iter()
                                .zip(src)
                                .map(|(a, b)| a * b)
                                .sum();
                        }
                    });
                o
            }
        };
        let mut result = vec![0.0f32; w * h];
        for (b, blk) in out.chunks(64).enumerate() {
            let (by, bx) = (b / bw, b % bw);
            for y in 0..8 {
                result[(by * 8 + y) * w + bx * 8..][..8].copy_from_slice(&blk[y * 8..y * 8 + 8]);
            }
        }
        Ok(result)
    }
}
