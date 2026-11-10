//! Image filters for texture pipelines as NPU matrix products: separable
//! Gaussian blur and 2× downsampling (mip chains), multi-scale bloom, and 3×3
//! stencils (Sobel edges, sharpen, normal maps from height maps).
//!
//! ![Bloom, edges and a normal map computed on the NPU (example `filters`)](https://raw.githubusercontent.com/aprilthepink/qanten/main/docs/images/filters.png)
//!
//! Images are one [`Matrix`] per channel, `height × width`, row-major.
//! Every filter takes a slice of channels and processes all of them in the
//! same NPU calls.
//!
//! # How it maps to the NPU
//!
//! - **Separable filters** ([`Separable`]): a 1-D filter along a row is a
//!   banded `out_w × w` matrix. Blurring every row of every channel is one
//!   product with that resident matrix; the columns are a second product
//!   with the `out_h × h` matrix. Two calls per image, however many channels.
//! - **Bloom** ([`Bloom`]): `Σ_k weight_k · blur_k(bright)` over several
//!   blur radii is still two calls. The row matrices of all scales are
//!   stacked (`K·w × w`), so each row comes back blurred at every scale; the
//!   weighted column matrices sit side by side (`h × K·h`), so the column
//!   pass sums the scales as it blurs.
//! - **Stencils** ([`Stencil3x3`]): the im2row formulation `ops::conv2d`
//!   uses: every pixel's 3×3 neighbourhood is a row of 9 values, and the
//!   resident `kernels × 9` matrix turns each into its filter responses. This
//!   is bandwidth-bound (9 values in per pixel), so it pays off mostly when
//!   batched with other NPU work.
//!
//! Edges are clamped (the border pixel repeats), and every filter has an
//! exact `*_cpu` twin used as the accuracy reference.

use glam::Vec3;
use qanten::compute::Matrix;
use rayon::prelude::*;

use crate::accel::{Accelerated, MatrixOp, accelerated_methods};
use crate::{Error, Result};

/// A 1-D Gaussian blur as an `n × n` matrix with clamped edges: taps that
/// fall outside the image land on the border pixel. Rows sum to 1.
pub fn gaussian_matrix(n: usize, sigma: f32) -> Matrix {
    let sigma = sigma.max(1e-3);
    let radius = (3.0 * sigma).ceil() as isize;
    let taps: Vec<f32> = (-radius..=radius)
        .map(|d| (-(d * d) as f32 / (2.0 * sigma * sigma)).exp())
        .collect();
    let total: f32 = taps.iter().sum();
    let mut m = Matrix::zeros(n, n);
    for o in 0..n {
        for (k, &t) in taps.iter().enumerate() {
            let i = (o as isize + k as isize - radius).clamp(0, n as isize - 1) as usize;
            m[(o, i)] += t / total;
        }
    }
    m
}

/// 2× downsampling along one axis as a `⌈n/2⌉ × n` matrix: a `[1 3 3 1] / 8`
/// tent centred between each pair of input samples, with clamped edges.
pub fn downsample_matrix(n: usize) -> Matrix {
    let out = n.div_ceil(2);
    let mut m = Matrix::zeros(out, n);
    for o in 0..out {
        for (k, w) in [1.0, 3.0, 3.0, 1.0].into_iter().enumerate() {
            let i = (2 * o as isize + k as isize - 1).clamp(0, n as isize - 1) as usize;
            m[(o, i)] += w / 8.0;
        }
    }
    m
}

/// Checks that all channels are `h × w` and returns `(h, w)`.
fn shape_of(channels: &[Matrix]) -> Result<(usize, usize)> {
    let first = channels
        .first()
        .ok_or_else(|| Error::InvalidArgument("no image channels".into()))?;
    let shape = first.shape();
    if channels.iter().any(|c| c.shape() != shape) {
        return Err(Error::Shape("all channels must have the same size".into()));
    }
    Ok(shape)
}

/// Stacks matrices with the same number of columns on top of each other.
fn vstack<'a>(parts: impl IntoIterator<Item = &'a Matrix>) -> Result<Matrix> {
    let mut data = Vec::new();
    let (mut rows, mut cols) = (0, None);
    for p in parts {
        if *cols.get_or_insert(p.cols()) != p.cols() {
            return Err(Error::Shape("vstack: column counts differ".into()));
        }
        rows += p.rows();
        data.extend_from_slice(p.as_slice());
    }
    Matrix::new(rows, cols.unwrap_or(0), data)
}

/// Splits `m` into `parts` equal row blocks.
fn split_rows(m: &Matrix, parts: usize) -> Vec<Matrix> {
    let n = m.rows() / parts.max(1);
    (0..parts)
        .map(|p| m.slice_rows(p * n..(p + 1) * n))
        .collect()
}

/// A filter that is a matrix along rows followed by a matrix along columns:
/// `out = C · X · Rᵀ` per channel.
#[derive(Debug)]
pub struct Separable {
    rows: MatrixOp,
    cols: MatrixOp,
}

impl Accelerated for Separable {
    fn ops(&self) -> Vec<&MatrixOp> {
        vec![&self.rows, &self.cols]
    }
    fn ops_mut(&mut self) -> Vec<&mut MatrixOp> {
        vec![&mut self.rows, &mut self.cols]
    }
}

accelerated_methods!(Separable);

impl Separable {
    /// `row` (`out_w × w`) filters along each row, `col` (`out_h × h`) along
    /// each column.
    pub fn new(row: Matrix, col: Matrix) -> Self {
        Self {
            rows: MatrixOp::new(row),
            cols: MatrixOp::new(col),
        }
    }

    /// A Gaussian blur of standard deviation `sigma` pixels for `width × height` images.
    pub fn gaussian(width: usize, height: usize, sigma: f32) -> Self {
        Self::new(
            gaussian_matrix(width, sigma),
            gaussian_matrix(height, sigma),
        )
    }

    /// 2× downsampling (one mip level) for `width × height` images.
    pub fn downsample(width: usize, height: usize) -> Self {
        Self::new(downsample_matrix(width), downsample_matrix(height))
    }

    /// Filters every channel: two NPU calls in total (rows, then columns).
    pub fn apply(&self, channels: &[Matrix]) -> Result<Vec<Matrix>> {
        self.run(channels, |op, x| op.apply(x))
    }

    /// Exact CPU reference for [`apply`](Self::apply).
    pub fn apply_cpu(&self, channels: &[Matrix]) -> Result<Vec<Matrix>> {
        self.run(channels, |op, x| op.apply_cpu(x))
    }

    fn run(
        &self,
        channels: &[Matrix],
        apply: impl Fn(&MatrixOp, &Matrix) -> Result<Matrix>,
    ) -> Result<Vec<Matrix>> {
        shape_of(channels)?;
        let c = channels.len();
        // Rows of every channel through the row matrix, then each channel's
        // columns (as rows of its transpose) through the column matrix.
        let rows = split_rows(&apply(&self.rows, &vstack(channels)?)?, c);
        let transposed: Vec<Matrix> = rows.iter().map(Matrix::transpose).collect();
        let cols = apply(&self.cols, &vstack(&transposed)?)?;
        Ok(split_rows(&cols, c).iter().map(Matrix::transpose).collect())
    }
}

/// Bloom: the bright parts of an image, blurred at several radii and added
/// back. Two NPU calls per image, whatever the number of radii and channels.
#[derive(Debug)]
pub struct Bloom {
    threshold: f32,
    intensity: f32,
    scales: usize,
    size: (usize, usize),
    /// Row blurs of every scale stacked: `K·w × w`.
    rows: MatrixOp,
    /// Weighted column blurs side by side: `h × K·h`.
    cols: MatrixOp,
}

impl Accelerated for Bloom {
    fn ops(&self) -> Vec<&MatrixOp> {
        vec![&self.rows, &self.cols]
    }
    fn ops_mut(&mut self) -> Vec<&mut MatrixOp> {
        vec![&mut self.rows, &mut self.cols]
    }
}

accelerated_methods!(Bloom);

impl Bloom {
    /// Bloom for `width × height` images: values above `threshold` are
    /// blurred with each of `sigmas` (in pixels, weighted equally) and added
    /// back scaled by `intensity`.
    pub fn new(
        width: usize,
        height: usize,
        sigmas: &[f32],
        threshold: f32,
        intensity: f32,
    ) -> Result<Self> {
        if sigmas.is_empty() || width == 0 || height == 0 {
            return Err(Error::InvalidArgument(
                "bloom needs a non-empty image and at least one radius".into(),
            ));
        }
        let k = sigmas.len();
        let row_blurs: Vec<Matrix> = sigmas.iter().map(|&s| gaussian_matrix(width, s)).collect();
        let col_blurs: Vec<Matrix> = sigmas.iter().map(|&s| gaussian_matrix(height, s)).collect();
        let rows = vstack(&row_blurs)?;
        let cols = Matrix::from_fn(height, k * height, |o, i| {
            col_blurs[i / height][(o, i % height)] / k as f32
        });
        Ok(Self {
            threshold,
            intensity,
            scales: k,
            size: (height, width),
            rows: MatrixOp::new(rows),
            cols: MatrixOp::new(cols),
        })
    }

    /// `image + intensity · glow` for every channel.
    pub fn apply(&self, channels: &[Matrix]) -> Result<Vec<Matrix>> {
        self.run(channels, |op, x| op.apply(x))
    }

    /// Exact CPU reference for [`apply`](Self::apply).
    pub fn apply_cpu(&self, channels: &[Matrix]) -> Result<Vec<Matrix>> {
        self.run(channels, |op, x| op.apply_cpu(x))
    }

    fn run(
        &self,
        channels: &[Matrix],
        apply: impl Fn(&MatrixOp, &Matrix) -> Result<Matrix>,
    ) -> Result<Vec<Matrix>> {
        let (h, w) = shape_of(channels)?;
        if (h, w) != self.size {
            return Err(Error::Shape(format!(
                "bloom was built for {}x{} images, got {h}x{w}",
                self.size.1, self.size.0
            )));
        }
        let (c, k, t) = (channels.len(), self.scales, self.threshold);
        let bright: Vec<Matrix> = channels
            .iter()
            .map(|m| m.map(|v| (v - t).max(0.0)))
            .collect();
        // Each row comes back blurred at every scale: [C·h, K·w].
        let rows = apply(&self.rows, &vstack(&bright)?)?;
        // Column pass input for (channel, x): the K blurred columns stacked.
        let cols_in = Matrix::from_fn(c * w, k * h, |r, i| {
            let (ch, x) = (r / w, r % w);
            let (scale, y) = (i / h, i % h);
            rows[(ch * h + y, scale * w + x)]
        });
        let glow = apply(&self.cols, &cols_in)?; // [C·w, h]
        Ok(channels
            .iter()
            .zip(split_rows(&glow, c))
            .map(|(img, g)| {
                let g = g.transpose();
                Matrix::from_fn(h, w, |y, x| img[(y, x)] + self.intensity * g[(y, x)])
            })
            .collect())
    }
}

/// Several 3×3 stencils applied at once (im2row × a resident `kernels × 9`
/// matrix), with clamped edges.
#[derive(Debug)]
pub struct Stencil3x3 {
    kernels: MatrixOp,
}

impl Accelerated for Stencil3x3 {
    fn ops(&self) -> Vec<&MatrixOp> {
        vec![&self.kernels]
    }
    fn ops_mut(&mut self) -> Vec<&mut MatrixOp> {
        vec![&mut self.kernels]
    }
}

accelerated_methods!(Stencil3x3);

impl Stencil3x3 {
    /// Kernels in row-major order (`k[dy + 1][dx + 1]` weights pixel `(y + dy, x + dx)`).
    pub fn new(kernels: &[[[f32; 3]; 3]]) -> Result<Self> {
        if kernels.is_empty() {
            return Err(Error::InvalidArgument("no kernels".into()));
        }
        Ok(Self::from_kernels(kernels))
    }

    /// Sobel: response 0 is ∂/∂x, response 1 is ∂/∂y (per pixel, ×8 scale
    /// removed so a unit ramp gives 1).
    pub fn sobel() -> Self {
        let gx = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]].map(|r| r.map(|v| v / 8.0));
        let gy = [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]].map(|r| r.map(|v| v / 8.0));
        Self::from_kernels(&[gx, gy])
    }

    /// Unsharp-style sharpening: `pixel + amount · (pixel − mean of its 4 neighbours)`.
    pub fn sharpen(amount: f32) -> Self {
        let a = amount;
        let n = -a / 4.0;
        Self::from_kernels(&[[[0.0, n, 0.0], [n, 1.0 + a, n], [0.0, n, 0.0]]])
    }

    fn from_kernels(kernels: &[[[f32; 3]; 3]]) -> Self {
        Self {
            kernels: MatrixOp::new(Matrix::from_fn(kernels.len(), 9, |r, c| {
                kernels[r][c / 3][c % 3]
            })),
        }
    }

    /// Number of kernels (responses per pixel).
    pub fn kernel_count(&self) -> usize {
        self.kernels.rows()
    }

    /// Every kernel on every channel: `out[channel][kernel]`, one NPU call.
    pub fn apply(&self, channels: &[Matrix]) -> Result<Vec<Vec<Matrix>>> {
        self.run(channels, |op, x| op.apply(x))
    }

    /// Exact CPU reference for [`apply`](Self::apply).
    pub fn apply_cpu(&self, channels: &[Matrix]) -> Result<Vec<Vec<Matrix>>> {
        self.run(channels, |op, x| op.apply_cpu(x))
    }

    fn run(
        &self,
        channels: &[Matrix],
        apply: impl Fn(&MatrixOp, &Matrix) -> Result<Matrix>,
    ) -> Result<Vec<Vec<Matrix>>> {
        let (h, w) = shape_of(channels)?;
        let k = self.kernel_count();
        let patches = Matrix::from_fn(channels.len() * h * w, 9, |r, t| {
            let (ch, p) = (r / (h * w), r % (h * w));
            let y = ((p / w) as isize + t as isize / 3 - 1).clamp(0, h as isize - 1) as usize;
            let x = ((p % w) as isize + t as isize % 3 - 1).clamp(0, w as isize - 1) as usize;
            channels[ch][(y, x)]
        });
        let out = apply(&self.kernels, &patches)?; // [C·h·w, k]
        Ok((0..channels.len())
            .map(|ch| {
                (0..k)
                    .map(|kk| Matrix::from_fn(h, w, |y, x| out[(ch * h * w + y * w + x, kk)]))
                    .collect()
            })
            .collect())
    }

    /// Edge strength `√(gx² + gy²)` of one channel (needs a [`sobel`](Self::sobel) stencil).
    pub fn edges(&self, image: &Matrix) -> Result<Matrix> {
        let g = self.gradients(image)?;
        Ok(Matrix::from_fn(image.rows(), image.cols(), |y, x| {
            g[0][(y, x)].hypot(g[1][(y, x)])
        }))
    }

    /// A tangent-space normal map from a height map: `normalize(−s·∂h/∂x,
    /// −s·∂h/∂y, 1)` per pixel, row-major (needs a [`sobel`](Self::sobel)
    /// stencil; `strength` is height units per pixel).
    pub fn normal_map(&self, height: &Matrix, strength: f32) -> Result<Vec<Vec3>> {
        let g = self.gradients(height)?;
        let (h, w) = height.shape();
        Ok((0..h * w)
            .into_par_iter()
            .map(|p| {
                let (y, x) = (p / w, p % w);
                Vec3::new(-strength * g[0][(y, x)], -strength * g[1][(y, x)], 1.0).normalize()
            })
            .collect())
    }

    fn gradients(&self, image: &Matrix) -> Result<Vec<Matrix>> {
        if self.kernel_count() != 2 {
            return Err(Error::InvalidArgument(
                "edges and normal maps need the two Sobel kernels".into(),
            ));
        }
        let mut out = self.apply(std::slice::from_ref(image))?;
        Ok(out.remove(0))
    }
}
