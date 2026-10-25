//! The host matrix type.

use std::fmt;
use std::ops::{Index, IndexMut, Range};

use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::rng::SplitMix64;
use crate::tensor::Tensor;

/// Element type of a [`Matrix`]: `f32` (what the NPU consumes) or `f64`
/// (exact host-side work such as residuals and eigenvectors).
pub trait Scalar: Copy + Default + PartialEq + fmt::Debug + Send + Sync + 'static {
    /// Zero.
    const ZERO: Self;
    /// One.
    const ONE: Self;
    /// Widen to `f64`.
    fn to_f64(self) -> f64;
    /// Narrow from `f64`.
    fn from_f64(v: f64) -> Self;
}

impl Scalar for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline]
    fn from_f64(v: f64) -> Self {
        v as f32
    }
}

impl Scalar for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    #[inline]
    fn to_f64(self) -> f64 {
        self
    }
    #[inline]
    fn from_f64(v: f64) -> Self {
        v
    }
}

/// A dense row-major matrix on the host. Use the aliases [`Matrix`] (`f32`)
/// and [`Matrix64`] (`f64`).
///
/// Products on the CPU ([`matmul`](Self::matmul), [`matvec`](Self::matvec))
/// accumulate in f64 and are the exact reference for everything the NPU
/// computes; [`matmul_on`](Self::matmul_on) can send large ones to the NPU.
#[derive(Clone, PartialEq)]
pub struct Mat<T: Scalar> {
    rows: usize,
    cols: usize,
    data: Vec<T>,
}

/// An `f32` matrix, the type the NPU consumes.
pub type Matrix = Mat<f32>;

/// An `f64` matrix, for exact host-side work (residuals, eigenvectors).
pub type Matrix64 = Mat<f64>;

impl<T: Scalar> fmt::Debug for Mat<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Mat {}x{}", self.rows, self.cols)?;
        if self.data.len() <= 64 {
            for row in self.iter_rows() {
                write!(f, "\n  {row:?}")?;
            }
        }
        Ok(())
    }
}

/// `Σ a·b` in f64, with eight independent partial sums so the compiler can
/// vectorise it (a single running sum is a serial dependency chain).
fn dot<T: Scalar, U: Scalar>(a: &[T], b: &[U]) -> f64 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let mut acc = [0.0f64; 8];
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    let tail: f64 = ca
        .remainder()
        .iter()
        .zip(cb.remainder())
        .map(|(x, y)| x.to_f64() * y.to_f64())
        .sum();
    for (x, y) in ca.zip(cb) {
        for k in 0..8 {
            acc[k] += x[k].to_f64() * y[k].to_f64();
        }
    }
    acc.iter().sum::<f64>() + tail
}

impl<T: Scalar> Mat<T> {
    /// From row-major data.
    pub fn new(rows: usize, cols: usize, data: Vec<T>) -> Result<Self> {
        if data.len() != rows * cols {
            return Err(Error::Shape(format!(
                "{} values do not make a {rows}x{cols} matrix",
                data.len()
            )));
        }
        Ok(Self { rows, cols, data })
    }

    /// All zeros.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![T::ZERO; rows * cols],
        }
    }

    /// The `n × n` identity.
    pub fn identity(n: usize) -> Self {
        Self::from_fn(n, n, |i, j| if i == j { T::ONE } else { T::ZERO })
    }

    /// Element `(i, j)` is `f(i, j)`.
    pub fn from_fn(rows: usize, cols: usize, mut f: impl FnMut(usize, usize) -> T) -> Self {
        let mut data = Vec::with_capacity(rows * cols);
        for i in 0..rows {
            data.extend((0..cols).map(|j| f(i, j)));
        }
        Self { rows, cols, data }
    }

    /// From equal-length rows.
    pub fn from_rows<R: AsRef<[T]>>(rows: &[R]) -> Result<Self> {
        let cols = rows.first().map_or(0, |r| r.as_ref().len());
        if rows.iter().any(|r| r.as_ref().len() != cols) {
            return Err(Error::Shape("rows have different lengths".into()));
        }
        Ok(Self {
            rows: rows.len(),
            cols,
            data: rows
                .iter()
                .flat_map(|r| r.as_ref().iter().copied())
                .collect(),
        })
    }

    /// Standard normal entries (deterministic for a seed).
    pub fn random_normal(rows: usize, cols: usize, seed: u64) -> Self {
        let mut rng = SplitMix64::new(seed);
        Self::from_fn(rows, cols, |_, _| T::from_f64(rng.next_normal() as f64))
    }

    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// `(rows, cols)`.
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Row-major values.
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Mutable row-major values.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// The row-major values.
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }

    /// Row `i`.
    pub fn row(&self, i: usize) -> &[T] {
        &self.data[i * self.cols..(i + 1) * self.cols]
    }

    /// Mutable row `i`.
    pub fn row_mut(&mut self, i: usize) -> &mut [T] {
        &mut self.data[i * self.cols..(i + 1) * self.cols]
    }

    /// The rows, in order.
    pub fn iter_rows(&self) -> impl ExactSizeIterator<Item = &[T]> + '_ {
        self.data.chunks_exact(self.cols.max(1)).take(self.rows)
    }

    /// The rows, mutably.
    pub fn iter_rows_mut(&mut self) -> impl ExactSizeIterator<Item = &mut [T]> + '_ {
        let rows = self.rows;
        self.data.chunks_exact_mut(self.cols.max(1)).take(rows)
    }

    /// Appends a row (its length must equal `cols`, or set it on an empty matrix).
    pub fn push_row(&mut self, row: &[T]) -> Result<()> {
        if self.rows == 0 && self.data.is_empty() {
            self.cols = row.len();
        }
        if row.len() != self.cols {
            return Err(Error::Shape(format!(
                "row of {} values for a matrix with {} columns",
                row.len(),
                self.cols
            )));
        }
        self.data.extend_from_slice(row);
        self.rows += 1;
        Ok(())
    }

    /// Rows `range` as a new matrix.
    pub fn slice_rows(&self, range: Range<usize>) -> Mat<T> {
        Mat {
            rows: range.len(),
            cols: self.cols,
            data: self.data[range.start * self.cols..range.end * self.cols].to_vec(),
        }
    }

    /// Columns `range` as a new matrix.
    pub fn columns(&self, range: Range<usize>) -> Mat<T> {
        Mat {
            rows: self.rows,
            cols: range.len(),
            data: self
                .iter_rows()
                .flat_map(|r| r[range.clone()].iter().copied())
                .collect(),
        }
    }

    /// The selected rows, in the given order.
    pub fn select_rows(&self, indices: &[usize]) -> Mat<T> {
        Mat {
            rows: indices.len(),
            cols: self.cols,
            data: indices
                .iter()
                .flat_map(|&i| self.row(i).iter().copied())
                .collect(),
        }
    }

    /// The transpose.
    pub fn transpose(&self) -> Mat<T> {
        Mat::from_fn(self.cols, self.rows, |i, j| self[(j, i)])
    }

    /// Elementwise map into any scalar type.
    pub fn map<U: Scalar>(&self, mut f: impl FnMut(T) -> U) -> Mat<U> {
        Mat {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().map(|&v| f(v)).collect(),
        }
    }

    /// Converts the element type (`f32` ⇄ `f64`).
    pub fn cast<U: Scalar>(&self) -> Mat<U> {
        self.map(|v| U::from_f64(v.to_f64()))
    }

    /// `self · other` on the CPU: multithreaded, accumulated in f64.
    pub fn matmul(&self, other: &Mat<T>) -> Result<Mat<T>> {
        if self.cols != other.rows {
            return Err(Error::Shape(format!(
                "matmul: {}x{} · {}x{}",
                self.rows, self.cols, other.rows, other.cols
            )));
        }
        self.matmul_transposed(&other.transpose())
    }

    /// `self · otherᵀ` on the CPU (every row of `self` against every row of
    /// `other`), without forming the transpose. This is "apply the matrix
    /// `other` to each row vector of `self`", the same layout as
    /// [`NpuMatrix::apply_batch`](super::NpuMatrix::apply_batch).
    pub fn matmul_transposed(&self, other: &Mat<T>) -> Result<Mat<T>> {
        if self.cols != other.cols {
            return Err(Error::Shape(format!(
                "matmul_transposed: {}x{} · ({}x{})ᵀ",
                self.rows, self.cols, other.rows, other.cols
            )));
        }
        let n = other.rows;
        let mut out = vec![T::ZERO; self.rows * n];
        out.par_chunks_mut(n.max(1))
            .zip(self.data.par_chunks(self.cols.max(1)))
            .for_each(|(dst, a)| {
                for (o, b) in dst.iter_mut().zip(other.iter_rows()) {
                    *o = T::from_f64(dot(a, b));
                }
            });
        Mat::new(self.rows, n, out)
    }

    /// `self · x` on the CPU, accumulated in f64.
    pub fn matvec<U: Scalar>(&self, x: &[U]) -> Result<Vec<T>> {
        if x.len() != self.cols {
            return Err(Error::Shape(format!(
                "matvec: {}x{} · vector of {}",
                self.rows,
                self.cols,
                x.len()
            )));
        }
        Ok(self.iter_rows().map(|r| T::from_f64(dot(r, x))).collect())
    }

    /// `selfᵀ · y` on the CPU, accumulated in f64 (without forming the transpose).
    pub fn matvec_transposed<U: Scalar>(&self, y: &[U]) -> Result<Vec<T>> {
        if y.len() != self.rows {
            return Err(Error::Shape(format!(
                "matvec_transposed: ({}x{})ᵀ · vector of {}",
                self.rows,
                self.cols,
                y.len()
            )));
        }
        let mut acc = vec![0.0f64; self.cols];
        for (row, &w) in self.iter_rows().zip(y) {
            for (a, v) in acc.iter_mut().zip(row) {
                *a += v.to_f64() * w.to_f64();
            }
        }
        Ok(acc.into_iter().map(T::from_f64).collect())
    }

    /// Frobenius norm.
    pub fn norm(&self) -> f64 {
        dot(&self.data, &self.data).sqrt()
    }

    /// `‖self − other‖ / ‖other‖` (Frobenius).
    pub fn relative_error(&self, other: &Mat<T>) -> f64 {
        let diff: f64 = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| (a.to_f64() - b.to_f64()).powi(2))
            .sum();
        diff.sqrt() / other.norm().max(f64::MIN_POSITIVE)
    }

    /// Column means.
    pub fn column_means(&self) -> Vec<f64> {
        let mut acc = vec![0.0f64; self.cols];
        for row in self.iter_rows() {
            for (a, v) in acc.iter_mut().zip(row) {
                *a += v.to_f64();
            }
        }
        let n = self.rows.max(1) as f64;
        acc.iter().map(|a| a / n).collect()
    }

    /// Eigenvalues and eigenvectors of a symmetric matrix, exactly in f64 on
    /// the CPU (cyclic Jacobi). Values are sorted largest first; eigenvector
    /// `i` is row `i` of the returned matrix. For small matrices, or as the
    /// final step of the NPU-driven
    /// [`LinearOperator::top_eigenpairs`](crate::linalg::LinearOperator::top_eigenpairs).
    pub fn symmetric_eigen(&self) -> Result<(Vec<f64>, Mat<f64>)> {
        if self.rows != self.cols {
            return Err(Error::Shape(format!(
                "eigendecomposition needs a square matrix, got {}x{}",
                self.rows, self.cols
            )));
        }
        let n = self.rows;
        let mut a: Mat<f64> = self.cast();
        let mut v = Mat::<f64>::identity(n);
        let scale = a.norm().max(f64::MIN_POSITIVE);
        for _sweep in 0..100 {
            let off: f64 = (0..n)
                .flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j)))
                .map(|(i, j)| a[(i, j)].powi(2))
                .sum();
            if off.sqrt() <= 1e-15 * scale {
                break;
            }
            for p in 0..n {
                for q in p + 1..n {
                    a.jacobi_rotate(&mut v, p, q);
                }
            }
        }
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&i, &j| a[(j, j)].total_cmp(&a[(i, i)]));
        let values = order.iter().map(|&i| a[(i, i)]).collect();
        // Columns of V are the eigenvectors; return them as rows.
        let vectors = Mat::from_fn(n, n, |r, k| v[(k, order[r])]);
        Ok((values, vectors))
    }
}

impl Mat<f64> {
    /// One Jacobi rotation zeroing `(p, q)`, accumulated into `v`.
    fn jacobi_rotate(&mut self, v: &mut Mat<f64>, p: usize, q: usize) {
        let apq = self[(p, q)];
        if apq.abs() < 1e-300 {
            return;
        }
        let theta = (self[(q, q)] - self[(p, p)]) / (2.0 * apq);
        let t = if theta == 0.0 {
            1.0
        } else {
            theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt())
        };
        let c = 1.0 / (t * t + 1.0).sqrt();
        let s = t * c;
        let rotate = |x: f64, y: f64| (c * x - s * y, s * x + c * y);
        for k in 0..self.rows {
            (self[(k, p)], self[(k, q)]) = rotate(self[(k, p)], self[(k, q)]);
        }
        for k in 0..self.cols {
            (self[(p, k)], self[(q, k)]) = rotate(self[(p, k)], self[(q, k)]);
        }
        for k in 0..v.rows {
            (v[(k, p)], v[(k, q)]) = rotate(v[(k, p)], v[(k, q)]);
        }
    }

    /// Orthonormalises the rows in place (modified Gram–Schmidt, twice for
    /// stability).
    pub fn orthonormalize_rows(&mut self) {
        for _ in 0..2 {
            for i in 0..self.rows {
                let (head, tail) = self.data.split_at_mut(i * self.cols);
                let row = &mut tail[..self.cols];
                for prev in head.chunks_exact(self.cols.max(1)) {
                    let d = dot(row, prev);
                    row.iter_mut().zip(prev).for_each(|(a, b)| *a -= d * b);
                }
                let n = dot(row, row).sqrt();
                if n > 0.0 {
                    row.iter_mut().for_each(|a| *a /= n);
                }
            }
        }
    }
}

impl Mat<f32> {
    /// As a `[rows, cols]` bf16 tensor.
    pub fn to_tensor(&self) -> Result<Tensor> {
        Tensor::from_f32(&self.data, [self.rows, self.cols])
    }

    /// From a rank-2 tensor.
    pub fn from_tensor(t: &Tensor) -> Result<Matrix> {
        let [r, c] = t.dims::<2>()?;
        Mat::new(r, c, t.to_f32_vec())
    }
}

impl<T: Scalar> Index<(usize, usize)> for Mat<T> {
    type Output = T;
    fn index(&self, (i, j): (usize, usize)) -> &T {
        &self.data[i * self.cols + j]
    }
}

impl<T: Scalar> IndexMut<(usize, usize)> for Mat<T> {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut T {
        &mut self.data[i * self.cols + j]
    }
}
