//! Linear algebra on the NPU with full-precision results.
//!
//! The NPU multiplies in bfloat16 (about 3 significant digits). The solvers
//! here use **mixed-precision iterative refinement**: the expensive inner
//! iterations (many matrix-vector products) run on the NPU in low precision,
//! and a few exact f64 residuals on the host correct the result. For
//! reasonably conditioned problems that converges to f64-level accuracy,
//! while the NPU does most of the multiply-accumulates.
//!
//! Everything starts from a [`LinearOperator`]: the matrix kept exactly on
//! the host and as a [`compute::NpuMatrix`](crate::compute::NpuMatrix) on the
//! device.
//!
//! | Method | Problem | Inner work on the NPU |
//! |---|---|---|
//! | [`LinearOperator::conjugate_gradient`] | `A·x = b`, `A` symmetric positive definite | CG iterations |
//! | [`LinearOperator::least_squares`] | `min ‖A·x − b‖`, any `A` | CGLS iterations (`A` and `Aᵀ` products) |
//! | [`LinearOperator::top_eigenpairs`] | largest eigenvalues/vectors of a symmetric `A` | block power iteration |
//! | [`Pca::fit`] | principal components of data | block power iteration on the covariance |
//!
//! ```
//! # fn main() -> qanten::Result<()> {
//! use qanten::compute::{Matrix, Matrix64};
//! use qanten::linalg::{LinearOperator, SolveOptions};
//! let npu = qanten::Npu::simulator();
//! // A small SPD system.
//! let a = Matrix64::from_rows(&[[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])?;
//! let op = LinearOperator::new(&npu, &a)?;
//! let (x, report) = op.conjugate_gradient(&[1.0, 2.0, 3.0], &SolveOptions::default())?;
//! assert!(report.converged && report.relative_residual < 1e-10);
//! # Ok(()) }
//! ```

use crate::compute::{Matrix, Matrix64, NpuMatrix};
use crate::device::Npu;
use crate::error::{Error, Result};

/// Stopping rules for the solvers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolveOptions {
    /// Target relative residual `‖b − A·x‖ / ‖b‖`, checked in exact f64.
    pub tol: f64,
    /// Maximum refinement (outer) steps.
    pub max_refinements: usize,
    /// Maximum NPU iterations per refinement step.
    pub inner_iters: usize,
    /// Relative accuracy each inner solve aims for (bf16 can't do much better than ~1e-2).
    pub inner_tol: f64,
}

impl Default for SolveOptions {
    fn default() -> Self {
        Self {
            tol: 1e-10,
            max_refinements: 30,
            inner_iters: 50,
            inner_tol: 1e-2,
        }
    }
}

/// What a solve did.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolveReport {
    /// `true` if `tol` was reached.
    pub converged: bool,
    /// Final exact relative residual.
    pub relative_residual: f64,
    /// Outer refinement steps taken (each costs one exact host product).
    pub refinements: usize,
    /// Total inner iterations, i.e. NPU matrix-vector products (CGLS: pairs).
    pub npu_iterations: usize,
}

/// Eigenvalues and eigenvectors of a symmetric matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct Eigen {
    /// Eigenvalues, largest magnitude first.
    pub values: Vec<f64>,
    /// Matching unit eigenvectors, one per row.
    pub vectors: Matrix64,
    /// Block power iterations on the NPU (each is one batched product).
    pub iterations: usize,
    /// Exact f64 subspace iterations on the host that polished the result.
    pub refinements: usize,
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum()
}

/// A dense matrix kept twice: exactly on the host (f64) for residuals, and on
/// the NPU (bf16, scaled per row and vector) for fast approximate products.
#[derive(Debug)]
pub struct LinearOperator {
    host: Matrix64,
    resident: NpuMatrix,
    resident_t: Option<NpuMatrix>,
}

impl LinearOperator {
    /// Uploads `a`.
    pub fn new(npu: &Npu, a: &Matrix64) -> Result<Self> {
        Ok(Self {
            resident: NpuMatrix::upload(npu, &a.cast())?,
            host: a.clone(),
            resident_t: None,
        })
    }

    /// Like [`new`](Self::new), and also uploads `Aᵀ` (needed by
    /// [`least_squares`](Self::least_squares)).
    pub fn with_transpose(npu: &Npu, a: &Matrix64) -> Result<Self> {
        let mut op = Self::new(npu, a)?;
        op.resident_t = Some(NpuMatrix::upload(npu, &a.transpose().cast())?);
        Ok(op)
    }

    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.host.rows()
    }

    /// Number of columns.
    pub fn cols(&self) -> usize {
        self.host.cols()
    }

    /// The exact matrix.
    pub fn matrix(&self) -> &Matrix64 {
        &self.host
    }

    /// Approximate `A·x` for every row `x` of `xs`, on the NPU.
    pub fn apply_npu(&self, xs: &Matrix) -> Result<Matrix> {
        self.resident.apply_batch(xs)
    }

    /// Approximate `Aᵀ·y` for every row `y` of `ys`, on the NPU (needs
    /// [`with_transpose`](Self::with_transpose)).
    pub fn apply_t_npu(&self, ys: &Matrix) -> Result<Matrix> {
        self.transposed()?.apply_batch(ys)
    }

    fn transposed(&self) -> Result<&NpuMatrix> {
        self.resident_t.as_ref().ok_or_else(|| {
            Error::InvalidArgument(
                "operator was built without its transpose (use with_transpose)".into(),
            )
        })
    }

    /// Exact `A·x` in f64 on the host.
    pub fn apply_exact(&self, x: &[f64]) -> Vec<f64> {
        self.host.matvec(x).unwrap_or_default()
    }

    /// Exact `Aᵀ·y` in f64 on the host.
    pub fn apply_t_exact(&self, y: &[f64]) -> Vec<f64> {
        self.host.matvec_transposed(y).unwrap_or_default()
    }

    /// Solves `A·x = b` for symmetric positive definite `A`.
    ///
    /// Outer loop: exact residual `r = b − A·x` in f64. Inner loop: conjugate
    /// gradient on `A·d = r` with NPU products until `inner_tol`, then `x += d`.
    pub fn conjugate_gradient(
        &self,
        b: &[f64],
        opts: &SolveOptions,
    ) -> Result<(Vec<f64>, SolveReport)> {
        if self.rows() != self.cols() || b.len() != self.rows() {
            return Err(Error::Shape(format!(
                "CG needs a square matrix and b of length {}",
                self.rows()
            )));
        }
        refine(
            b,
            opts,
            |x| self.apply_exact(x),
            |r| {
                inner_cg(r, opts, |p| {
                    let ap = self.resident.apply(p)?;
                    let pap = dot(p, &ap);
                    Ok((ap, pap))
                })
            },
        )
    }

    /// Solves `min ‖A·x − b‖₂` for any `rows × cols` `A` (build the operator
    /// with [`with_transpose`](Self::with_transpose)).
    ///
    /// Refinement on the normal equations: the exact residual of
    /// `Aᵀ(b − A·x)` is computed in f64; the inner solve is CGLS with `A` and
    /// `Aᵀ` products on the NPU.
    pub fn least_squares(&self, b: &[f64], opts: &SolveOptions) -> Result<(Vec<f64>, SolveReport)> {
        if b.len() != self.rows() {
            return Err(Error::Shape(format!(
                "b has {} values, A has {} rows",
                b.len(),
                self.rows()
            )));
        }
        let at = self.transposed()?;
        refine(
            &self.apply_t_exact(b),
            opts,
            |x| self.apply_t_exact(&self.apply_exact(x)),
            |g| {
                // CG on AᵀA·d = g; pᵀAᵀAp = ‖Ap‖² is better conditioned in low precision.
                inner_cg(g, opts, |p| {
                    let ap = self.resident.apply(p)?;
                    let norm2 = dot(&ap, &ap);
                    Ok((at.apply(&ap)?, norm2))
                })
            },
        )
    }

    /// The `k` eigenpairs of largest magnitude of a symmetric matrix, by block
    /// power (subspace) iteration: each step is one batched NPU product of
    /// `k + 4` vectors, followed by f64 orthonormalisation and Rayleigh–Ritz
    /// on the host. Once the NPU iterations stall at bf16 accuracy, a few
    /// exact f64 subspace iterations finish the job (usually 3–10, reported
    /// in `refinements`).
    pub fn top_eigenpairs(&self, k: usize, max_iters: usize, seed: u64) -> Result<Eigen> {
        let n = self.rows();
        if self.cols() != n {
            return Err(Error::Shape("eigenpairs need a square matrix".into()));
        }
        let k = k.clamp(1, n);
        // Oversample the subspace for faster, more reliable convergence.
        let mut q = Matrix64::random_normal((k + 4).min(n), n, seed);
        q.orthonormalize_rows();
        let mut ritz = RitzTracker::new(k);
        let mut iterations = 0;
        for _ in 0..max_iters {
            iterations += 1;
            q = self.apply_npu(&q.cast())?.cast();
            q.orthonormalize_rows();
            // bf16 products can't resolve changes much below ~1e-5; hand
            // over to the exact refinement once there.
            if ritz.update(&self.rayleigh_ritz(&q)?.0) < 1e-5 {
                break;
            }
        }
        // The NPU subspace is accurate to about bf16 precision; a few exact
        // f64 subspace iterations polish it to full precision.
        let mut refinements = 0;
        for _ in 0..30 {
            refinements += 1;
            q = self.host.matmul(&q.transpose())?.transpose();
            q.orthonormalize_rows();
            if ritz.update(&self.rayleigh_ritz(&q)?.0) < 1e-14 {
                break;
            }
        }
        let (values, vectors) = self.rayleigh_ritz(&q)?;
        Ok(Eigen {
            values: values.into_iter().take(k).collect(),
            vectors: vectors.slice_rows(0..k),
            iterations,
            refinements,
        })
    }

    /// Projects the exact matrix onto the row space of `q`, diagonalises the
    /// small projected matrix, and returns Ritz values/vectors sorted by
    /// magnitude (vectors as rows).
    fn rayleigh_ritz(&self, q: &Matrix64) -> Result<(Vec<f64>, Matrix64)> {
        let aq = self.host.matmul(&q.transpose())?; // n × m
        let h = q.matmul(&aq)?; // m × m
        let (vals, vecs) = h.symmetric_eigen()?;
        let mut order: Vec<usize> = (0..vals.len()).collect();
        order.sort_by(|&a, &b| vals[b].abs().total_cmp(&vals[a].abs()));
        let values = order.iter().map(|&i| vals[i]).collect();
        let mut ritz = vecs.select_rows(&order).matmul(q)?;
        ritz.orthonormalize_rows();
        Ok((values, ritz))
    }
}

/// Watches the leading Ritz values for convergence.
#[derive(Debug)]
struct RitzTracker {
    previous: Vec<f64>,
}

impl RitzTracker {
    fn new(k: usize) -> Self {
        Self {
            previous: vec![0.0; k],
        }
    }

    /// Largest relative change of the leading values since the last call.
    fn update(&mut self, values: &[f64]) -> f64 {
        let change = values
            .iter()
            .zip(&self.previous)
            .map(|(a, b)| ((a - b) / a.abs().max(1e-300)).abs())
            .fold(0.0, f64::max);
        self.previous = values.iter().take(self.previous.len()).copied().collect();
        change
    }
}

/// The refinement loop shared by the solvers: exact residuals on the host,
/// corrections from an approximate inner solve.
fn refine(
    b: &[f64],
    opts: &SolveOptions,
    exact: impl Fn(&[f64]) -> Vec<f64>,
    mut inner: impl FnMut(&[f64]) -> Result<(Vec<f64>, usize)>,
) -> Result<(Vec<f64>, SolveReport)> {
    let nb = norm(b).max(f64::MIN_POSITIVE);
    let mut x = vec![0.0; b.len()];
    let mut report = SolveReport {
        converged: false,
        relative_residual: 1.0,
        refinements: 0,
        npu_iterations: 0,
    };
    loop {
        let r: Vec<f64> = b.iter().zip(exact(&x)).map(|(b, a)| b - a).collect();
        report.relative_residual = norm(&r) / nb;
        report.converged = report.relative_residual <= opts.tol;
        if report.converged || report.refinements == opts.max_refinements {
            return Ok((x, report));
        }
        let (d, used) = inner(&r)?;
        report.npu_iterations += used;
        x.iter_mut().zip(&d).for_each(|(xi, di)| *xi += di);
        report.refinements += 1;
    }
}

/// Approximate conjugate gradient on `M·d = r` in f32, where `apply(p)`
/// returns `(M·p, pᵀ·M·p)` (computed on the NPU). Works on the normalised
/// system `r / ‖r‖` and returns `(d, products used)`.
fn inner_cg(
    r0: &[f64],
    opts: &SolveOptions,
    apply: impl Fn(&[f32]) -> Result<(Vec<f32>, f64)>,
) -> Result<(Vec<f64>, usize)> {
    let scale = norm(r0).max(f64::MIN_POSITIVE);
    let mut r: Vec<f32> = r0.iter().map(|v| (v / scale) as f32).collect();
    let mut p = r.clone();
    let mut d = vec![0.0f32; r.len()];
    let mut rr = dot(&r, &r);
    let mut used = 0;
    while used < opts.inner_iters && rr.sqrt() > opts.inner_tol {
        let (mp, pmp) = apply(&p)?;
        used += 1;
        if pmp <= 0.0 {
            break; // lost positive definiteness in low precision; refinement continues
        }
        let alpha = (rr / pmp) as f32;
        d.iter_mut().zip(&p).for_each(|(d, p)| *d += alpha * p);
        r.iter_mut().zip(&mp).for_each(|(r, m)| *r -= alpha * m);
        let rr_new = dot(&r, &r);
        let beta = (rr_new / rr) as f32;
        p.iter_mut().zip(&r).for_each(|(p, r)| *p = r + beta * *p);
        rr = rr_new;
    }
    Ok((d.iter().map(|v| *v as f64 * scale).collect(), used))
}

/// Principal component analysis.
#[derive(Debug, Clone, PartialEq)]
pub struct Pca {
    /// Column means that were subtracted.
    pub mean: Vec<f64>,
    /// Principal directions (unit vectors, one per row), most variance first.
    pub components: Matrix64,
    /// Variance along each component.
    pub explained_variance: Vec<f64>,
    /// Fraction of the total variance each component explains.
    pub explained_ratio: Vec<f64>,
}

impl Pca {
    /// The top `k` principal components of `data` (one sample per row).
    ///
    /// The covariance `XᵀX / (n−1)` of the centred data is computed exactly
    /// on the host; its top eigenpairs come from
    /// [`LinearOperator::top_eigenpairs`], whose power iterations run on the NPU.
    pub fn fit(npu: &Npu, data: &Matrix64, k: usize) -> Result<Pca> {
        if data.rows() < 2 {
            return Err(Error::Shape("PCA needs at least 2 samples".into()));
        }
        let mean = data.column_means();
        let mut centred = data.clone();
        for row in centred.iter_rows_mut() {
            row.iter_mut().zip(&mean).for_each(|(v, m)| *v -= m);
        }
        let n1 = (data.rows() - 1) as f64;
        let cov = centred.transpose().matmul(&centred)?.map(|v| v / n1);
        let total_var: f64 = (0..cov.rows()).map(|i| cov[(i, i)]).sum();
        let eig = LinearOperator::new(npu, &cov)?.top_eigenpairs(k, 200, 0x9CA)?;
        Ok(Pca {
            mean,
            explained_ratio: eig
                .values
                .iter()
                .map(|v| v / total_var.max(f64::MIN_POSITIVE))
                .collect(),
            explained_variance: eig.values,
            components: eig.vectors,
        })
    }

    /// Projects each row of `data` onto the components (`samples × k`).
    pub fn transform(&self, data: &Matrix64) -> Result<Matrix64> {
        let mut centred = data.clone();
        for row in centred.iter_rows_mut() {
            row.iter_mut().zip(&self.mean).for_each(|(v, m)| *v -= m);
        }
        centred.matmul(&self.components.transpose())
    }
}
