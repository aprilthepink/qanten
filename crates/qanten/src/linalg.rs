//! Linear algebra on the NPU with full-precision results.
//!
//! The NPU multiplies in bfloat16 (about 3 significant digits). The solvers
//! here use **mixed-precision iterative refinement**: the expensive inner
//! iterations (many matrix-vector products) run on the NPU in low precision,
//! and a few exact f64 residuals on the host correct the result. For
//! reasonably conditioned problems that converges to f64-level accuracy,
//! while the NPU does most of the multiply-accumulates.
//!
//! | Function | Problem | Inner work on the NPU |
//! |---|---|---|
//! | [`conjugate_gradient`] | `A·x = b`, `A` symmetric positive definite | CG iterations |
//! | [`least_squares`] | `min ‖A·x − b‖`, any `A` | CGLS iterations (`A` and `Aᵀ` products) |
//! | [`top_eigenpairs`] | largest eigenvalues/vectors of a symmetric `A` | block power iteration |
//! | [`pca`] | principal components of data | block power iteration on the covariance |
//!
//! ```
//! # fn main() -> qanten::Result<()> {
//! use qanten::linalg::{LinearOperator, SolveOptions, conjugate_gradient};
//! let npu = qanten::Npu::simulator();
//! // A small SPD system.
//! let a = [4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0];
//! let op = LinearOperator::new(&npu, &a, 3, 3)?;
//! let (x, report) = conjugate_gradient(&op, &[1.0, 2.0, 3.0], &SolveOptions::default())?;
//! assert!(report.converged && report.relative_residual < 1e-10);
//! # Ok(()) }
//! ```

use crate::device::Npu;
use crate::error::{Error, Result};
use crate::ops::LinearWeights;
use crate::rng::SplitMix64;
use crate::tensor::Tensor;

/// A dense matrix kept twice: resident on the NPU (bf16) for fast approximate
/// products, and on the host in f64 for exact residuals.
#[derive(Debug)]
pub struct LinearOperator {
    rows: usize,
    cols: usize,
    host: Vec<f64>,
    npu: LinearWeights,
    npu_t: Option<LinearWeights>,
    scale: f64,
}

impl LinearOperator {
    /// Uploads the row-major `rows × cols` matrix `a`.
    pub fn new(npu: &Npu, a: &[f64], rows: usize, cols: usize) -> Result<Self> {
        Self::build(npu, a, rows, cols, false)
    }

    /// Like [`new`](Self::new), and also uploads `Aᵀ` (needed by [`least_squares`]).
    pub fn with_transpose(npu: &Npu, a: &[f64], rows: usize, cols: usize) -> Result<Self> {
        Self::build(npu, a, rows, cols, true)
    }

    fn build(npu: &Npu, a: &[f64], rows: usize, cols: usize, transpose: bool) -> Result<Self> {
        if a.len() != rows * cols || rows == 0 || cols == 0 {
            return Err(Error::Shape(format!(
                "matrix has {} values, expected {rows}x{cols}",
                a.len()
            )));
        }
        // Scale entries to unit range before bf16 conversion (keeps the
        // relative error uniform even for tiny or huge matrices).
        let scale = a
            .iter()
            .fold(0.0f64, |m, v| m.max(v.abs()))
            .max(f64::MIN_POSITIVE);
        let scaled: Vec<f32> = a.iter().map(|v| (v / scale) as f32).collect();
        let npu_w = LinearWeights::upload(npu, &Tensor::from_f32(&scaled, [rows, cols])?)?;
        let npu_t = if transpose {
            let mut t = vec![0.0f32; rows * cols];
            for r in 0..rows {
                for c in 0..cols {
                    t[c * rows + r] = scaled[r * cols + c];
                }
            }
            Some(LinearWeights::upload(
                npu,
                &Tensor::from_f32(&t, [cols, rows])?,
            )?)
        } else {
            None
        };
        Ok(Self {
            rows,
            cols,
            host: a.to_vec(),
            npu: npu_w,
            npu_t,
            scale,
        })
    }

    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Approximate `A·x` for several vectors at once (`xs` is `k × cols`), on the NPU.
    pub fn apply_npu(&self, xs: &[f32]) -> Result<Vec<f32>> {
        npu_apply(&self.npu, xs, self.cols, self.scale)
    }

    /// Approximate `Aᵀ·y` for `k × rows` vectors on the NPU (needs [`with_transpose`](Self::with_transpose)).
    pub fn apply_t_npu(&self, ys: &[f32]) -> Result<Vec<f32>> {
        let t = self.npu_t.as_ref().ok_or_else(|| {
            Error::InvalidArgument(
                "operator was built without its transpose (use with_transpose)".into(),
            )
        })?;
        npu_apply(t, ys, self.rows, self.scale)
    }

    /// Exact `A·x` in f64 on the host.
    pub fn apply_exact(&self, x: &[f64]) -> Vec<f64> {
        self.host
            .chunks(self.cols)
            .map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum())
            .collect()
    }

    /// Exact `Aᵀ·y` in f64 on the host.
    pub fn apply_t_exact(&self, y: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.cols];
        for (row, &yr) in self.host.chunks(self.cols).zip(y) {
            for (o, a) in out.iter_mut().zip(row) {
                *o += a * yr;
            }
        }
        out
    }
}

/// Runs `xs` (k × cols) through `w`, normalising each vector to unit range
/// first so bf16 sees O(1) values, then undoing both scalings.
fn npu_apply(w: &LinearWeights, xs: &[f32], cols: usize, scale: f64) -> Result<Vec<f32>> {
    let k = xs.len() / cols.max(1);
    let mut norm = Vec::with_capacity(k);
    let mut scaled = Vec::with_capacity(xs.len());
    for v in xs.chunks(cols) {
        let m = v
            .iter()
            .fold(0.0f32, |a, x| a.max(x.abs()))
            .max(f32::MIN_POSITIVE);
        norm.push(m);
        scaled.extend(v.iter().map(|x| x / m));
    }
    let y = w
        .forward(&Tensor::from_f32(&scaled, [k, cols])?)?
        .to_f32_vec();
    let out_len = y.len() / k.max(1);
    Ok(y.chunks(out_len.max(1))
        .zip(&norm)
        .flat_map(|(row, &m)| {
            row.iter()
                .map(move |v| (*v as f64 * m as f64 * scale) as f32)
        })
        .collect())
}

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

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn dot32(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum()
}

/// Solves `A·x = b` for symmetric positive definite `A`.
///
/// Outer loop: exact residual `r = b − A·x` in f64. Inner loop: conjugate
/// gradient on `A·d = r` with NPU products until `inner_tol`, then `x += d`.
pub fn conjugate_gradient(
    op: &LinearOperator,
    b: &[f64],
    opts: &SolveOptions,
) -> Result<(Vec<f64>, SolveReport)> {
    if op.rows != op.cols || b.len() != op.rows {
        return Err(Error::Shape(format!(
            "CG needs a square matrix and b of length {}",
            op.rows
        )));
    }
    refine(
        b,
        opts,
        |x| op.apply_exact(x),
        |r, iters, tol| inner_cg(op, r, iters, tol),
    )
}

fn inner_cg(op: &LinearOperator, r0: &[f64], iters: usize, tol: f64) -> Result<(Vec<f64>, usize)> {
    let n = r0.len();
    let scale = norm(r0).max(f64::MIN_POSITIVE);
    // Work on the normalised system A·d = r/‖r‖ in f32.
    let rhs: Vec<f32> = r0.iter().map(|v| (v / scale) as f32).collect();
    let mut d = vec![0.0f32; n];
    let mut r = rhs.clone();
    let mut p = rhs;
    let mut rr = dot32(&r, &r);
    let mut used = 0;
    for _ in 0..iters {
        if rr.sqrt() <= tol {
            break;
        }
        let ap = op.apply_npu(&p)?;
        used += 1;
        let pap = dot32(&p, &ap);
        if pap <= 0.0 {
            break; // lost positive definiteness in low precision; let refinement continue
        }
        let alpha = (rr / pap) as f32;
        for i in 0..n {
            d[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        let rr_new = dot32(&r, &r);
        let beta = (rr_new / rr) as f32;
        for i in 0..n {
            p[i] = r[i] + beta * p[i];
        }
        rr = rr_new;
    }
    Ok((d.iter().map(|v| *v as f64 * scale).collect(), used))
}

/// Generic refinement loop shared by the solvers.
fn refine(
    b: &[f64],
    opts: &SolveOptions,
    exact: impl Fn(&[f64]) -> Vec<f64>,
    mut inner: impl FnMut(&[f64], usize, f64) -> Result<(Vec<f64>, usize)>,
) -> Result<(Vec<f64>, SolveReport)> {
    let nb = norm(b).max(f64::MIN_POSITIVE);
    let mut x = vec![0.0; b.len()];
    let mut report = SolveReport {
        converged: false,
        relative_residual: 1.0,
        refinements: 0,
        npu_iterations: 0,
    };
    for _ in 0..=opts.max_refinements {
        let ax = exact(&x);
        let r: Vec<f64> = b.iter().zip(&ax).map(|(b, a)| b - a).collect();
        report.relative_residual = norm(&r) / nb;
        if report.relative_residual <= opts.tol {
            report.converged = true;
            break;
        }
        if report.refinements == opts.max_refinements {
            break;
        }
        let (d, used) = inner(&r, opts.inner_iters, opts.inner_tol)?;
        report.npu_iterations += used;
        for (xi, di) in x.iter_mut().zip(&d) {
            *xi += di;
        }
        report.refinements += 1;
    }
    Ok((x, report))
}

/// Solves `min ‖A·x − b‖₂` for any `rows × cols` `A` (build the operator with
/// [`LinearOperator::with_transpose`]).
///
/// Refinement on the normal equations: the exact residual of `Aᵀ(b − A·x)` is
/// computed in f64; the inner solve is CGLS with `A` and `Aᵀ` products on the NPU.
pub fn least_squares(
    op: &LinearOperator,
    b: &[f64],
    opts: &SolveOptions,
) -> Result<(Vec<f64>, SolveReport)> {
    if b.len() != op.rows {
        return Err(Error::Shape(format!(
            "b has {} values, A has {} rows",
            b.len(),
            op.rows
        )));
    }
    let atb = op.apply_t_exact(b);
    let (x, report) = refine(
        &atb,
        opts,
        |x| op.apply_t_exact(&op.apply_exact(x)),
        |g, iters, tol| inner_cgls_normal(op, g, iters, tol),
    )?;
    Ok((x, report))
}

/// Approximately solves `AᵀA·d = g` with CG, applying `A` then `Aᵀ` on the NPU.
fn inner_cgls_normal(
    op: &LinearOperator,
    g0: &[f64],
    iters: usize,
    tol: f64,
) -> Result<(Vec<f64>, usize)> {
    let n = g0.len();
    let scale = norm(g0).max(f64::MIN_POSITIVE);
    let rhs: Vec<f32> = g0.iter().map(|v| (v / scale) as f32).collect();
    let (mut d, mut r, mut p) = (vec![0.0f32; n], rhs.clone(), rhs);
    let mut rr = dot32(&r, &r);
    let mut used = 0;
    for _ in 0..iters {
        if rr.sqrt() <= tol {
            break;
        }
        let ap = op.apply_npu(&p)?;
        let atap = op.apply_t_npu(&ap)?;
        used += 1;
        let pap = dot32(&ap, &ap);
        if pap <= 0.0 {
            break;
        }
        let alpha = (rr / pap) as f32;
        for i in 0..n {
            d[i] += alpha * p[i];
            r[i] -= alpha * atap[i];
        }
        let rr_new = dot32(&r, &r);
        let beta = (rr_new / rr) as f32;
        for i in 0..n {
            p[i] = r[i] + beta * p[i];
        }
        rr = rr_new;
    }
    Ok((d.iter().map(|v| *v as f64 * scale).collect(), used))
}

/// Eigenvalues and eigenvectors of a symmetric matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct Eigen {
    /// Eigenvalues, largest magnitude first.
    pub values: Vec<f64>,
    /// Matching unit eigenvectors, one row of length `n` each.
    pub vectors: Vec<Vec<f64>>,
    /// Block power iterations on the NPU (each is one batched product).
    pub iterations: usize,
    /// Exact f64 subspace iterations on the host that polished the result.
    pub refinements: usize,
}

/// The `k` eigenpairs of largest magnitude of a symmetric matrix, by block
/// power (subspace) iteration: each step is one batched NPU product of `k + 4`
/// vectors, followed by f64 orthonormalisation and Rayleigh-Ritz on the host.
/// Once the NPU iterations stall at bf16 accuracy, a few exact f64 subspace
/// iterations finish the job (usually 3–10, reported in `refinements`).
pub fn top_eigenpairs(op: &LinearOperator, k: usize, max_iters: usize, seed: u64) -> Result<Eigen> {
    let n = op.rows;
    if op.cols != n {
        return Err(Error::Shape("eigenpairs need a square matrix".into()));
    }
    let k = k.clamp(1, n);
    // Oversample the subspace for faster, more reliable convergence.
    let m = (k + 4).min(n);
    let mut rng = SplitMix64::new(seed);
    let mut q: Vec<Vec<f64>> = (0..m)
        .map(|_| (0..n).map(|_| rng.next_normal() as f64).collect())
        .collect();
    orthonormalize(&mut q);
    let mut prev = vec![0.0; k];
    let mut iterations = 0;
    for _ in 0..max_iters {
        iterations += 1;
        let flat: Vec<f32> = q.iter().flat_map(|v| v.iter().map(|&x| x as f32)).collect();
        let z = op.apply_npu(&flat)?;
        q = z
            .chunks(n)
            .map(|c| c.iter().map(|&v| v as f64).collect())
            .collect();
        orthonormalize(&mut q);
        let (vals, _) = rayleigh_ritz(op, &q);
        let change = vals
            .iter()
            .take(k)
            .zip(&prev)
            .map(|(a, b)| ((a - b) / a.abs().max(1e-300)).abs())
            .fold(0.0, f64::max);
        prev = vals.iter().take(k).copied().collect();
        // bf16 products can't resolve changes much below ~1e-5; hand over to
        // the exact refinement once there.
        if change < 1e-5 {
            break;
        }
    }
    // Refinement: the NPU subspace is accurate to about bf16 precision; a few
    // exact f64 subspace iterations on the host polish it to full precision.
    let mut refinements = 0;
    for _ in 0..30 {
        refinements += 1;
        q = q.iter().map(|v| op.apply_exact(v)).collect();
        orthonormalize(&mut q);
        let (vals, _) = rayleigh_ritz(op, &q);
        let change = vals
            .iter()
            .take(k)
            .zip(&prev)
            .map(|(a, b)| ((a - b) / a.abs().max(1e-300)).abs())
            .fold(0.0, f64::max);
        prev = vals.iter().take(k).copied().collect();
        if change < 1e-14 {
            break;
        }
    }
    let (values, vectors) = rayleigh_ritz(op, &q);
    Ok(Eigen {
        refinements,
        values: values.into_iter().take(k).collect(),
        vectors: vectors.into_iter().take(k).collect(),
        iterations,
    })
}

/// Modified Gram-Schmidt in f64 (twice, for stability).
fn orthonormalize(q: &mut [Vec<f64>]) {
    for _ in 0..2 {
        for i in 0..q.len() {
            for j in 0..i {
                let d: f64 = q[i].iter().zip(&q[j]).map(|(a, b)| a * b).sum();
                let (head, tail) = q.split_at_mut(i);
                for (a, b) in tail[0].iter_mut().zip(&head[j]) {
                    *a -= d * b;
                }
            }
            let nrm = norm(&q[i]);
            if nrm > 0.0 {
                q[i].iter_mut().for_each(|v| *v /= nrm);
            }
        }
    }
}

/// Projects the exact matrix onto span(q), diagonalises the small m×m matrix
/// with Jacobi rotations, and returns Ritz values/vectors sorted by magnitude.
fn rayleigh_ritz(op: &LinearOperator, q: &[Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let m = q.len();
    let aq: Vec<Vec<f64>> = q.iter().map(|v| op.apply_exact(v)).collect();
    let mut h = vec![vec![0.0; m]; m];
    for i in 0..m {
        for j in 0..m {
            h[i][j] = q[i].iter().zip(&aq[j]).map(|(a, b)| a * b).sum();
        }
    }
    let (vals, vecs) = jacobi_eigen(h);
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| vals[b].abs().total_cmp(&vals[a].abs()));
    let n = q[0].len();
    let values = order.iter().map(|&i| vals[i]).collect();
    let vectors = order
        .iter()
        .map(|&i| {
            let mut v = vec![0.0; n];
            for (j, qj) in q.iter().enumerate() {
                for (o, x) in v.iter_mut().zip(qj) {
                    *o += vecs[j][i] * x;
                }
            }
            let nrm = norm(&v).max(f64::MIN_POSITIVE);
            v.iter_mut().for_each(|x| *x /= nrm);
            v
        })
        .collect();
    (values, vectors)
}

/// Cyclic Jacobi eigenvalue algorithm for a small symmetric matrix.
/// Returns `(eigenvalues, eigenvector matrix V)` with eigenvector `i` in column `i`.
#[allow(clippy::needless_range_loop)] // rotation kernel: index form mirrors the math
fn jacobi_eigen(mut a: Vec<Vec<f64>>) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = a.len();
    let mut v: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect();
    for _ in 0..100 {
        let off: f64 = (0..n)
            .flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j)))
            .map(|(i, j)| a[i][j] * a[i][j])
            .sum();
        if off < 1e-30 {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                if a[p][q].abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (akp, akq) = (a[k][p], a[k][q]);
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let (apk, aqk) = (a[p][k], a[q][k]);
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
                for row in v.iter_mut() {
                    let (vkp, vkq) = (row[p], row[q]);
                    row[p] = c * vkp - s * vkq;
                    row[q] = s * vkp + c * vkq;
                }
            }
        }
    }
    ((0..n).map(|i| a[i][i]).collect(), v)
}

/// Principal component analysis result.
#[derive(Debug, Clone, PartialEq)]
pub struct Pca {
    /// Column means that were subtracted.
    pub mean: Vec<f64>,
    /// Principal directions (unit vectors), most variance first.
    pub components: Vec<Vec<f64>>,
    /// Variance along each component.
    pub explained_variance: Vec<f64>,
    /// Fraction of the total variance each component explains.
    pub explained_ratio: Vec<f64>,
}

impl Pca {
    /// Projects `samples × features` rows onto the components → `samples × k`.
    pub fn transform(&self, data: &[f64]) -> Vec<f64> {
        let d = self.mean.len();
        data.chunks(d)
            .flat_map(|row| {
                self.components.iter().map(move |c| {
                    row.iter()
                        .zip(&self.mean)
                        .zip(c)
                        .map(|((x, m), w)| (x - m) * w)
                        .sum::<f64>()
                })
            })
            .collect()
    }
}

/// The top `k` principal components of `data` (`samples × features`, row-major).
///
/// The covariance matrix `XᵀX / (n−1)` is computed exactly on the host; its
/// top eigenpairs come from [`top_eigenpairs`], whose power iterations run on
/// the NPU.
pub fn pca(npu: &Npu, data: &[f64], samples: usize, features: usize, k: usize) -> Result<Pca> {
    if data.len() != samples * features || samples < 2 {
        return Err(Error::Shape(format!(
            "data must be {samples}x{features} with at least 2 samples"
        )));
    }
    let mean: Vec<f64> = (0..features)
        .map(|j| (0..samples).map(|i| data[i * features + j]).sum::<f64>() / samples as f64)
        .collect();
    // Centred data, transposed: Xᵀ is features × samples.
    let mut xt = vec![0.0f64; features * samples];
    for i in 0..samples {
        for j in 0..features {
            xt[j * samples + i] = data[i * features + j] - mean[j];
        }
    }
    let total_var: f64 = xt.iter().map(|v| v * v).sum::<f64>() / (samples - 1) as f64;
    let mut cov = vec![0.0f64; features * features];
    for a in 0..features {
        for b in a..features {
            let v: f64 = xt[a * samples..(a + 1) * samples]
                .iter()
                .zip(&xt[b * samples..(b + 1) * samples])
                .map(|(x, y)| x * y)
                .sum::<f64>()
                / (samples - 1) as f64;
            cov[a * features + b] = v;
            cov[b * features + a] = v;
        }
    }
    let op = LinearOperator::new(npu, &cov, features, features)?;
    let eig = top_eigenpairs(&op, k, 200, 0x9CA)?;
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
