//! General matrix and float computation on the NPU.
//!
//! # What the NPU is good at
//!
//! The NPU computes matrix–vector products. A matrix of `r × c` values sits
//! on the device and every vector streamed through it costs `c` values in and
//! `r` out, for `r · c` multiply-accumulates. So the work per byte moved grows
//! with the matrix size, and it pays off when **one matrix is reused for many
//! vectors**: a big batch through a matrix kept on the device ([`NpuMatrix`]),
//! or a large product [`gemm`] where one operand stays put while the other
//! streams through.
//!
//! Elementwise operations ([`mul`], [`cos`], [`sin`]) do one operation per
//! value moved, so a CPU is usually as fast for those on their own. They are
//! here because `cos` is native to the photonic hardware, and for pipelines
//! that keep data near the device.
//!
//! | Use | Type / function |
//! |---|---|
//! | Host matrix, exact CPU product, reference | [`Matrix`] |
//! | Matrix kept on one or more NPUs, tiled and scaled | [`NpuMatrix`] |
//! | `C = A·B` / `y = A·x` that decides NPU vs CPU | [`gemm`], [`gemv`], [`Placement`] |
//! | Cheap accuracy check of a product | [`check_product`] |
//! | Elementwise `cos`, `sin`, `a·b` on slices | [`cos`], [`sin`], [`mul`] |
//!
//! # Precision
//!
//! Inputs and results on the NPU are bfloat16, about 3 significant digits;
//! sums are accumulated at higher precision. [`NpuMatrix`] scales every
//! matrix row and every input vector to unit range before sending it (which
//! matters on analog hardware), and adds tile results on the host in f32.
//! Expect relative errors around `1e-2`–`1e-3` per product, so this is for
//! work where that is enough (simulation, statistics, search, signal
//! processing, Monte Carlo, iterative methods with exact correction; see
//! [`linalg`](crate::linalg) for solvers that get f64 accuracy back).
//! [`check_product`] measures the error of a result cheaply.
//!
//! ```
//! # fn main() -> qanten::Result<()> {
//! use qanten::compute::{Matrix, NpuMatrix, check_product};
//! let npu = qanten::Npu::simulator();
//! let a = Matrix::from_fn(256, 128, |i, j| ((i * 7 + j * 3) % 11) as f32 - 5.0);
//! let x = Matrix::from_fn(1000, 128, |i, j| ((i + j) % 5) as f32 * 0.1);
//!
//! let resident = NpuMatrix::upload(&npu, &a)?; // upload once
//! let y = resident.apply_batch(&x)?;            // 1000 vectors: y[i] = A · x[i]
//! assert_eq!((y.rows(), y.cols()), (1000, 256));
//!
//! // Freivalds-style check on the CPU in O(n²): y must equal x·Aᵀ.
//! let err = check_product(&x, &a.transpose(), &y, 4);
//! assert!(err < 1e-2, "relative error {err}");
//! # Ok(()) }
//! ```

use std::f64::consts::TAU;
use std::fmt;
use std::ops::{Index, IndexMut};

use rayon::prelude::*;

use crate::device::Npu;
use crate::error::{Error, Result};
use crate::ops::{self, LinearWeights};
use crate::pool::NpuPool;
use crate::rng::SplitMix64;
use crate::tensor::Tensor;

// ---------------------------------------------------------------------------
// Host matrix
// ---------------------------------------------------------------------------

/// A dense row-major `f32` matrix on the host.
#[derive(Clone, PartialEq)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

impl fmt::Debug for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Matrix {}x{}", self.rows, self.cols)?;
        if self.rows * self.cols <= 64 {
            for r in 0..self.rows {
                write!(f, "\n  {:?}", self.row(r))?;
            }
        }
        Ok(())
    }
}

impl Matrix {
    /// From row-major data.
    pub fn new(rows: usize, cols: usize, data: Vec<f32>) -> Result<Self> {
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
            data: vec![0.0; rows * cols],
        }
    }

    /// The `n × n` identity.
    pub fn identity(n: usize) -> Self {
        Self::from_fn(n, n, |i, j| if i == j { 1.0 } else { 0.0 })
    }

    /// Element `(i, j)` is `f(i, j)`.
    pub fn from_fn(rows: usize, cols: usize, mut f: impl FnMut(usize, usize) -> f32) -> Self {
        let mut data = Vec::with_capacity(rows * cols);
        for i in 0..rows {
            for j in 0..cols {
                data.push(f(i, j));
            }
        }
        Self { rows, cols, data }
    }

    /// From equal-length rows.
    pub fn from_rows(rows: &[Vec<f32>]) -> Result<Self> {
        let cols = rows.first().map_or(0, Vec::len);
        if rows.iter().any(|r| r.len() != cols) {
            return Err(Error::Shape("rows have different lengths".into()));
        }
        Ok(Self {
            rows: rows.len(),
            cols,
            data: rows.concat(),
        })
    }

    /// Standard normal entries (deterministic for a seed).
    pub fn random_normal(rows: usize, cols: usize, seed: u64) -> Self {
        let mut rng = SplitMix64::new(seed);
        Self::from_fn(rows, cols, |_, _| rng.next_normal())
    }

    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Row-major values.
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// Mutable row-major values.
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// The row-major values.
    pub fn into_vec(self) -> Vec<f32> {
        self.data
    }

    /// Row `i`.
    pub fn row(&self, i: usize) -> &[f32] {
        &self.data[i * self.cols..(i + 1) * self.cols]
    }

    /// The transpose.
    pub fn transpose(&self) -> Matrix {
        let mut t = vec![0.0; self.data.len()];
        for i in 0..self.rows {
            for j in 0..self.cols {
                t[j * self.rows + i] = self.data[i * self.cols + j];
            }
        }
        Matrix {
            rows: self.cols,
            cols: self.rows,
            data: t,
        }
    }

    /// Columns `range` as a new matrix.
    pub fn columns(&self, range: std::ops::Range<usize>) -> Matrix {
        let w = range.len();
        let mut data = Vec::with_capacity(self.rows * w);
        for i in 0..self.rows {
            data.extend_from_slice(&self.row(i)[range.clone()]);
        }
        Matrix {
            rows: self.rows,
            cols: w,
            data,
        }
    }

    /// `self · other` on the CPU (multithreaded, f32 with f64 accumulation).
    pub fn matmul_cpu(&self, other: &Matrix) -> Result<Matrix> {
        if self.cols != other.rows {
            return Err(Error::Shape(format!(
                "matmul: {}x{} · {}x{}",
                self.rows, self.cols, other.rows, other.cols
            )));
        }
        let bt = other.transpose();
        let (k, n) = (self.cols, other.cols);
        let mut out = vec![0.0f32; self.rows * n];
        out.par_chunks_mut(n.max(1))
            .enumerate()
            .for_each(|(i, row)| {
                let a = &self.data[i * k..(i + 1) * k];
                for (j, o) in row.iter_mut().enumerate() {
                    let b = &bt.data[j * k..(j + 1) * k];
                    *o = a
                        .iter()
                        .zip(b)
                        .map(|(x, y)| *x as f64 * *y as f64)
                        .sum::<f64>() as f32;
                }
            });
        Matrix::new(self.rows, n, out)
    }

    /// `self · x` on the CPU.
    pub fn matvec_cpu(&self, x: &[f32]) -> Result<Vec<f32>> {
        if x.len() != self.cols {
            return Err(Error::Shape(format!(
                "matvec: {}x{} · vector of {}",
                self.rows,
                self.cols,
                x.len()
            )));
        }
        Ok((0..self.rows)
            .map(|i| {
                self.row(i)
                    .iter()
                    .zip(x)
                    .map(|(a, b)| *a as f64 * *b as f64)
                    .sum::<f64>() as f32
            })
            .collect())
    }

    /// Frobenius norm.
    pub fn norm(&self) -> f64 {
        self.data
            .iter()
            .map(|v| (*v as f64).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// `‖self − other‖ / ‖other‖` (Frobenius).
    pub fn relative_error(&self, other: &Matrix) -> f64 {
        let diff: f64 = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        diff / other.norm().max(f64::MIN_POSITIVE)
    }

    /// As a `[rows, cols]` bf16 tensor.
    pub fn to_tensor(&self) -> Result<Tensor> {
        Tensor::from_f32(&self.data, [self.rows, self.cols])
    }

    /// From a rank-2 tensor.
    pub fn from_tensor(t: &Tensor) -> Result<Matrix> {
        let [r, c] = t.dims::<2>()?;
        Matrix::new(r, c, t.to_f32_vec())
    }
}

impl Index<(usize, usize)> for Matrix {
    type Output = f32;
    fn index(&self, (i, j): (usize, usize)) -> &f32 {
        &self.data[i * self.cols + j]
    }
}

impl IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut f32 {
        &mut self.data[i * self.cols + j]
    }
}

// ---------------------------------------------------------------------------
// Resident matrix
// ---------------------------------------------------------------------------

/// How to cut an [`NpuMatrix`] into tiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Tiling {
    /// At most this many rows per tile (`None` = no limit). Row blocks are
    /// spread over the NPUs round-robin, so with several NPUs set this or use
    /// [`NpuMatrix::upload_pool`].
    pub max_rows: Option<usize>,
    /// At most this many columns per tile (`None` = no limit). Column tiles
    /// produce partial sums that are added on the host in f32.
    pub max_cols: Option<usize>,
}

/// A tile index and its `k × rows` partial result.
type TileOutput = (usize, Vec<f32>);

#[derive(Debug)]
struct Tile {
    npu: usize,
    row0: usize,
    col0: usize,
    weights: LinearWeights,
    /// Per-row scale that was divided out before upload.
    row_scale: Vec<f32>,
}

/// A matrix kept on one or more NPUs, for streaming many vectors through it.
///
/// Upload once, then call [`apply_batch`](Self::apply_batch) (or
/// [`apply`](Self::apply) / [`matmul`](Self::matmul)) as often as needed:
/// only the vectors move. Large matrices can be split into tiles (to respect
/// device memory) and across NPUs (tiles on different NPUs run in parallel).
#[derive(Debug)]
pub struct NpuMatrix {
    rows: usize,
    cols: usize,
    npus: Vec<Npu>,
    tiles: Vec<Tile>,
}

impl NpuMatrix {
    /// Uploads `a` to `npu` as a single tile.
    pub fn upload(npu: &Npu, a: &Matrix) -> Result<Self> {
        Self::upload_tiled(std::slice::from_ref(npu), a, Tiling::default())
    }

    /// Uploads `a` split into one row block per NPU of `pool`; products run
    /// on all of them in parallel.
    pub fn upload_pool(pool: &NpuPool, a: &Matrix) -> Result<Self> {
        let per = a.rows.div_ceil(pool.len().max(1)).max(1);
        Self::upload_tiled(
            pool.npus(),
            a,
            Tiling {
                max_rows: Some(per),
                max_cols: None,
            },
        )
    }

    /// Uploads `a` cut into tiles of at most `tiling` size, row blocks spread
    /// over `npus` round-robin.
    pub fn upload_tiled(npus: &[Npu], a: &Matrix, tiling: Tiling) -> Result<Self> {
        if npus.is_empty() {
            return Err(Error::InvalidArgument("no NPUs given".into()));
        }
        if a.rows == 0 || a.cols == 0 {
            return Err(Error::Shape("cannot upload an empty matrix".into()));
        }
        let tr = tiling.max_rows.unwrap_or(a.rows).clamp(1, a.rows);
        let tc = tiling.max_cols.unwrap_or(a.cols).clamp(1, a.cols);
        let mut tiles = Vec::new();
        for (block, row0) in (0..a.rows).step_by(tr).enumerate() {
            let rows = tr.min(a.rows - row0);
            let npu = block % npus.len();
            for col0 in (0..a.cols).step_by(tc) {
                let cols = tc.min(a.cols - col0);
                let mut vals = Vec::with_capacity(rows * cols);
                let mut row_scale = Vec::with_capacity(rows);
                for r in row0..row0 + rows {
                    let src = &a.row(r)[col0..col0 + cols];
                    let m = src.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                    let m = if m > 0.0 { m } else { 1.0 };
                    row_scale.push(m);
                    vals.extend(src.iter().map(|v| v / m));
                }
                let weights =
                    LinearWeights::upload(&npus[npu], &Tensor::from_f32(&vals, [rows, cols])?)?;
                tiles.push(Tile {
                    npu,
                    row0,
                    col0,
                    weights,
                    row_scale,
                });
            }
        }
        Ok(Self {
            rows: a.rows,
            cols: a.cols,
            npus: npus.to_vec(),
            tiles,
        })
    }

    /// Rows of the matrix (length of each output vector).
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Columns of the matrix (length of each input vector).
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Number of tiles.
    pub fn tiles(&self) -> usize {
        self.tiles.len()
    }

    /// `y = A · x` for one vector. Prefer [`apply_batch`](Self::apply_batch):
    /// a single vector moves `cols + rows` values for `rows · cols` MACs.
    pub fn apply(&self, x: &[f32]) -> Result<Vec<f32>> {
        Ok(self
            .apply_batch(&Matrix::new(1, x.len(), x.to_vec())?)?
            .into_vec())
    }

    /// `Y[i] = A · X[i]` for every row of `X` (`k × cols` → `k × rows`),
    /// i.e. `Y = X · Aᵀ`. Tiles on different NPUs run in parallel.
    pub fn apply_batch(&self, x: &Matrix) -> Result<Matrix> {
        if x.cols != self.cols {
            return Err(Error::Shape(format!(
                "vectors have {} values, the matrix has {} columns",
                x.cols, self.cols
            )));
        }
        let k = x.rows;
        if k == 0 {
            return Ok(Matrix::zeros(0, self.rows));
        }
        // Run each NPU's tiles on its own thread; collect (tile, k × rows partial).
        let partials: Vec<Result<Vec<TileOutput>>> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..self.npus.len())
                .map(|npu| {
                    s.spawn(move || {
                        self.tiles
                            .iter()
                            .enumerate()
                            .filter(|(_, t)| t.npu == npu)
                            .map(|(i, t)| Ok((i, run_tile(t, x)?)))
                            .collect::<Result<Vec<_>>>()
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| {
                    h.join()
                        .unwrap_or_else(|_| Err(Error::Unsupported("NPU thread panicked".into())))
                })
                .collect()
        });
        let mut y = Matrix::zeros(k, self.rows);
        for part in partials {
            for (i, vals) in part? {
                let t = &self.tiles[i];
                let rows = t.row_scale.len();
                for v in 0..k {
                    let dst = &mut y.data[v * self.rows + t.row0..v * self.rows + t.row0 + rows];
                    for (d, s) in dst.iter_mut().zip(&vals[v * rows..(v + 1) * rows]) {
                        *d += s;
                    }
                }
            }
        }
        Ok(y)
    }

    /// `A · B` with `B` of shape `cols × n`: streams the `n` columns of `B`.
    pub fn matmul(&self, b: &Matrix) -> Result<Matrix> {
        Ok(self.apply_batch(&b.transpose())?.transpose())
    }

    /// `A^steps · x`, keeping `A` on the device between steps (square
    /// matrices only). Each step's input is scaled to unit range before it
    /// goes to the NPU, so growing or shrinking vectors stay accurate in
    /// bf16. Useful for Markov chains, power iteration, linear recurrences
    /// and time-stepping.
    pub fn apply_repeated(&self, x: &[f32], steps: usize) -> Result<Vec<f32>> {
        if self.rows != self.cols {
            return Err(Error::Shape("apply_repeated needs a square matrix".into()));
        }
        let mut v = x.to_vec();
        for _ in 0..steps {
            v = self.apply(&v)?;
        }
        Ok(v)
    }
}

/// Runs the column slice of `x` that belongs to tile `t`; returns `k × t.rows`
/// with the row and vector scales undone.
fn run_tile(t: &Tile, x: &Matrix) -> Result<Vec<f32>> {
    let cols = t.weights.in_features();
    let rows = t.row_scale.len();
    let k = x.rows;
    let mut scaled = Vec::with_capacity(k * cols);
    let mut vscale = Vec::with_capacity(k);
    for v in 0..k {
        let src = &x.row(v)[t.col0..t.col0 + cols];
        let m = src.iter().fold(0.0f32, |m, a| m.max(a.abs()));
        let m = if m > 0.0 { m } else { 1.0 };
        vscale.push(m);
        scaled.extend(src.iter().map(|a| a / m));
    }
    let y = t
        .weights
        .forward(&Tensor::from_f32(&scaled, [k, cols])?)?
        .to_f32_vec();
    let mut out = Vec::with_capacity(k * rows);
    for (v, row) in y.chunks(rows).enumerate() {
        out.extend(row.iter().zip(&t.row_scale).map(|(a, r)| a * r * vscale[v]));
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// Where [`gemm`] and [`gemv`] run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Placement {
    /// NPU when the product is big and reuses its matrix enough (see
    /// [`AutoPolicy`]), CPU otherwise.
    #[default]
    Auto,
    /// Always the NPU.
    Npu,
    /// Always the CPU (exact f32).
    Cpu,
}

/// The rule behind [`Placement::Auto`].
///
/// The NPU wins when there is a lot of arithmetic (`min_macs`) *and* the
/// resident matrix is reused for enough vectors (`min_reuse`). Otherwise the
/// fixed cost of uploading the matrix and calling the driver dominates. The
/// defaults are conservative guesses; on hardware, measure with
/// [`FallbackPolicy::calibrate`](crate::fallback::FallbackPolicy::calibrate)
/// and the `qanten bench` command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AutoPolicy {
    /// Products with fewer multiply-accumulates run on the CPU.
    pub min_macs: u64,
    /// Products that stream fewer vectors through the resident matrix run on the CPU.
    pub min_reuse: usize,
}

impl Default for AutoPolicy {
    fn default() -> Self {
        Self {
            min_macs: 1 << 20,
            min_reuse: 16,
        }
    }
}

impl AutoPolicy {
    /// `true` if an `m × k · k × n` product should run on the NPU.
    pub fn use_npu(&self, m: usize, k: usize, n: usize) -> bool {
        let macs = m as u64 * k as u64 * n as u64;
        // The larger operand stays resident; the other side is the reuse.
        let reuse = if m * k >= k * n { n } else { m };
        macs >= self.min_macs && reuse >= self.min_reuse
    }
}

/// `C = A · B`. With [`Placement::Auto`] the default [`AutoPolicy`] decides
/// between NPU and CPU.
///
/// On the NPU the larger operand is uploaded and kept resident while the
/// other is streamed through it (`A` resident: columns of `B` stream; `B`
/// resident: rows of `A` stream). To reuse a matrix across calls, upload it
/// yourself with [`NpuMatrix`].
pub fn gemm(npu: &Npu, a: &Matrix, b: &Matrix, placement: Placement) -> Result<Matrix> {
    gemm_with(npu, a, b, placement, &AutoPolicy::default())
}

/// [`gemm`] with a custom [`AutoPolicy`].
pub fn gemm_with(
    npu: &Npu,
    a: &Matrix,
    b: &Matrix,
    placement: Placement,
    policy: &AutoPolicy,
) -> Result<Matrix> {
    if a.cols != b.rows {
        return Err(Error::Shape(format!(
            "gemm: {}x{} · {}x{}",
            a.rows, a.cols, b.rows, b.cols
        )));
    }
    let (m, k, n) = (a.rows, a.cols, b.cols);
    let on_npu = match placement {
        Placement::Npu => true,
        Placement::Cpu => false,
        Placement::Auto => policy.use_npu(m, k, n),
    };
    if !on_npu || m == 0 || n == 0 || k == 0 {
        return a.matmul_cpu(b);
    }
    if m * k >= k * n {
        // A resident, stream B's columns.
        NpuMatrix::upload(npu, a)?.matmul(b)
    } else {
        // Bᵀ resident, stream A's rows: C = A · B, row i = Bᵀ · a_i.
        NpuMatrix::upload(npu, &b.transpose())?.apply_batch(a)
    }
}

/// `y = A · x`. A single matrix–vector product never reuses the matrix, so
/// `Auto` runs it on the CPU; upload an [`NpuMatrix`] when the same `A` is
/// applied to many vectors.
pub fn gemv(npu: &Npu, a: &Matrix, x: &[f32], placement: Placement) -> Result<Vec<f32>> {
    match placement {
        Placement::Npu => NpuMatrix::upload(npu, a)?.apply(x),
        Placement::Cpu | Placement::Auto => a.matvec_cpu(x),
    }
}

/// Estimates the relative error of a claimed product `C ≈ A · B` in `O(n²)`
/// on the CPU (Freivalds' method): for `trials` random vectors `v`, compares
/// `A·(B·v)` with `C·v` and returns the largest `‖A·B·v − C·v‖ / ‖A·B·v‖`.
/// A dimension mismatch returns `f64::INFINITY`.
pub fn check_product(a: &Matrix, b: &Matrix, c: &Matrix, trials: usize) -> f64 {
    if a.cols != b.rows || c.rows != a.rows || c.cols != b.cols {
        return f64::INFINITY;
    }
    let mut rng = SplitMix64::new(0x00C0_FFEE);
    let mut worst = 0.0f64;
    let mv = |m: &Matrix, v: &[f64]| -> Vec<f64> {
        (0..m.rows)
            .map(|i| m.row(i).iter().zip(v).map(|(a, b)| *a as f64 * b).sum())
            .collect()
    };
    for _ in 0..trials.max(1) {
        let v: Vec<f64> = (0..b.cols).map(|_| rng.next_normal() as f64).collect();
        let exact = mv(a, &mv(b, &v));
        let claimed = mv(c, &v);
        let num: f64 = exact
            .iter()
            .zip(&claimed)
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt();
        let den = exact.iter().map(|x| x * x).sum::<f64>().sqrt();
        worst = worst.max(num / den.max(f64::MIN_POSITIVE));
    }
    worst
}

// ---------------------------------------------------------------------------
// Elementwise
// ---------------------------------------------------------------------------

/// `a[i] · b[i]` on the NPU.
pub fn mul(npu: &Npu, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
    if a.len() != b.len() {
        return Err(Error::Shape(format!(
            "mul: lengths {} and {} differ",
            a.len(),
            b.len()
        )));
    }
    if a.is_empty() {
        return Ok(Vec::new());
    }
    let n = a.len();
    Ok(ops::mul(npu, &Tensor::from_f32(a, [n])?, &Tensor::from_f32(b, [n])?)?.to_f32_vec())
}

/// `amplitude[i] · cos(x[i] + phase)` with the NPU's native periodic
/// nonlinearity. `x` is reduced to `[−π, π)` on the host in f64 first, so
/// large arguments keep their accuracy.
fn periodic(npu: &Npu, x: &[f32], phase: f64) -> Result<Vec<f32>> {
    if x.is_empty() {
        return Ok(Vec::new());
    }
    let reduced: Vec<f32> = x
        .iter()
        .map(|&v| {
            let t = v as f64 + phase;
            (t - TAU * ((t + std::f64::consts::PI) / TAU).floor()) as f32
        })
        .collect();
    let n = x.len();
    let u = Tensor::from_f32(&reduced, [n])?;
    Ok(ops::periodic_nl(npu, &u, &Tensor::ones([n]))?.to_f32_vec())
}

/// `cos(x[i])` on the NPU (exact cos on the simulator; on hardware the
/// optical transfer curve, so accuracy depends on calibration). About
/// `±0.006` absolute error from the bf16 phase.
pub fn cos(npu: &Npu, x: &[f32]) -> Result<Vec<f32>> {
    periodic(npu, x, 0.0)
}

/// `sin(x[i]) = cos(x[i] − π/2)` on the NPU; see [`cos`].
pub fn sin(npu: &Npu, x: &[f32]) -> Result<Vec<f32>> {
    periodic(npu, x, -std::f64::consts::FRAC_PI_2)
}
