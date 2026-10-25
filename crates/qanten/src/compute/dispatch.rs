//! Choosing between NPU and CPU, and checking results.

use crate::device::Npu;
use crate::error::{Error, Result};
use crate::rng::SplitMix64;

use super::matrix::{Mat, Matrix, Scalar};
use super::resident::NpuMatrix;

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
        let reuse = if m >= n { n } else { m };
        macs >= self.min_macs && reuse >= self.min_reuse
    }
}

/// Where a product runs.
#[derive(Debug, Clone, Copy)]
pub enum Placement<'a> {
    /// On the CPU, exactly.
    Cpu,
    /// Always on this NPU.
    Npu(&'a Npu),
    /// On this NPU if the policy says the product is worth it, else the CPU.
    Auto(&'a Npu, AutoPolicy),
}

impl<'a> Placement<'a> {
    /// [`Placement::Auto`] with the default [`AutoPolicy`].
    pub fn auto(npu: &'a Npu) -> Self {
        Placement::Auto(npu, AutoPolicy::default())
    }

    /// The NPU an `m × k · k × n` product should run on, if any.
    pub fn choose(&self, m: usize, k: usize, n: usize) -> Option<&'a Npu> {
        match *self {
            Placement::Cpu => None,
            Placement::Npu(npu) => Some(npu),
            Placement::Auto(npu, policy) => policy.use_npu(m, k, n).then_some(npu),
        }
    }
}

impl Mat<f32> {
    /// `self · other`, on the NPU or the CPU as `placement` decides.
    ///
    /// On the NPU the larger operand is uploaded and kept resident while the
    /// other is streamed through it. To reuse a matrix across calls, upload
    /// it yourself with [`NpuMatrix`].
    pub fn matmul_on(&self, other: &Matrix, placement: Placement<'_>) -> Result<Matrix> {
        if self.cols() != other.rows() {
            return Err(Error::Shape(format!(
                "matmul: {}x{} · {}x{}",
                self.rows(),
                self.cols(),
                other.rows(),
                other.cols()
            )));
        }
        let (m, k, n) = (self.rows(), self.cols(), other.cols());
        match placement.choose(m, k, n) {
            None => self.matmul(other),
            _ if m == 0 || k == 0 || n == 0 => self.matmul(other),
            // A resident, stream B's columns.
            Some(npu) if m >= n => NpuMatrix::upload(npu, self)?.matmul(other),
            // Bᵀ resident, stream A's rows: row i of C = Bᵀ · aᵢ.
            Some(npu) => NpuMatrix::upload(npu, &other.transpose())?.apply_batch(self),
        }
    }

    /// `self · x`. A single matrix–vector product never reuses the matrix,
    /// so only [`Placement::Npu`] sends it to the NPU; upload an
    /// [`NpuMatrix`] when the same matrix is applied to many vectors.
    pub fn matvec_on(&self, x: &[f32], placement: Placement<'_>) -> Result<Vec<f32>> {
        match placement {
            Placement::Npu(npu) => NpuMatrix::upload(npu, self)?.apply(x),
            Placement::Cpu | Placement::Auto(..) => self.matvec(x),
        }
    }
}

impl<T: Scalar> Mat<T> {
    /// How far `self` is from the exact product `a · b`, measured in `O(n²)`
    /// on the CPU (Freivalds' method): for `trials` random vectors `v`,
    /// compares `a·(b·v)` with `self·v` and returns the largest
    /// `‖a·b·v − self·v‖ / ‖a·b·v‖`. Mismatched shapes give `f64::INFINITY`.
    pub fn product_error(&self, a: &Mat<T>, b: &Mat<T>, trials: usize) -> f64 {
        if a.cols() != b.rows() || self.shape() != (a.rows(), b.cols()) {
            return f64::INFINITY;
        }
        let mut rng = SplitMix64::new(0x00C0_FFEE);
        let (a, b, c) = (a.cast::<f64>(), b.cast::<f64>(), self.cast::<f64>());
        (0..trials.max(1))
            .map(|_| {
                let v: Vec<f64> = (0..b.cols()).map(|_| rng.next_normal() as f64).collect();
                let bv = b.matvec(&v).unwrap_or_default();
                let exact = a.matvec(&bv).unwrap_or_default();
                let claimed = c.matvec(&v).unwrap_or_default();
                let diff = exact
                    .iter()
                    .zip(&claimed)
                    .map(|(x, y)| (x - y).powi(2))
                    .sum::<f64>();
                let size = exact.iter().map(|x| x * x).sum::<f64>();
                (diff / size.max(f64::MIN_POSITIVE)).sqrt()
            })
            .fold(0.0, f64::max)
    }
}
