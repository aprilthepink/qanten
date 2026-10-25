//! Glue between the graphics modules and [`qanten::compute`].

use qanten::compute::{Matrix, NpuMatrix};
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

/// A constant matrix applied to batches of row vectors: `Y = X · Aᵀ`.
///
/// It runs exactly on the CPU until [`upload`](Self::upload) puts it on the
/// NPU(s) as a [`NpuMatrix`], and keeps the host copy as the reference for
/// the module's `*_cpu` methods and for self-checks. `NpuMatrix` scales every
/// matrix row and input vector to unit range before it reaches the NPU, so
/// very large or small values (HDR radiance, world-space offsets) keep their
/// relative precision.
#[derive(Debug)]
pub(crate) struct MatrixOp {
    host: Matrix,
    resident: Option<NpuMatrix>,
}

impl MatrixOp {
    pub(crate) fn new(host: Matrix) -> Self {
        Self {
            host,
            resident: None,
        }
    }

    pub(crate) fn upload(&mut self, target: Target<'_>) -> Result<()> {
        self.resident = Some(match target {
            Target::Npu(npu) => NpuMatrix::upload(npu, &self.host)?,
            Target::Pool(pool) => NpuMatrix::upload_replicated(pool, &self.host)?,
        });
        Ok(())
    }

    pub(crate) fn is_accelerated(&self) -> bool {
        self.resident.is_some()
    }

    /// Every row of `x` through the matrix, on the NPU if uploaded.
    pub(crate) fn apply(&self, x: &Matrix) -> Result<Matrix> {
        match &self.resident {
            Some(m) => m.apply_batch(x),
            None => self.apply_cpu(x),
        }
    }

    /// Every row of `x` through the matrix, exactly on the CPU.
    pub(crate) fn apply_cpu(&self, x: &Matrix) -> Result<Matrix> {
        x.matmul_transposed(&self.host)
    }

    /// Relative error of the NPU product on random vectors, measured on the
    /// CPU with [`Matrix::product_error`] in `O(rows · cols)`; `None` before upload.
    pub(crate) fn self_check(&self) -> Result<Option<f64>> {
        let Some(resident) = &self.resident else {
            return Ok(None);
        };
        let x = Matrix::random_normal(16, self.host.cols(), 1);
        let y = resident.apply_batch(&x)?;
        Ok(Some(y.product_error(&x, &self.host.transpose(), 4)))
    }
}

/// A graphics type whose heavy lifting is one or more [`MatrixOp`]s.
pub(crate) trait Accelerated: Sized {
    fn ops(&self) -> Vec<&MatrixOp>;
    fn ops_mut(&mut self) -> Vec<&mut MatrixOp>;
    /// Builds the ops that exist only on the NPU path (default: none).
    fn prepare(&mut self) {}

    fn upload_to(mut self, target: Target<'_>) -> Result<Self> {
        self.prepare();
        for op in self.ops_mut() {
            op.upload(target)?;
        }
        Ok(self)
    }

    /// Worst self-check error over all ops; `None` on the CPU path.
    fn worst_check(&self) -> Result<Option<f64>> {
        let mut worst: Option<f64> = None;
        for op in self.ops() {
            if let Some(e) = op.self_check()? {
                worst = Some(worst.map_or(e, |w| w.max(e)));
            }
        }
        Ok(worst)
    }
}

/// The public `with_npu` / `with_pool` / `self_check` / `is_accelerated`
/// methods every graphics type shares.
macro_rules! accelerated_methods {
    ($ty:ty) => {
        impl $ty {
            /// Uploads the constant matrices to `npu`; later calls run there.
            pub fn with_npu(self, npu: &$crate::Npu) -> $crate::Result<Self> {
                $crate::accel::Accelerated::upload_to(self, $crate::accel::Target::Npu(npu))
            }

            /// Like [`with_npu`](Self::with_npu), but with a full copy on every
            /// NPU of `pool`; each batch is split between them, so they all work
            /// at once.
            pub fn with_pool(self, pool: &qanten::pool::NpuPool) -> $crate::Result<Self> {
                $crate::accel::Accelerated::upload_to(self, $crate::accel::Target::Pool(pool))
            }

            /// `true` after [`with_npu`](Self::with_npu) or [`with_pool`](Self::with_pool).
            pub fn is_accelerated(&self) -> bool {
                $crate::accel::Accelerated::ops(self)
                    .iter()
                    .any(|op| op.is_accelerated())
            }

            /// NPU self-test: relative error of the resident matrices on random
            /// vectors, checked on the CPU with
            /// [`Matrix::product_error`](qanten::compute::Matrix::product_error).
            /// `None` on the CPU path. Run it now and then on hardware: a rising
            /// value means the NPU needs recalibration.
            pub fn self_check(&self) -> $crate::Result<Option<f64>> {
                $crate::accel::Accelerated::worst_check(self)
            }
        }
    };
}

pub(crate) use accelerated_methods;
