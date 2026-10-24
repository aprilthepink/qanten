//! Device-resident memory.
//!
//! Uploading weights once and reusing them is the main way to speed up
//! repeated inference: the per-call transfer shrinks to the features.
//!
//! The data role is part of the type, so mixing up weights and features, or
//! downloading weights (which the hardware does not support), fails to compile:
//!
//! ```compile_fail
//! # let npu = qanten::Npu::simulator();
//! let w = qanten::DeviceBuffer::<qanten::Weights>::alloc(&npu, 2, 2).unwrap();
//! w.download(); // no `download` on weights
//! ```

use std::marker::PhantomData;

use half::bf16;

use crate::backend::{DataRole, RawHandle};
use crate::device::Npu;
use crate::error::{Error, Result, ensure_shape};
use crate::tensor::Tensor;

mod sealed {
    pub trait Sealed {}
}

/// Marker trait for [`Weights`] and [`Features`].
pub trait Role: sealed::Sealed + Send + Sync + 'static {
    const ROLE: DataRole;
}

/// Marker: the matrix side of a product. Upload only.
#[derive(Debug, Clone, Copy)]
pub enum Weights {}
/// Marker: vector inputs and results. Upload and download.
#[derive(Debug, Clone, Copy)]
pub enum Features {}

impl sealed::Sealed for Weights {}
impl sealed::Sealed for Features {}
impl Role for Weights {
    const ROLE: DataRole = DataRole::Weights;
}
impl Role for Features {
    const ROLE: DataRole = DataRole::Features;
}

/// A `rows × cols` block of `bf16` values in NPU memory. Freed on drop.
pub struct DeviceBuffer<R: Role> {
    npu: Npu,
    raw: RawHandle,
    rows: usize,
    cols: usize,
    _role: PhantomData<R>,
}

impl<R: Role> std::fmt::Debug for DeviceBuffer<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceBuffer")
            .field("role", &R::ROLE)
            .field("npu", &self.npu.id())
            .field("rows", &self.rows)
            .field("cols", &self.cols)
            .finish()
    }
}

impl<R: Role> DeviceBuffer<R> {
    /// Allocates an uninitialised (zeroed on the simulator) block.
    pub fn alloc(npu: &Npu, rows: usize, cols: usize) -> Result<Self> {
        let raw = npu.backend().allocate(npu.id(), rows, cols, R::ROLE)?;
        Ok(Self {
            npu: npu.clone(),
            raw,
            rows,
            cols,
            _role: PhantomData,
        })
    }

    /// Allocates a block sized for a 2-D tensor and uploads it.
    pub fn from_tensor(npu: &Npu, tensor: &Tensor) -> Result<Self> {
        let [rows, cols] = tensor.dims::<2>()?;
        let buf = Self::alloc(npu, rows, cols)?;
        buf.upload(tensor)?;
        Ok(buf)
    }

    /// Overwrites the block with a tensor of the same `rows × cols` shape.
    pub fn upload(&self, tensor: &Tensor) -> Result<()> {
        ensure_shape!(
            tensor.shape() == [self.rows, self.cols],
            "upload: tensor shape {:?} does not match buffer {}x{}",
            tensor.shape(),
            self.rows,
            self.cols
        );
        self.upload_raw(tensor.data())
    }

    /// Overwrites the block from raw row-major data.
    pub fn upload_raw(&self, data: &[bf16]) -> Result<()> {
        ensure_shape!(
            data.len() == self.rows * self.cols,
            "upload: {} values do not fit buffer {}x{}",
            data.len(),
            self.rows,
            self.cols
        );
        self.npu.backend().upload(self.raw, data)
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn npu(&self) -> &Npu {
        &self.npu
    }

    /// Size in bytes on the device.
    pub fn size_bytes(&self) -> usize {
        self.rows * self.cols * std::mem::size_of::<bf16>()
    }
}

impl DeviceBuffer<Features> {
    /// Copies the block back to the host as a `rows × cols` tensor.
    pub fn download(&self) -> Result<Tensor> {
        let mut out = vec![bf16::ZERO; self.rows * self.cols];
        self.npu.backend().download(self.raw, &mut out)?;
        Tensor::new(out, [self.rows, self.cols])
    }
}

impl<R: Role> Drop for DeviceBuffer<R> {
    fn drop(&mut self) {
        if let Err(err) = self.npu.backend().free(self.raw) {
            tracing::warn!(npu = self.npu.id(), %err, "failed to free device memory");
        }
    }
}

/// `result = vectors × weightsᵀ`, everything resident on the NPU.
///
/// Shapes: `weights` is `out × in`, `vectors` is `batch × in`, `result` is `batch × out`.
pub fn matvec_resident(
    weights: &DeviceBuffer<Weights>,
    vectors: &DeviceBuffer<Features>,
    result: &mut DeviceBuffer<Features>,
) -> Result<()> {
    if !weights.npu.same_device(&vectors.npu) || !weights.npu.same_device(&result.npu) {
        return Err(Error::DeviceMismatch(
            "weights, vectors and result must live on the same NPU".into(),
        ));
    }
    ensure_shape!(
        weights.cols == vectors.cols,
        "weights have {} columns but vectors have {}",
        weights.cols,
        vectors.cols
    );
    ensure_shape!(
        result.rows == vectors.rows && result.cols == weights.rows,
        "result must be {}x{}, got {}x{}",
        vectors.rows,
        weights.rows,
        result.rows,
        result.cols
    );
    weights
        .npu
        .backend()
        .matvec_resident(weights.raw, vectors.raw, result.raw)
}
