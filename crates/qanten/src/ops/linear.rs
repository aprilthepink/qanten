use half::bf16;

use crate::device::Npu;
use crate::error::{Error, Result, ensure_shape};
use crate::memory::{DeviceBuffer, Features, Weights, matvec_resident};
use crate::tensor::Tensor;

/// Batched matrix-vector product on the NPU: `out[b, r] = Σ_c matrix[r, c] · vectors[b, c]`.
///
/// `matrix` is `[rows, cols]`, `vectors` is `[batch, cols]` (or `[cols]`),
/// the result is `[batch, rows]` (or `[rows]`).
pub fn matvec(npu: &Npu, matrix: &Tensor, vectors: &Tensor) -> Result<Tensor> {
    let [rows, cols] = matrix.dims::<2>()?;
    let (batch, single) = batch_dims("matvec", vectors, cols)?;
    let mut out = vec![bf16::ZERO; batch * rows];
    npu.backend().matvec_batched(
        npu.id(),
        matrix.data(),
        vectors.data(),
        &mut out,
        rows,
        cols,
    )?;
    if single {
        Tensor::new(out, [rows])
    } else {
        Tensor::new(out, [batch, rows])
    }
}

/// Splits a `[cols]` or `[batch, cols]` input into `(batch, is_1d)`.
pub(crate) fn batch_dims(op: &str, x: &Tensor, cols: usize) -> Result<(usize, bool)> {
    let (batch, single, c) = match x.shape() {
        [c] => (1, true, *c),
        [b, c] => (*b, false, *c),
        s => {
            return Err(Error::Shape(format!(
                "{op}: input must be 1-D or 2-D, got {s:?}"
            )));
        }
    };
    ensure_shape!(c == cols, "{op}: expected {cols} input features, got {c}");
    Ok((batch, single))
}

/// Matrix product `a @ b` on the NPU, `[m, k] @ [k, n] -> [m, n]`.
///
/// Transposes `b` on the host first. For repeated products with the same
/// right-hand side, use [`LinearWeights`] with the transposed matrix.
pub fn matmul(npu: &Npu, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let [m, k] = a.dims::<2>()?;
    let [k2, n] = b.dims::<2>()?;
    ensure_shape!(
        k == k2,
        "matmul: inner dimensions differ ({m}x{k} @ {k2}x{n})"
    );
    let bt = transpose2d(b.data(), k, n);
    matvec(npu, &Tensor::new(bt, [n, k])?, a)
}

pub(crate) fn transpose2d(data: &[bf16], rows: usize, cols: usize) -> Vec<bf16> {
    let mut out = vec![bf16::ZERO; data.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

/// Fully connected layer: `x @ weightᵀ (+ bias)`, PyTorch's `F.linear`.
///
/// `x` is `[batch, in]` or `[in]`, `weight` is `[out, in]`, `bias` is `[out]`.
pub fn linear(npu: &Npu, x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    let y = matvec(npu, weight, x)?;
    match bias {
        Some(b) => super::add_bias(&y, b),
        None => Ok(y),
    }
}

/// Linear weights uploaded once and kept in NPU memory.
#[derive(Debug)]
pub struct LinearWeights {
    buffer: DeviceBuffer<Weights>,
}

impl LinearWeights {
    /// Uploads a `[out, in]` weight matrix to `npu`.
    pub fn upload(npu: &Npu, weight: &Tensor) -> Result<Self> {
        Ok(Self {
            buffer: DeviceBuffer::from_tensor(npu, weight)?,
        })
    }

    /// Output size (matrix rows).
    pub fn out_features(&self) -> usize {
        self.buffer.rows()
    }

    /// Input size (matrix columns).
    pub fn in_features(&self) -> usize {
        self.buffer.cols()
    }

    /// The NPU holding the weights.
    pub fn npu(&self) -> &Npu {
        self.buffer.npu()
    }

    /// `x @ weightᵀ` with `x` of shape `[batch, in]` or `[in]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, single) = batch_dims("linear", x, self.in_features())?;
        let y = self.forward_rows(x.data(), batch)?;
        if single {
            y.reshape([self.out_features()])
        } else {
            Ok(y)
        }
    }

    /// Multiplies `batch` row-major input rows; returns `[batch, out]`.
    pub(crate) fn forward_rows(&self, rows: &[bf16], batch: usize) -> Result<Tensor> {
        let npu = self.npu();
        let input = DeviceBuffer::<Features>::alloc(npu, batch, self.in_features())?;
        input.upload_raw(rows)?;
        let mut out = DeviceBuffer::<Features>::alloc(npu, batch, self.out_features())?;
        matvec_resident(&self.buffer, &input, &mut out)?;
        out.download()
    }
}
