//! A small owned n-dimensional tensor of `bf16` values (row-major).
//!
//! The NPU computes in bfloat16, so that is the storage type. Constructors
//! and accessors convert from and to `f32` for convenience.

use std::fmt;

use half::bf16;

use crate::error::{Error, Result, bail_shape, ensure_shape};

/// Owned, contiguous, row-major tensor of `bf16` values with a dynamic rank.
#[derive(Clone, PartialEq)]
pub struct Tensor {
    data: Vec<bf16>,
    shape: Vec<usize>,
}

fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

impl Tensor {
    /// Creates a tensor from `bf16` data. Fails if `data.len()` does not match `shape`.
    pub fn new(data: Vec<bf16>, shape: impl Into<Vec<usize>>) -> Result<Self> {
        let shape = shape.into();
        ensure_shape!(
            data.len() == numel(&shape),
            "data has {} elements but shape {:?} needs {}",
            data.len(),
            shape,
            numel(&shape)
        );
        Ok(Self { data, shape })
    }

    /// Creates a tensor from `f32` data (rounded to the nearest `bf16`).
    pub fn from_f32(data: &[f32], shape: impl Into<Vec<usize>>) -> Result<Self> {
        Self::new(data.iter().map(|&x| bf16::from_f32(x)).collect(), shape)
    }

    /// Creates a tensor from any data that converts to `f64`.
    pub fn from_slice<T: Copy + Into<f64>>(
        data: &[T],
        shape: impl Into<Vec<usize>>,
    ) -> Result<Self> {
        Self::new(
            data.iter().map(|&x| bf16::from_f64(x.into())).collect(),
            shape,
        )
    }

    /// A 1-D tensor holding `data`.
    pub fn vector(data: &[f32]) -> Self {
        Self {
            data: data.iter().map(|&x| bf16::from_f32(x)).collect(),
            shape: vec![data.len()],
        }
    }

    /// Tensor filled with `value`.
    pub fn full(shape: impl Into<Vec<usize>>, value: f32) -> Self {
        let shape = shape.into();
        Self {
            data: vec![bf16::from_f32(value); numel(&shape)],
            shape,
        }
    }

    /// Tensor filled with zeros.
    pub fn zeros(shape: impl Into<Vec<usize>>) -> Self {
        Self::full(shape, 0.0)
    }

    /// Tensor filled with ones.
    pub fn ones(shape: impl Into<Vec<usize>>) -> Self {
        Self::full(shape, 1.0)
    }

    /// Tensor of uniform random values in `[low, high)`, reproducible from `seed`.
    pub fn rand_uniform(shape: impl Into<Vec<usize>>, low: f32, high: f32, seed: u64) -> Self {
        let shape = shape.into();
        let mut rng = crate::rng::SplitMix64::new(seed);
        let data = (0..numel(&shape))
            .map(|_| bf16::from_f32(low + (high - low) * rng.next_f32()))
            .collect();
        Self { data, shape }
    }

    /// The shape (size of each dimension).
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Number of dimensions.
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Total number of elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// `true` if the tensor has no elements.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// The raw `bf16` data in row-major order.
    pub fn data(&self) -> &[bf16] {
        &self.data
    }

    /// Mutable access to the raw data. The length cannot change, so the shape stays valid.
    pub fn data_mut(&mut self) -> &mut [bf16] {
        &mut self.data
    }

    /// Consumes the tensor and returns its data.
    pub fn into_data(self) -> Vec<bf16> {
        self.data
    }

    /// Consumes the tensor and returns `(data, shape)`.
    pub fn into_parts(self) -> (Vec<bf16>, Vec<usize>) {
        (self.data, self.shape)
    }

    /// The data converted to `f32`.
    pub fn to_f32_vec(&self) -> Vec<f32> {
        self.data.iter().map(|x| x.to_f32()).collect()
    }

    /// Element at a multi-dimensional index, as `f32`.
    pub fn get(&self, index: &[usize]) -> Option<f32> {
        if index.len() != self.rank() {
            return None;
        }
        let mut flat = 0;
        for (&i, &d) in index.iter().zip(&self.shape) {
            if i >= d {
                return None;
            }
            flat = flat * d + i;
        }
        self.data.get(flat).map(|x| x.to_f32())
    }

    /// Returns the tensor with a new shape holding the same number of elements.
    /// One dimension may be `usize::MAX` to infer it (like `-1` in PyTorch).
    pub fn reshape(mut self, shape: impl Into<Vec<usize>>) -> Result<Self> {
        let mut shape = shape.into();
        let inferred: Vec<usize> = (0..shape.len())
            .filter(|&i| shape[i] == usize::MAX)
            .collect();
        match inferred.as_slice() {
            [] => {}
            [i] => {
                let known: usize = shape.iter().filter(|&&d| d != usize::MAX).product();
                ensure_shape!(
                    known != 0 && self.len().is_multiple_of(known),
                    "cannot infer dimension: {} elements do not fit {:?}",
                    self.len(),
                    shape
                );
                shape[*i] = self.len() / known;
            }
            _ => bail_shape!("at most one dimension can be inferred, got {:?}", shape),
        }
        ensure_shape!(
            numel(&shape) == self.len(),
            "cannot reshape {:?} ({} elements) into {:?}",
            self.shape,
            self.len(),
            shape
        );
        self.shape = shape;
        Ok(self)
    }

    /// Like [`Tensor::reshape`] but clones the data.
    pub fn reshaped(&self, shape: impl Into<Vec<usize>>) -> Result<Self> {
        self.clone().reshape(shape)
    }

    /// Flattens all dimensions from `start_dim` on into one (like `torch.flatten`).
    pub fn flatten_from(self, start_dim: usize) -> Result<Self> {
        ensure_shape!(
            start_dim < self.rank().max(1),
            "flatten start_dim {start_dim} out of range for rank {}",
            self.rank()
        );
        let mut shape: Vec<usize> = self.shape[..start_dim].to_vec();
        shape.push(self.shape[start_dim..].iter().product());
        self.reshape(shape)
    }

    /// Adds a leading dimension of size 1.
    pub fn unsqueeze0(mut self) -> Self {
        self.shape.insert(0, 1);
        self
    }

    /// Applies `f` to every element (computed in `f32`).
    pub fn map(&self, f: impl Fn(f32) -> f32) -> Self {
        Self {
            data: self
                .data
                .iter()
                .map(|&x| bf16::from_f32(f(x.to_f32())))
                .collect(),
            shape: self.shape.clone(),
        }
    }

    /// Smallest and largest value (`None` for an empty tensor).
    pub fn min_max(&self) -> Option<(f32, f32)> {
        self.data.iter().map(|v| v.to_f32()).fold(None, |acc, x| {
            Some(acc.map_or((x, x), |(lo, hi): (f32, f32)| (lo.min(x), hi.max(x))))
        })
    }

    /// Mean of all values (`0` for an empty tensor).
    pub fn mean(&self) -> f32 {
        self.data.iter().map(|v| v.to_f32()).sum::<f32>() / self.len().max(1) as f32
    }

    /// Index of the maximum along the last dimension, for every row.
    pub fn argmax_last(&self) -> Vec<usize> {
        let Some(&last) = self.shape.last() else {
            return vec![0];
        };
        if last == 0 {
            return Vec::new();
        }
        self.data
            .chunks(last)
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map_or(0, |(i, _)| i)
            })
            .collect()
    }

    /// Largest absolute elementwise difference to `other` (shapes must match).
    pub fn max_abs_diff(&self, other: &Tensor) -> Result<f32> {
        ensure_shape!(
            self.shape == other.shape,
            "shapes differ: {:?} vs {:?}",
            self.shape,
            other.shape
        );
        Ok(self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| (a.to_f32() - b.to_f32()).abs())
            .fold(0.0, f32::max))
    }

    /// Splits along the first dimension into at most `parts` pieces of
    /// near-equal size (sizes differ by at most one; empty pieces are omitted).
    pub fn split_batch(&self, parts: usize) -> Vec<Tensor> {
        let Some(&n) = self.shape.first() else {
            return vec![self.clone()];
        };
        let parts = parts.clamp(1, n.max(1));
        let row = self.len() / n.max(1);
        let (base, extra) = (n / parts, n % parts);
        let mut out = Vec::with_capacity(parts);
        let mut start = 0;
        for p in 0..parts {
            let rows = base + usize::from(p < extra);
            if rows == 0 {
                continue;
            }
            let mut shape = self.shape.clone();
            shape[0] = rows;
            out.push(Tensor {
                data: self.data[start * row..(start + rows) * row].to_vec(),
                shape,
            });
            start += rows;
        }
        if out.is_empty() {
            out.push(self.clone());
        }
        out
    }

    /// Concatenates tensors along the first dimension. All other dimensions must match.
    pub fn concat_batch(parts: &[Tensor]) -> Result<Tensor> {
        let Some(first) = parts.first() else {
            bail_shape!("concat_batch: no tensors given");
        };
        ensure_shape!(
            first.rank() > 0,
            "concat_batch: tensors must have at least one dimension"
        );
        let mut shape = first.shape.clone();
        shape[0] = 0;
        let mut data = Vec::with_capacity(parts.iter().map(Tensor::len).sum());
        for p in parts {
            ensure_shape!(
                p.rank() == first.rank() && p.shape[1..] == first.shape[1..],
                "concat_batch: shape {:?} does not match {:?}",
                p.shape,
                first.shape
            );
            shape[0] += p.shape[0];
            data.extend_from_slice(&p.data);
        }
        Tensor::new(data, shape)
    }

    /// The shape as a fixed-size array, if the rank is `D`.
    pub fn dims<const D: usize>(&self) -> Result<[usize; D]> {
        self.shape.as_slice().try_into().map_err(|_| {
            Error::Shape(format!(
                "expected a {D}-D tensor, got shape {:?}",
                self.shape
            ))
        })
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const MAX: usize = 8;
        write!(f, "Tensor(shape={:?}, data=[", self.shape)?;
        for (i, x) in self.data.iter().take(MAX).enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{x}")?;
        }
        if self.data.len() > MAX {
            write!(f, ", … {} more", self.data.len() - MAX)?;
        }
        write!(f, "])")
    }
}

#[cfg(feature = "ndarray")]
mod ndarray_impl {
    use super::*;
    use ndarray::{ArrayD, IxDyn};

    impl From<&ArrayD<f32>> for Tensor {
        fn from(a: &ArrayD<f32>) -> Self {
            Tensor {
                data: a.iter().map(|&x| bf16::from_f32(x)).collect(),
                shape: a.shape().to_vec(),
            }
        }
    }

    impl From<&ArrayD<bf16>> for Tensor {
        fn from(a: &ArrayD<bf16>) -> Self {
            Tensor {
                data: a.iter().copied().collect(),
                shape: a.shape().to_vec(),
            }
        }
    }

    impl Tensor {
        /// Converts to an `ndarray` array of `f32`.
        pub fn to_ndarray(&self) -> ArrayD<f32> {
            #[allow(clippy::expect_used)]
            ArrayD::from_shape_vec(IxDyn(&self.shape), self.to_f32_vec())
                .expect("tensor invariant: data length matches shape")
        }

        /// Converts to an `ndarray` array of `bf16` without changing values.
        pub fn into_ndarray_bf16(self) -> ArrayD<bf16> {
            #[allow(clippy::expect_used)]
            ArrayD::from_shape_vec(IxDyn(&self.shape), self.data)
                .expect("tensor invariant: data length matches shape")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construct_and_reshape() {
        let t = Tensor::from_f32(&[1., 2., 3., 4., 5., 6.], [2, 3]).unwrap();
        assert_eq!(t.get(&[1, 2]), Some(6.0));
        assert!(Tensor::from_f32(&[1.], [2]).is_err());
        let r = t.clone().reshape([3, usize::MAX]).unwrap();
        assert_eq!(r.shape(), &[3, 2]);
        assert_eq!(t.clone().flatten_from(0).unwrap().shape(), &[6]);
        assert!(t.reshaped([4, 2]).is_err());
    }

    #[test]
    fn argmax() {
        let t = Tensor::from_f32(&[0., 3., 1., 9., 2., 1.], [2, 3]).unwrap();
        assert_eq!(t.argmax_last(), vec![1, 0]);
    }

    #[test]
    fn rand_is_deterministic() {
        let a = Tensor::rand_uniform([4, 4], -1., 1., 7);
        let b = Tensor::rand_uniform([4, 4], -1., 1., 7);
        assert_eq!(a, b);
        assert!(a.to_f32_vec().iter().all(|x| (-1.0..=1.0).contains(x)));
    }
}
