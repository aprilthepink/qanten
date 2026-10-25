use std::collections::BTreeMap;

use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// Named tensors, like a PyTorch `state_dict`.
#[derive(Debug, Clone, Default)]
pub struct StateDict {
    tensors: BTreeMap<String, Tensor>,
}

impl StateDict {
    /// An empty state dict.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds or replaces a tensor; returns the previous one.
    pub fn insert(&mut self, name: impl Into<String>, tensor: Tensor) -> Option<Tensor> {
        self.tensors.insert(name.into(), tensor)
    }

    /// The tensor called `name`, or [`Error::MissingTensor`].
    pub fn get(&self, name: &str) -> Result<&Tensor> {
        self.tensors
            .get(name)
            .ok_or_else(|| Error::MissingTensor(name.to_owned()))
    }

    /// The tensor called `name`, if present.
    pub fn get_opt(&self, name: &str) -> Option<&Tensor> {
        self.tensors.get(name)
    }

    /// Removes and returns a tensor.
    pub fn remove(&mut self, name: &str) -> Option<Tensor> {
        self.tensors.remove(name)
    }

    /// `true` if a tensor called `name` exists.
    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// All tensor names, sorted.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    /// All `(name, tensor)` pairs, sorted by name.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Tensor)> {
        self.tensors.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Number of tensors.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// `true` if there are no tensors.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Only the tensors under `prefix.`, with the prefix stripped.
    pub fn sub(&self, prefix: &str) -> StateDict {
        let p = format!("{prefix}.");
        StateDict {
            tensors: self
                .tensors
                .iter()
                .filter_map(|(k, v)| k.strip_prefix(&p).map(|s| (s.to_owned(), v.clone())))
                .collect(),
        }
    }
}

impl FromIterator<(String, Tensor)> for StateDict {
    fn from_iter<I: IntoIterator<Item = (String, Tensor)>>(iter: I) -> Self {
        Self {
            tensors: iter.into_iter().collect(),
        }
    }
}

impl StateDict {
    /// Reads a weight file by its extension: `.safetensors` (feature
    /// `safetensors`) or `.npz` (feature `numpy`).
    pub fn load_file(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let path = path.as_ref();
        let ext = path
            .extension()
            .map(|e| e.to_string_lossy().to_ascii_lowercase())
            .unwrap_or_default();
        match ext.as_str() {
            #[cfg(feature = "safetensors")]
            "safetensors" => Self::load_safetensors(path),
            #[cfg(feature = "numpy")]
            "npz" => Self::load_npz(path),
            other => Err(Error::Unsupported(format!(
                "cannot read weights from `.{other}` files (supported: .safetensors with the \
                 `safetensors` feature, .npz with the `numpy` feature)"
            ))),
        }
    }
}

#[cfg(feature = "safetensors")]
mod st {
    use half::{bf16, f16};
    use safetensors::{Dtype, SafeTensors, tensor::TensorView};

    use super::*;

    fn convert(name: &str, view: &TensorView<'_>) -> Result<Tensor> {
        let bytes = view.data();
        macro_rules! decode {
            ($ty:ty, $n:expr, $conv:expr) => {
                bytes
                    .chunks_exact($n)
                    .map(|c| {
                        let mut a = [0u8; $n];
                        a.copy_from_slice(c);
                        #[allow(clippy::redundant_closure_call)]
                        ($conv)(<$ty>::from_le_bytes(a))
                    })
                    .collect::<Vec<bf16>>()
            };
        }
        let data = match view.dtype() {
            Dtype::BF16 => decode!(u16, 2, bf16::from_bits),
            Dtype::F16 => decode!(u16, 2, |b| bf16::from_f32(f16::from_bits(b).to_f32())),
            Dtype::F32 => decode!(f32, 4, bf16::from_f32),
            Dtype::F64 => decode!(f64, 8, bf16::from_f64),
            Dtype::I64 => decode!(i64, 8, |v| bf16::from_f64(v as f64)),
            Dtype::I32 => decode!(i32, 4, |v| bf16::from_f64(v as f64)),
            Dtype::I16 => decode!(i16, 2, |v| bf16::from_f32(v as f32)),
            Dtype::I8 => decode!(i8, 1, |v| bf16::from_f32(v as f32)),
            Dtype::U8 => decode!(u8, 1, |v| bf16::from_f32(v as f32)),
            Dtype::BOOL => decode!(u8, 1, |v| bf16::from_f32(if v != 0 { 1.0 } else { 0.0 })),
            other => {
                return Err(Error::Format(format!(
                    "tensor `{name}`: unsupported dtype {other:?}"
                )));
            }
        };
        Tensor::new(data, view.shape().to_vec())
    }

    impl StateDict {
        /// Parses a `.safetensors` file. Any float or int dtype is converted to `bf16`.
        ///
        /// In PyTorch: `safetensors.torch.save_file(model.state_dict(), "model.safetensors")`.
        pub fn from_safetensors_bytes(bytes: &[u8]) -> Result<Self> {
            let st = SafeTensors::deserialize(bytes).map_err(|e| Error::Format(e.to_string()))?;
            st.tensors()
                .into_iter()
                .map(|(name, view)| {
                    let t = convert(&name, &view)?;
                    Ok((name, t))
                })
                .collect()
        }

        /// Reads a `.safetensors` file from disk.
        pub fn load_safetensors(path: impl AsRef<std::path::Path>) -> Result<Self> {
            let bytes = std::fs::read(path)?;
            Self::from_safetensors_bytes(&bytes)
        }

        /// Serialises all tensors as `bf16` safetensors.
        pub fn to_safetensors_bytes(&self) -> Result<Vec<u8>> {
            let raw: Vec<(String, Vec<u8>, Vec<usize>)> = self
                .tensors
                .iter()
                .map(|(k, t)| {
                    let bytes = t
                        .data()
                        .iter()
                        .flat_map(|v| v.to_bits().to_le_bytes())
                        .collect();
                    (k.clone(), bytes, t.shape().to_vec())
                })
                .collect();
            let views = raw
                .iter()
                .map(|(k, b, s)| {
                    TensorView::new(Dtype::BF16, s.clone(), b)
                        .map(|v| (k.clone(), v))
                        .map_err(|e| Error::Format(e.to_string()))
                })
                .collect::<Result<Vec<_>>>()?;
            safetensors::serialize(views, None).map_err(|e| Error::Format(e.to_string()))
        }

        /// Writes all tensors to a `.safetensors` file.
        pub fn save_safetensors(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
            std::fs::write(path, self.to_safetensors_bytes()?)?;
            Ok(())
        }
    }
}
