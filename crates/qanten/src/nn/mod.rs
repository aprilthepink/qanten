//! Neural-network layers for inference, PyTorch style.
//!
//! Layers that hold weights upload them to the NPU when constructed, so
//! every `forward` only moves activations. Names and tensor layouts match
//! PyTorch, so a `state_dict` saved with `safetensors.torch.save_file`
//! loads directly:
//!
//! ```
//! # fn main() -> qanten::Result<()> {
//! use qanten::nn::{self, Module};
//! let npu = qanten::Npu::simulator();
//! let mut sd = nn::StateDict::new();
//! sd.insert("0.weight", qanten::Tensor::rand_uniform([16, 4], -0.5, 0.5, 1));
//! sd.insert("0.bias", qanten::Tensor::zeros([16]));
//! sd.insert("2.weight", qanten::Tensor::rand_uniform([3, 16], -0.5, 0.5, 2));
//! sd.insert("2.bias", qanten::Tensor::zeros([3]));
//!
//! let model = nn::Sequential::new()
//!     .add(nn::Linear::load(&npu, &sd, "0")?)
//!     .add(nn::ReLU)
//!     .add(nn::Linear::load(&npu, &sd, "2")?)
//!     .add(nn::Softmax);
//! let probs = model.forward(&qanten::Tensor::rand_uniform([8, 4], 0., 1., 3))?;
//! assert_eq!(probs.shape(), &[8, 3]);
//! # Ok(()) }
//! ```

mod layers;
mod state_dict;

pub use layers::*;
pub use state_dict::StateDict;

use crate::error::Result;
use crate::tensor::Tensor;

/// Anything that maps a tensor to a tensor.
pub trait Module: Send + Sync + std::fmt::Debug {
    /// Runs the layer on `x`.
    fn forward(&self, x: &Tensor) -> Result<Tensor>;

    /// Short layer name used in debug output.
    fn name(&self) -> &'static str {
        "module"
    }
}

impl<M: Module + ?Sized> Module for Box<M> {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        (**self).forward(x)
    }
    fn name(&self) -> &'static str {
        (**self).name()
    }
}

impl<M: Module + ?Sized> Module for std::sync::Arc<M> {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        (**self).forward(x)
    }
    fn name(&self) -> &'static str {
        (**self).name()
    }
}

/// Runs layers one after another.
#[derive(Debug, Default)]
pub struct Sequential {
    layers: Vec<Box<dyn Module>>,
}

impl Sequential {
    /// An empty model.
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a layer (builder style).
    #[allow(clippy::should_implement_trait)]
    pub fn add(mut self, layer: impl Module + 'static) -> Self {
        self.layers.push(Box::new(layer));
        self
    }

    /// Appends a layer.
    pub fn push(&mut self, layer: impl Module + 'static) {
        self.layers.push(Box::new(layer));
    }

    /// Number of layers.
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// `true` if there are no layers.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// The layers, in order.
    pub fn layers(&self) -> &[Box<dyn Module>] {
        &self.layers
    }

    /// Runs the model and returns the output of every layer (for debugging).
    pub fn forward_trace(&self, x: &Tensor) -> Result<Vec<Tensor>> {
        let mut outs = Vec::with_capacity(self.layers.len());
        let mut cur = x.clone();
        for layer in &self.layers {
            cur = layer.forward(&cur)?;
            outs.push(cur.clone());
        }
        Ok(outs)
    }
}

impl Module for Sequential {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut layers = self.layers.iter();
        let Some(first) = layers.next() else {
            return Ok(x.clone());
        };
        let mut cur = first.forward(x)?;
        for layer in layers {
            cur = layer.forward(&cur)?;
        }
        Ok(cur)
    }

    fn name(&self) -> &'static str {
        "Sequential"
    }
}
