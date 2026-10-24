use std::fmt;

use super::{Module, StateDict};
use crate::device::Npu;
use crate::error::Result;
use crate::ops::{
    self, Conv2dParams, ConvTranspose2dParams, ConvTransposeWeights, ConvWeights, Hw,
    LinearWeights, Pool2dParams,
};
use crate::tensor::Tensor;

fn key(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_owned()
    } else {
        format!("{prefix}.{name}")
    }
}

/// Fully connected layer, `y = x Wᵀ + b`. Weights stay on the NPU.
#[derive(Debug)]
pub struct Linear {
    weights: LinearWeights,
    bias: Option<Tensor>,
}

impl Linear {
    /// `weight` is `[out, in]`, `bias` is `[out]`.
    pub fn new(npu: &Npu, weight: &Tensor, bias: Option<Tensor>) -> Result<Self> {
        let weights = LinearWeights::upload(npu, weight)?;
        if let Some(b) = &bias {
            crate::error::ensure_shape!(
                b.len() == weights.out_features(),
                "Linear: bias has {} elements, expected {}",
                b.len(),
                weights.out_features()
            );
        }
        Ok(Self { weights, bias })
    }

    /// Loads `{prefix}.weight` and optional `{prefix}.bias`.
    pub fn load(npu: &Npu, sd: &StateDict, prefix: &str) -> Result<Self> {
        Self::new(
            npu,
            sd.get(&key(prefix, "weight"))?,
            sd.get_opt(&key(prefix, "bias")).cloned(),
        )
    }

    pub fn in_features(&self) -> usize {
        self.weights.in_features()
    }

    pub fn out_features(&self) -> usize {
        self.weights.out_features()
    }
}

impl Module for Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.weights.forward(x)?;
        match &self.bias {
            Some(b) => ops::add_bias(&y, b),
            None => Ok(y),
        }
    }
    fn name(&self) -> &'static str {
        "Linear"
    }
}

/// 2-D convolution. Weights stay on the NPU.
#[derive(Debug)]
pub struct Conv2d {
    weights: ConvWeights,
    bias: Option<Tensor>,
    params: Conv2dParams,
}

impl Conv2d {
    /// `weight` is `[out, in / groups, kh, kw]`, `bias` is `[out]`.
    pub fn new(
        npu: &Npu,
        weight: &Tensor,
        bias: Option<Tensor>,
        params: Conv2dParams,
    ) -> Result<Self> {
        Ok(Self {
            weights: ConvWeights::upload(npu, weight, params.groups)?,
            bias,
            params,
        })
    }

    /// Loads `{prefix}.weight` and optional `{prefix}.bias`.
    pub fn load(npu: &Npu, sd: &StateDict, prefix: &str, params: Conv2dParams) -> Result<Self> {
        Self::new(
            npu,
            sd.get(&key(prefix, "weight"))?,
            sd.get_opt(&key(prefix, "bias")).cloned(),
            params,
        )
    }

    pub fn params(&self) -> &Conv2dParams {
        &self.params
    }
}

impl Module for Conv2d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.weights.forward(x, &self.params)?;
        match &self.bias {
            Some(b) => ops::add_channel_bias(&y, b),
            None => Ok(y),
        }
    }
    fn name(&self) -> &'static str {
        "Conv2d"
    }
}

/// 2-D transposed convolution. Weights stay on the NPU.
#[derive(Debug)]
pub struct ConvTranspose2d {
    weights: ConvTransposeWeights,
    bias: Option<Tensor>,
    params: ConvTranspose2dParams,
}

impl ConvTranspose2d {
    /// `weight` is `[in, out / groups, kh, kw]`, `bias` is `[out]`.
    pub fn new(
        npu: &Npu,
        weight: &Tensor,
        bias: Option<Tensor>,
        params: ConvTranspose2dParams,
    ) -> Result<Self> {
        Ok(Self {
            weights: ConvTransposeWeights::upload(npu, weight, params.groups)?,
            bias,
            params,
        })
    }

    pub fn load(
        npu: &Npu,
        sd: &StateDict,
        prefix: &str,
        params: ConvTranspose2dParams,
    ) -> Result<Self> {
        Self::new(
            npu,
            sd.get(&key(prefix, "weight"))?,
            sd.get_opt(&key(prefix, "bias")).cloned(),
            params,
        )
    }
}

impl Module for ConvTranspose2d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.weights.forward(x, &self.params)?;
        match &self.bias {
            Some(b) => ops::add_channel_bias(&y, b),
            None => Ok(y),
        }
    }
    fn name(&self) -> &'static str {
        "ConvTranspose2d"
    }
}

/// Inference-mode batch normalisation for `NCHW` tensors.
#[derive(Debug)]
pub struct BatchNorm2d {
    npu: Npu,
    running_mean: Tensor,
    running_var: Tensor,
    weight: Option<Tensor>,
    bias: Option<Tensor>,
    eps: f32,
}

impl BatchNorm2d {
    pub fn new(
        npu: &Npu,
        running_mean: Tensor,
        running_var: Tensor,
        weight: Option<Tensor>,
        bias: Option<Tensor>,
        eps: f32,
    ) -> Self {
        Self {
            npu: npu.clone(),
            running_mean,
            running_var,
            weight,
            bias,
            eps,
        }
    }

    /// Loads `{prefix}.running_mean`, `.running_var` and optional `.weight` / `.bias`.
    /// PyTorch's default `eps` is `1e-5`.
    pub fn load(npu: &Npu, sd: &StateDict, prefix: &str, eps: f32) -> Result<Self> {
        Ok(Self::new(
            npu,
            sd.get(&key(prefix, "running_mean"))?.clone(),
            sd.get(&key(prefix, "running_var"))?.clone(),
            sd.get_opt(&key(prefix, "weight")).cloned(),
            sd.get_opt(&key(prefix, "bias")).cloned(),
            eps,
        ))
    }
}

impl Module for BatchNorm2d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        ops::batch_norm2d(
            &self.npu,
            x,
            &self.running_mean,
            &self.running_var,
            self.weight.as_ref(),
            self.bias.as_ref(),
            self.eps,
        )
    }
    fn name(&self) -> &'static str {
        "BatchNorm2d"
    }
}

/// Layer normalisation over the last dimension.
#[derive(Debug, Clone)]
pub struct LayerNorm {
    pub weight: Option<Tensor>,
    pub bias: Option<Tensor>,
    pub eps: f32,
}

impl LayerNorm {
    pub fn load(sd: &StateDict, prefix: &str, eps: f32) -> Self {
        Self {
            weight: sd.get_opt(&key(prefix, "weight")).cloned(),
            bias: sd.get_opt(&key(prefix, "bias")).cloned(),
            eps,
        }
    }
}

impl Module for LayerNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        ops::layer_norm(x, self.weight.as_ref(), self.bias.as_ref(), self.eps)
    }
    fn name(&self) -> &'static str {
        "LayerNorm"
    }
}

/// Photonic KAN layer (see [`ops::kan_layer`]).
#[derive(Debug)]
pub struct Kan {
    npu: Npu,
    pub phis: Tensor,
    pub ampls: Tensor,
    pub ks: Tensor,
}

impl Kan {
    pub fn new(npu: &Npu, phis: Tensor, ampls: Tensor, ks: Tensor) -> Self {
        Self {
            npu: npu.clone(),
            phis,
            ampls,
            ks,
        }
    }

    /// Loads `{prefix}.phis`, `{prefix}.ampls` and `{prefix}.ks`.
    pub fn load(npu: &Npu, sd: &StateDict, prefix: &str) -> Result<Self> {
        Ok(Self::new(
            npu,
            sd.get(&key(prefix, "phis"))?.clone(),
            sd.get(&key(prefix, "ampls"))?.clone(),
            sd.get(&key(prefix, "ks"))?.clone(),
        ))
    }
}

impl Module for Kan {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        ops::kan_layer(&self.npu, x, &self.phis, &self.ampls, &self.ks)
    }
    fn name(&self) -> &'static str {
        "Kan"
    }
}

/// Max pooling.
#[derive(Debug, Clone, Copy)]
pub struct MaxPool2d(pub Pool2dParams);

impl MaxPool2d {
    pub fn new(kernel: impl Into<Hw>) -> Self {
        Self(Pool2dParams::new(kernel))
    }
}

impl Module for MaxPool2d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        ops::max_pool2d(x, self.0)
    }
    fn name(&self) -> &'static str {
        "MaxPool2d"
    }
}

/// Average pooling.
#[derive(Debug, Clone, Copy)]
pub struct AvgPool2d(pub Pool2dParams);

impl AvgPool2d {
    pub fn new(kernel: impl Into<Hw>) -> Self {
        Self(Pool2dParams::new(kernel))
    }
}

impl Module for AvgPool2d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        ops::avg_pool2d(x, self.0)
    }
    fn name(&self) -> &'static str {
        "AvgPool2d"
    }
}

/// Adaptive average pooling to a fixed output size.
#[derive(Debug, Clone, Copy)]
pub struct AdaptiveAvgPool2d(pub Hw);

impl AdaptiveAvgPool2d {
    pub fn new(output: impl Into<Hw>) -> Self {
        Self(output.into())
    }
}

impl Module for AdaptiveAvgPool2d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        ops::adaptive_avg_pool2d(x, self.0)
    }
    fn name(&self) -> &'static str {
        "AdaptiveAvgPool2d"
    }
}

/// Flattens dimensions from `start_dim` on (PyTorch default: 1).
#[derive(Debug, Clone, Copy)]
pub struct Flatten {
    pub start_dim: usize,
}

impl Default for Flatten {
    fn default() -> Self {
        Self { start_dim: 1 }
    }
}

impl Module for Flatten {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.clone().flatten_from(self.start_dim)
    }
    fn name(&self) -> &'static str {
        "Flatten"
    }
}

macro_rules! activation {
    ($(#[$doc:meta])* $name:ident, $f:expr) => {
        $(#[$doc])*
        #[derive(Debug, Clone, Copy, Default)]
        pub struct $name;

        impl Module for $name {
            fn forward(&self, x: &Tensor) -> Result<Tensor> {
                #[allow(clippy::redundant_closure_call)]
                ($f)(x)
            }
            fn name(&self) -> &'static str {
                stringify!($name)
            }
        }
    };
}

activation!(
    /// `max(x, 0)`.
    ReLU, |x| Ok(ops::relu(x))
);
activation!(
    /// Logistic sigmoid.
    Sigmoid, |x| Ok(ops::sigmoid(x))
);
activation!(
    /// Hyperbolic tangent.
    Tanh, |x| Ok(ops::tanh(x))
);
activation!(
    /// GELU (tanh approximation).
    Gelu, |x| Ok(ops::gelu(x))
);
activation!(
    /// SiLU / swish.
    Silu, |x| Ok(ops::silu(x))
);
activation!(
    /// Softmax over the last dimension.
    Softmax, ops::softmax
);

/// Leaky ReLU with a configurable negative slope.
#[derive(Debug, Clone, Copy)]
pub struct LeakyReLU(pub f32);

impl Module for LeakyReLU {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(ops::leaky_relu(x, self.0))
    }
    fn name(&self) -> &'static str {
        "LeakyReLU"
    }
}

/// Wraps a closure as a layer, for custom glue between layers.
pub struct Lambda<F> {
    name: &'static str,
    f: F,
}

impl<F> Lambda<F>
where
    F: Fn(&Tensor) -> Result<Tensor> + Send + Sync,
{
    pub fn new(name: &'static str, f: F) -> Self {
        Self { name, f }
    }
}

impl<F> fmt::Debug for Lambda<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Lambda({})", self.name)
    }
}

impl<F> Module for Lambda<F>
where
    F: Fn(&Tensor) -> Result<Tensor> + Send + Sync,
{
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        (self.f)(x)
    }
    fn name(&self) -> &'static str {
        self.name
    }
}
