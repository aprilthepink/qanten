//! Building MLPs straight from weight files ([`Sequential::load_mlp`]).
//!
//! Every 2-D tensor is a Linear layer `[out, in]`, ordered by the first number
//! in its name (`0.weight`, `fc1.weight`, `l1_w`, …). Its bias is the
//! matching `.bias` / `_b` / `bias` tensor if present. The activation goes
//! between layers (not after the last). Consecutive sizes must fit.

use super::{Gelu, Linear, Module, ReLU, Sequential, Sigmoid, Silu, StateDict, Tanh};
use crate::device::Npu;
use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// Activation placed between the layers of [`Sequential::load_mlp`].
///
/// Parses from and prints as `relu`, `tanh`, `sigmoid`, `gelu`, `silu`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActivationKind {
    /// `max(x, 0)`.
    Relu,
    /// Hyperbolic tangent.
    Tanh,
    /// Logistic sigmoid.
    Sigmoid,
    /// GELU (tanh approximation).
    Gelu,
    /// SiLU / swish.
    Silu,
}

impl ActivationKind {
    /// Every kind.
    pub const ALL: [ActivationKind; 5] = [
        ActivationKind::Relu,
        ActivationKind::Tanh,
        ActivationKind::Sigmoid,
        ActivationKind::Gelu,
        ActivationKind::Silu,
    ];

    /// Lower-case name (`relu`, …).
    pub fn name(self) -> &'static str {
        match self {
            ActivationKind::Relu => "relu",
            ActivationKind::Tanh => "tanh",
            ActivationKind::Sigmoid => "sigmoid",
            ActivationKind::Gelu => "gelu",
            ActivationKind::Silu => "silu",
        }
    }

    /// A boxed layer of this kind.
    pub fn layer(self) -> Box<dyn Module> {
        match self {
            ActivationKind::Relu => Box::new(ReLU),
            ActivationKind::Tanh => Box::new(Tanh),
            ActivationKind::Sigmoid => Box::new(Sigmoid),
            ActivationKind::Gelu => Box::new(Gelu),
            ActivationKind::Silu => Box::new(Silu),
        }
    }
}

impl std::fmt::Display for ActivationKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

impl std::str::FromStr for ActivationKind {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, String> {
        Self::ALL
            .into_iter()
            .find(|k| k.name().eq_ignore_ascii_case(s))
            .ok_or_else(|| {
                let names: Vec<_> = Self::ALL.iter().map(|k| k.name()).collect();
                format!(
                    "unknown activation `{s}` (expected one of {})",
                    names.join(", ")
                )
            })
    }
}

/// One layer found by [`Sequential::load_mlp`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlpLayerInfo {
    /// Name of the weight tensor.
    pub weight: String,
    /// Output size.
    pub out_features: usize,
    /// Input size.
    pub in_features: usize,
    /// Whether a bias tensor was found.
    pub has_bias: bool,
}

/// Works for PyTorch `nn.Sequential` state dicts, qanten's training exports
/// and Q.ANT's example weights.
impl Sequential {
    /// Builds an MLP from a state dict without knowing its layer names.
    ///
    /// Every 2-D tensor is a Linear layer `[out, in]`, ordered by the first
    /// number in its name (`0.weight`, `fc1.weight`, `l1_w`, …). Its bias is
    /// the matching `.bias` / `_b` / `bias` tensor if present. `activation`
    /// goes between layers (not after the last). Consecutive sizes must fit.
    /// Returns the model and what was found.
    pub fn load_mlp(
        npu: &Npu,
        sd: &StateDict,
        activation: ActivationKind,
    ) -> Result<(Sequential, Vec<MlpLayerInfo>)> {
        load_mlp(npu, sd, activation)
    }
}

fn load_mlp(
    npu: &Npu,
    sd: &StateDict,
    activation: ActivationKind,
) -> Result<(Sequential, Vec<MlpLayerInfo>)> {
    let number = |k: &str| -> Option<usize> {
        let digits: String = k
            .chars()
            .skip_while(|c| !c.is_ascii_digit())
            .take_while(char::is_ascii_digit)
            .collect();
        digits.parse().ok()
    };
    let mut layers: Vec<(usize, String)> = sd
        .iter()
        .filter(|(_, t)| t.rank() == 2)
        .filter_map(|(k, _)| Some((number(k)?, k.to_owned())))
        .collect();
    layers.sort();
    if layers.is_empty() {
        return Err(Error::MissingTensor(
            "numbered 2-D weight matrices (e.g. `0.weight` or `l1_w`)".into(),
        ));
    }
    let bias_of = |w: &str| -> Option<Tensor> {
        [
            w.replace(".weight", ".bias"),
            w.replace("_w", "_b"),
            w.replace("weight", "bias"),
        ]
        .iter()
        .filter(|b| b.as_str() != w)
        .find_map(|b| sd.get_opt(b).cloned())
    };
    let mut model = Sequential::new();
    let mut info = Vec::with_capacity(layers.len());
    for (i, (_, name)) in layers.iter().enumerate() {
        let w = sd.get(name)?;
        let (out, inp) = (w.shape()[0], w.shape()[1]);
        if let Some(prev) = info.last().map(|l: &MlpLayerInfo| l.out_features)
            && prev != inp
        {
            return Err(Error::Shape(format!(
                "layer `{name}` expects {inp} inputs but the previous layer has {prev} outputs"
            )));
        }
        let bias = bias_of(name);
        info.push(MlpLayerInfo {
            weight: name.clone(),
            out_features: out,
            in_features: inp,
            has_bias: bias.is_some(),
        });
        model.push(Linear::new(npu, w, bias)?);
        if i + 1 < layers.len() {
            model.push_boxed(activation.layer());
        }
    }
    Ok((model, info))
}
