//! Tensor operations.
//!
//! Functions that take an `&Npu` run their heavy part (multiplications,
//! matrix-vector products, the periodic nonlinearity) on the NPU. The rest
//! (pooling, activations, bias, reshuffling data) runs on the host CPU, the
//! same split Q.ANT's official toolkit uses.
//!
//! Layout conventions follow PyTorch: images are `NCHW`, linear weights are
//! `[out_features, in_features]`, conv weights are `[out, in/groups, kh, kw]`,
//! transposed-conv weights are `[in, out, kh, kw]`.

mod activation;
mod conv;
mod elementwise;
mod kan;
mod linear;
mod norm;
mod pool;

pub use activation::{gelu, leaky_relu, relu, relu_inplace, sigmoid, silu, softmax, tanh};
pub use conv::{
    Conv2dParams, ConvTranspose2dParams, ConvTransposeWeights, ConvWeights, conv_transpose2d,
    conv2d,
};
pub use elementwise::{add, add_bias, add_channel_bias, mul, periodic_nl, scale};
pub use kan::kan_layer;
pub use linear::{LinearWeights, linear, matmul, matvec};
pub use norm::{batch_norm2d, layer_norm};
pub use pool::{
    Pool2dParams, adaptive_avg_pool2d, adaptive_max_pool2d, avg_pool2d, global_avg_pool2d,
    max_pool2d,
};

/// `(height, width)` pair used by 2-D ops. Build one from a `usize` for a
/// square value, or from a tuple.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Hw {
    pub h: usize,
    pub w: usize,
}

impl Hw {
    pub const fn new(h: usize, w: usize) -> Self {
        Self { h, w }
    }
}

impl From<usize> for Hw {
    fn from(v: usize) -> Self {
        Self { h: v, w: v }
    }
}

impl From<(usize, usize)> for Hw {
    fn from((h, w): (usize, usize)) -> Self {
        Self { h, w }
    }
}

impl From<[usize; 2]> for Hw {
    fn from([h, w]: [usize; 2]) -> Self {
        Self { h, w }
    }
}
