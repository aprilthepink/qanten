//! Error type used across the crate.

use std::fmt;

/// Every fallible operation in this crate returns this error.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// The driver (or simulator) rejected an operation.
    #[error("NPU {npu}: driver error {code}: {message}")]
    Driver {
        npu: u32,
        code: i32,
        message: String,
    },

    /// The Q.ANT driver library could not be loaded.
    #[error("{0}")]
    DriverUnavailable(String),

    /// The crate was built without the `driver` feature.
    #[error("hardware access requires the `driver` feature of the `qanten` crate")]
    DriverFeatureDisabled,

    /// Tensor shapes do not fit the operation.
    #[error("shape error: {0}")]
    Shape(String),

    /// A parameter is out of range.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    /// The operation exists but this combination of parameters is not supported.
    #[error("unsupported: {0}")]
    Unsupported(String),

    /// Inputs live on different NPUs.
    #[error("device mismatch: {0}")]
    DeviceMismatch(String),

    /// A tensor needed by a model is missing from a state dict.
    #[error("missing tensor `{0}` in state dict")]
    MissingTensor(String),

    /// File parsing errors (safetensors, ...).
    #[error("format error: {0}")]
    Format(String),

    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// `Result` alias with [`Error`] as the error type.
pub type Result<T, E = Error> = std::result::Result<T, E>;

pub(crate) fn shape_err(args: fmt::Arguments<'_>) -> Error {
    Error::Shape(args.to_string())
}

macro_rules! bail_shape {
    ($($arg:tt)*) => {
        return Err($crate::error::shape_err(format_args!($($arg)*)))
    };
}
pub(crate) use bail_shape;

macro_rules! ensure_shape {
    ($cond:expr, $($arg:tt)*) => {
        if !$cond {
            $crate::error::bail_shape!($($arg)*);
        }
    };
}
pub(crate) use ensure_shape;
