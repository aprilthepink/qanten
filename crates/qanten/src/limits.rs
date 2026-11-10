//! Size limits for data from untrusted sources: model files, weight files and
//! network requests.
//!
//! Parsers and the ONNX interpreter check every element count that comes from
//! outside before allocating, so a small malicious file or request cannot ask
//! for terabytes or overflow `usize` arithmetic.

use crate::error::{Error, Result};

/// Largest number of elements accepted for one tensor from an untrusted
/// source: 2²⁸ (512 MiB as bf16, 1 GiB as f32, 2 GiB as i64/f64).
pub const MAX_ELEMENTS: usize = 1 << 28;

/// Largest total size of the values an ONNX evaluation keeps alive at once
/// (4 GiB). Without it, a small graph of many large nodes could hold each at
/// the per-tensor limit.
pub const MAX_LIVE_BYTES: usize = 1 << 32;

/// Largest number of inner-loop steps one host-side operator may take on
/// untrusted parameters (about a few seconds of CPU). Bounds ops whose cost
/// grows with an attribute rather than with their output, such as pooling
/// windows.
pub const MAX_WORK: usize = 1 << 32;

/// Largest rank accepted from an untrusted source.
pub const MAX_RANK: usize = 32;

/// Element count of an untrusted `shape`, checked for overflow (including
/// the partial products ops compute), rank and [`MAX_ELEMENTS`].
pub fn elements(shape: &[usize]) -> Result<usize> {
    if shape.len() > MAX_RANK {
        return Err(Error::Shape(format!(
            "rank {} exceeds the limit of {MAX_RANK}",
            shape.len()
        )));
    }
    let n = crate::Tensor::checked_numel(shape)
        .ok_or_else(|| Error::Shape(format!("shape {shape:?} overflows usize")))?;
    // Zero-sized dimensions hide the size of the others; bound those too so
    // every partial product an op computes stays small.
    let nonzero: usize = shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d.max(1)))
        .unwrap_or(usize::MAX);
    if n > MAX_ELEMENTS || nonzero > MAX_ELEMENTS {
        return Err(Error::Shape(format!(
            "shape {shape:?} has more than {MAX_ELEMENTS} elements"
        )));
    }
    Ok(n)
}

/// Checks that an operator's estimated cost, `factors` multiplied together,
/// stays within [`MAX_WORK`].
pub fn work(what: &str, factors: &[usize]) -> Result<()> {
    let total = factors
        .iter()
        .try_fold(1usize, |acc, &f| acc.checked_mul(f))
        .filter(|&t| t <= MAX_WORK);
    match total {
        Some(_) => Ok(()),
        None => Err(Error::InvalidArgument(format!(
            "{what} would take more than {MAX_WORK} steps"
        ))),
    }
}
