//! 2-D pooling (host CPU, parallel).

use half::bf16;
use rayon::prelude::*;

use super::Hw;
use crate::error::{Error, Result, ensure_shape};
use crate::tensor::Tensor;

/// Parameters of max / average pooling (PyTorch semantics).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pool2dParams {
    pub kernel: Hw,
    /// Defaults to the kernel size.
    pub stride: Hw,
    pub padding: Hw,
    /// Average pooling only: count padded zeros in the divisor (PyTorch default `true`).
    pub count_include_pad: bool,
}

impl Pool2dParams {
    pub fn new(kernel: impl Into<Hw>) -> Self {
        let kernel = kernel.into();
        Self {
            kernel,
            stride: kernel,
            padding: Hw::from(0),
            count_include_pad: true,
        }
    }
    pub fn stride(mut self, s: impl Into<Hw>) -> Self {
        self.stride = s.into();
        self
    }
    pub fn padding(mut self, p: impl Into<Hw>) -> Self {
        self.padding = p.into();
        self
    }
    pub fn count_include_pad(mut self, yes: bool) -> Self {
        self.count_include_pad = yes;
        self
    }

    fn output_size(&self, h: usize, w: usize) -> Result<Hw> {
        let (k, s, p) = (self.kernel, self.stride, self.padding);
        if k.h == 0 || k.w == 0 || s.h == 0 || s.w == 0 {
            return Err(Error::InvalidArgument(
                "pool kernel and stride must be > 0".into(),
            ));
        }
        if 2 * p.h > k.h || 2 * p.w > k.w {
            return Err(Error::InvalidArgument(format!(
                "pool padding {p:?} must be at most half the kernel {k:?}"
            )));
        }
        ensure_shape!(
            h + 2 * p.h >= k.h && w + 2 * p.w >= k.w,
            "pool kernel {k:?} is larger than the padded input {}x{}",
            h + 2 * p.h,
            w + 2 * p.w
        );
        Ok(Hw::new(
            (h + 2 * p.h - k.h) / s.h + 1,
            (w + 2 * p.w - k.w) / s.w + 1,
        ))
    }
}

#[derive(Clone, Copy)]
enum Mode {
    Max,
    Avg { count_include_pad: bool },
}

fn pool(x: &Tensor, prm: &Pool2dParams, mode: Mode) -> Result<Tensor> {
    let [n, c, h, w] = x.dims::<4>()?;
    let out = prm.output_size(h, w)?;
    let data = x.data();
    let plane_out = out.h * out.w;
    let mut result = vec![bf16::ZERO; n * c * plane_out];
    result
        .par_chunks_mut(plane_out.max(1))
        .enumerate()
        .for_each(|(nc, dst)| {
            let plane = &data[nc * h * w..][..h * w];
            for oy in 0..out.h {
                for ox in 0..out.w {
                    let y0 = (oy * prm.stride.h) as isize - prm.padding.h as isize;
                    let x0 = (ox * prm.stride.w) as isize - prm.padding.w as isize;
                    let mut max = f32::NEG_INFINITY;
                    let mut sum = 0.0f32;
                    let mut real = 0usize;
                    for ky in 0..prm.kernel.h as isize {
                        for kx in 0..prm.kernel.w as isize {
                            let (iy, ix) = (y0 + ky, x0 + kx);
                            if iy >= 0 && ix >= 0 && iy < h as isize && ix < w as isize {
                                let v = plane[iy as usize * w + ix as usize].to_f32();
                                max = max.max(v);
                                sum += v;
                                real += 1;
                            }
                        }
                    }
                    dst[oy * out.w + ox] = bf16::from_f32(match mode {
                        Mode::Max => max,
                        Mode::Avg { count_include_pad } => {
                            // PyTorch divides by the window clipped to the padded input.
                            let divisor = if count_include_pad {
                                let hend =
                                    (y0 + prm.kernel.h as isize).min((h + prm.padding.h) as isize);
                                let wend =
                                    (x0 + prm.kernel.w as isize).min((w + prm.padding.w) as isize);
                                ((hend - y0) * (wend - x0)) as usize
                            } else {
                                real
                            };
                            sum / divisor.max(1) as f32
                        }
                    });
                }
            }
        });
    Tensor::new(result, [n, c, out.h, out.w])
}

/// Max pooling over an `NCHW` tensor.
pub fn max_pool2d(x: &Tensor, params: Pool2dParams) -> Result<Tensor> {
    pool(x, &params, Mode::Max)
}

/// Average pooling over an `NCHW` tensor.
pub fn avg_pool2d(x: &Tensor, params: Pool2dParams) -> Result<Tensor> {
    pool(
        x,
        &params,
        Mode::Avg {
            count_include_pad: params.count_include_pad,
        },
    )
}

fn adaptive(x: &Tensor, out: Hw, max: bool) -> Result<Tensor> {
    let [n, c, h, w] = x.dims::<4>()?;
    if out.h == 0 || out.w == 0 {
        return Err(Error::InvalidArgument(
            "adaptive pool output size must be > 0".into(),
        ));
    }
    let data = x.data();
    let plane_out = out.h * out.w;
    let mut result = vec![bf16::ZERO; n * c * plane_out];
    result
        .par_chunks_mut(plane_out)
        .enumerate()
        .for_each(|(nc, dst)| {
            let plane = &data[nc * h * w..][..h * w];
            for oy in 0..out.h {
                let (y0, y1) = (oy * h / out.h, ((oy + 1) * h).div_ceil(out.h));
                for ox in 0..out.w {
                    let (x0, x1) = (ox * w / out.w, ((ox + 1) * w).div_ceil(out.w));
                    let mut acc = if max { f32::NEG_INFINITY } else { 0.0 };
                    for iy in y0..y1 {
                        for ix in x0..x1 {
                            let v = plane[iy * w + ix].to_f32();
                            acc = if max { acc.max(v) } else { acc + v };
                        }
                    }
                    if !max {
                        acc /= ((y1 - y0) * (x1 - x0)).max(1) as f32;
                    }
                    dst[oy * out.w + ox] = bf16::from_f32(acc);
                }
            }
        });
    Tensor::new(result, [n, c, out.h, out.w])
}

/// Adaptive average pooling to a fixed output size (any input size).
pub fn adaptive_avg_pool2d(x: &Tensor, output: impl Into<Hw>) -> Result<Tensor> {
    adaptive(x, output.into(), false)
}

/// Adaptive max pooling to a fixed output size (any input size).
pub fn adaptive_max_pool2d(x: &Tensor, output: impl Into<Hw>) -> Result<Tensor> {
    adaptive(x, output.into(), true)
}

/// Mean over each `H × W` plane: `[N, C, H, W] -> [N, C]`.
pub fn global_avg_pool2d(x: &Tensor) -> Result<Tensor> {
    let [n, c, _, _] = x.dims::<4>()?;
    adaptive(x, Hw::from(1), false)?.reshape([n, c])
}
