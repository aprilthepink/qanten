//! 2-D convolution and transposed convolution.
//!
//! Both are lowered to one batched matrix-vector product per group on the
//! NPU: the input is unfolded into patch rows on the host (im2row), the
//! weights are the matrix, and the result is folded back into `NCHW`.

use half::bf16;
use rayon::prelude::*;

use super::Hw;
use super::linear::LinearWeights;
use crate::device::Npu;
use crate::error::{Error, Result, ensure_shape};
use crate::tensor::Tensor;

/// Parameters of a 2-D convolution (PyTorch semantics, zero padding).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Conv2dParams {
    /// Step between kernel positions.
    pub stride: Hw,
    /// Zero padding added on each side.
    pub padding: Hw,
    /// Spacing between kernel taps (1 = dense).
    pub dilation: Hw,
    /// Channel groups (equal to the channel count for depthwise).
    pub groups: usize,
}

impl Default for Conv2dParams {
    fn default() -> Self {
        Self {
            stride: Hw::from(1),
            padding: Hw::from(0),
            dilation: Hw::from(1),
            groups: 1,
        }
    }
}

impl Conv2dParams {
    /// Stride 1, no padding, dilation 1, one group.
    pub fn new() -> Self {
        Self::default()
    }
    /// Sets the stride (a `usize` for both axes, or `(h, w)`).
    pub fn stride(mut self, s: impl Into<Hw>) -> Self {
        self.stride = s.into();
        self
    }
    /// Sets the zero padding.
    pub fn padding(mut self, p: impl Into<Hw>) -> Self {
        self.padding = p.into();
        self
    }
    /// Sets the dilation.
    pub fn dilation(mut self, d: impl Into<Hw>) -> Self {
        self.dilation = d.into();
        self
    }
    /// Sets the number of groups.
    pub fn groups(mut self, g: usize) -> Self {
        self.groups = g;
        self
    }

    fn validate(&self) -> Result<()> {
        if self.stride.h == 0 || self.stride.w == 0 {
            return Err(Error::InvalidArgument("stride must be > 0".into()));
        }
        if self.dilation.h == 0 || self.dilation.w == 0 {
            return Err(Error::InvalidArgument("dilation must be > 0".into()));
        }
        if self.groups == 0 {
            return Err(Error::InvalidArgument("groups must be > 0".into()));
        }
        Ok(())
    }

    /// Output spatial size for an input of size `input` and a kernel of size `kernel`.
    pub fn output_size(&self, input: Hw, kernel: Hw) -> Result<Hw> {
        self.validate()?;
        let one = |i: usize, k: usize, p: usize, d: usize, s: usize, dim: &str| -> Result<usize> {
            let span = d * (k.max(1) - 1) + 1;
            let padded = i + 2 * p;
            ensure_shape!(
                k > 0 && padded >= span,
                "conv2d: kernel {dim} {k} (dilated span {span}) is larger than the padded input {dim} {padded}"
            );
            Ok((padded - span) / s + 1)
        };
        Ok(Hw {
            h: one(
                input.h,
                kernel.h,
                self.padding.h,
                self.dilation.h,
                self.stride.h,
                "height",
            )?,
            w: one(
                input.w,
                kernel.w,
                self.padding.w,
                self.dilation.w,
                self.stride.w,
                "width",
            )?,
        })
    }
}

/// Parameters of a 2-D transposed convolution (PyTorch semantics).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvTranspose2dParams {
    /// Step between kernel positions.
    pub stride: Hw,
    /// Zero padding added on each side.
    pub padding: Hw,
    /// Extra rows/columns added to one side of the output; must be smaller than stride or dilation.
    pub output_padding: Hw,
    /// Spacing between kernel taps (1 = dense).
    pub dilation: Hw,
    /// Channel groups (equal to the channel count for depthwise).
    pub groups: usize,
}

impl Default for ConvTranspose2dParams {
    fn default() -> Self {
        Self {
            stride: Hw::from(1),
            padding: Hw::from(0),
            output_padding: Hw::from(0),
            dilation: Hw::from(1),
            groups: 1,
        }
    }
}

impl ConvTranspose2dParams {
    /// Stride 1, no padding, dilation 1, one group.
    pub fn new() -> Self {
        Self::default()
    }
    /// Sets the stride (a `usize` for both axes, or `(h, w)`).
    pub fn stride(mut self, s: impl Into<Hw>) -> Self {
        self.stride = s.into();
        self
    }
    /// Sets the zero padding.
    pub fn padding(mut self, p: impl Into<Hw>) -> Self {
        self.padding = p.into();
        self
    }
    /// Sets the output padding.
    pub fn output_padding(mut self, p: impl Into<Hw>) -> Self {
        self.output_padding = p.into();
        self
    }
    /// Sets the dilation.
    pub fn dilation(mut self, d: impl Into<Hw>) -> Self {
        self.dilation = d.into();
        self
    }
    /// Sets the number of groups.
    pub fn groups(mut self, g: usize) -> Self {
        self.groups = g;
        self
    }

    /// Output spatial size for an input of size `input` and a kernel of size `kernel`.
    pub fn output_size(&self, input: Hw, kernel: Hw) -> Result<Hw> {
        if self.stride.h == 0 || self.stride.w == 0 || self.dilation.h == 0 || self.dilation.w == 0
        {
            return Err(Error::InvalidArgument(
                "stride and dilation must be > 0".into(),
            ));
        }
        if self.groups == 0 {
            return Err(Error::InvalidArgument("groups must be > 0".into()));
        }
        let one = |i: usize, k: usize, p: usize, op: usize, d: usize, s: usize| -> Result<usize> {
            if op >= s.max(d) {
                return Err(Error::InvalidArgument(format!(
                    "output_padding ({op}) must be smaller than stride ({s}) or dilation ({d})"
                )));
            }
            let size = (i as i64 - 1) * s as i64 - 2 * p as i64
                + d as i64 * (k as i64 - 1)
                + op as i64
                + 1;
            ensure_shape!(
                i > 0 && k > 0 && size > 0,
                "conv_transpose2d: computed output size {size} is not positive"
            );
            Ok(size as usize)
        };
        Ok(Hw {
            h: one(
                input.h,
                kernel.h,
                self.padding.h,
                self.output_padding.h,
                self.dilation.h,
                self.stride.h,
            )?,
            w: one(
                input.w,
                kernel.w,
                self.padding.w,
                self.output_padding.w,
                self.dilation.w,
                self.stride.w,
            )?,
        })
    }
}

/// Convolution weights uploaded to the NPU once (one matrix per group).
#[derive(Debug)]
pub struct ConvWeights {
    per_group: Vec<LinearWeights>,
    shape: [usize; 4],
}

impl ConvWeights {
    /// Uploads `[out_channels, in_channels / groups, kh, kw]` weights.
    pub fn upload(npu: &Npu, weight: &Tensor, groups: usize) -> Result<Self> {
        let shape @ [oc, icg, kh, kw] = weight.dims::<4>()?;
        if groups == 0 {
            return Err(Error::InvalidArgument("groups must be > 0".into()));
        }
        ensure_shape!(
            oc % groups == 0,
            "conv2d: {oc} output channels are not divisible by {groups} groups"
        );
        let ocg = oc / groups;
        let cols = icg * kh * kw;
        let per_group = (0..groups)
            .map(|g| {
                let slice = &weight.data()[g * ocg * cols..(g + 1) * ocg * cols];
                LinearWeights::upload(npu, &Tensor::new(slice.to_vec(), [ocg, cols])?)
            })
            .collect::<Result<_>>()?;
        Ok(Self { per_group, shape })
    }

    /// `[out_channels, in_channels / groups, kh, kw]`.
    pub fn shape(&self) -> [usize; 4] {
        self.shape
    }

    /// Number of groups the weights were uploaded for.
    pub fn groups(&self) -> usize {
        self.per_group.len()
    }

    /// The NPU holding the weights (`None` only for zero groups, which `upload` rejects).
    pub fn npu(&self) -> Option<&Npu> {
        self.per_group.first().map(LinearWeights::npu)
    }

    /// Runs the convolution on an `NCHW` input. `params.groups` must match the upload.
    pub fn forward(&self, x: &Tensor, params: &Conv2dParams) -> Result<Tensor> {
        let [n, c, h, w] = x.dims::<4>()?;
        let [oc, icg, kh, kw] = self.shape;
        let groups = self.groups();
        ensure_shape!(
            params.groups == groups,
            "conv2d: weights were uploaded for {groups} groups, params say {}",
            params.groups
        );
        ensure_shape!(
            c == icg * groups,
            "conv2d: input has {c} channels, weights expect {} ({icg} per group x {groups} groups)",
            icg * groups
        );
        let out = params.output_size(Hw::new(h, w), Hw::new(kh, kw))?;
        let ocg = oc / groups;
        let positions = out.h * out.w;
        let mut result = vec![bf16::ZERO; n * oc * positions];
        for (g, weights) in self.per_group.iter().enumerate() {
            let patches = im2row(
                x.data(),
                [n, c, h, w],
                g * icg,
                icg,
                Hw::new(kh, kw),
                params,
                out,
            );
            let y = weights.forward_rows(&patches, n * positions)?;
            scatter_nhwc(y.data(), &mut result, n, oc, g * ocg, ocg, positions);
        }
        Tensor::new(result, [n, oc, out.h, out.w])
    }
}

/// Transposed-convolution weights uploaded to the NPU once.
#[derive(Debug)]
pub struct ConvTransposeWeights {
    per_group: Vec<LinearWeights>,
    shape: [usize; 4],
}

impl ConvTransposeWeights {
    /// Uploads `[in_channels, out_channels / groups, kh, kw]` weights (PyTorch layout).
    pub fn upload(npu: &Npu, weight: &Tensor, groups: usize) -> Result<Self> {
        let shape @ [ic, ocg, kh, kw] = weight.dims::<4>()?;
        if groups == 0 {
            return Err(Error::InvalidArgument("groups must be > 0".into()));
        }
        ensure_shape!(
            ic % groups == 0,
            "conv_transpose2d: {ic} input channels are not divisible by {groups} groups"
        );
        let icg = ic / groups;
        let k = kh * kw;
        let src = weight.data();
        let per_group = (0..groups)
            .map(|g| {
                // Matrix row o, column (ci, ky, kx) = W[g*icg + ci, o, ky, kx].
                let mut m = vec![bf16::ZERO; ocg * icg * k];
                for ci in 0..icg {
                    for o in 0..ocg {
                        let s = ((g * icg + ci) * ocg + o) * k;
                        let d = o * icg * k + ci * k;
                        m[d..d + k].copy_from_slice(&src[s..s + k]);
                    }
                }
                LinearWeights::upload(npu, &Tensor::new(m, [ocg, icg * k])?)
            })
            .collect::<Result<_>>()?;
        Ok(Self { per_group, shape })
    }

    /// `[in_channels, out_channels / groups, kh, kw]`.
    pub fn shape(&self) -> [usize; 4] {
        self.shape
    }

    /// Number of groups the weights were uploaded for.
    pub fn groups(&self) -> usize {
        self.per_group.len()
    }

    /// Runs the transposed convolution on an `NCHW` input.
    pub fn forward(&self, x: &Tensor, params: &ConvTranspose2dParams) -> Result<Tensor> {
        let [n, c, h, w] = x.dims::<4>()?;
        let [ic, ocg, kh, kw] = self.shape;
        let groups = self.groups();
        ensure_shape!(
            params.groups == groups,
            "conv_transpose2d: weights were uploaded for {groups} groups, params say {}",
            params.groups
        );
        ensure_shape!(
            c == ic,
            "conv_transpose2d: input has {c} channels, weights expect {ic}"
        );
        let icg = ic / groups;
        let oc = ocg * groups;
        let out = params.output_size(Hw::new(h, w), Hw::new(kh, kw))?;
        let positions = out.h * out.w;
        let mut result = vec![bf16::ZERO; n * oc * positions];
        for (g, weights) in self.per_group.iter().enumerate() {
            let patches = im2row_transpose(
                x.data(),
                [n, c, h, w],
                g * icg,
                icg,
                Hw::new(kh, kw),
                params,
                out,
            );
            let y = weights.forward_rows(&patches, n * positions)?;
            scatter_nhwc(y.data(), &mut result, n, oc, g * ocg, ocg, positions);
        }
        Tensor::new(result, [n, oc, out.h, out.w])
    }
}

/// 2-D convolution on the NPU (PyTorch's `F.conv2d`).
///
/// `x` is `[N, C, H, W]`, `weight` is `[OC, C / groups, KH, KW]`, `bias` is `[OC]`.
/// Uploads the weights for this one call; keep a [`ConvWeights`] (or use
/// [`nn::Conv2d`](crate::nn::Conv2d)) to reuse them.
pub fn conv2d(
    npu: &Npu,
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    params: Conv2dParams,
) -> Result<Tensor> {
    params.validate()?;
    let y = ConvWeights::upload(npu, weight, params.groups)?.forward(x, &params)?;
    match bias {
        Some(b) => super::add_channel_bias(&y, b),
        None => Ok(y),
    }
}

/// 2-D transposed convolution on the NPU (PyTorch's `F.conv_transpose2d`).
///
/// `x` is `[N, C, H, W]`, `weight` is `[C, OC / groups, KH, KW]`, `bias` is `[OC]`.
pub fn conv_transpose2d(
    npu: &Npu,
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    params: ConvTranspose2dParams,
) -> Result<Tensor> {
    let y = ConvTransposeWeights::upload(npu, weight, params.groups)?.forward(x, &params)?;
    match bias {
        Some(b) => super::add_channel_bias(&y, b),
        None => Ok(y),
    }
}

/// Unfolds input channels `c0..c0+cg` into rows `(n, oy, ox)` × columns `(ci, ky, kx)`.
fn im2row(
    x: &[bf16],
    [_, c, h, w]: [usize; 4],
    c0: usize,
    cg: usize,
    k: Hw,
    p: &Conv2dParams,
    out: Hw,
) -> Vec<bf16> {
    let cols = cg * k.h * k.w;
    let positions = out.h * out.w;
    let mut rows = vec![bf16::ZERO; x.len() / (c * h * w).max(1) * positions * cols];
    rows.par_chunks_mut(cols.max(1))
        .enumerate()
        .for_each(|(row, dst)| {
            let b = row / positions;
            let oy = (row % positions) / out.w;
            let ox = row % out.w;
            for ci in 0..cg {
                let plane = &x[(b * c + c0 + ci) * h * w..][..h * w];
                for ky in 0..k.h {
                    let iy = (oy * p.stride.h + ky * p.dilation.h) as isize - p.padding.h as isize;
                    if iy < 0 || iy >= h as isize {
                        continue;
                    }
                    for kx in 0..k.w {
                        let ix =
                            (ox * p.stride.w + kx * p.dilation.w) as isize - p.padding.w as isize;
                        if ix >= 0 && ix < w as isize {
                            dst[(ci * k.h + ky) * k.w + kx] = plane[iy as usize * w + ix as usize];
                        }
                    }
                }
            }
        });
    rows
}

/// Like [`im2row`] but gathers the inputs that contribute to each output
/// position of a transposed convolution.
fn im2row_transpose(
    x: &[bf16],
    [_, c, h, w]: [usize; 4],
    c0: usize,
    cg: usize,
    k: Hw,
    p: &ConvTranspose2dParams,
    out: Hw,
) -> Vec<bf16> {
    let cols = cg * k.h * k.w;
    let positions = out.h * out.w;
    let n = x.len() / (c * h * w).max(1);
    let mut rows = vec![bf16::ZERO; n * positions * cols];
    // Input index for output index `o` and kernel tap `kk`, if it exists.
    let src = |o: usize,
               kk: usize,
               pad: usize,
               dil: usize,
               stride: usize,
               size: usize|
     -> Option<usize> {
        let num = o as isize + pad as isize - (kk * dil) as isize;
        if num < 0 || num % stride as isize != 0 {
            return None;
        }
        let i = (num / stride as isize) as usize;
        (i < size).then_some(i)
    };
    rows.par_chunks_mut(cols.max(1))
        .enumerate()
        .for_each(|(row, dst)| {
            let b = row / positions;
            let oy = (row % positions) / out.w;
            let ox = row % out.w;
            for ky in 0..k.h {
                let Some(iy) = src(oy, ky, p.padding.h, p.dilation.h, p.stride.h, h) else {
                    continue;
                };
                for kx in 0..k.w {
                    let Some(ix) = src(ox, kx, p.padding.w, p.dilation.w, p.stride.w, w) else {
                        continue;
                    };
                    for ci in 0..cg {
                        dst[(ci * k.h + ky) * k.w + kx] = x[((b * c + c0 + ci) * h + iy) * w + ix];
                    }
                }
            }
        });
    rows
}

/// Writes `y` (`[n * positions, ocg]`, i.e. NHWC for one group) into channels
/// `oc0..oc0+ocg` of the NCHW `out` buffer.
fn scatter_nhwc(
    y: &[bf16],
    out: &mut [bf16],
    n: usize,
    oc: usize,
    oc0: usize,
    ocg: usize,
    positions: usize,
) {
    for b in 0..n {
        for pos in 0..positions {
            let src = &y[(b * positions + pos) * ocg..][..ocg];
            for (o, &v) in src.iter().enumerate() {
                out[(b * oc + oc0 + o) * positions + pos] = v;
            }
        }
    }
}
