//! Training on the NPU.
//!
//! Mixed precision: every layer keeps **f32 master weights** on the host;
//! forward and backward matrix products run on the NPU in bf16, and the
//! optimizer updates the f32 copies. The heavy parts of backpropagation are
//! the same primitive as inference:
//!
//! | Layer | Forward | Input gradient | Weight gradient |
//! |---|---|---|---|
//! | [`Dense`] | `x·Wᵀ` (NPU) | `g·W` (NPU) | `gᵀ·x` (NPU) |
//! | [`Kan`] | `tcos(k·x+φ)·a` (NPU) | `−a·k·sin(…)` (NPU) | `cos(…)`, `−a·sin(…)` (NPU) |
//! | activations | host | host | – |
//!
//! Because the NPU is in the loop, training against
//! [`Npu::simulator_with`](crate::Npu::simulator_with) with noise makes the
//! model robust to analog imprecision ("hardware-aware training").
//!
//! ```
//! # fn main() -> qanten::Result<()> {
//! use qanten::train::{self, Adam, Dense, Model, Relu};
//! use qanten::{Npu, Tensor};
//!
//! let npu = Npu::simulator();
//! let mut model = Model::new()
//!     .add(Dense::new(&npu, 2, 16, 1)?)
//!     .add(Relu::default())
//!     .add(Dense::new(&npu, 16, 1, 2)?);
//! let mut opt = Adam::new(0.01);
//! // Learn y = x0 - x1.
//! let x = Tensor::rand_uniform([64, 2], -1., 1., 3);
//! let y = Tensor::from_f32(&x.to_f32_vec().chunks(2).map(|p| p[0] - p[1]).collect::<Vec<_>>(), [64, 1])?;
//! let mut first = None;
//! for _ in 0..200 {
//!     let pred = model.forward(&x)?;
//!     let (loss, grad) = train::mse(&pred, &y)?;
//!     first.get_or_insert(loss);
//!     model.backward(&grad)?;
//!     model.step(&mut opt)?;
//! }
//! let (last, _) = train::mse(&model.forward(&x)?, &y)?;
//! assert!(last < first.unwrap_or(1.0) / 10.0);
//! # Ok(()) }
//! ```

use std::f32::consts::FRAC_PI_2;

use crate::device::Npu;
use crate::error::{Error, Result, ensure_shape};
use crate::nn::StateDict;
use crate::ops;
use crate::rng::SplitMix64;
use crate::tensor::Tensor;

/// A trainable parameter: f32 values and their accumulated gradient.
#[derive(Debug)]
pub struct Param<'a> {
    /// Name inside the layer (`"weight"`, `"bias"`, …).
    pub name: &'static str,
    /// Current values (f32 master copy).
    pub value: &'a mut [f32],
    /// Gradient accumulated by `backward` since the last `zero_grad`.
    pub grad: &'a mut [f32],
    /// Shape of the parameter.
    pub shape: Vec<usize>,
}

/// A layer that can be trained.
pub trait Layer: Send + std::fmt::Debug {
    /// Forward pass; remembers what `backward` needs.
    fn forward(&mut self, x: &Tensor) -> Result<Tensor>;
    /// Given `∂loss/∂output`, accumulates parameter gradients and returns `∂loss/∂input`.
    fn backward(&mut self, grad: &Tensor) -> Result<Tensor>;
    /// The trainable parameters (empty for activations).
    fn params(&mut self) -> Vec<Param<'_>> {
        Vec::new()
    }
    /// Non-trainable tensors to export with the parameters (like PyTorch buffers).
    fn buffers(&self) -> Vec<(&'static str, Tensor)> {
        Vec::new()
    }
    /// Called after the optimizer changed the parameters (e.g. to refresh the bf16 copy).
    fn sync(&mut self) -> Result<()> {
        Ok(())
    }
    /// Short name for state-dict export.
    fn name(&self) -> &'static str;
}

fn to_bf16(v: &[f32], shape: &[usize]) -> Result<Tensor> {
    Tensor::from_f32(v, shape.to_vec())
}

fn transpose(v: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut t = vec![0.0; v.len()];
    for r in 0..rows {
        for c in 0..cols {
            t[c * rows + r] = v[r * cols + c];
        }
    }
    t
}

fn take<T>(slot: &mut Option<T>, what: &str) -> Result<T> {
    slot.take().ok_or_else(|| {
        Error::InvalidArgument(format!("{what}: backward called without a forward pass"))
    })
}

/// Fully connected layer `y = x·Wᵀ + b` with `W` `[out, in]`.
#[derive(Debug)]
pub struct Dense {
    npu: Npu,
    n_in: usize,
    n_out: usize,
    weight: Vec<f32>,
    bias: Vec<f32>,
    grad_w: Vec<f32>,
    grad_b: Vec<f32>,
    weight_bf16: Tensor,
    input: Option<Tensor>,
}

impl Dense {
    /// PyTorch's default initialisation: uniform in `±1/√in` for weight and bias.
    pub fn new(npu: &Npu, n_in: usize, n_out: usize, seed: u64) -> Result<Self> {
        let bound = 1.0 / (n_in.max(1) as f32).sqrt();
        let mut rng = SplitMix64::new(seed);
        let mut u = || (rng.next_f32() * 2.0 - 1.0) * bound;
        let weight: Vec<f32> = (0..n_in * n_out).map(|_| u()).collect();
        let bias: Vec<f32> = (0..n_out).map(|_| u()).collect();
        Self::from_weights(npu, weight, bias, n_in, n_out)
    }

    /// From existing f32 weights `[out, in]` and bias `[out]`.
    pub fn from_weights(
        npu: &Npu,
        weight: Vec<f32>,
        bias: Vec<f32>,
        n_in: usize,
        n_out: usize,
    ) -> Result<Self> {
        ensure_shape!(
            weight.len() == n_in * n_out && bias.len() == n_out,
            "Dense: weight must be {n_out}x{n_in} and bias {n_out}"
        );
        let weight_bf16 = to_bf16(&weight, &[n_out, n_in])?;
        Ok(Self {
            npu: npu.clone(),
            n_in,
            n_out,
            grad_w: vec![0.0; weight.len()],
            grad_b: vec![0.0; n_out],
            weight,
            bias,
            weight_bf16,
            input: None,
        })
    }
}

impl Layer for Dense {
    fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        let [_, n] = x.dims::<2>()?;
        ensure_shape!(
            n == self.n_in,
            "Dense: expected {} inputs, got {n}",
            self.n_in
        );
        let y = ops::linear(
            &self.npu,
            x,
            &self.weight_bf16,
            Some(&Tensor::from_f32(&self.bias, [self.n_out])?),
        )?;
        self.input = Some(x.clone());
        Ok(y)
    }

    fn backward(&mut self, grad: &Tensor) -> Result<Tensor> {
        let x = take(&mut self.input, "Dense")?;
        let [b, o] = grad.dims::<2>()?;
        ensure_shape!(
            o == self.n_out,
            "Dense: gradient has {o} outputs, expected {}",
            self.n_out
        );
        let g = grad.to_f32_vec();
        for row in g.chunks(self.n_out) {
            for (acc, v) in self.grad_b.iter_mut().zip(row) {
                *acc += v;
            }
        }
        // Gradients are often tiny (1/batch). Scale them to unit range before
        // they go to the NPU and undo it afterwards, so bf16 rounding and
        // analog noise act on O(1) values (per-tensor gradient scaling).
        let gmax = g.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        if gmax == 0.0 {
            return Ok(Tensor::zeros([b, self.n_in]));
        }
        let gs: Vec<f32> = g.iter().map(|v| v / gmax).collect();
        // ∂W = gᵀ·x: matrix xᵀ [in, b] times vectors gᵀ [out, b] → [out, in]  (NPU)
        let xt = transpose(&x.to_f32_vec(), b, self.n_in);
        let gt = transpose(&gs, b, self.n_out);
        let dw = ops::matvec(
            &self.npu,
            &to_bf16(&xt, &[self.n_in, b])?,
            &to_bf16(&gt, &[self.n_out, b])?,
        )?;
        for (acc, v) in self.grad_w.iter_mut().zip(dw.data()) {
            *acc += v.to_f32() * gmax;
        }
        // ∂x = g·W: matrix Wᵀ [in, out] times vectors g [b, out] → [b, in]  (NPU)
        let wt = transpose(&self.weight, self.n_out, self.n_in);
        let dx = ops::matvec(
            &self.npu,
            &to_bf16(&wt, &[self.n_in, self.n_out])?,
            &to_bf16(&gs, &[b, self.n_out])?,
        )?;
        Ok(dx.map(|v| v * gmax))
    }

    fn params(&mut self) -> Vec<Param<'_>> {
        vec![
            Param {
                name: "weight",
                value: &mut self.weight,
                grad: &mut self.grad_w,
                shape: vec![self.n_out, self.n_in],
            },
            Param {
                name: "bias",
                value: &mut self.bias,
                grad: &mut self.grad_b,
                shape: vec![self.n_out],
            },
        ]
    }

    fn sync(&mut self) -> Result<()> {
        self.weight_bf16 = to_bf16(&self.weight, &[self.n_out, self.n_in])?;
        Ok(())
    }

    fn name(&self) -> &'static str {
        "Dense"
    }
}

/// Q.ANT's photonic KAN layer, trainable:
/// `y[b, j] = Σ_{i,k} tcos(ks[k]·x[b,i] + φ[j,i,k]) · a[j,i,k]`.
/// Amplitudes and phases are learned; the frequencies `ks` are fixed.
#[derive(Debug)]
pub struct Kan {
    npu: Npu,
    n_in: usize,
    n_out: usize,
    ks: Vec<f32>,
    phis: Vec<f32>,
    ampls: Vec<f32>,
    grad_phis: Vec<f32>,
    grad_ampls: Vec<f32>,
    input: Option<Tensor>,
}

impl Kan {
    /// Random phases in `[−π, π)`, amplitudes in `±1/√(in·n_k)`.
    pub fn new(npu: &Npu, n_in: usize, n_out: usize, ks: &[f32], seed: u64) -> Result<Self> {
        if ks.is_empty() {
            return Err(Error::InvalidArgument(
                "Kan needs at least one frequency".into(),
            ));
        }
        let n = n_out * n_in * ks.len();
        let mut rng = SplitMix64::new(seed);
        let scale = 1.0 / ((n_in * ks.len()) as f32).sqrt();
        let phis = (0..n)
            .map(|_| (rng.next_f32() * 2.0 - 1.0) * std::f32::consts::PI)
            .collect();
        let ampls = (0..n)
            .map(|_| (rng.next_f32() * 2.0 - 1.0) * scale)
            .collect();
        Ok(Self {
            npu: npu.clone(),
            n_in,
            n_out,
            ks: ks.to_vec(),
            phis,
            ampls,
            grad_phis: vec![0.0; n],
            grad_ampls: vec![0.0; n],
            input: None,
        })
    }

    /// Arguments `k·x + φ` (reduced to `[−π, π)`) for every `(b, j, i, k)`, flat.
    fn arguments(&self, x: &[f32], batch: usize, shift: f32) -> Vec<f32> {
        let nk = self.ks.len();
        let mut args = Vec::with_capacity(batch * self.n_out * self.n_in * nk);
        for b in 0..batch {
            for j in 0..self.n_out {
                for i in 0..self.n_in {
                    for (k, &kk) in self.ks.iter().enumerate() {
                        let a = kk * x[b * self.n_in + i]
                            + self.phis[(j * self.n_in + i) * nk + k]
                            + shift;
                        args.push(
                            (a + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU)
                                - std::f32::consts::PI,
                        );
                    }
                }
            }
        }
        args
    }

    fn tiled(&self, values: &[f32], batch: usize) -> Vec<f32> {
        (0..batch).flat_map(|_| values.iter().copied()).collect()
    }

    /// `tcos(args) · v` on the NPU.
    fn nl(&self, args: &[f32], v: &[f32]) -> Result<Vec<f32>> {
        let n = args.len();
        Ok(ops::periodic_nl(
            &self.npu,
            &Tensor::from_f32(args, [n])?,
            &Tensor::from_f32(v, [n])?,
        )?
        .to_f32_vec())
    }
}

impl Layer for Kan {
    fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        let [batch, n] = x.dims::<2>()?;
        ensure_shape!(
            n == self.n_in,
            "Kan: expected {} inputs, got {n}",
            self.n_in
        );
        let xs = x.to_f32_vec();
        let terms = self.nl(
            &self.arguments(&xs, batch, 0.0),
            &self.tiled(&self.ampls, batch),
        )?;
        let per = self.n_in * self.ks.len();
        let y: Vec<f32> = terms.chunks(per).map(|c| c.iter().sum()).collect();
        self.input = Some(x.clone());
        Tensor::from_f32(&y, [batch, self.n_out])
    }

    fn backward(&mut self, grad: &Tensor) -> Result<Tensor> {
        let x = take(&mut self.input, "Kan")?;
        let [batch, o] = grad.dims::<2>()?;
        ensure_shape!(
            o == self.n_out,
            "Kan: gradient has {o} outputs, expected {}",
            self.n_out
        );
        let g = grad.to_f32_vec();
        let xs = x.to_f32_vec();
        let nk = self.ks.len();
        let n = self.n_out * self.n_in * nk;
        let ones = vec![1.0; batch * n];
        // cos(θ) and −sin(θ) = cos(θ + π/2), both on the NPU.
        let cos = self.nl(&self.arguments(&xs, batch, 0.0), &ones)?;
        let neg_sin = self.nl(&self.arguments(&xs, batch, FRAC_PI_2), &ones)?;
        let mut dx = vec![0.0f32; batch * self.n_in];
        for b in 0..batch {
            for j in 0..self.n_out {
                let gj = g[b * self.n_out + j];
                for i in 0..self.n_in {
                    for k in 0..nk {
                        let p = (j * self.n_in + i) * nk + k;
                        let t = b * n + p;
                        self.grad_ampls[p] += gj * cos[t];
                        self.grad_phis[p] += gj * self.ampls[p] * neg_sin[t];
                        dx[b * self.n_in + i] += gj * self.ampls[p] * self.ks[k] * neg_sin[t];
                    }
                }
            }
        }
        Tensor::from_f32(&dx, [batch, self.n_in])
    }

    fn params(&mut self) -> Vec<Param<'_>> {
        let shape = vec![self.n_out, self.n_in, self.ks.len()];
        vec![
            Param {
                name: "phis",
                value: &mut self.phis,
                grad: &mut self.grad_phis,
                shape: shape.clone(),
            },
            Param {
                name: "ampls",
                value: &mut self.ampls,
                grad: &mut self.grad_ampls,
                shape,
            },
        ]
    }

    fn buffers(&self) -> Vec<(&'static str, Tensor)> {
        vec![("ks", Tensor::vector(&self.ks))]
    }

    fn name(&self) -> &'static str {
        "Kan"
    }
}

macro_rules! activation {
    ($(#[$doc:meta])* $name:ident, |$x:ident| $f:expr, |$y:ident, $xx:ident| $df:expr) => {
        $(#[$doc])*
        #[derive(Debug, Default)]
        pub struct $name {
            input: Option<Vec<f32>>,
            output: Option<Vec<f32>>,
        }

        impl Layer for $name {
            fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
                let xs = x.to_f32_vec();
                let ys: Vec<f32> = xs.iter().map(|&$x| $f).collect();
                let out = Tensor::from_f32(&ys, x.shape().to_vec())?;
                self.input = Some(xs);
                self.output = Some(ys);
                Ok(out)
            }

            fn backward(&mut self, grad: &Tensor) -> Result<Tensor> {
                let xs = take(&mut self.input, stringify!($name))?;
                let ys = take(&mut self.output, stringify!($name))?;
                let g = grad.to_f32_vec();
                let d: Vec<f32> = g
                    .iter()
                    .zip(xs.iter().zip(&ys))
                    .map(|(g, (&$xx, &$y))| g * $df)
                    .collect();
                Tensor::from_f32(&d, grad.shape().to_vec())
            }

            fn name(&self) -> &'static str {
                stringify!($name)
            }
        }
    };
}

activation!(
    /// `max(x, 0)`.
    Relu, |x| x.max(0.0), |_y, x| if x > 0.0 { 1.0 } else { 0.0 }
);
activation!(
    /// Hyperbolic tangent.
    Tanh, |x| x.tanh(), |y, _x| 1.0 - y * y
);
activation!(
    /// Logistic sigmoid.
    Sigmoid, |x| 1.0 / (1.0 + (-x).exp()), |y, _x| y * (1.0 - y)
);

/// `sin(ω₀·x)` (the SIREN activation), computed on the host.
#[derive(Debug)]
pub struct Sine {
    /// Frequency `ω₀` (SIREN uses 30 for the first layer).
    pub omega0: f32,
    input: Option<Vec<f32>>,
}

impl Sine {
    /// `sin(omega0 · x)`.
    pub fn new(omega0: f32) -> Self {
        Self {
            omega0,
            input: None,
        }
    }
}

impl Layer for Sine {
    fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        let xs = x.to_f32_vec();
        let ys: Vec<f32> = xs.iter().map(|v| (self.omega0 * v).sin()).collect();
        self.input = Some(xs);
        Tensor::from_f32(&ys, x.shape().to_vec())
    }

    fn backward(&mut self, grad: &Tensor) -> Result<Tensor> {
        let xs = take(&mut self.input, "Sine")?;
        let d: Vec<f32> = grad
            .to_f32_vec()
            .iter()
            .zip(&xs)
            .map(|(g, x)| g * self.omega0 * (self.omega0 * x).cos())
            .collect();
        Tensor::from_f32(&d, grad.shape().to_vec())
    }

    fn name(&self) -> &'static str {
        "Sine"
    }
}

/// Layers applied in order, trained together.
#[derive(Debug, Default)]
pub struct Model {
    layers: Vec<Box<dyn Layer>>,
}

impl Model {
    /// An empty model.
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a layer (builder style).
    #[allow(clippy::should_implement_trait)]
    pub fn add(mut self, layer: impl Layer + 'static) -> Self {
        self.layers.push(Box::new(layer));
        self
    }

    /// Forward pass through all layers (keeps what backward needs).
    pub fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        let mut cur = x.clone();
        for l in &mut self.layers {
            cur = l.forward(&cur)?;
        }
        Ok(cur)
    }

    /// Backward pass from `∂loss/∂output`; accumulates gradients.
    pub fn backward(&mut self, grad: &Tensor) -> Result<Tensor> {
        let mut g = grad.clone();
        for l in self.layers.iter_mut().rev() {
            g = l.backward(&g)?;
        }
        Ok(g)
    }

    /// Applies one optimizer step, clears the gradients and refreshes the layers.
    pub fn step(&mut self, opt: &mut dyn Optimizer) -> Result<()> {
        let mut slot = 0;
        for l in &mut self.layers {
            for p in l.params() {
                opt.update(slot, p.value, p.grad);
                p.grad.fill(0.0);
                slot += 1;
            }
            l.sync()?;
        }
        opt.finish_step();
        Ok(())
    }

    /// Scales all gradients so their combined L2 norm is at most `max_norm`
    /// (like `torch.nn.utils.clip_grad_norm_`). Returns the norm before
    /// clipping. Call between `backward` and `step`; it keeps training stable
    /// when gradients are noisy, e.g. on analog hardware.
    pub fn clip_grad_norm(&mut self, max_norm: f32) -> f32 {
        let mut sq = 0.0f64;
        for l in &mut self.layers {
            for p in l.params() {
                sq += p.grad.iter().map(|g| (*g as f64).powi(2)).sum::<f64>();
            }
        }
        let norm = sq.sqrt() as f32;
        if norm > max_norm && norm > 0.0 {
            let scale = max_norm / norm;
            for l in &mut self.layers {
                for p in l.params() {
                    p.grad.iter_mut().for_each(|g| *g *= scale);
                }
            }
        }
        norm
    }

    /// Total number of trainable values.
    pub fn num_params(&mut self) -> usize {
        self.layers
            .iter_mut()
            .flat_map(|l| l.params())
            .map(|p| p.value.len())
            .sum()
    }

    /// Parameters as a state dict named `"{layer index}.{param}"`, the
    /// convention of PyTorch's `nn.Sequential`. Inference layers load it, e.g.
    /// `nn::Linear::load(&npu, &sd, "0")`.
    pub fn state_dict(&mut self) -> Result<StateDict> {
        let mut sd = StateDict::new();
        for (i, l) in self.layers.iter_mut().enumerate() {
            for (name, t) in l.buffers() {
                sd.insert(format!("{i}.{name}"), t);
            }
            for p in l.params() {
                sd.insert(
                    format!("{i}.{}", p.name),
                    Tensor::from_f32(p.value, p.shape.clone())?,
                );
            }
        }
        Ok(sd)
    }
}

/// Updates parameters from their gradients.
pub trait Optimizer {
    /// Updates parameter number `slot` (stable across steps) in place.
    fn update(&mut self, slot: usize, value: &mut [f32], grad: &[f32]);
    /// Called once after all parameters of a step were updated.
    fn finish_step(&mut self) {}
}

/// Stochastic gradient descent with optional momentum.
#[derive(Debug, Clone)]
pub struct Sgd {
    /// Learning rate.
    pub lr: f32,
    /// Momentum factor (0 = plain SGD).
    pub momentum: f32,
    velocity: Vec<Vec<f32>>,
}

impl Sgd {
    /// SGD with learning rate `lr` and momentum `momentum`.
    pub fn new(lr: f32, momentum: f32) -> Self {
        Self {
            lr,
            momentum,
            velocity: Vec::new(),
        }
    }
}

impl Optimizer for Sgd {
    fn update(&mut self, slot: usize, value: &mut [f32], grad: &[f32]) {
        if self.velocity.len() <= slot {
            self.velocity.resize(slot + 1, Vec::new());
        }
        let v = &mut self.velocity[slot];
        if v.len() != value.len() {
            *v = vec![0.0; value.len()];
        }
        for ((w, g), v) in value.iter_mut().zip(grad).zip(v.iter_mut()) {
            *v = self.momentum * *v + g;
            *w -= self.lr * *v;
        }
    }
}

/// Adam (Kingma & Ba) with PyTorch's defaults `β = (0.9, 0.999)`, `ε = 1e-8`.
#[derive(Debug, Clone)]
pub struct Adam {
    /// Learning rate.
    pub lr: f32,
    /// Decay of the first moment.
    pub beta1: f32,
    /// Decay of the second moment.
    pub beta2: f32,
    /// Numerical stabiliser.
    pub eps: f32,
    /// Decoupled weight decay (AdamW); 0 disables it.
    pub weight_decay: f32,
    t: i32,
    m: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
}

impl Adam {
    /// Adam with learning rate `lr`.
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            t: 0,
            m: Vec::new(),
            v: Vec::new(),
        }
    }
}

impl Optimizer for Adam {
    fn update(&mut self, slot: usize, value: &mut [f32], grad: &[f32]) {
        if self.m.len() <= slot {
            self.m.resize(slot + 1, Vec::new());
            self.v.resize(slot + 1, Vec::new());
        }
        if self.m[slot].len() != value.len() {
            self.m[slot] = vec![0.0; value.len()];
            self.v[slot] = vec![0.0; value.len()];
        }
        let t = self.t + 1;
        let (c1, c2) = (1.0 - self.beta1.powi(t), 1.0 - self.beta2.powi(t));
        for (((w, g), m), v) in value
            .iter_mut()
            .zip(grad)
            .zip(self.m[slot].iter_mut())
            .zip(self.v[slot].iter_mut())
        {
            *m = self.beta1 * *m + (1.0 - self.beta1) * g;
            *v = self.beta2 * *v + (1.0 - self.beta2) * g * g;
            *w -= self.lr * ((*m / c1) / ((*v / c2).sqrt() + self.eps) + self.weight_decay * *w);
        }
    }

    fn finish_step(&mut self) {
        self.t += 1;
    }
}

/// Mean squared error and its gradient with respect to `pred`.
pub fn mse(pred: &Tensor, target: &Tensor) -> Result<(f32, Tensor)> {
    ensure_shape!(
        pred.shape() == target.shape(),
        "mse: shapes differ: {:?} vs {:?}",
        pred.shape(),
        target.shape()
    );
    let (p, t) = (pred.to_f32_vec(), target.to_f32_vec());
    let n = p.len().max(1) as f32;
    let loss = p.iter().zip(&t).map(|(a, b)| (a - b).powi(2)).sum::<f32>() / n;
    let grad: Vec<f32> = p.iter().zip(&t).map(|(a, b)| 2.0 * (a - b) / n).collect();
    Ok((loss, Tensor::from_f32(&grad, pred.shape().to_vec())?))
}

/// Softmax cross-entropy for `logits` `[batch, classes]` and integer labels;
/// returns the mean loss and its gradient with respect to the logits.
pub fn cross_entropy(logits: &Tensor, labels: &[usize]) -> Result<(f32, Tensor)> {
    let [b, c] = logits.dims::<2>()?;
    ensure_shape!(
        labels.len() == b,
        "cross_entropy: {} labels for a batch of {b}",
        labels.len()
    );
    let l = logits.to_f32_vec();
    let mut grad = vec![0.0f32; b * c];
    let mut loss = 0.0;
    for (i, &y) in labels.iter().enumerate() {
        if y >= c {
            return Err(Error::InvalidArgument(format!(
                "label {y} out of range for {c} classes"
            )));
        }
        let row = &l[i * c..(i + 1) * c];
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = row.iter().map(|v| (v - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        loss += sum.ln() + max - row[y];
        for (k, e) in exps.iter().enumerate() {
            grad[i * c + k] = (e / sum - if k == y { 1.0 } else { 0.0 }) / b as f32;
        }
    }
    Ok((loss / b as f32, Tensor::from_f32(&grad, [b, c])?))
}

/// Fraction of rows whose largest logit is at the label.
pub fn accuracy(logits: &Tensor, labels: &[usize]) -> f32 {
    let pred = logits.argmax_last();
    let hits = pred.iter().zip(labels).filter(|(p, l)| p == l).count();
    hits as f32 / labels.len().max(1) as f32
}
