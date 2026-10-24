//! Decoder-only transformers (GPT style).
//!
//! The layer and parameter names match the TinyGPT in Q.ANT's community
//! examples (`small_lm.py`), which in turn follows nanoGPT: a model trained
//! there and saved with `safetensors.torch.save_file(model.state_dict(), ...)`
//! loads with [`Gpt::load`].
//!
//! What runs where: the Q/K/V, output and MLP projections and the
//! `Q·Kᵀ` / `P·V` attention products run on the NPU; embeddings, layer norm,
//! softmax, masking and residual additions run on the host.
//!
//! ```
//! # fn main() -> qanten::Result<()> {
//! use qanten::nn::transformer::{Gpt, GptConfig, Sampling};
//! let npu = qanten::Npu::simulator();
//! let model = Gpt::random(&npu, GptConfig::tiny(32), 7)?;   // or Gpt::load(&npu, &state_dict, config)
//! let logits = model.logits(&[1, 2, 3], 1)?;                  // [batch, time, vocab]
//! assert_eq!(logits.shape(), &[1, 3, 32]);
//! let tokens = model.generate(&[1, 2, 3], 5, &Sampling::greedy())?;
//! assert_eq!(tokens.len(), 3 + 5);
//! # Ok(()) }
//! ```

use half::bf16;

use super::{LayerNorm, Linear, Module, StateDict};
use crate::device::Npu;
use crate::error::{Error, Result, ensure_shape};
use crate::ops;
use crate::rng::SplitMix64;
use crate::tensor::Tensor;

fn key(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_owned()
    } else {
        format!("{prefix}.{name}")
    }
}

/// Activation used inside the transformer MLP.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// `max(x, 0)` (Q.ANT's TinyGPT).
    Relu,
    /// GELU, tanh approximation (GPT-2, nanoGPT).
    Gelu,
}

/// Hyper-parameters of a [`Gpt`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GptConfig {
    /// Number of distinct tokens.
    pub vocab_size: usize,
    /// Maximum context length (size of the position embedding).
    pub block_size: usize,
    /// Embedding (model) dimension.
    pub n_embd: usize,
    /// Attention heads per block (`n_embd` must be divisible by it).
    pub n_head: usize,
    /// Number of transformer blocks.
    pub n_layer: usize,
    /// MLP activation.
    pub activation: Activation,
    /// Layer-norm epsilon (PyTorch default `1e-5`).
    pub layer_norm_eps: f32,
}

impl GptConfig {
    /// The defaults of Q.ANT's TinyGPT: context 64, width 64, 4 heads, 2 layers, ReLU.
    pub fn tiny(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            block_size: 64,
            n_embd: 64,
            n_head: 4,
            n_layer: 2,
            activation: Activation::Relu,
            layer_norm_eps: 1e-5,
        }
    }

    fn validate(&self) -> Result<()> {
        if self.n_head == 0 || !self.n_embd.is_multiple_of(self.n_head) {
            return Err(Error::InvalidArgument(format!(
                "n_embd ({}) must be a positive multiple of n_head ({})",
                self.n_embd, self.n_head
            )));
        }
        if self.vocab_size == 0 || self.block_size == 0 {
            return Err(Error::InvalidArgument(
                "vocab_size and block_size must be > 0".into(),
            ));
        }
        Ok(())
    }
}

/// A lookup table of vectors, one row per token (PyTorch's `nn.Embedding`).
#[derive(Debug, Clone)]
pub struct Embedding {
    weight: Tensor,
}

impl Embedding {
    /// `weight` is `[num_embeddings, dim]`.
    pub fn new(weight: Tensor) -> Result<Self> {
        weight.dims::<2>()?;
        Ok(Self { weight })
    }

    /// Loads `{name}` (e.g. `"token_embedding.weight"`).
    pub fn load(sd: &StateDict, name: &str) -> Result<Self> {
        Self::new(sd.get(name)?.clone())
    }

    /// Number of rows.
    pub fn num_embeddings(&self) -> usize {
        self.weight.shape()[0]
    }

    /// Vector size.
    pub fn dim(&self) -> usize {
        self.weight.shape()[1]
    }

    /// Rows for `ids` → `[ids.len(), dim]`. Errors on an out-of-range id.
    pub fn lookup(&self, ids: &[usize]) -> Result<Tensor> {
        let d = self.dim();
        let mut data = Vec::with_capacity(ids.len() * d);
        for &id in ids {
            if id >= self.num_embeddings() {
                return Err(Error::InvalidArgument(format!(
                    "token id {id} out of range (vocabulary has {})",
                    self.num_embeddings()
                )));
            }
            data.extend_from_slice(&self.weight.data()[id * d..(id + 1) * d]);
        }
        Tensor::new(data, [ids.len(), d])
    }
}

/// Cached keys and values of one attention layer, for incremental decoding
/// of a single sequence.
#[derive(Debug, Clone, Default)]
pub struct KvCache {
    /// Per head: `[t, head_dim]` keys, row-major.
    keys: Vec<Vec<bf16>>,
    /// Per head: `[t, head_dim]` values, row-major.
    values: Vec<Vec<bf16>>,
    len: usize,
}

impl KvCache {
    /// Number of cached positions.
    pub fn len(&self) -> usize {
        self.len
    }

    /// `true` if nothing is cached.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Forgets everything.
    pub fn clear(&mut self) {
        *self = Self::default();
    }
}

/// Multi-head causal self-attention with a fused `qkv` projection
/// (`[3·d, d]`) and an output projection `proj` (`[d, d]`).
#[derive(Debug)]
pub struct CausalSelfAttention {
    npu: Npu,
    qkv: Linear,
    proj: Linear,
    heads: usize,
}

impl CausalSelfAttention {
    /// From already-built projections.
    pub fn new(npu: &Npu, qkv: Linear, proj: Linear, heads: usize) -> Result<Self> {
        let d = proj.out_features();
        ensure_shape!(
            qkv.out_features() == 3 * d && qkv.in_features() == d && proj.in_features() == d,
            "attention: qkv must be [3d, d] and proj [d, d], got qkv {}x{} and proj {}x{}",
            qkv.out_features(),
            qkv.in_features(),
            proj.out_features(),
            proj.in_features()
        );
        if heads == 0 || !d.is_multiple_of(heads) {
            return Err(Error::InvalidArgument(format!(
                "{d} is not divisible by {heads} heads"
            )));
        }
        Ok(Self {
            npu: npu.clone(),
            qkv,
            proj,
            heads,
        })
    }

    /// Loads `{prefix}.qkv.weight/bias` and `{prefix}.proj.weight/bias`.
    pub fn load(npu: &Npu, sd: &StateDict, prefix: &str, heads: usize) -> Result<Self> {
        Self::new(
            npu,
            Linear::load(npu, sd, &key(prefix, "qkv"))?,
            Linear::load(npu, sd, &key(prefix, "proj"))?,
            heads,
        )
    }

    fn dim(&self) -> usize {
        self.proj.out_features()
    }

    /// Full causal attention over `x` (`[batch, time, d]`).
    pub fn forward_full(&self, x: &Tensor) -> Result<Tensor> {
        let [b, t, d] = x.dims::<3>()?;
        ensure_shape!(
            d == self.dim(),
            "attention: expected width {}, got {d}",
            self.dim()
        );
        let qkv = self.qkv.forward(x)?.to_f32_vec(); // [b, t, 3d]
        let mut y = vec![0.0f32; b * t * d];
        for bi in 0..b {
            let rows = &qkv[bi * t * 3 * d..(bi + 1) * t * 3 * d];
            for h in 0..self.heads {
                let (q, k, v) = self.split_head(rows, t, h);
                let out = self.attend(&q, &k, &v, t, 0)?;
                self.write_head(&mut y[bi * t * d..(bi + 1) * t * d], &out, t, h);
            }
        }
        self.proj.forward(&Tensor::from_f32(&y, [b, t, d])?)
    }

    /// Attention for `x` (`[1, t_new, d]`) continuing a sequence whose earlier
    /// keys and values are in `cache`; appends the new ones.
    pub fn forward_cached(&self, x: &Tensor, cache: &mut KvCache) -> Result<Tensor> {
        let [b, t, d] = x.dims::<3>()?;
        ensure_shape!(
            b == 1,
            "cached attention handles one sequence, got batch {b}"
        );
        ensure_shape!(
            d == self.dim(),
            "attention: expected width {}, got {d}",
            self.dim()
        );
        let qkv = self.qkv.forward(x)?.to_f32_vec();
        if cache.keys.len() != self.heads {
            cache.keys = vec![Vec::new(); self.heads];
            cache.values = vec![Vec::new(); self.heads];
        }
        let offset = cache.len;
        let mut y = vec![0.0f32; t * d];
        for h in 0..self.heads {
            let (q, k, v) = self.split_head(&qkv, t, h);
            cache.keys[h].extend(k.iter().map(|&x| bf16::from_f32(x)));
            cache.values[h].extend(v.iter().map(|&x| bf16::from_f32(x)));
            let k_all: Vec<f32> = cache.keys[h].iter().map(|x| x.to_f32()).collect();
            let v_all: Vec<f32> = cache.values[h].iter().map(|x| x.to_f32()).collect();
            let out = self.attend(&q, &k_all, &v_all, t, offset)?;
            self.write_head(&mut y, &out, t, h);
        }
        cache.len += t;
        self.proj.forward(&Tensor::from_f32(&y, [1, t, d])?)
    }

    /// `(q, k, v)` of head `h` from `[t, 3d]` rows, each `[t, head_dim]`.
    fn split_head(&self, rows: &[f32], t: usize, h: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let d = self.dim();
        let hd = d / self.heads;
        let take = |part: usize| -> Vec<f32> {
            (0..t)
                .flat_map(|i| {
                    let start = i * 3 * d + part * d + h * hd;
                    rows[start..start + hd].iter().copied()
                })
                .collect()
        };
        (take(0), take(1), take(2))
    }

    fn write_head(&self, y: &mut [f32], out: &[f32], t: usize, h: usize) {
        let d = self.dim();
        let hd = d / self.heads;
        for i in 0..t {
            y[i * d + h * hd..i * d + (h + 1) * hd].copy_from_slice(&out[i * hd..(i + 1) * hd]);
        }
    }

    /// softmax(q·kᵀ/√hd + causal mask)·v for one head. `q` holds `t` queries
    /// at absolute positions `offset..offset+t`; `k`, `v` hold all keys so far.
    fn attend(&self, q: &[f32], k: &[f32], v: &[f32], t: usize, offset: usize) -> Result<Vec<f32>> {
        let hd = self.dim() / self.heads;
        let tk = k.len() / hd;
        let scale = 1.0 / (hd as f32).sqrt();
        let q_scaled: Vec<f32> = q.iter().map(|x| x * scale).collect();
        // scores[i, j] = q_i · k_j  (NPU)
        let scores = ops::matvec(
            &self.npu,
            &Tensor::from_f32(k, [tk, hd])?,
            &Tensor::from_f32(&q_scaled, [t, hd])?,
        )?
        .to_f32_vec();
        // Causal softmax on the host.
        let mut p = vec![0.0f32; t * tk];
        for i in 0..t {
            let visible = (offset + i + 1).min(tk);
            let row = &scores[i * tk..i * tk + visible];
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = row.iter().map(|s| (s - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            for (j, e) in exps.into_iter().enumerate() {
                p[i * tk + j] = e / sum;
            }
        }
        // out = p · v  (NPU): matrix = vᵀ [hd, tk], vectors = p [t, tk]
        let mut v_t = vec![0.0f32; hd * tk];
        for j in 0..tk {
            for c in 0..hd {
                v_t[c * tk + j] = v[j * hd + c];
            }
        }
        Ok(ops::matvec(
            &self.npu,
            &Tensor::from_f32(&v_t, [hd, tk])?,
            &Tensor::from_f32(&p, [t, tk])?,
        )?
        .to_f32_vec())
    }
}

impl Module for CausalSelfAttention {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_full(x)
    }
    fn name(&self) -> &'static str {
        "CausalSelfAttention"
    }
}

/// Two-layer MLP: `proj(act(fc(x)))`, `fc` is `[4d, d]` in GPT models.
#[derive(Debug)]
pub struct Mlp {
    fc: Linear,
    proj: Linear,
    activation: Activation,
}

impl Mlp {
    /// From built layers.
    pub fn new(fc: Linear, proj: Linear, activation: Activation) -> Self {
        Self {
            fc,
            proj,
            activation,
        }
    }

    /// Loads `{prefix}.fc.*` and `{prefix}.proj.*`.
    pub fn load(npu: &Npu, sd: &StateDict, prefix: &str, activation: Activation) -> Result<Self> {
        Ok(Self::new(
            Linear::load(npu, sd, &key(prefix, "fc"))?,
            Linear::load(npu, sd, &key(prefix, "proj"))?,
            activation,
        ))
    }
}

impl Module for Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.fc.forward(x)?;
        let h = match self.activation {
            Activation::Relu => ops::relu(&h),
            Activation::Gelu => ops::gelu(&h),
        };
        self.proj.forward(&h)
    }
    fn name(&self) -> &'static str {
        "Mlp"
    }
}

/// Pre-LayerNorm transformer block: `x + attn(ln_1(x))`, then `x + mlp(ln_2(x))`.
#[derive(Debug)]
pub struct TransformerBlock {
    ln_1: LayerNorm,
    attn: CausalSelfAttention,
    ln_2: LayerNorm,
    mlp: Mlp,
}

impl TransformerBlock {
    /// From built parts.
    pub fn new(ln_1: LayerNorm, attn: CausalSelfAttention, ln_2: LayerNorm, mlp: Mlp) -> Self {
        Self {
            ln_1,
            attn,
            ln_2,
            mlp,
        }
    }

    /// Loads `{prefix}.ln_1`, `.attn`, `.ln_2`, `.mlp`.
    pub fn load(npu: &Npu, sd: &StateDict, prefix: &str, config: &GptConfig) -> Result<Self> {
        Ok(Self::new(
            LayerNorm::load(sd, &key(prefix, "ln_1"), config.layer_norm_eps),
            CausalSelfAttention::load(npu, sd, &key(prefix, "attn"), config.n_head)?,
            LayerNorm::load(sd, &key(prefix, "ln_2"), config.layer_norm_eps),
            Mlp::load(npu, sd, &key(prefix, "mlp"), config.activation)?,
        ))
    }

    /// Like `forward`, continuing a cached sequence (batch 1).
    pub fn forward_cached(&self, x: &Tensor, cache: &mut KvCache) -> Result<Tensor> {
        let x = ops::add(x, &self.attn.forward_cached(&self.ln_1.forward(x)?, cache)?)?;
        ops::add(&x, &self.mlp.forward(&self.ln_2.forward(&x)?)?)
    }
}

impl Module for TransformerBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = ops::add(x, &self.attn.forward_full(&self.ln_1.forward(x)?)?)?;
        ops::add(&x, &self.mlp.forward(&self.ln_2.forward(&x)?)?)
    }
    fn name(&self) -> &'static str {
        "TransformerBlock"
    }
}

/// How [`Gpt::generate`] picks the next token.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sampling {
    /// `0` = always the most likely token; higher = more random.
    pub temperature: f32,
    /// Only sample among the `k` most likely tokens.
    pub top_k: Option<usize>,
    /// Random seed (generation is reproducible).
    pub seed: u64,
}

impl Sampling {
    /// Always pick the most likely token.
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: None,
            seed: 0,
        }
    }

    /// Sample with `temperature`, optionally restricted to the `top_k` tokens.
    pub fn with_temperature(temperature: f32, top_k: Option<usize>, seed: u64) -> Self {
        Self {
            temperature,
            top_k,
            seed,
        }
    }

    fn pick(&self, logits: &[f32], rng: &mut SplitMix64) -> usize {
        let argmax = || {
            logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map_or(0, |(i, _)| i)
        };
        if self.temperature <= 0.0 {
            return argmax();
        }
        let mut order: Vec<usize> = (0..logits.len()).collect();
        order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
        order.truncate(self.top_k.unwrap_or(logits.len()).max(1));
        let max = logits[order[0]];
        let weights: Vec<f32> = order
            .iter()
            .map(|&i| ((logits[i] - max) / self.temperature).exp())
            .collect();
        let total: f32 = weights.iter().sum();
        let mut r = rng.next_f32() * total;
        for (i, w) in order.iter().zip(&weights) {
            if r < *w {
                return *i;
            }
            r -= w;
        }
        order[0]
    }
}

/// A decoder-only transformer language model.
#[derive(Debug)]
pub struct Gpt {
    config: GptConfig,
    token_embedding: Embedding,
    position_embedding: Embedding,
    blocks: Vec<TransformerBlock>,
    ln_f: LayerNorm,
    lm_head: Linear,
}

impl Gpt {
    /// Loads a model with TinyGPT / nanoGPT-style names:
    /// `token_embedding.weight`, `position_embedding.weight`,
    /// `blocks.{i}.{ln_1, attn.qkv, attn.proj, ln_2, mlp.fc, mlp.proj}`,
    /// `ln_f`, `lm_head`.
    pub fn load(npu: &Npu, sd: &StateDict, config: GptConfig) -> Result<Self> {
        config.validate()?;
        let token_embedding = Embedding::load(sd, "token_embedding.weight")?;
        let position_embedding = Embedding::load(sd, "position_embedding.weight")?;
        ensure_shape!(
            token_embedding.dim() == config.n_embd && position_embedding.dim() == config.n_embd,
            "embedding width does not match n_embd {}",
            config.n_embd
        );
        ensure_shape!(
            position_embedding.num_embeddings() >= config.block_size,
            "position embedding has {} rows, block_size is {}",
            position_embedding.num_embeddings(),
            config.block_size
        );
        let blocks = (0..config.n_layer)
            .map(|i| TransformerBlock::load(npu, sd, &format!("blocks.{i}"), &config))
            .collect::<Result<_>>()?;
        Ok(Self {
            config,
            token_embedding,
            position_embedding,
            blocks,
            ln_f: LayerNorm::load(sd, "ln_f", config.layer_norm_eps),
            lm_head: Linear::load(npu, sd, "lm_head")?,
        })
    }

    /// A randomly initialised model (for tests, benchmarks and demos).
    pub fn random(npu: &Npu, config: GptConfig, seed: u64) -> Result<Self> {
        config.validate()?;
        let d = config.n_embd;
        let mut sd = StateDict::new();
        let mut s = seed;
        let mut rand = |shape: &[usize], scale: f32| {
            s = s.wrapping_add(1);
            Tensor::rand_uniform(shape.to_vec(), -scale, scale, s)
        };
        sd.insert("token_embedding.weight", rand(&[config.vocab_size, d], 0.5));
        sd.insert(
            "position_embedding.weight",
            rand(&[config.block_size, d], 0.5),
        );
        let w = 1.0 / (d as f32).sqrt();
        for i in 0..config.n_layer {
            let p = format!("blocks.{i}");
            for ln in ["ln_1", "ln_2"] {
                sd.insert(format!("{p}.{ln}.weight"), Tensor::ones([d]));
                sd.insert(format!("{p}.{ln}.bias"), Tensor::zeros([d]));
            }
            sd.insert(format!("{p}.attn.qkv.weight"), rand(&[3 * d, d], w));
            sd.insert(format!("{p}.attn.qkv.bias"), Tensor::zeros([3 * d]));
            sd.insert(format!("{p}.attn.proj.weight"), rand(&[d, d], w));
            sd.insert(format!("{p}.attn.proj.bias"), Tensor::zeros([d]));
            sd.insert(format!("{p}.mlp.fc.weight"), rand(&[4 * d, d], w));
            sd.insert(format!("{p}.mlp.fc.bias"), Tensor::zeros([4 * d]));
            sd.insert(format!("{p}.mlp.proj.weight"), rand(&[d, 4 * d], w / 2.0));
            sd.insert(format!("{p}.mlp.proj.bias"), Tensor::zeros([d]));
        }
        sd.insert("ln_f.weight", Tensor::ones([d]));
        sd.insert("ln_f.bias", Tensor::zeros([d]));
        sd.insert("lm_head.weight", rand(&[config.vocab_size, d], w));
        Self::load(npu, &sd, config)
    }

    /// The model's hyper-parameters.
    pub fn config(&self) -> &GptConfig {
        &self.config
    }

    fn embed(&self, ids: &[usize], batch: usize, offset: usize) -> Result<Tensor> {
        let t = ids.len() / batch.max(1);
        ensure_shape!(
            batch > 0 && t * batch == ids.len(),
            "{} token ids do not split into {batch} sequences",
            ids.len()
        );
        ensure_shape!(
            offset + t <= self.config.block_size,
            "sequence of {} exceeds the context of {}",
            offset + t,
            self.config.block_size
        );
        let tok = self.token_embedding.lookup(ids)?;
        let positions: Vec<usize> = (0..batch).flat_map(|_| offset..offset + t).collect();
        let pos = self.position_embedding.lookup(&positions)?;
        ops::add(&tok, &pos)?.reshape([batch, t, self.config.n_embd])
    }

    /// Next-token logits for every position: `ids` holds `batch` sequences of
    /// equal length, row-major; returns `[batch, time, vocab]`.
    pub fn logits(&self, ids: &[usize], batch: usize) -> Result<Tensor> {
        let mut x = self.embed(ids, batch, 0)?;
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        self.lm_head.forward(&self.ln_f.forward(&x)?)
    }

    /// A fresh cache for [`step`](Self::step), one entry per block.
    pub fn new_cache(&self) -> Vec<KvCache> {
        vec![KvCache::default(); self.blocks.len()]
    }

    /// Feeds `ids` (one sequence) after what `cache` already holds and returns
    /// the logits of the last position (`[vocab]`). Only the new tokens are
    /// processed.
    pub fn step(&self, ids: &[usize], cache: &mut [KvCache]) -> Result<Tensor> {
        ensure_shape!(
            cache.len() == self.blocks.len(),
            "cache has {} layers, model {}",
            cache.len(),
            self.blocks.len()
        );
        let offset = cache.first().map_or(0, KvCache::len);
        let mut x = self.embed(ids, 1, offset)?;
        for (block, c) in self.blocks.iter().zip(cache.iter_mut()) {
            x = block.forward_cached(&x, c)?;
        }
        let logits = self.lm_head.forward(&self.ln_f.forward(&x)?)?;
        let v = self.config.vocab_size;
        let last = logits.data()[(ids.len() - 1) * v..ids.len() * v].to_vec();
        Tensor::new(last, [v])
    }

    /// Continues `prompt` by `max_new_tokens` tokens and returns prompt +
    /// continuation. Uses a key/value cache, so each new token only costs one
    /// position of work. When the context fills up, the oldest tokens are
    /// dropped and the cache is rebuilt from the most recent ones.
    pub fn generate(
        &self,
        prompt: &[usize],
        max_new_tokens: usize,
        sampling: &Sampling,
    ) -> Result<Vec<usize>> {
        if prompt.is_empty() {
            return Err(Error::InvalidArgument(
                "generate needs a non-empty prompt".into(),
            ));
        }
        let ctx = self.config.block_size;
        let mut tokens = prompt.to_vec();
        let mut rng = SplitMix64::new(sampling.seed);
        let mut cache = self.new_cache();
        let start = tokens.len().saturating_sub(ctx);
        let mut logits = self.step(&tokens[start..], &mut cache)?;
        for _ in 0..max_new_tokens {
            let next = sampling.pick(&logits.to_f32_vec(), &mut rng);
            tokens.push(next);
            if cache[0].len() >= ctx {
                // Slide the window: keep the most recent half of the context.
                cache.iter_mut().for_each(KvCache::clear);
                let keep = (ctx / 2).max(1);
                logits = self.step(&tokens[tokens.len() - keep..], &mut cache)?;
            } else {
                logits = self.step(&[next], &mut cache)?;
            }
        }
        Ok(tokens)
    }
}
