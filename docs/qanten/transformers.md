# Transformers (`qanten::nn::transformer`)

Decoder-only (GPT-style) language models: embeddings, multi-head causal self-attention
with a key/value cache, pre-LayerNorm blocks, and text generation.

Parameter names follow the **TinyGPT in Q.ANT's community examples**
(`qant_native_computing_toolkit_examples/community/python/small_lm`), which follows
nanoGPT. A model trained there loads directly.

## Running a model

```rust
use qanten::nn::transformer::{Gpt, GptConfig, Sampling};
use qanten::Npu;

fn main() -> qanten::Result<()> {
    let npu = Npu::auto(0)?;
    let config = GptConfig::tiny(65);             // 65 characters, Q.ANT's TinyGPT defaults
    let model = Gpt::random(&npu, config, 1)?;     // or Gpt::load(&npu, &state_dict, config)

    let logits = model.logits(&[10, 20, 30, 40, 50, 60, 1, 2], 2)?;  // 2 sequences of 4
    assert_eq!(logits.shape(), &[2, 4, 65]);

    let out = model.generate(&[10, 20], 50, &Sampling::with_temperature(0.8, Some(10), 7))?;
    assert_eq!(out.len(), 52);
    Ok(())
}
```

## Loading Q.ANT's TinyGPT

After running their `train.py` (which saves `small_lm.pt`), convert once in Python:

```python
import torch
from safetensors.torch import save_file
ck = torch.load("small_lm.pt")
save_file(ck["state_dict"], "small_lm.safetensors")
```

Then:

```sh
cargo run --release -p qanten --example tiny_gpt --features safetensors -- \
    --weights small_lm.safetensors --text input.txt --prompt "ROMEO:" --tokens 300
```

`--text` is the Tiny Shakespeare file the model was trained on; its sorted distinct
characters are the vocabulary, as in their `CharTokenizer`.

## Configuration

| `GptConfig` field | Meaning | `tiny()` |
|---|---|---|
| `vocab_size` | number of tokens | argument |
| `block_size` | maximum context (position embedding rows) | 64 |
| `n_embd` | model width | 64 |
| `n_head` | attention heads (`n_embd % n_head == 0`) | 4 |
| `n_layer` | transformer blocks | 2 |
| `activation` | `Activation::Relu` (TinyGPT) or `Gelu` (GPT-2 / nanoGPT) | `Relu` |
| `layer_norm_eps` | layer-norm epsilon | `1e-5` |

Expected tensors: `token_embedding.weight`, `position_embedding.weight`,
`blocks.{i}.ln_1.*`, `blocks.{i}.attn.qkv.*` (`[3d, d]`, fused Q/K/V),
`blocks.{i}.attn.proj.*`, `blocks.{i}.ln_2.*`, `blocks.{i}.mlp.fc.*`, `blocks.{i}.mlp.proj.*`,
`ln_f.*`, `lm_head.weight` (bias optional).

## Building blocks

| Type | Purpose |
|---|---|
| `Embedding` | `lookup(&[ids])` → `[n, dim]` rows (`nn.Embedding`) |
| `CausalSelfAttention` | multi-head causal attention; `forward_full` for whole sequences, `forward_cached` with a `KvCache` |
| `Mlp` | `proj(act(fc(x)))` |
| `TransformerBlock` | `x + attn(ln_1(x))`, then `x + mlp(ln_2(x))` |
| `Gpt` | the full model: `logits`, `step` (cached), `generate` |
| `Sampling` | `greedy()` or `with_temperature(t, top_k, seed)`; reproducible |

All blocks implement `Module` on `[batch, time, dim]` tensors.

## What runs where

| NPU | Host |
|---|---|
| Q/K/V projection, output projection, MLP (resident weights), the `Q·Kᵀ` and `P·V` products of every head | embedding lookup, layer norm, causal mask + softmax, residual adds, sampling |

`Linear` layers accept any number of leading dimensions (`[..., in]` → `[..., out]`),
so `[batch, time, dim]` goes through one NPU call per projection.

## Generation and the KV cache

`generate` feeds the prompt once, then each new token on its own: keys and values of
earlier positions are cached per layer and head, so a step costs one position of work
instead of the whole sequence. When the context is full, the oldest tokens are dropped
and the cache is rebuilt from the most recent half of the window. `step(&ids, &mut cache)`
exposes the same mechanism for custom decoding loops.

## Accuracy

Tests compare logits against a plain f32 reference written from the PyTorch
definitions: within 5 % of the logit scale, and the cached path matches a full forward
pass within 0.05.
