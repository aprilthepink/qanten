//! Character-level text generation with a GPT on the NPU.
//!
//! Without arguments it runs a randomly initialised model (gibberish, but it
//! exercises the whole pipeline). With weights trained by Q.ANT's community
//! example (`community/python/small_lm/train.py`) it writes Shakespeare-ish text:
//!
//! ```sh
//! # in Python, once: convert the checkpoint to safetensors
//! #   import torch; from safetensors.torch import save_file
//! #   ck = torch.load("small_lm.pt"); save_file(ck["state_dict"], "small_lm.safetensors")
//! cargo run --release -p qanten --example tiny_gpt --features safetensors -- \
//!     --weights small_lm.safetensors --text input.txt --prompt "ROMEO:" --tokens 300
//! ```
//! `--text` is the training text (Tiny Shakespeare); its sorted characters are the vocabulary.

use std::time::Instant;

use anyhow::{Context, Result, bail};
use qanten::Npu;
use qanten::nn::StateDict;
use qanten::nn::transformer::{Gpt, GptConfig, Sampling};

fn main() -> Result<()> {
    let (mut weights, mut text, mut prompt, mut tokens) =
        (None, None, "ROMEO:".to_owned(), 200usize);
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = || args.next().with_context(|| format!("{a} needs a value"));
        match a.as_str() {
            "--weights" => weights = Some(val()?),
            "--text" => text = Some(std::fs::read_to_string(val()?)?),
            "--prompt" => prompt = val()?,
            "--tokens" => tokens = val()?.parse()?,
            other => bail!("unknown argument {other}"),
        }
    }

    // Character vocabulary: sorted distinct characters of the training text.
    let mut chars: Vec<char> = match &text {
        Some(t) => t.chars().collect(),
        None => "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
            .chars()
            .collect(),
    };
    chars.sort_unstable();
    chars.dedup();
    let encode = |s: &str| -> Result<Vec<usize>> {
        s.chars()
            .map(|c| {
                chars
                    .binary_search(&c)
                    .map_err(|_| anyhow::anyhow!("{c:?} is not in the vocabulary"))
            })
            .collect()
    };

    let npu = Npu::auto(0)?;
    println!("backend: {}", npu.backend().name());
    let config = GptConfig::tiny(chars.len());
    let model = match &weights {
        Some(path) => Gpt::load(&npu, &StateDict::load_safetensors(path)?, config)?,
        None => {
            println!("no --weights given: using a random model (expect gibberish)");
            Gpt::random(&npu, config, 1)?
        }
    };

    let start = Instant::now();
    let out = model.generate(
        &encode(&prompt)?,
        tokens,
        &Sampling::with_temperature(0.8, Some(10), 7),
    )?;
    let secs = start.elapsed().as_secs_f64();
    let text: String = out.iter().map(|&i| chars[i]).collect();
    println!("---\n{text}\n---");
    println!(
        "{tokens} tokens in {secs:.2} s ({:.1} tokens/s)",
        tokens as f64 / secs
    );
    Ok(())
}
