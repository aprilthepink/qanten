//! The NPU transformer against a plain f32 reference written from the PyTorch definitions.

use qanten::nn::StateDict;
use qanten::nn::transformer::{Activation, Gpt, GptConfig, Sampling};
use qanten::{Npu, Tensor};

fn get(sd: &StateDict, k: &str) -> Vec<f32> {
    sd.get(k).unwrap().to_f32_vec()
}

fn linear(x: &[f32], w: &[f32], b: &[f32], n_in: usize, n_out: usize) -> Vec<f32> {
    let rows = x.len() / n_in;
    let mut y = vec![0.0; rows * n_out];
    for r in 0..rows {
        for o in 0..n_out {
            y[r * n_out + o] = b[o]
                + (0..n_in)
                    .map(|i| x[r * n_in + i] * w[o * n_in + i])
                    .sum::<f32>();
        }
    }
    y
}

fn layer_norm(x: &[f32], g: &[f32], b: &[f32], d: usize) -> Vec<f32> {
    x.chunks(d)
        .flat_map(|row| {
            let m = row.iter().sum::<f32>() / d as f32;
            let v = row.iter().map(|x| (x - m).powi(2)).sum::<f32>() / d as f32;
            row.iter()
                .enumerate()
                .map(move |(i, x)| (x - m) / (v + 1e-5).sqrt() * g[i] + b[i])
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Reference forward of TinyGPT for one sequence; returns [t, vocab] logits.
fn reference(sd: &StateDict, c: &GptConfig, ids: &[usize]) -> Vec<f32> {
    let (d, t, h) = (c.n_embd, ids.len(), c.n_head);
    let hd = d / h;
    let (te, pe) = (
        get(sd, "token_embedding.weight"),
        get(sd, "position_embedding.weight"),
    );
    let mut x: Vec<f32> = (0..t)
        .flat_map(|i| (0..d).map(move |j| (i, j)))
        .map(|(i, j)| te[ids[i] * d + j] + pe[i * d + j])
        .collect();
    for l in 0..c.n_layer {
        let p = |n: &str| get(sd, &format!("blocks.{l}.{n}"));
        let a = layer_norm(&x, &p("ln_1.weight"), &p("ln_1.bias"), d);
        let qkv = linear(&a, &p("attn.qkv.weight"), &p("attn.qkv.bias"), d, 3 * d);
        let mut att = vec![0.0; t * d];
        for head in 0..h {
            for i in 0..t {
                let q = |r: usize, c: usize| qkv[r * 3 * d + head * hd + c];
                let k = |r: usize, c: usize| qkv[r * 3 * d + d + head * hd + c];
                let v = |r: usize, c: usize| qkv[r * 3 * d + 2 * d + head * hd + c];
                let s: Vec<f32> = (0..=i)
                    .map(|j| {
                        (0..hd).map(|cc| q(i, cc) * k(j, cc)).sum::<f32>() / (hd as f32).sqrt()
                    })
                    .collect();
                let m = s.iter().copied().fold(f32::MIN, f32::max);
                let e: Vec<f32> = s.iter().map(|x| (x - m).exp()).collect();
                let z: f32 = e.iter().sum();
                for cc in 0..hd {
                    att[i * d + head * hd + cc] = (0..=i).map(|j| e[j] / z * v(j, cc)).sum();
                }
            }
        }
        let o = linear(&att, &p("attn.proj.weight"), &p("attn.proj.bias"), d, d);
        x.iter_mut().zip(o).for_each(|(a, b)| *a += b);
        let a = layer_norm(&x, &p("ln_2.weight"), &p("ln_2.bias"), d);
        let mut hdn = linear(&a, &p("mlp.fc.weight"), &p("mlp.fc.bias"), d, 4 * d);
        hdn.iter_mut().for_each(|v| *v = v.max(0.0));
        let o = linear(&hdn, &p("mlp.proj.weight"), &p("mlp.proj.bias"), 4 * d, d);
        x.iter_mut().zip(o).for_each(|(a, b)| *a += b);
    }
    let x = layer_norm(&x, &get(sd, "ln_f.weight"), &get(sd, "ln_f.bias"), d);
    linear(
        &x,
        &get(sd, "lm_head.weight"),
        &vec![0.0; c.vocab_size],
        d,
        c.vocab_size,
    )
}

fn state_dict(c: &GptConfig) -> StateDict {
    let d = c.n_embd;
    let mut sd = StateDict::new();
    let mut s = 0;
    let mut r = |shape: Vec<usize>, sc: f32| {
        s += 1;
        Tensor::rand_uniform(shape, -sc, sc, s)
    };
    sd.insert("token_embedding.weight", r(vec![c.vocab_size, d], 0.5));
    sd.insert("position_embedding.weight", r(vec![c.block_size, d], 0.5));
    for l in 0..c.n_layer {
        for (n, shape, sc) in [
            ("ln_1.weight", vec![d], 0.2),
            ("ln_1.bias", vec![d], 0.1),
            ("ln_2.weight", vec![d], 0.2),
            ("ln_2.bias", vec![d], 0.1),
            ("attn.qkv.weight", vec![3 * d, d], 0.25),
            ("attn.qkv.bias", vec![3 * d], 0.1),
            ("attn.proj.weight", vec![d, d], 0.25),
            ("attn.proj.bias", vec![d], 0.1),
            ("mlp.fc.weight", vec![4 * d, d], 0.25),
            ("mlp.fc.bias", vec![4 * d], 0.1),
            ("mlp.proj.weight", vec![d, 4 * d], 0.15),
            ("mlp.proj.bias", vec![d], 0.1),
        ] {
            let mut t = r(shape, sc);
            if n.ends_with("ln_1.weight") || n.ends_with("ln_2.weight") {
                t = t.map(|v| 1.0 + v);
            }
            sd.insert(format!("blocks.{l}.{n}"), t);
        }
    }
    sd.insert("ln_f.weight", Tensor::ones([d]));
    sd.insert("ln_f.bias", Tensor::zeros([d]));
    sd.insert("lm_head.weight", r(vec![c.vocab_size, d], 0.25));
    sd
}

fn config() -> GptConfig {
    GptConfig {
        vocab_size: 20,
        block_size: 16,
        n_embd: 32,
        n_head: 4,
        n_layer: 2,
        activation: Activation::Relu,
        layer_norm_eps: 1e-5,
    }
}

#[test]
fn logits_match_reference() {
    let c = config();
    let sd = state_dict(&c);
    let model = Gpt::load(&Npu::simulator(), &sd, c).unwrap();
    let ids = [3, 1, 4, 1, 5, 9, 2, 6];
    let want = reference(&sd, &c, &ids);
    let got = model.logits(&ids, 1).unwrap();
    assert_eq!(got.shape(), &[1, 8, 20]);
    let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let err = got
        .to_f32_vec()
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    assert!(
        err < 0.05 * scale.max(1.0),
        "max error {err} (logit scale {scale})"
    );
    // Batched: two sequences give the same rows as one at a time.
    let both = model
        .logits(&[3, 1, 4, 1, 5, 9, 2, 6, 3, 1, 4, 1, 5, 9, 2, 6], 2)
        .unwrap()
        .to_f32_vec();
    assert_eq!(&both[..160], &both[160..]);
}

#[test]
fn kv_cache_matches_full_forward() {
    let c = config();
    let model = Gpt::load(&Npu::simulator(), &state_dict(&c), c).unwrap();
    let ids = [2usize, 7, 1, 8, 2, 8];
    let full = model.logits(&ids, 1).unwrap().to_f32_vec();
    let mut cache = model.new_cache();
    model.step(&ids[..3], &mut cache).unwrap();
    let mut last = Vec::new();
    for &t in &ids[3..] {
        last = model.step(&[t], &mut cache).unwrap().to_f32_vec();
    }
    let want = &full[5 * 20..6 * 20];
    let err = last
        .iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    assert!(err < 0.05, "cached vs full: {err}");
    assert_eq!(cache[0].len(), 6);
}

#[test]
fn generation() {
    let c = config();
    let model = Gpt::random(&Npu::simulator(), c, 1).unwrap();
    let a = model
        .generate(&[1, 2], 30, &Sampling::with_temperature(0.8, Some(5), 42))
        .unwrap();
    let b = model
        .generate(&[1, 2], 30, &Sampling::with_temperature(0.8, Some(5), 42))
        .unwrap();
    assert_eq!(a, b, "same seed, same text");
    assert_eq!(
        a.len(),
        32,
        "longer than the 16-token context: sliding window works"
    );
    assert!(a.iter().all(|&t| t < 20));
    let g = model.generate(&[1, 2], 3, &Sampling::greedy()).unwrap();
    let first = model.logits(&[1, 2], 1).unwrap().to_f32_vec()[20..40].to_vec();
    let argmax = first
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0;
    assert_eq!(g[2], argmax);
    assert!(model.generate(&[], 1, &Sampling::greedy()).is_err());
    assert!(model.logits(&[99], 1).is_err(), "token id out of range");
}

#[test]
fn rejects_bad_config() {
    let mut c = config();
    c.n_head = 5;
    assert!(Gpt::random(&Npu::simulator(), c, 0).is_err());
}
