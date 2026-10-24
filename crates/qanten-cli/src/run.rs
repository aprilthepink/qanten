//! `qanten run` and `qanten inspect`.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use qanten::nn::{self, Module, StateDict};
use qanten::onnx::{OnnxModel, OnnxValue};
use qanten::{Npu, Tensor};

/// Activation between layers for `--arch mlp`.
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum Activation {
    Relu,
    Tanh,
    Sigmoid,
    Gelu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Arch {
    /// Linear layers named `<n>.weight` / `<n>.bias` (PyTorch nn.Sequential), activation between.
    Mlp,
}

fn ext(path: &Path) -> String {
    path.extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .unwrap_or_default()
}

fn load_state_dict(path: &Path) -> Result<StateDict> {
    Ok(match ext(path).as_str() {
        "safetensors" => StateDict::load_safetensors(path)?,
        "npz" => StateDict::load_npz(path)?,
        other => bail!("cannot read weights from .{other} files (use .safetensors or .npz)"),
    })
}

/// `qanten inspect FILE`
pub fn inspect(path: &Path, npu: &Npu) -> Result<()> {
    if ext(path) == "onnx" {
        let model = OnnxModel::load(npu, path)?;
        println!("{}", model.summary());
        return Ok(());
    }
    if ext(path) == "npy" {
        let t = qanten::numpy::load_npy(path)?;
        println!(
            "{}: shape {:?}, {} values",
            path.display(),
            t.shape(),
            t.len()
        );
        return Ok(());
    }
    let sd = load_state_dict(path)?;
    let total: usize = sd.iter().map(|(_, t)| t.len()).sum();
    println!("{}: {} tensors, {} values", path.display(), sd.len(), total);
    let width = sd.keys().map(str::len).max().unwrap_or(0);
    for (name, t) in sd.iter() {
        let v = t.to_f32_vec();
        let (lo, hi) = v
            .iter()
            .fold((f32::MAX, f32::MIN), |(a, b), &x| (a.min(x), b.max(x)));
        println!(
            "  {name:<width$}  {:<16} min {lo:>9.4}  max {hi:>9.4}",
            format!("{:?}", t.shape())
        );
    }
    Ok(())
}

/// Builds an MLP from a weights file: every 2-D tensor is a layer `[out, in]`,
/// ordered by the first number in its name (`0.weight`, `fc1.weight`,
/// `l1_w`, …); its bias is the matching `.bias` / `_b` / `bias` tensor if
/// present. Returns the model and its input size.
fn mlp_from_state_dict(
    npu: &Npu,
    sd: &StateDict,
    act: Activation,
) -> Result<(nn::Sequential, usize)> {
    let number = |k: &str| -> Option<usize> {
        let digits: String = k
            .chars()
            .skip_while(|c| !c.is_ascii_digit())
            .take_while(char::is_ascii_digit)
            .collect();
        digits.parse().ok()
    };
    let mut layers: Vec<(usize, String)> = sd
        .iter()
        .filter(|(_, t)| t.rank() == 2)
        .filter_map(|(k, _)| Some((number(k)?, k.to_owned())))
        .collect();
    layers.sort();
    if layers.is_empty() {
        bail!("no numbered 2-D weight matrices found (expected names like `0.weight` or `l1_w`)");
    }
    let bias_of = |w: &str| -> Option<Tensor> {
        [
            w.replace(".weight", ".bias"),
            w.replace("_w", "_b"),
            w.replace("weight", "bias"),
        ]
        .iter()
        .filter(|b| b.as_str() != w)
        .find_map(|b| sd.get_opt(b).cloned())
    };
    let mut model = nn::Sequential::new();
    let mut prev_out = None;
    for (i, (_, name)) in layers.iter().enumerate() {
        let w = sd.get(name)?;
        let [out, inp] = [w.shape()[0], w.shape()[1]];
        if let Some(p) = prev_out
            && p != inp
        {
            bail!("layer `{name}` expects {inp} inputs but the previous layer has {p} outputs");
        }
        prev_out = Some(out);
        println!(
            "  layer {i}: {name} [{out}, {inp}]{}",
            if bias_of(name).is_some() {
                " + bias"
            } else {
                ""
            }
        );
        model.push(nn::Linear::new(npu, w, bias_of(name))?);
        if i + 1 < layers.len() {
            match act {
                Activation::Relu => model.push(nn::ReLU),
                Activation::Tanh => model.push(nn::Tanh),
                Activation::Sigmoid => model.push(nn::Sigmoid),
                Activation::Gelu => model.push(nn::Gelu),
            }
        }
    }
    let in_features = sd.get(&layers[0].1)?.shape()[1];
    Ok((model, in_features))
}

fn parse_inputs(specs: &[String]) -> Result<Vec<(Option<String>, Tensor)>> {
    specs
        .iter()
        .map(|s| {
            let (name, path) = match s.split_once('=') {
                Some((n, p)) => (Some(n.to_owned()), p),
                None => (None, s.as_str()),
            };
            let t =
                qanten::numpy::load_npy(path).with_context(|| format!("reading input {path}"))?;
            Ok((name, t))
        })
        .collect()
}

fn describe(name: &str, t: &Tensor) {
    let v = t.to_f32_vec();
    let (lo, hi) = v
        .iter()
        .fold((f32::MAX, f32::MIN), |(a, b), &x| (a.min(x), b.max(x)));
    let mean = v.iter().sum::<f32>() / v.len().max(1) as f32;
    println!(
        "  {name}: shape {:?}, min {lo:.4}, max {hi:.4}, mean {mean:.4}",
        t.shape()
    );
    if t.rank() == 2 && t.shape()[1] > 1 && t.shape()[0] <= 16 {
        println!("    argmax per row: {:?}", t.argmax_last());
    }
}

/// Settings for `qanten run`.
pub struct RunArgs {
    pub model: PathBuf,
    pub inputs: Vec<String>,
    pub output_dir: Option<PathBuf>,
    pub arch: Option<Arch>,
    pub activation: Activation,
    pub batch: usize,
    pub iters: usize,
    pub profile: bool,
}

/// `qanten run MODEL`
pub fn run(base: &Npu, args: RunArgs) -> Result<()> {
    let (npu, profiler) = base.profiled();
    let inputs = parse_inputs(&args.inputs)?;
    println!("backend: {}", base.backend().name());

    let start = Instant::now();
    let (outputs, per_iter) = match ext(&args.model).as_str() {
        "onnx" => {
            let model = OnnxModel::load(&npu, &args.model)?;
            println!(
                "loaded {} in {:.1} ms",
                args.model.display(),
                start.elapsed().as_secs_f64() * 1e3
            );
            let feeds: Vec<(String, OnnxValue)> = model
                .inputs()
                .iter()
                .enumerate()
                .map(|(i, spec)| {
                    let given = inputs
                        .iter()
                        .find(|(n, _)| n.as_deref() == Some(spec.name.as_str()))
                        .or_else(|| inputs.get(i).filter(|(n, _)| n.is_none()));
                    let t = match given {
                        Some((_, t)) => t.clone(),
                        None => {
                            let shape: Vec<usize> = spec
                                .shape
                                .iter()
                                .enumerate()
                                .map(|(d, s)| s.unwrap_or(if d == 0 { args.batch } else { 1 }))
                                .collect();
                            println!(
                                "  no input for `{}`: using random data of shape {shape:?}",
                                spec.name
                            );
                            if spec.is_integer {
                                Tensor::zeros(shape)
                            } else {
                                Tensor::rand_uniform(shape, 0.0, 1.0, 42)
                            }
                        }
                    };
                    let v = if spec.is_integer {
                        OnnxValue::Int {
                            data: t.to_f32_vec().iter().map(|&x| x.round() as i64).collect(),
                            shape: t.shape().to_vec(),
                        }
                    } else {
                        OnnxValue::Float(t)
                    };
                    (spec.name.clone(), v)
                })
                .collect();
            let refs: Vec<(&str, OnnxValue)> =
                feeds.iter().map(|(n, v)| (n.as_str(), v.clone())).collect();
            profiler.reset();
            let t0 = Instant::now();
            let mut out = Vec::new();
            for _ in 0..args.iters.max(1) {
                out = model.run(&refs)?;
            }
            let per = t0.elapsed() / args.iters.max(1) as u32;
            let tensors = out
                .into_iter()
                .map(|(n, v)| Ok((n, v.into_tensor()?)))
                .collect::<Result<Vec<_>>>()?;
            (tensors, per)
        }
        "safetensors" | "npz" => {
            let sd = load_state_dict(&args.model)?;
            let arch = args
                .arch
                .context("weight files need an architecture: add --arch mlp")?;
            let Arch::Mlp = arch;
            let (model, in_features) = mlp_from_state_dict(&npu, &sd, args.activation)?;
            println!(
                "loaded {}-layer MLP from {}",
                model.len().div_ceil(2),
                args.model.display()
            );
            let x = match inputs.first() {
                Some((_, t)) => t.clone(),
                None => {
                    println!(
                        "  no input given: using random data of shape [{}, {in_features}]",
                        args.batch
                    );
                    Tensor::rand_uniform([args.batch, in_features], 0.0, 1.0, 42)
                }
            };
            profiler.reset();
            let t0 = Instant::now();
            let mut y = Tensor::zeros([0]);
            for _ in 0..args.iters.max(1) {
                y = model.forward(&x)?;
            }
            (
                vec![("output".to_owned(), y)],
                t0.elapsed() / args.iters.max(1) as u32,
            )
        }
        other => bail!("unsupported model format .{other} (use .onnx, .safetensors or .npz)"),
    };

    println!(
        "ran in {:.3} ms per run ({} run{})",
        per_iter.as_secs_f64() * 1e3,
        args.iters.max(1),
        if args.iters > 1 { "s" } else { "" }
    );
    for (name, t) in &outputs {
        describe(name, t);
    }
    if let Some(dir) = &args.output_dir {
        std::fs::create_dir_all(dir)?;
        for (name, t) in &outputs {
            let safe: String = name
                .chars()
                .map(|c| {
                    if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                        c
                    } else {
                        '_'
                    }
                })
                .collect();
            let path = dir.join(format!("{safe}.npy"));
            qanten::numpy::save_npy(&path, t)?;
            println!("  wrote {}", path.display());
        }
    }
    if args.profile {
        println!("\n{}", profiler.report());
    }
    Ok(())
}
