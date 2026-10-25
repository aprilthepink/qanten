//! `qanten`: command-line tool for Q.ANT photonic NPUs.

mod run;

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use qanten::backend::Backend;
use qanten::ops::{self, Conv2dParams, ConvTranspose2dParams};
use qanten::{Npu, Tensor};

#[derive(Parser)]
#[command(
    name = "qanten",
    version,
    about = "Inspect, benchmark and test Q.ANT photonic NPUs"
)]
struct Cli {
    /// Use the built-in simulator instead of hardware.
    #[arg(long, global = true, env = "QANTEN_SIM")]
    sim: bool,

    /// Print machine-readable JSON.
    #[arg(long, global = true)]
    json: bool,

    /// More log output (-v info, -vv debug, -vvv trace).
    #[arg(short, long, global = true, action = clap::ArgAction::Count)]
    verbose: u8,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// List NPUs visible to the driver.
    Devices,
    /// Driver, firmware and board info of an NPU.
    Info {
        #[arg(short, long, default_value_t = 0)]
        npu: u32,
    },
    /// Read board sensors (temperatures, voltages, current).
    Sensors {
        #[arg(short, long, default_value_t = 0)]
        npu: u32,
        /// Refresh every N seconds until interrupted.
        #[arg(short, long, value_name = "SECS")]
        watch: Option<f64>,
    },
    /// Show (and optionally reset) performance counters.
    Perf {
        #[arg(short, long, default_value_t = 0)]
        npu: u32,
        #[arg(long)]
        reset: bool,
    },
    /// Measure matrix-vector throughput.
    Bench {
        #[arg(short, long, default_value_t = 0)]
        npu: u32,
        /// Output features (matrix rows).
        #[arg(long, default_value_t = 1024)]
        rows: usize,
        /// Input features (matrix columns).
        #[arg(long, default_value_t = 1024)]
        cols: usize,
        /// Vectors per call.
        #[arg(long, default_value_t = 64)]
        batch: usize,
        #[arg(long, default_value_t = 20)]
        iters: usize,
        /// Print a per-operation profile (calls, time, bytes moved, MACs).
        #[arg(long)]
        profile: bool,
    },
    /// Run every op on the NPU and compare against the simulator.
    Selftest {
        #[arg(short, long, default_value_t = 0)]
        npu: u32,
        /// Allowed error relative to the largest reference value.
        #[arg(long, default_value_t = 0.05)]
        tolerance: f32,
    },
    /// Check the driver installation and explain what is missing.
    Doctor,
    /// Run a model file (.onnx, or .safetensors/.npz with --arch) on the NPU.
    Run {
        /// Model file.
        model: PathBuf,
        /// Input as `name=file.npy` or just `file.npy` (repeatable). Random data if omitted.
        #[arg(short, long = "input")]
        inputs: Vec<String>,
        /// Write every output to `<dir>/<name>.npy`.
        #[arg(short, long)]
        output_dir: Option<PathBuf>,
        /// Architecture for weight files (.safetensors / .npz).
        #[arg(long, value_enum)]
        arch: Option<run::Arch>,
        /// Activation between layers for `--arch mlp`: relu, tanh, sigmoid, gelu or silu.
        #[arg(long, default_value = "relu")]
        activation: qanten::nn::ActivationKind,
        /// Batch size for random inputs.
        #[arg(long, default_value_t = 1)]
        batch: usize,
        /// Repeat the run N times and report the average time.
        #[arg(long, default_value_t = 1)]
        iters: usize,
        /// Print a per-operation profile.
        #[arg(long)]
        profile: bool,
        #[arg(short, long, default_value_t = 0)]
        npu: u32,
    },
    /// List the tensors of a .safetensors/.npz/.npy file, or summarise an .onnx model.
    Inspect {
        /// File to inspect.
        file: PathBuf,
    },
}

fn open(sim: bool, id: u32) -> Result<Npu> {
    if sim {
        return Ok(Npu::simulator());
    }
    Npu::open(id)
        .with_context(|| format!("cannot open NPU {id} (use --sim to run without hardware)"))
}

fn print_json(v: &impl serde::Serialize) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(v)?);
    Ok(())
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let level = match cli.verbose {
        0 => "warn",
        1 => "info",
        2 => "debug",
        _ => "trace",
    };
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(level)),
        )
        .with_writer(std::io::stderr)
        .init();
    match run(cli) {
        Ok(code) => code,
        Err(err) => {
            eprintln!("error: {err:#}");
            ExitCode::FAILURE
        }
    }
}

fn run(cli: Cli) -> Result<ExitCode> {
    match cli.command {
        Command::Devices => devices(cli.sim, cli.json)?,
        Command::Info { npu } => info(&open(cli.sim, npu)?, cli.json)?,
        Command::Sensors { npu, watch } => sensors(&open(cli.sim, npu)?, cli.json, watch)?,
        Command::Perf { npu, reset } => perf(&open(cli.sim, npu)?, cli.json, reset)?,
        Command::Bench {
            npu,
            rows,
            cols,
            batch,
            iters,
            profile,
        } => {
            let base = open(cli.sim, npu)?;
            if profile {
                let (npu, profiler) = base.profiled();
                bench(&npu, cli.json, rows, cols, batch, iters)?;
                if cli.json {
                    print_json(&profiler.report())?;
                } else {
                    println!("\n{}", profiler.report());
                }
            } else {
                bench(&base, cli.json, rows, cols, batch, iters)?;
            }
        }
        Command::Selftest { npu, tolerance } => {
            return selftest(&open(cli.sim, npu)?, cli.json, tolerance);
        }
        Command::Doctor => return doctor(cli.json),
        Command::Run {
            model,
            inputs,
            output_dir,
            arch,
            activation,
            batch,
            iters,
            profile,
            npu,
        } => run::run(
            &open(cli.sim, npu)?,
            run::RunArgs {
                model,
                inputs,
                output_dir,
                arch,
                activation,
                batch,
                iters,
                profile,
            },
        )?,
        Command::Inspect { file } => run::inspect(&file, &Npu::simulator())?,
    }
    Ok(ExitCode::SUCCESS)
}

fn devices(sim: bool, json: bool) -> Result<()> {
    let list = if sim {
        qanten::backend::Simulator::default().available_npus()?
    } else {
        qanten::devices().context("cannot list devices (use --sim to run without hardware)")?
    };
    if json {
        return print_json(&list);
    }
    if list.is_empty() {
        println!("no NPUs found");
    }
    for d in list {
        println!("npu {:>2}  serial {}", d.id, d.serial);
    }
    Ok(())
}

fn info(npu: &Npu, json: bool) -> Result<()> {
    let info = npu.device_info()?;
    let version = npu.driver_version()?;
    if json {
        #[derive(serde::Serialize)]
        struct Out<'a> {
            npu: u32,
            backend: String,
            driver_version: &'a str,
            info: &'a qanten::info::DeviceInfo,
        }
        return print_json(&Out {
            npu: npu.id(),
            backend: npu.backend().name(),
            driver_version: &version,
            info: &info,
        });
    }
    println!("NPU            {}", npu.id());
    println!("backend        {}", npu.backend().name());
    println!("driver         {version}");
    println!("firmware       {}", info.version.firmware);
    println!("zephyr         {}", info.version.zephyr);
    println!("board serial   {}", info.version.board_serial);
    println!("12V power      {:.1} W", info.sensors.power_12v_watts());
    Ok(())
}

fn sensors(npu: &Npu, json: bool, watch: Option<f64>) -> Result<()> {
    loop {
        let s = npu.device_info()?.sensors;
        if json {
            print_json(&s)?;
        } else {
            for (name, value, unit) in s.rows() {
                println!("{name:<14} {value:>9.3} {unit}");
            }
            println!("{:<14} {:>9.3} W", "power_12v", s.power_12v_watts());
        }
        match watch {
            Some(secs) if secs > 0.0 => {
                std::thread::sleep(Duration::from_secs_f64(secs));
                if !json {
                    println!();
                }
            }
            _ => return Ok(()),
        }
    }
}

fn perf(npu: &Npu, json: bool, reset: bool) -> Result<()> {
    let p = npu.perf_counters()?;
    if json {
        #[derive(serde::Serialize)]
        struct Out {
            elapsed_s: f64,
            stalled_s: f64,
            utilization: f64,
        }
        print_json(&Out {
            elapsed_s: p.elapsed.as_secs_f64(),
            stalled_s: p.stalled.as_secs_f64(),
            utilization: p.utilization(),
        })?;
    } else {
        println!("elapsed      {:.3} s", p.elapsed.as_secs_f64());
        println!("stalled      {:.3} s", p.stalled.as_secs_f64());
        println!("utilization  {:.1} %", 100.0 * p.utilization());
    }
    if reset {
        npu.reset_perf_counters()?;
        if !json {
            println!("counters reset");
        }
    }
    Ok(())
}

fn time_it(iters: usize, mut f: impl FnMut() -> qanten::Result<()>) -> Result<Duration> {
    f()?; // warm-up
    let start = Instant::now();
    for _ in 0..iters {
        f()?;
    }
    Ok(start.elapsed() / iters.max(1) as u32)
}

fn bench(
    npu: &Npu,
    json: bool,
    rows: usize,
    cols: usize,
    batch: usize,
    iters: usize,
) -> Result<()> {
    if rows == 0 || cols == 0 || batch == 0 {
        bail!("rows, cols and batch must be > 0");
    }
    let w = Tensor::rand_uniform([rows, cols], -1., 1., 1);
    let x = Tensor::rand_uniform([batch, cols], -1., 1., 2);
    let macs = (rows * cols * batch) as f64;

    let streamed = time_it(iters, || ops::matvec(npu, &w, &x).map(drop))?;
    let resident = ops::LinearWeights::upload(npu, &w)?;
    let preloaded = time_it(iters, || resident.forward(&x).map(drop))?;

    let rate = |d: Duration| macs / d.as_secs_f64() / 1e9;
    if json {
        #[derive(serde::Serialize)]
        struct Out {
            backend: String,
            rows: usize,
            cols: usize,
            batch: usize,
            streamed_ms: f64,
            streamed_gmacs: f64,
            resident_ms: f64,
            resident_gmacs: f64,
        }
        return print_json(&Out {
            backend: npu.backend().name(),
            rows,
            cols,
            batch,
            streamed_ms: streamed.as_secs_f64() * 1e3,
            streamed_gmacs: rate(streamed),
            resident_ms: preloaded.as_secs_f64() * 1e3,
            resident_gmacs: rate(preloaded),
        });
    }
    println!("backend: {}", npu.backend().name());
    println!("matvec {rows}x{cols}, batch {batch}, {iters} iterations\n");
    println!("{:<26} {:>10} {:>12}", "mode", "ms/call", "GMAC/s");
    println!(
        "{:<26} {:>10.3} {:>12.3}",
        "weights uploaded per call",
        streamed.as_secs_f64() * 1e3,
        rate(streamed)
    );
    println!(
        "{:<26} {:>10.3} {:>12.3}",
        "weights resident",
        preloaded.as_secs_f64() * 1e3,
        rate(preloaded)
    );
    Ok(())
}

#[derive(serde::Serialize)]
struct Check {
    op: &'static str,
    max_error: f32,
    passed: bool,
}

fn selftest(npu: &Npu, json: bool, tol: f32) -> Result<ExitCode> {
    let reference = Npu::simulator();
    type Op = Box<dyn Fn(&Npu) -> qanten::Result<Tensor>>;
    let a = Tensor::rand_uniform([4, 64], -1., 1., 1);
    let b = Tensor::rand_uniform([4, 64], -1., 1., 2);
    let w = Tensor::rand_uniform([32, 64], -0.5, 0.5, 3);
    let img = Tensor::rand_uniform([1, 3, 12, 12], 0., 1., 4);
    let k = Tensor::rand_uniform([8, 3, 3, 3], -0.5, 0.5, 5);
    let kt = Tensor::rand_uniform([3, 4, 3, 3], -0.5, 0.5, 6);
    let mean = Tensor::rand_uniform([3], 0., 0.5, 7);
    let var = Tensor::rand_uniform([3], 0.5, 1.5, 8);
    let phis = Tensor::rand_uniform([4, 8, 3], -1., 1., 9);
    let ampls = Tensor::rand_uniform([4, 8, 3], -0.5, 0.5, 10);
    let ks = Tensor::vector(&[0.5, 1.0, 1.5]);
    let xk = Tensor::rand_uniform([2, 8], -1., 1., 11);

    let tests: Vec<(&'static str, Op)> = vec![
        (
            "mul",
            Box::new({
                let (a, b) = (a.clone(), b.clone());
                move |n| ops::mul(n, &a, &b)
            }),
        ),
        (
            "periodic_nl",
            Box::new({
                let (a, b) = (a.clone(), b.clone());
                move |n| ops::periodic_nl(n, &a, &b)
            }),
        ),
        (
            "linear",
            Box::new({
                let (a, w) = (a.clone(), w.clone());
                move |n| ops::linear(n, &a, &w, None)
            }),
        ),
        (
            "linear (resident)",
            Box::new({
                let (a, w) = (a.clone(), w.clone());
                move |n| ops::LinearWeights::upload(n, &w)?.forward(&a)
            }),
        ),
        (
            "conv2d",
            Box::new({
                let (img, k) = (img.clone(), k.clone());
                move |n| ops::conv2d(n, &img, &k, None, Conv2dParams::new().padding(1).stride(2))
            }),
        ),
        (
            "conv_transpose2d",
            Box::new({
                let (img, kt) = (img.clone(), kt.clone());
                move |n| {
                    ops::conv_transpose2d(
                        n,
                        &img,
                        &kt,
                        None,
                        ConvTranspose2dParams::new().stride(2).padding(1),
                    )
                }
            }),
        ),
        (
            "batch_norm2d",
            Box::new({
                let (img, mean, var) = (img.clone(), mean.clone(), var.clone());
                move |n| ops::batch_norm2d(n, &img, &mean, &var, None, None, 1e-5)
            }),
        ),
        (
            "kan_layer",
            Box::new(move |n| ops::kan_layer(n, &xk, &phis, &ampls, &ks)),
        ),
    ];

    let mut results = Vec::new();
    for (name, op) in &tests {
        let want = op(&reference)?;
        let got = op(npu).with_context(|| format!("{name} failed on the NPU"))?;
        let scale = want
            .to_f32_vec()
            .iter()
            .fold(1e-6f32, |m, v| m.max(v.abs()));
        let err = got.max_abs_diff(&want)? / scale;
        results.push(Check {
            op: name,
            max_error: err,
            passed: err <= tol,
        });
    }
    let ok = results.iter().all(|c| c.passed);
    if json {
        print_json(&results)?;
    } else {
        println!("backend: {}  (tolerance {tol})\n", npu.backend().name());
        for c in &results {
            println!(
                "{:<20} {:>9.5}  {}",
                c.op,
                c.max_error,
                if c.passed { "ok" } else { "FAIL" }
            );
        }
        println!(
            "\n{}",
            if ok {
                "all checks passed"
            } else {
                "some checks FAILED"
            }
        );
    }
    Ok(if ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    })
}

fn doctor(json: bool) -> Result<ExitCode> {
    #[derive(serde::Serialize)]
    struct Item {
        check: &'static str,
        ok: bool,
        detail: String,
    }
    let mut items = Vec::new();

    let env = std::env::var(qanten::sys::DRIVER_LIB_ENV).ok();
    items.push(Item {
        check: "QANTEN_DRIVER_LIB",
        ok: true,
        detail: env
            .clone()
            .unwrap_or_else(|| "not set (using default search path)".into()),
    });

    let driver = qanten::backend::DriverBackend::load();
    items.push(Item {
        check: "driver library",
        ok: driver.is_ok(),
        detail: match &driver {
            Ok(d) => d.library().path().to_string_lossy().into_owned(),
            Err(e) => e.to_string(),
        },
    });

    let pkg = std::process::Command::new("dpkg-query")
        .args(["-W", "-f=${Version}", "qant_native_computing_driver"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).into_owned());
    items.push(Item {
        check: "driver package",
        ok: pkg.is_some(),
        detail: pkg.unwrap_or_else(|| "qant_native_computing_driver not installed via dpkg".into()),
    });

    let cfg_dir = std::path::Path::new("/etc/qant-native-computing-driver");
    let configs: Vec<String> = std::fs::read_dir(cfg_dir)
        .map(|d| {
            d.filter_map(|e| e.ok())
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .filter(|n| n.ends_with(".toml"))
                .collect()
        })
        .unwrap_or_default();
    items.push(Item {
        check: "device configs",
        ok: !configs.is_empty(),
        detail: if configs.is_empty() {
            format!("no *.toml in {}", cfg_dir.display())
        } else {
            configs.join(", ")
        },
    });

    if let Ok(d) = &driver {
        let devs = d.available_npus();
        items.push(Item {
            check: "NPUs",
            ok: devs.as_ref().is_ok_and(|v| !v.is_empty()),
            detail: match devs {
                Ok(v) if v.is_empty() => "driver reports no NPUs".into(),
                Ok(v) => v
                    .iter()
                    .map(|d| format!("{} ({})", d.id, d.serial))
                    .collect::<Vec<_>>()
                    .join(", "),
                Err(e) => e.to_string(),
            },
        });
    }

    let ok = driver.is_ok();
    if json {
        print_json(&items)?;
    } else {
        for i in &items {
            println!(
                "[{}] {:<16} {}",
                if i.ok { " ok " } else { "warn" },
                i.check,
                i.detail
            );
        }
        println!();
        if ok {
            println!("driver found. Try `qanten selftest`.");
        } else {
            println!("driver not found. Install Q.ANT's qant_native_computing_driver package,");
            println!(
                "or set {} to the .so path. `qanten --sim ...` works without it.",
                qanten::sys::DRIVER_LIB_ENV
            );
        }
    }
    Ok(if ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    })
}
