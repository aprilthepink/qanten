//! `machine-monitor`: train a fault classifier, watch a simulated plant, or
//! serve diagnoses over HTTP. See README.md.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use machine_monitor::features::FeatureExtractor;
use machine_monitor::fleet::{Diagnosis, Monitor, Plant};
use machine_monitor::model::{self, Confusion, Dataset, TrainConfig};
use machine_monitor::{Assessment, Condition, Diagnoser, INPUT_WIDTH, Machine, OUTPUT_WIDTH};
use qanten::Npu;
use qanten::compute::Matrix;
use qanten::nn::{Module, StateDict};
use qanten::pool::NpuPool;
use qanten::profile::Profiler;

#[derive(Parser)]
#[command(
    version,
    about = "Vibration-based predictive maintenance on Q.ANT NPUs"
)]
struct Cli {
    /// Use simulated NPUs even if the Q.ANT driver is installed.
    #[arg(long, global = true)]
    sim: bool,
    /// How many simulated NPUs (with --sim, or when no driver is found).
    #[arg(long, global = true, default_value_t = 2)]
    sim_npus: u32,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Simulate labelled vibration data, train the classifier on the NPU, evaluate it.
    Train {
        /// Where to write the weights.
        #[arg(short, long, default_value = "model.safetensors")]
        out: PathBuf,
        /// Training windows per condition.
        #[arg(long, default_value_t = 400)]
        per_class: usize,
        #[arg(long, default_value_t = 40)]
        epochs: usize,
    },
    /// Watch a simulated plant for weeks of operation and raise maintenance alarms.
    Monitor {
        #[arg(short, long, default_value = "model.safetensors")]
        model: PathBuf,
        /// Machines in the plant.
        #[arg(long, default_value_t = 24)]
        machines: usize,
        /// Hours to simulate (one measurement per machine per hour).
        #[arg(long, default_value_t = 336)]
        hours: usize,
        #[arg(long, default_value_t = 7)]
        seed: u64,
    },
    /// Serve the diagnoser over HTTP (one replica per NPU, dynamic batching).
    Serve {
        #[arg(short, long, default_value = "model.safetensors")]
        model: PathBuf,
        #[arg(long, default_value = "127.0.0.1:8080")]
        addr: String,
    },
    /// Print an example request body for `serve`.
    Sample {
        /// One of: healthy, imbalance, misalignment, looseness, outer-race, inner-race, cavitation.
        #[arg(long, default_value = "outer-race")]
        condition: Condition,
        /// Fault severity, 0–1.
        #[arg(long, default_value_t = 0.7)]
        severity: f32,
        #[arg(long, default_value_t = 1780.0)]
        rpm: f32,
        #[arg(long, default_value_t = 1)]
        seed: u64,
    },
}

fn open_pool(cli: &Cli) -> Result<NpuPool> {
    if !cli.sim && qanten::driver_available() {
        let pool = NpuPool::open_all().context("opening the Q.ANT NPUs")?;
        eprintln!("using {} Q.ANT NPU(s)", pool.len());
        return Ok(pool);
    }
    if !cli.sim {
        eprintln!("no Q.ANT driver found, using the simulator");
    }
    eprintln!("using {} simulated NPU(s)", cli.sim_npus);
    Ok(NpuPool::simulated(cli.sim_npus)?)
}

fn train(npu: &Npu, out: &Path, per_class: usize, epochs: usize) -> Result<()> {
    let extractor = FeatureExtractor::new(npu)?;
    let t = Instant::now();
    let data = Dataset::generate(&extractor, per_class, (0.2, 1.0), 1)?;
    println!(
        "simulated and featurised {} training windows in {:.1} s",
        data.len(),
        t.elapsed().as_secs_f64()
    );

    let t = Instant::now();
    let config = TrainConfig {
        epochs,
        ..Default::default()
    };
    let mut sd = model::train(npu, &data, &config, |e, loss, acc| {
        if e == 1 || e % 10 == 0 || e == epochs {
            println!(
                "  epoch {e:>3}: loss {loss:.3}, train accuracy {:.1}%",
                100.0 * acc
            );
        }
    })?;
    println!("trained in {:.1} s", t.elapsed().as_secs_f64());

    // Novelty detector: PCA model of the training features (qanten::compute).
    let test_size = per_class / 4 + 1;
    let validation = Dataset::generate(&extractor, test_size, (0.2, 1.0), 7)?;
    let fit = model::add_novelty(npu, &mut sd, &data, &validation)?;
    println!(
        "novelty detector: {} components explain {:.1}% of the variance; covariance on the {} \
         (checked error {:.1e}); thresholds SPE {:.1}, T² {:.1}",
        fit.components,
        100.0 * fit.explained,
        if fit.covariance_on_npu { "NPU" } else { "CPU" },
        fit.covariance_error,
        fit.thresholds[0],
        fit.thresholds[1]
    );
    model::save(&sd, out)?;
    println!("wrote {}", out.display());

    // Evaluate the deployed pipeline on machines it has never seen.
    let diag = Diagnoser::load(npu, &sd)?;
    for (name, severity) in [
        ("all severities", (0.2, 1.0)),
        ("early faults (severity 0.2-0.4)", (0.2, 0.4)),
    ] {
        let test = Dataset::generate(&extractor, test_size, severity, 99)?;
        let confusion = Confusion::new(&diag.classify_features(&test.features)?, &test.labels);
        println!(
            "\ntest set, {name}: {} windows, accuracy {:.1}%\n{confusion}",
            test.len(),
            100.0 * confusion.accuracy()
        );
    }

    // How well the novelty score separates known conditions from an unseen fault.
    let flagged = |scores: Vec<f32>| {
        scores.iter().filter(|s| **s > 1.0).count() as f32 / scores.len().max(1) as f32
    };
    let known = Dataset::generate(&extractor, test_size, (0.2, 1.0), 99)?;
    let unseen =
        Dataset::generate_condition(&extractor, Condition::Cavitation, test_size, (0.3, 1.0), 5)?;
    let mut called = [0usize; Condition::ALL.len()];
    for p in diag.classify_features(&unseen.features)?.argmax_last() {
        called[p] += 1;
    }
    let calls: Vec<String> = Condition::ALL
        .iter()
        .zip(called)
        .filter(|(_, n)| *n > 0)
        .map(|(c, n)| format!("{c} {n}"))
        .collect();
    println!(
        "novelty (score > 1): {:.1}% of known-condition windows, {:.1}% of cavitation windows \
         (a fault the classifier never saw; it called them: {})",
        100.0 * flagged(diag.novelty_scores(&known.features)?),
        100.0 * flagged(diag.novelty_scores(&unseen.features)?),
        calls.join(", ")
    );
    Ok(())
}

fn clock(hour: usize) -> String {
    format!("day {:>2} {:02}:00", hour / 24 + 1, hour % 24)
}

fn monitor(pool: &NpuPool, path: &Path, machines: usize, hours: usize, seed: u64) -> Result<()> {
    let sd = StateDict::load_safetensors(path)
        .with_context(|| format!("{} (run `machine-monitor train` first)", path.display()))?;
    let (npus, profilers): (Vec<Npu>, Vec<Profiler>) =
        pool.npus().iter().map(Npu::profiled).unzip();
    let pool = NpuPool::new(npus)?;
    // One copy of the pipeline per NPU; each hour's batch is split across them.
    let diagnoser = pool.replicate(|npu| Diagnoser::load(npu, &sd))?;
    if !diagnoser.copies()[0].has_novelty() {
        println!("note: model has no novelty detector (retrain to get one)");
    }
    let plant = Plant::simulate(machines, hours, seed);
    let mut monitor = Monitor::new(machines);
    let mut npu_time = Duration::ZERO;
    let mut worst_check = 0.0f64;

    println!(
        "monitoring {machines} machines for {hours} hours ({} days)\n",
        hours / 24
    );
    for hour in 0..hours {
        // Daily NPU self-test: random vectors through each NPU's resident
        // novelty matrix, verified on the CPU with Matrix::product_error.
        if hour % 24 == 0 {
            for (n, copy) in diagnoser.copies().iter().enumerate() {
                let Some(err) = copy.self_check(hour as u64)? else {
                    continue;
                };
                worst_check = worst_check.max(err);
                if err > 0.05 {
                    println!(
                        "{}  NPU {n} self-check error {:.1}%: recalibrate",
                        clock(hour),
                        100.0 * err
                    );
                }
            }
        }
        let rows: Vec<Vec<f32>> = plant
            .measure(hour)
            .into_iter()
            .map(|m| std::iter::once(m.shaft_hz).chain(m.samples).collect())
            .collect();
        let t = Instant::now();
        let out =
            Matrix::from_tensor(&diagnoser.forward(&Matrix::from_rows(&rows)?.to_tensor()?)?)?;
        npu_time += t.elapsed();
        let assessments: Vec<Assessment> = out.iter_rows().map(Assessment::from_row).collect();
        debug_assert_eq!(out.cols(), OUTPUT_WIDTH);
        for n in monitor.step(hour, &assessments) {
            let strength = match n.diagnosis {
                Diagnosis::Unrecognised => format!("novelty {:.1}", n.confidence),
                Diagnosis::Fault(_) => format!("{:.0}%", 100.0 * n.confidence),
            };
            println!(
                "{}  {}  {:<14} {:<13} {strength:<12} -> {}",
                clock(hour),
                if n.new_alarm { "ALARM " } else { "UPDATE" },
                plant.assets[n.asset].name,
                n.diagnosis.to_string(),
                n.diagnosis.action()
            );
        }
    }

    println!(
        "\n{:<14} {:<32} {:<32} {:>9} {:>7}  result",
        "machine", "truth", "first alarm", "severity", "health"
    );
    let outcomes: Vec<_> = (0..machines).map(|i| monitor.outcome(&plant, i)).collect();
    for (i, (asset, outcome)) in plant.assets.iter().zip(&outcomes).enumerate() {
        let truth = asset.degradation.map_or("healthy".into(), |d| {
            format!("{} from {}", d.condition, clock(d.onset_hour))
        });
        let alarm = monitor
            .first_alarm(i)
            .map_or("-".into(), |(h, d)| format!("{d} at {}", clock(h)));
        let severity = outcome
            .severity()
            .map_or("-".into(), |s| format!("{:.0}%", 100.0 * s));
        println!(
            "{:<14} {truth:<32} {alarm:<32} {severity:>9} {:>6.0}%  {}",
            asset.name,
            100.0 * monitor.health(i),
            outcome.label()
        );
    }
    use machine_monitor::fleet::Outcome::*;
    let count =
        |f: fn(&machine_monitor::fleet::Outcome) -> bool| outcomes.iter().filter(|o| f(o)).count();
    println!(
        "\n'severity' is how far the fault had developed when the alarm fired (lower = earlier warning)."
    );
    println!(
        "{}/{} faults found with the right diagnosis, {} with the wrong one, {} missed, {} false alarm(s)",
        count(|o| matches!(o, Detected { .. })),
        plant
            .assets
            .iter()
            .filter(|a| a.degradation.is_some())
            .count(),
        count(|o| matches!(o, WrongDiagnosis { .. })),
        count(|o| matches!(o, Missed)),
        count(|o| matches!(o, FalseAlarm)),
    );

    let windows = machines * hours;
    println!(
        "\n{windows} windows diagnosed in {:.2} s on the NPUs ({:.0} windows/s, {:.2} ms per batch of {machines})",
        npu_time.as_secs_f64(),
        windows as f64 / npu_time.as_secs_f64(),
        npu_time.as_secs_f64() * 1e3 / hours as f64
    );
    if diagnoser.copies()[0].has_novelty() {
        println!(
            "daily NPU self-checks: worst relative error {:.2}%",
            100.0 * worst_check
        );
    }
    for (i, p) in profilers.iter().enumerate() {
        let t = p.report().total();
        println!(
            "  NPU {i}: {} calls, {:.2} GMAC, {:.1} MB sent",
            t.calls,
            t.macs as f64 / 1e9,
            t.bytes_to_device as f64 / 1e6
        );
    }
    Ok(())
}

async fn serve(pool: NpuPool, path: PathBuf, addr: String) -> Result<()> {
    let sd = StateDict::load_safetensors(&path)
        .with_context(|| format!("{} (run `machine-monitor train` first)", path.display()))?;
    let mut state = qanten_serve::AppState::new(&pool, qanten_serve::ServerConfig::default())?;
    state.add_module("diagnose", INPUT_WIDTH, |npu| Diagnoser::load(npu, &sd))?;
    let mut classes: Vec<_> = Condition::ALL
        .iter()
        .map(|c| serde_json::json!({ "index": c.index(), "condition": c.label(), "action": c.action() }))
        .collect();
    classes.push(serde_json::json!({
        "index": 6,
        "novelty": "max(SPE, T²) over their thresholds; above 1 = unlike every known condition",
        "action": Diagnosis::Unrecognised.action(),
    }));
    let app = qanten_serve::router(Arc::new(state)).route(
        "/v1/conditions",
        axum::routing::get(move || async move { axum::Json(classes.clone()) }),
    );
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    eprintln!("serving on http://{addr}  (POST /v1/models/diagnose/infer, GET /v1/conditions)");
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}

fn sample(condition: Condition, severity: f32, rpm: f32, seed: u64) -> Result<()> {
    let m = Machine::random_at(rpm / 60.0, seed).measure(condition, severity, seed + 1);
    let rounded: Vec<f64> = std::iter::once(m.shaft_hz)
        .chain(m.samples)
        .map(|v| (v as f64 * 1e5).round() / 1e5)
        .collect();
    println!(
        "{}",
        serde_json::json!({ "shape": [1, INPUT_WIDTH], "data": rounded })
    );
    Ok(())
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();
    let cli = Cli::parse();
    match &cli.command {
        Command::Train {
            out,
            per_class,
            epochs,
        } => {
            let pool = open_pool(&cli)?;
            train(&pool.npus()[0], out, *per_class, *epochs)
        }
        Command::Monitor {
            model,
            machines,
            hours,
            seed,
        } => monitor(&open_pool(&cli)?, model, *machines, *hours, *seed),
        Command::Serve { model, addr } => {
            let pool = open_pool(&cli)?;
            tokio::runtime::Runtime::new()?.block_on(serve(pool, model.clone(), addr.clone()))
        }
        Command::Sample {
            condition,
            severity,
            rpm,
            seed,
        } => sample(*condition, *severity, *rpm, *seed),
    }
}
