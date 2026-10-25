//! `machine-monitor`: train a fault classifier, watch a simulated plant, or
//! serve diagnoses over HTTP. See README.md.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use machine_monitor::features::FeatureExtractor;
use machine_monitor::fleet::{self, AlarmTracker, Diagnosis, Event};
use machine_monitor::model::{self, Dataset, TrainConfig};
use machine_monitor::{Assessment, Condition, Diagnoser, INPUT_WIDTH, Machine, OUTPUT_WIDTH};
use qanten::nn::{Module, StateDict};
use qanten::pool::NpuPool;
use qanten::profile::Profiler;
use qanten::{Npu, Tensor};

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

fn print_confusion(m: &[Vec<usize>]) {
    print!("{:>14}", "true \\ pred");
    for c in Condition::ALL {
        print!("{:>13}", c.label());
    }
    println!("{:>9}", "recall");
    for (c, row) in Condition::ALL.iter().zip(m) {
        print!("{:>14}", c.label());
        for v in row {
            print!("{v:>13}");
        }
        let total: usize = row.iter().sum();
        println!(
            "{:>8.1}%",
            100.0 * row[c.index()] as f32 / total.max(1) as f32
        );
    }
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
    let validation = Dataset::generate(&extractor, per_class / 4 + 1, (0.2, 1.0), 7)?;
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
    for (name, range) in [
        ("all severities", (0.2, 1.0)),
        ("early faults (severity 0.2-0.4)", (0.2, 0.4)),
    ] {
        let test = Dataset::generate(&extractor, per_class / 4 + 1, range, 99)?;
        let probs = diag.classify_features(&test.features)?;
        let acc = qanten::train::accuracy(&probs, &test.labels);
        println!(
            "\ntest set, {name}: {} windows, accuracy {:.1}%",
            test.len(),
            100.0 * acc
        );
        print_confusion(&model::confusion(&probs, &test.labels));
    }

    // How well the novelty score separates known conditions from an unseen fault.
    let known = Dataset::generate(&extractor, per_class / 4 + 1, (0.2, 1.0), 99)?;
    let known_flagged = flagged(&diag.novelty_scores(&known.features)?);
    let mut rng = machine_monitor::sim::Rng::new(5);
    let unseen: Vec<(f32, Vec<f32>)> = (0..per_class / 4 + 1)
        .map(|_| {
            let m = Machine::random(rng.next_u64());
            let sev = rng.uniform(0.3, 1.0);
            (
                m.shaft_hz,
                m.measure(Condition::Cavitation, sev, rng.next_u64()),
            )
        })
        .collect();
    let rows: Vec<(f32, &[f32])> = unseen.iter().map(|(f, x)| (*f, x.as_slice())).collect();
    let assessed = diag.assess(&rows)?;
    let novelty: Vec<f32> = assessed.iter().map(|a| a.novelty).collect();
    let mut called = [0usize; 6];
    for a in &assessed {
        let best = (0..6)
            .max_by(|&i, &j| a.probs[i].total_cmp(&a.probs[j]))
            .unwrap_or(0);
        called[best] += 1;
    }
    let calls: Vec<String> = Condition::ALL
        .iter()
        .zip(called)
        .filter(|(_, n)| *n > 0)
        .map(|(c, n)| format!("{c} {n}"))
        .collect();
    println!(
        "\nnovelty (score > 1): {:.1}% of known-condition windows, {:.1}% of cavitation windows \
         (a fault the classifier never saw; it called them: {})",
        100.0 * known_flagged,
        100.0 * flagged(&novelty),
        calls.join(", ")
    );
    Ok(())
}

fn flagged(scores: &[f32]) -> f32 {
    scores.iter().filter(|s| **s > 1.0).count() as f32 / scores.len().max(1) as f32
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
    let assets = fleet::plant(machines, hours, seed);
    let mut trackers = vec![AlarmTracker::default(); machines];
    let mut first_alarm: Vec<Option<(usize, Diagnosis)>> = vec![None; machines];
    let mut final_call: Vec<Option<Diagnosis>> = vec![None; machines];
    let mut npu_time = Duration::ZERO;
    let mut worst_check = 0.0f64;

    println!(
        "monitoring {machines} machines for {hours} hours ({} days)\n",
        hours / 24
    );
    for hour in 0..hours {
        // Daily NPU self-test: random vectors through each NPU's resident
        // novelty projector, verified on the CPU with compute::check_product.
        if hour % 24 == 0 {
            for (n, copy) in diagnoser.copies().iter().enumerate() {
                if let Some(err) = copy.self_check(hour as u64)? {
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
        }
        let mut batch = Vec::with_capacity(machines * INPUT_WIDTH);
        for a in &assets {
            batch.push(a.machine.shaft_hz);
            batch.extend(a.measure(hour, seed));
        }
        let x = Tensor::from_f32(&batch, [machines, INPUT_WIDTH])?;
        let t = Instant::now();
        let out = diagnoser.forward(&x)?.to_f32_vec();
        npu_time += t.elapsed();
        for (i, row) in out.chunks(OUTPUT_WIDTH).enumerate() {
            let event = trackers[i].update(&Assessment::from_row(row));
            let a = &assets[i];
            let (kind, diagnosis, confidence) = match event {
                Event::Alarm {
                    diagnosis,
                    confidence,
                } => {
                    first_alarm[i] = Some((hour, diagnosis));
                    ("ALARM ", diagnosis, confidence)
                }
                Event::Reclassified {
                    diagnosis,
                    confidence,
                } => ("UPDATE", diagnosis, confidence),
                Event::None => continue,
            };
            final_call[i] = Some(diagnosis);
            let strength = match diagnosis {
                Diagnosis::Unrecognised => format!("novelty {confidence:.1}"),
                Diagnosis::Fault(_) => format!("{:.0}%", 100.0 * confidence),
            };
            println!(
                "{}  {kind}  {:<14} {:<13} {strength:<12} -> {}",
                clock(hour),
                a.name,
                diagnosis.to_string(),
                diagnosis.action()
            );
        }
    }

    println!(
        "\n{:<14} {:<32} {:<32} {:>9} {:>7}  result",
        "machine", "truth", "first alarm", "severity", "health"
    );
    let (mut hits, mut wrong, mut missed, mut false_alarms, mut faults) = (0, 0, 0, 0, 0);
    for (i, a) in assets.iter().enumerate() {
        let truth = match a.degradation {
            Some(d) => format!("{} from {}", d.condition, clock(d.onset_hour)),
            None => "healthy".into(),
        };
        let alarm = first_alarm[i]
            .map(|(h, d)| format!("{d} at {}", clock(h)))
            .unwrap_or_else(|| "-".into());
        let (severity, result) = match (a.degradation, first_alarm[i]) {
            (Some(d), Some((h, _))) => {
                faults += 1;
                let expected = match d.condition.class() {
                    Some(_) => Diagnosis::Fault(d.condition),
                    None => Diagnosis::Unrecognised,
                };
                let ok = final_call[i] == Some(expected);
                if ok {
                    hits += 1
                } else {
                    wrong += 1
                }
                let result = match (ok, expected) {
                    (true, Diagnosis::Unrecognised) => "detected (new fault type)",
                    (true, _) => "detected",
                    (false, _) => "detected, wrong diagnosis",
                };
                (format!("{:.0}%", 100.0 * a.state_at(h).1), result)
            }
            (Some(_), None) => {
                faults += 1;
                let reached = a.state_at(hours - 1).1;
                if reached > 0.3 {
                    missed += 1
                }
                (
                    "-".into(),
                    if reached > 0.3 {
                        "MISSED"
                    } else {
                        "not developed yet"
                    },
                )
            }
            (None, Some(_)) => {
                false_alarms += 1;
                ("-".into(), "FALSE ALARM")
            }
            (None, None) => ("-".into(), "ok"),
        };
        println!(
            "{:<14} {truth:<32} {alarm:<32} {severity:>9} {:>6.0}%  {result}",
            a.name,
            100.0 * trackers[i].health()
        );
    }
    println!(
        "\n'severity' is how far the fault had developed when the alarm fired (lower = earlier warning)."
    );
    println!(
        "{hits}/{faults} faults found with the right diagnosis, {wrong} with the wrong one, \
         {missed} missed, {false_alarms} false alarm(s)"
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
    let machine = Machine::random_at(rpm / 60.0, seed);
    let mut data = vec![machine.shaft_hz];
    data.extend(machine.measure(condition, severity, seed + 1));
    let rounded: Vec<f64> = data
        .iter()
        .map(|v| (*v as f64 * 1e5).round() / 1e5)
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
