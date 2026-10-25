//! `qanten-serve`: HTTP inference server for Q.ANT NPUs.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use clap::Parser;
use qanten::nn::ActivationKind;
use qanten::pool::NpuPool;
use qanten::{Npu, SimConfig};
use qanten_serve::{AppState, ServerConfig, router};

#[derive(Parser)]
#[command(
    name = "qanten-serve",
    version,
    about = "HTTP inference server for Q.ANT photonic NPUs"
)]
struct Args {
    /// Model to serve as `name=path` (.onnx, or .safetensors/.npz MLP). Repeatable.
    #[arg(short, long = "model", required = true)]
    models: Vec<String>,
    /// Address to listen on.
    #[arg(long, default_value = "127.0.0.1:8080", env = "QANTEN_SERVE_ADDR")]
    addr: String,
    /// Maximum rows per dynamic batch.
    #[arg(long, default_value_t = 64)]
    max_batch: usize,
    /// Milliseconds the batcher waits for more requests.
    #[arg(long, default_value_t = 2)]
    max_wait_ms: u64,
    /// Activation between layers of MLP weight files: relu, tanh, sigmoid, gelu or silu.
    #[arg(long, default_value = "relu")]
    activation: ActivationKind,
    /// Use simulated NPUs instead of hardware.
    #[arg(long, env = "QANTEN_SIM")]
    sim: bool,
    /// Number of simulated NPUs (with --sim).
    #[arg(long, default_value_t = 1)]
    sim_npus: u32,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();
    let args = Args::parse();

    let pool = if args.sim {
        if args.sim_npus <= 1 {
            NpuPool::new(vec![Npu::simulator_with(SimConfig::default())])?
        } else {
            NpuPool::simulated(args.sim_npus)?
        }
    } else {
        NpuPool::open_all()
            .context("no Q.ANT NPUs available (use --sim to run without hardware)")?
    };
    let config = ServerConfig {
        max_batch_rows: args.max_batch.max(1),
        max_wait: Duration::from_millis(args.max_wait_ms),
    };
    let mut state = AppState::new(&pool, config)?;
    let activation = args.activation;
    for spec in &args.models {
        let Some((name, path)) = spec.split_once('=') else {
            bail!("--model expects name=path, got `{spec}`");
        };
        state
            .add_model_file(name, &PathBuf::from(path), activation)
            .with_context(|| format!("loading model `{name}` from {path}"))?;
        tracing::info!(model = name, path, "model loaded");
    }

    let listener = tokio::net::TcpListener::bind(&args.addr)
        .await
        .with_context(|| format!("binding {}", args.addr))?;
    tracing::info!(addr = %args.addr, npus = pool.len(), backend = %pool.npus()[0].backend().name(), "qanten-serve listening");
    axum::serve(listener, router(Arc::new(state)))
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
            tracing::info!("shutting down");
        })
        .await?;
    Ok(())
}
