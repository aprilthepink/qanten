//! # qanten-serve
//!
//! An HTTP inference server for Q.ANT photonic NPUs.
//!
//! - Serves ONNX models and MLP weight files from one process.
//! - **Multi-NPU**: every model is loaded once per NPU of an
//!   [`NpuPool`]; requests go to whichever NPU is free.
//! - **Dynamic batching**: concurrent requests to a single-input model are
//!   merged into one NPU batch (up to `max_batch_rows` rows or `max_wait`),
//!   then split back per request.
//! - JSON API, `/health`, and Prometheus `/metrics` with request and NPU counters.
//!
//! The binary (`qanten-serve`) wraps [`AppState`] and [`router`]; both are
//! public so the server can be embedded in another axum application.

#![warn(missing_docs)]

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use qanten::executor::NpuExecutor;
use qanten::nn::{self, ActivationKind, Module, StateDict};
use qanten::onnx::{OnnxModel, OnnxValue};
use qanten::pool::NpuPool;
use qanten::profile::{OpKind, Profiler};
use qanten::{Npu, Tensor};
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, oneshot};

/// Server tuning.
#[derive(Debug, Clone, Copy)]
pub struct ServerConfig {
    /// Maximum rows merged into one NPU batch.
    pub max_batch_rows: usize,
    /// How long the batcher waits for more requests after the first one.
    pub max_wait: Duration,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            max_batch_rows: 64,
            max_wait: Duration::from_millis(2),
        }
    }
}

/// Input description returned by `/v1/models`.
#[derive(Debug, Clone, Serialize)]
pub struct InputSpec {
    /// Input name.
    pub name: String,
    /// Shape; `null` for dynamic dimensions.
    pub shape: Vec<Option<usize>>,
    /// `true` for integer inputs (token ids, masks).
    pub integer: bool,
}

type Outputs = Vec<(String, OnnxValue)>;

/// A loaded model: kind, inputs, outputs, one runner per NPU.
type Loaded = (
    &'static str,
    Vec<InputSpec>,
    Vec<String>,
    Vec<Arc<dyn Runner>>,
);

/// Something that can be served.
trait Runner: Send + Sync {
    fn run(&self, inputs: &[(&str, OnnxValue)]) -> qanten::Result<Outputs>;
}

impl Runner for OnnxModel {
    fn run(&self, inputs: &[(&str, OnnxValue)]) -> qanten::Result<Outputs> {
        OnnxModel::run(self, inputs)
    }
}

struct ModuleRunner(Box<dyn Module>);

impl Runner for ModuleRunner {
    fn run(&self, inputs: &[(&str, OnnxValue)]) -> qanten::Result<Outputs> {
        let x = inputs
            .first()
            .ok_or_else(|| qanten::Error::InvalidArgument("missing input".into()))?
            .1
            .clone()
            .into_tensor()?;
        Ok(vec![(
            "output".to_owned(),
            OnnxValue::Float(self.0.forward(&x)?),
        )])
    }
}

#[derive(Default)]
struct Stats {
    requests: AtomicU64,
    errors: AtomicU64,
    batches: AtomicU64,
    batched_rows: AtomicU64,
    latency_us: AtomicU64,
}

struct BatchItem {
    input: Tensor,
    reply: oneshot::Sender<Result<(Outputs, usize), String>>,
}

struct ModelEntry {
    kind: &'static str,
    inputs: Vec<InputSpec>,
    outputs: Vec<String>,
    /// One copy per NPU of the pool, in pool order.
    replicas: Vec<Arc<dyn Runner>>,
    batcher: Option<mpsc::Sender<BatchItem>>,
    stats: Stats,
}

/// Shared server state: NPUs, executor, models and metrics.
pub struct AppState {
    pool: NpuPool,
    profilers: Vec<Profiler>,
    exec: Arc<NpuExecutor>,
    models: BTreeMap<String, Arc<ModelEntry>>,
    config: ServerConfig,
    started: Instant,
    backend: String,
}

impl std::fmt::Debug for AppState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AppState")
            .field("npus", &self.pool.len())
            .field("models", &self.models.keys().collect::<Vec<_>>())
            .finish()
    }
}

fn replica_index(pool: &NpuPool, npu: &Npu) -> usize {
    pool.npus()
        .iter()
        .position(|n| n.same_device(npu))
        .unwrap_or(0)
}

impl AppState {
    /// State for `pool`. Each NPU is wrapped in a profiler for `/metrics`.
    pub fn new(pool: &NpuPool, config: ServerConfig) -> anyhow::Result<Self> {
        let (npus, profilers): (Vec<Npu>, Vec<Profiler>) =
            pool.npus().iter().map(Npu::profiled).unzip();
        let backend = pool.npus()[0].backend().name();
        let pool = NpuPool::new(npus)?;
        Ok(Self {
            exec: Arc::new(NpuExecutor::for_pool(&pool)),
            pool,
            profilers,
            models: BTreeMap::new(),
            config,
            started: Instant::now(),
            backend,
        })
    }

    /// Loads a model file under `name`: `.onnx`, or `.safetensors` / `.npz`
    /// (an MLP, see [`nn::mlp_from_state_dict`]). Must be called inside a tokio runtime.
    pub fn add_model_file(
        &mut self,
        name: &str,
        path: &std::path::Path,
        activation: ActivationKind,
    ) -> anyhow::Result<()> {
        let ext = path
            .extension()
            .map(|e| e.to_string_lossy().to_lowercase())
            .unwrap_or_default();
        let (kind, inputs, outputs, replicas): Loaded = match ext.as_str() {
            "onnx" => {
                let bytes = std::fs::read(path)?;
                let models = self
                    .pool
                    .npus()
                    .iter()
                    .map(|npu| OnnxModel::from_bytes(npu, &bytes))
                    .collect::<qanten::Result<Vec<_>>>()?;
                let inputs = models[0]
                    .inputs()
                    .iter()
                    .map(|i| InputSpec {
                        name: i.name.clone(),
                        shape: i.shape.clone(),
                        integer: i.is_integer,
                    })
                    .collect();
                let outputs = models[0].outputs().to_vec();
                (
                    "onnx",
                    inputs,
                    outputs,
                    models
                        .into_iter()
                        .map(|m| Arc::new(m) as Arc<dyn Runner>)
                        .collect(),
                )
            }
            "safetensors" | "npz" => {
                let sd = if ext == "npz" {
                    StateDict::load_npz(path)?
                } else {
                    StateDict::load_safetensors(path)?
                };
                let mut replicas: Vec<Arc<dyn Runner>> = Vec::new();
                let mut in_features = 0;
                for npu in self.pool.npus() {
                    let (model, info) = nn::mlp_from_state_dict(npu, &sd, activation)?;
                    in_features = info[0].in_features;
                    replicas.push(Arc::new(ModuleRunner(Box::new(model))));
                }
                let inputs = vec![InputSpec {
                    name: "input".into(),
                    shape: vec![None, Some(in_features)],
                    integer: false,
                }];
                ("mlp", inputs, vec!["output".into()], replicas)
            }
            other => {
                anyhow::bail!("unsupported model format .{other} (use .onnx, .safetensors or .npz)")
            }
        };
        self.insert(name, kind, inputs, outputs, replicas);
        Ok(())
    }

    /// Serves an already-built module (single input called `input`, output `output`)
    /// built once per NPU by `build`. Must be called inside a tokio runtime.
    pub fn add_module<M, F>(
        &mut self,
        name: &str,
        in_features: usize,
        build: F,
    ) -> anyhow::Result<()>
    where
        M: Module + 'static,
        F: Fn(&Npu) -> qanten::Result<M>,
    {
        let replicas = self
            .pool
            .npus()
            .iter()
            .map(|npu| Ok(Arc::new(ModuleRunner(Box::new(build(npu)?))) as Arc<dyn Runner>))
            .collect::<qanten::Result<Vec<_>>>()?;
        let inputs = vec![InputSpec {
            name: "input".into(),
            shape: vec![None, Some(in_features)],
            integer: false,
        }];
        self.insert(name, "module", inputs, vec!["output".into()], replicas);
        Ok(())
    }

    fn insert(
        &mut self,
        name: &str,
        kind: &'static str,
        inputs: Vec<InputSpec>,
        outputs: Vec<String>,
        replicas: Vec<Arc<dyn Runner>>,
    ) {
        // Batch only single float-input models.
        let batchable = inputs.len() == 1 && !inputs[0].integer;
        let mut entry = ModelEntry {
            kind,
            inputs,
            outputs,
            replicas,
            batcher: None,
            stats: Stats::default(),
        };
        if batchable {
            let (tx, rx) = mpsc::channel(1024);
            entry.batcher = Some(tx);
            let entry = Arc::new(entry);
            tokio::spawn(batcher(
                Arc::clone(&entry),
                rx,
                Arc::clone(&self.exec),
                self.pool.clone(),
                self.config,
            ));
            self.models.insert(name.to_owned(), entry);
        } else {
            self.models.insert(name.to_owned(), Arc::new(entry));
        }
    }
}

/// Collects requests into batches and runs them.
async fn batcher(
    entry: Arc<ModelEntry>,
    mut rx: mpsc::Receiver<BatchItem>,
    exec: Arc<NpuExecutor>,
    pool: NpuPool,
    cfg: ServerConfig,
) {
    while let Some(first) = rx.recv().await {
        let mut items = vec![first];
        let deadline = tokio::time::Instant::now() + cfg.max_wait;
        let rows = |items: &[BatchItem]| {
            items
                .iter()
                .map(|i| i.input.shape().first().copied().unwrap_or(1))
                .sum::<usize>()
        };
        while rows(&items) < cfg.max_batch_rows {
            match tokio::time::timeout_at(deadline, rx.recv()).await {
                Ok(Some(item)) => items.push(item),
                _ => break,
            }
        }
        // Requests whose trailing shape differs from the first are run on their own.
        let trailing = items[0].input.shape().get(1..).map(<[usize]>::to_vec);
        let (same, other): (Vec<BatchItem>, Vec<BatchItem>) = items.into_iter().partition(|i| {
            i.input.rank() >= 1 && i.input.shape().get(1..).map(<[usize]>::to_vec) == trailing
        });
        // Each batch runs as its own task, so several NPUs can work on one model at once.
        for group in std::iter::once(same).chain(other.into_iter().map(|i| vec![i])) {
            if group.is_empty() {
                continue;
            }
            tokio::spawn(run_batch(
                Arc::clone(&entry),
                Arc::clone(&exec),
                pool.clone(),
                group,
            ));
        }
    }
}

async fn run_batch(
    entry: Arc<ModelEntry>,
    exec: Arc<NpuExecutor>,
    pool: NpuPool,
    group: Vec<BatchItem>,
) {
    let sizes: Vec<usize> = group.iter().map(|i| i.input.shape()[0]).collect();
    let total: usize = sizes.iter().sum();
    let parts: Vec<Tensor> = group.iter().map(|i| i.input.clone()).collect();
    let input_name = entry.inputs[0].name.clone();
    let model = Arc::clone(&entry);
    let result = match Tensor::concat_batch(&parts) {
        Ok(batch) => exec
            .run(move |npu| {
                let r = &model.replicas[replica_index(&pool, npu)];
                r.run(&[(input_name.as_str(), OnnxValue::Float(batch))])
            })
            .await
            .map_err(|e| e.to_string()),
        Err(e) => Err(e.to_string()),
    };
    entry.stats.batches.fetch_add(1, Ordering::Relaxed);
    entry
        .stats
        .batched_rows
        .fetch_add(total as u64, Ordering::Relaxed);
    match result {
        Ok(outputs) => {
            // Split every output along its first dimension, if it is the batch dimension.
            let mut per_item: Vec<Outputs> = vec![Vec::new(); group.len()];
            for (name, value) in outputs {
                let t = match value.into_tensor() {
                    Ok(t) => t,
                    Err(e) => {
                        for item in group {
                            let _ = item.reply.send(Err(e.to_string()));
                        }
                        return;
                    }
                };
                if t.shape().first() == Some(&total) {
                    let row = t.len() / total.max(1);
                    let mut start = 0;
                    for (k, &n) in sizes.iter().enumerate() {
                        let mut shape = t.shape().to_vec();
                        shape[0] = n;
                        let piece =
                            Tensor::new(t.data()[start * row..(start + n) * row].to_vec(), shape);
                        start += n;
                        if let Ok(piece) = piece {
                            per_item[k].push((name.clone(), OnnxValue::Float(piece)));
                        }
                    }
                } else {
                    for p in &mut per_item {
                        p.push((name.clone(), OnnxValue::Float(t.clone())));
                    }
                }
            }
            for (item, out) in group.into_iter().zip(per_item) {
                let _ = item.reply.send(Ok((out, total)));
            }
        }
        Err(e) => {
            for item in group {
                let _ = item.reply.send(Err(e.clone()));
            }
        }
    }
}

// ---------------------------------------------------------------- HTTP

/// A tensor in JSON: `{"shape": [...], "data": [...]}`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct JsonTensor {
    /// Dimensions.
    pub shape: Vec<usize>,
    /// Values, row-major.
    pub data: Vec<f64>,
}

/// Body of `POST /v1/models/{name}/infer`: either `{"inputs": {"x": tensor, ...}}`
/// or, for single-input models, just a tensor.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum InferRequest {
    /// Named inputs.
    Named {
        /// Input name → tensor.
        inputs: BTreeMap<String, JsonTensor>,
    },
    /// The only input.
    Single(JsonTensor),
}

/// Response of an inference request.
#[derive(Debug, Serialize, Deserialize)]
pub struct InferResponse {
    /// Model name.
    pub model: String,
    /// Output name → tensor.
    pub outputs: BTreeMap<String, JsonTensor>,
    /// Rows in the NPU batch this request was part of (> its own rows when batched).
    pub batch_rows: usize,
    /// Server-side time for this request.
    pub latency_ms: f64,
}

struct ApiError(StatusCode, String);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(serde_json::json!({ "error": self.1 }))).into_response()
    }
}

fn bad_request(msg: impl ToString) -> ApiError {
    ApiError(StatusCode::BAD_REQUEST, msg.to_string())
}

fn to_value(t: &JsonTensor, integer: bool) -> Result<OnnxValue, ApiError> {
    let n: usize = t.shape.iter().product();
    if n != t.data.len() {
        return Err(bad_request(format!(
            "shape {:?} needs {n} values, got {}",
            t.shape,
            t.data.len()
        )));
    }
    Ok(if integer {
        OnnxValue::Int {
            data: t.data.iter().map(|&v| v as i64).collect(),
            shape: t.shape.clone(),
        }
    } else {
        let data: Vec<f32> = t.data.iter().map(|&v| v as f32).collect();
        OnnxValue::Float(Tensor::from_f32(&data, t.shape.clone()).map_err(bad_request)?)
    })
}

fn to_json(v: OnnxValue) -> Result<JsonTensor, ApiError> {
    Ok(match v {
        OnnxValue::Float(t) => JsonTensor {
            shape: t.shape().to_vec(),
            data: t.to_f32_vec().into_iter().map(f64::from).collect(),
        },
        OnnxValue::Int { data, shape } => JsonTensor {
            shape,
            data: data.into_iter().map(|v| v as f64).collect(),
        },
    })
}

async fn health(State(s): State<Arc<AppState>>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "ok",
        "backend": s.backend,
        "npus": s.pool.len(),
        "models": s.models.len(),
        "uptime_s": s.started.elapsed().as_secs_f64(),
    }))
}

#[derive(Serialize)]
struct ModelInfo<'a> {
    name: &'a str,
    kind: &'a str,
    inputs: &'a [InputSpec],
    outputs: &'a [String],
    dynamic_batching: bool,
}

async fn list_models(State(s): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let models: Vec<ModelInfo<'_>> = s
        .models
        .iter()
        .map(|(name, m)| ModelInfo {
            name,
            kind: m.kind,
            inputs: &m.inputs,
            outputs: &m.outputs,
            dynamic_batching: m.batcher.is_some(),
        })
        .collect();
    Json(serde_json::json!({ "models": models }))
}

async fn infer(
    State(s): State<Arc<AppState>>,
    Path(name): Path<String>,
    Json(req): Json<InferRequest>,
) -> Result<Json<InferResponse>, ApiError> {
    let start = Instant::now();
    let entry = s
        .models
        .get(&name)
        .cloned()
        .ok_or_else(|| ApiError(StatusCode::NOT_FOUND, format!("no model named `{name}`")))?;
    entry.stats.requests.fetch_add(1, Ordering::Relaxed);
    let result = infer_inner(&s, &entry, req).await;
    entry
        .stats
        .latency_us
        .fetch_add(start.elapsed().as_micros() as u64, Ordering::Relaxed);
    match result {
        Ok((outputs, batch_rows)) => Ok(Json(InferResponse {
            model: name,
            outputs: outputs
                .into_iter()
                .map(|(k, v)| Ok((k, to_json(v)?)))
                .collect::<Result<_, ApiError>>()?,
            batch_rows,
            latency_ms: start.elapsed().as_secs_f64() * 1e3,
        })),
        Err(e) => {
            entry.stats.errors.fetch_add(1, Ordering::Relaxed);
            Err(e)
        }
    }
}

async fn infer_inner(
    s: &Arc<AppState>,
    entry: &Arc<ModelEntry>,
    req: InferRequest,
) -> Result<(Outputs, usize), ApiError> {
    let named: BTreeMap<String, JsonTensor> = match req {
        InferRequest::Named { inputs } => inputs,
        InferRequest::Single(t) => {
            let first = entry
                .inputs
                .first()
                .ok_or_else(|| bad_request("model has no inputs"))?;
            BTreeMap::from([(first.name.clone(), t)])
        }
    };
    let mut values = Vec::with_capacity(entry.inputs.len());
    for spec in &entry.inputs {
        let t = named
            .get(&spec.name)
            .ok_or_else(|| bad_request(format!("missing input `{}`", spec.name)))?;
        values.push((spec.name.clone(), to_value(t, spec.integer)?));
    }
    if let Some(tx) = &entry.batcher {
        let input = match values.pop() {
            Some((_, OnnxValue::Float(t))) if t.rank() >= 1 => t,
            _ => {
                return Err(bad_request(
                    "input must be a float tensor with a batch dimension",
                ));
            }
        };
        let (reply, wait) = oneshot::channel();
        tx.send(BatchItem { input, reply })
            .await
            .map_err(|_| ApiError(StatusCode::SERVICE_UNAVAILABLE, "batcher stopped".into()))?;
        let r = wait
            .await
            .map_err(|_| ApiError(StatusCode::INTERNAL_SERVER_ERROR, "batch dropped".into()))?;
        return r.map_err(bad_request);
    }
    let model = Arc::clone(entry);
    let pool = s.pool.clone();
    let rows = values.first().map_or(1, |(_, v)| match v {
        OnnxValue::Float(t) => t.shape().first().copied().unwrap_or(1),
        OnnxValue::Int { shape, .. } => shape.first().copied().unwrap_or(1),
    });
    let out = s
        .exec
        .run(move |npu| {
            let refs: Vec<(&str, OnnxValue)> = values
                .iter()
                .map(|(n, v)| (n.as_str(), v.clone()))
                .collect();
            model.replicas[replica_index(&pool, npu)].run(&refs)
        })
        .await
        .map_err(bad_request)?;
    Ok((out, rows))
}

async fn metrics(State(s): State<Arc<AppState>>) -> impl IntoResponse {
    use std::fmt::Write;
    let mut m = String::new();
    let mut metric = |name: &str, help: &str, kind: &str, rows: Vec<(String, f64)>| {
        let _ = writeln!(m, "# HELP {name} {help}\n# TYPE {name} {kind}");
        for (labels, v) in rows {
            let _ = writeln!(m, "{name}{labels} {v}");
        }
    };
    let per_model = |f: &dyn Fn(&Stats) -> f64| -> Vec<(String, f64)> {
        s.models
            .iter()
            .map(|(n, e)| (format!("{{model=\"{n}\"}}"), f(&e.stats)))
            .collect()
    };
    let load = |a: &AtomicU64| a.load(Ordering::Relaxed) as f64;
    metric(
        "qanten_uptime_seconds",
        "Seconds since the server started.",
        "gauge",
        vec![(String::new(), s.started.elapsed().as_secs_f64())],
    );
    metric(
        "qanten_requests_total",
        "Inference requests received.",
        "counter",
        per_model(&|st| load(&st.requests)),
    );
    metric(
        "qanten_request_errors_total",
        "Inference requests that failed.",
        "counter",
        per_model(&|st| load(&st.errors)),
    );
    metric(
        "qanten_request_latency_seconds_sum",
        "Total server-side request time.",
        "counter",
        per_model(&|st| load(&st.latency_us) / 1e6),
    );
    metric(
        "qanten_batches_total",
        "NPU batches executed by the dynamic batcher.",
        "counter",
        per_model(&|st| load(&st.batches)),
    );
    metric(
        "qanten_batched_rows_total",
        "Rows processed in dynamic batches.",
        "counter",
        per_model(&|st| load(&st.batched_rows)),
    );
    let reports: Vec<_> = s.profilers.iter().map(Profiler::report).collect();
    let per_npu = |f: &dyn Fn(&qanten::profile::Report) -> f64| -> Vec<(String, f64)> {
        reports
            .iter()
            .enumerate()
            .map(|(i, r)| (format!("{{npu=\"{i}\"}}"), f(r)))
            .collect()
    };
    metric(
        "qanten_npu_macs_total",
        "Multiply-accumulates performed on the NPU.",
        "counter",
        per_npu(&|r| r.total().macs as f64),
    );
    metric(
        "qanten_npu_busy_seconds_total",
        "Time spent inside the NPU backend.",
        "counter",
        per_npu(&|r| r.total().time.as_secs_f64()),
    );
    metric(
        "qanten_npu_bytes_to_device_total",
        "Bytes sent to the NPU.",
        "counter",
        per_npu(&|r| r.total().bytes_to_device as f64),
    );
    metric(
        "qanten_npu_bytes_from_device_total",
        "Bytes read back from the NPU.",
        "counter",
        per_npu(&|r| r.total().bytes_from_device as f64),
    );
    metric(
        "qanten_npu_resident_matvec_calls_total",
        "Matrix-vector calls on resident weights.",
        "counter",
        per_npu(&|r| r.op(OpKind::MatVecResident).calls as f64),
    );
    metric(
        "qanten_npu_device_memory_bytes",
        "Device memory currently allocated.",
        "gauge",
        per_npu(&|r| r.live_device_bytes as f64),
    );
    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4",
        )],
        m,
    )
}

/// The HTTP routes:
///
/// | Method | Path | |
/// |---|---|---|
/// | GET | `/health` | status, backend, NPU and model counts |
/// | GET | `/v1/models` | models with their inputs and outputs |
/// | POST | `/v1/models/{name}/infer` | run a model (JSON) |
/// | GET | `/metrics` | Prometheus metrics |
pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/models/{name}/infer", post(infer))
        .route("/metrics", get(metrics))
        .with_state(state)
}
