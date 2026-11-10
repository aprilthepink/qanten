//! # qanten-serve
//!
//! An HTTP inference server for Q.ANT photonic NPUs.
//!
//! - Serves ONNX models and MLP weight files from one process.
//! - **Multi-NPU**: every model is loaded once per NPU of an
//!   [`NpuPool`]; requests go to whichever NPU is free.
//! - **Dynamic batching**: concurrent requests to a single-input model are
//!   merged into one NPU batch (up to `max_batch_rows` rows or `max_wait`),
//!   then split back per request. ONNX models are batched only when
//!   [`OnnxModel::rows_are_independent`] proves no output row depends on
//!   another caller's input.
//! - JSON API, `/health`, and Prometheus `/metrics` with request and NPU counters.
//!
//! The binary (`qanten-serve`) wraps [`AppState`] and [`router`]; both are
//! public so the server can be embedded in another axum application.

#![warn(missing_docs)]

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use axum::extract::{Path, Request, State};
use axum::http::StatusCode;
use axum::middleware::{self, Next};
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
use tokio::sync::{Semaphore, mpsc, oneshot};

/// Server tuning and limits.
#[derive(Debug, Clone, Copy)]
pub struct ServerConfig {
    /// Maximum rows merged into one NPU batch.
    pub max_batch_rows: usize,
    /// How long the batcher waits for more requests after the first one.
    pub max_wait: Duration,
    /// Largest accepted request body.
    pub max_body_bytes: usize,
    /// An inference request that takes longer fails with 504, and its work
    /// is cancelled between model nodes.
    pub request_timeout: Duration,
    /// Inference requests handled at once; more get 503 straight away. This
    /// bounds the work queued per NPU and the memory held by request bodies.
    pub max_concurrent_requests: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            max_batch_rows: 64,
            max_wait: Duration::from_millis(2),
            max_body_bytes: 16 << 20,
            request_timeout: Duration::from_secs(30),
            max_concurrent_requests: 256,
        }
    }
}

/// Model names: 1–64 characters of `[A-Za-z0-9._-]`. They appear in URLs,
/// error messages and Prometheus labels, so nothing else is allowed.
fn check_model_name(name: &str) -> anyhow::Result<()> {
    let ok = (1..=64).contains(&name.len())
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'));
    anyhow::ensure!(
        ok,
        "invalid model name {name:?}: use 1-64 characters from A-Z, a-z, 0-9, '.', '_', '-'"
    );
    Ok(())
}

/// Escapes a Prometheus label value (`\`, `"`, newline).
fn label(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
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

/// A loaded model: kind, inputs, outputs, one runner per NPU, and whether
/// requests from different callers may share a batch.
type Loaded = (
    &'static str,
    Vec<InputSpec>,
    Vec<String>,
    Vec<Arc<dyn Runner>>,
    bool,
);

/// Sets a cancellation flag when dropped, i.e. when the request that owns it
/// finishes or is abandoned (timeout, client gone).
struct CancelOnDrop(Arc<AtomicBool>);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

/// Something that can be served.
trait Runner: Send + Sync {
    /// Runs the model; stops early with [`qanten::Error::Cancelled`] once
    /// `cancel` is set, where the model can.
    fn run(&self, inputs: &[(&str, OnnxValue)], cancel: &AtomicBool) -> qanten::Result<Outputs>;
}

impl Runner for OnnxModel {
    fn run(&self, inputs: &[(&str, OnnxValue)], cancel: &AtomicBool) -> qanten::Result<Outputs> {
        self.run_cancellable(inputs, cancel)
    }
}

struct ModuleRunner(Box<dyn Module>);

impl Runner for ModuleRunner {
    fn run(&self, inputs: &[(&str, OnnxValue)], cancel: &AtomicBool) -> qanten::Result<Outputs> {
        if cancel.load(Ordering::Relaxed) {
            return Err(qanten::Error::Cancelled);
        }
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
    /// Set once the request is abandoned.
    cancelled: Arc<AtomicBool>,
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
    /// Permits for in-flight inference requests.
    in_flight: Arc<Semaphore>,
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
            in_flight: Arc::new(Semaphore::new(config.max_concurrent_requests)),
            started: Instant::now(),
            backend,
        })
    }

    /// Loads a model file under `name`: `.onnx`, or `.safetensors` / `.npz`
    /// (an MLP, see [`nn::Sequential::load_mlp`]). Must be called inside a tokio runtime.
    pub fn add_model_file(
        &mut self,
        name: &str,
        path: &std::path::Path,
        activation: ActivationKind,
    ) -> anyhow::Result<()> {
        check_model_name(name)?;
        let ext = path
            .extension()
            .map(|e| e.to_string_lossy().to_lowercase())
            .unwrap_or_default();
        let (kind, inputs, outputs, replicas, batchable): Loaded = match ext.as_str() {
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
                let batchable = models[0].rows_are_independent();
                (
                    "onnx",
                    inputs,
                    outputs,
                    models
                        .into_iter()
                        .map(|m| Arc::new(m) as Arc<dyn Runner>)
                        .collect(),
                    batchable,
                )
            }
            "safetensors" | "npz" => {
                let sd = StateDict::load_file(path)?;
                let mut replicas: Vec<Arc<dyn Runner>> = Vec::new();
                let mut in_features = 0;
                for npu in self.pool.npus() {
                    let (model, info) = nn::Sequential::load_mlp(npu, &sd, activation)?;
                    in_features = info[0].in_features;
                    replicas.push(Arc::new(ModuleRunner(Box::new(model))));
                }
                let inputs = vec![InputSpec {
                    name: "input".into(),
                    shape: vec![None, Some(in_features)],
                    integer: false,
                }];
                // Dense layers and activations are row-wise.
                ("mlp", inputs, vec!["output".into()], replicas, true)
            }
            other => {
                anyhow::bail!("unsupported model format .{other} (use .onnx, .safetensors or .npz)")
            }
        };
        self.insert(name, kind, inputs, outputs, replicas, batchable);
        Ok(())
    }

    /// Serves an already-built module (single input called `input`, output `output`)
    /// built once per NPU by `build`. Must be called inside a tokio runtime.
    ///
    /// Requests are batched, so the module must process the rows of its
    /// input independently (true for MLPs and convolutional nets in
    /// inference mode); otherwise one caller's result would depend on, and
    /// could reveal, another caller's input.
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
        check_model_name(name)?;
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
        self.insert(
            name,
            "module",
            inputs,
            vec!["output".into()],
            replicas,
            true,
        );
        Ok(())
    }

    fn insert(
        &mut self,
        name: &str,
        kind: &'static str,
        inputs: Vec<InputSpec>,
        outputs: Vec<String>,
        replicas: Vec<Arc<dyn Runner>>,
        rows_independent: bool,
    ) {
        // Batch only single float-input models that never mix rows.
        let batchable = rows_independent && inputs.len() == 1 && !inputs[0].integer;
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
                .fold(0usize, usize::saturating_add)
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

/// Runs `input` on whichever NPU is free, stopping early once `cancel` is set.
async fn run_on_pool(
    entry: &Arc<ModelEntry>,
    exec: &NpuExecutor,
    pool: &NpuPool,
    input: Tensor,
    cancel: Arc<AtomicBool>,
) -> Result<Outputs, String> {
    let model = Arc::clone(entry);
    let pool = pool.clone();
    exec.run(move |npu| {
        let name = model.inputs[0].name.as_str();
        model.replicas[replica_index(&pool, npu)].run(&[(name, OnnxValue::Float(input))], &cancel)
    })
    .await
    .map_err(|e| e.to_string())
}

/// Splits every output of a batch along axis 0 into one piece per request.
/// `None` if an output's first dimension is not the batch: the model did not
/// keep rows apart for this input, so the batch must not be shared.
fn split_outputs(outputs: Outputs, sizes: &[usize]) -> Result<Option<Vec<Outputs>>, String> {
    let total: usize = sizes.iter().sum();
    let mut per_item: Vec<Outputs> = vec![Vec::new(); sizes.len()];
    for (name, value) in outputs {
        let t = value.into_tensor().map_err(|e| e.to_string())?;
        if t.shape().first() != Some(&total) {
            return Ok(None);
        }
        let row = t.len() / total.max(1);
        let mut start = 0;
        for (k, &n) in sizes.iter().enumerate() {
            let mut shape = t.shape().to_vec();
            shape[0] = n;
            let piece = Tensor::new(t.data()[start * row..(start + n) * row].to_vec(), shape)
                .map_err(|e| e.to_string())?;
            start += n;
            per_item[k].push((name.clone(), OnnxValue::Float(piece)));
        }
    }
    Ok(Some(per_item))
}

async fn run_batch(
    entry: Arc<ModelEntry>,
    exec: Arc<NpuExecutor>,
    pool: NpuPool,
    group: Vec<BatchItem>,
) {
    let sizes: Vec<usize> = group.iter().map(|i| i.input.shape()[0]).collect();
    let total: usize = sizes.iter().sum();
    entry.stats.batches.fetch_add(1, Ordering::Relaxed);
    entry
        .stats
        .batched_rows
        .fetch_add(total as u64, Ordering::Relaxed);
    // The batch is cancelled once every request in it has been abandoned.
    let cancel = Arc::new(AtomicBool::new(false));
    let watcher = {
        let flags: Vec<Arc<AtomicBool>> = group.iter().map(|i| Arc::clone(&i.cancelled)).collect();
        let cancel = Arc::clone(&cancel);
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(Duration::from_millis(50));
            while !flags.iter().all(|f| f.load(Ordering::Relaxed)) {
                tick.tick().await;
            }
            cancel.store(true, Ordering::Relaxed);
        })
    };
    let parts: Vec<Tensor> = group.iter().map(|i| i.input.clone()).collect();
    let result = match Tensor::concat_batch(&parts) {
        Ok(batch) => run_on_pool(&entry, &exec, &pool, batch, Arc::clone(&cancel)).await,
        Err(e) => Err(e.to_string()),
    };
    watcher.abort();
    let mut group = group;
    if group.len() == 1 {
        // A batch of one is the request itself, whatever its output shapes.
        if let Some(item) = group.pop() {
            let _ = item.reply.send(result.map(|o| (o, total)));
        }
        return;
    }
    match result.and_then(|out| split_outputs(out, &sizes)) {
        Ok(Some(per_item)) => {
            for (item, out) in group.into_iter().zip(per_item) {
                let _ = item.reply.send(Ok((out, total)));
            }
        }
        // Rows were not kept apart: never hand one caller another's data.
        // Run each request on its own instead.
        Ok(None) => {
            for item in group {
                let rows = item.input.shape()[0];
                let out = run_on_pool(&entry, &exec, &pool, item.input, item.cancelled).await;
                let _ = item.reply.send(out.map(|o| (o, rows)));
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
#[serde(try_from = "RawInferRequest")]
pub enum InferRequest {
    /// Named inputs.
    Named {
        /// Input name → tensor.
        inputs: BTreeMap<String, JsonTensor>,
    },
    /// The only input.
    Single(JsonTensor),
}

/// Both request forms as one struct. Deserializing it streams the body
/// straight into the vectors; `#[serde(untagged)]` would first buffer the
/// whole body as generic JSON, using many times its size in memory.
#[derive(Deserialize)]
struct RawInferRequest {
    inputs: Option<BTreeMap<String, JsonTensor>>,
    shape: Option<Vec<usize>>,
    data: Option<Vec<f64>>,
}

impl TryFrom<RawInferRequest> for InferRequest {
    type Error = &'static str;

    fn try_from(r: RawInferRequest) -> Result<Self, Self::Error> {
        match r {
            RawInferRequest {
                inputs: Some(inputs),
                shape: None,
                data: None,
            } => Ok(Self::Named { inputs }),
            RawInferRequest {
                inputs: None,
                shape: Some(shape),
                data: Some(data),
            } => Ok(Self::Single(JsonTensor { shape, data })),
            _ => Err(r#"expected {"inputs": {...}} or {"shape": [...], "data": [...]}"#),
        }
    }
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
    // Checked and capped: a shape is never trusted to be small or to fit usize.
    let n = qanten::limits::elements(&t.shape).map_err(bad_request)?;
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
    let entry = s.models.get(&name).cloned().ok_or_else(|| {
        let shown: String = name.chars().take(64).collect();
        ApiError(StatusCode::NOT_FOUND, format!("no model named `{shown}`"))
    })?;
    entry.stats.requests.fetch_add(1, Ordering::Relaxed);
    let result =
        match tokio::time::timeout(s.config.request_timeout, infer_inner(&s, &entry, req)).await {
            Ok(r) => r,
            Err(_) => Err(ApiError(
                StatusCode::GATEWAY_TIMEOUT,
                "inference timed out".into(),
            )),
        };
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
    // Stops the model between nodes once this request is abandoned (the
    // timeout in `infer` drops this future) or done.
    let cancelled = Arc::new(AtomicBool::new(false));
    let _cancel_on_drop = CancelOnDrop(Arc::clone(&cancelled));
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
        tx.send(BatchItem {
            input,
            reply,
            cancelled: Arc::clone(&cancelled),
        })
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
            model.replicas[replica_index(&pool, npu)].run(&refs, &cancelled)
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
            .map(|(n, e)| (format!("{{model=\"{}\"}}", label(n)), f(&e.stats)))
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

/// Rejects an inference request with 503 when `max_concurrent_requests` are
/// already in flight. It runs before the body is read, so waiting requests
/// hold no memory and the per-NPU queues stay short.
async fn limit_in_flight(State(s): State<Arc<AppState>>, req: Request, next: Next) -> Response {
    match Arc::clone(&s.in_flight).try_acquire_owned() {
        Ok(_permit) => next.run(req).await,
        Err(_) => ApiError(
            StatusCode::SERVICE_UNAVAILABLE,
            "server busy, retry later".into(),
        )
        .into_response(),
    }
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
    let body_limit = axum::extract::DefaultBodyLimit::max(state.config.max_body_bytes);
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route(
            "/v1/models/{name}/infer",
            post(infer).layer(middleware::from_fn_with_state(
                Arc::clone(&state),
                limit_in_flight,
            )),
        )
        .route("/metrics", get(metrics))
        .layer(body_limit)
        .with_state(state)
}
