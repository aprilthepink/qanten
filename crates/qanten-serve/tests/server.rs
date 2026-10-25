use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use qanten::nn::{ActivationKind, StateDict};
use qanten::onnx::OnnxModel;
use qanten::pool::NpuPool;
use qanten::{Npu, Tensor};
use qanten_serve::{AppState, InferResponse, ServerConfig, router};
use tower::ServiceExt;

const ONNX: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../qanten/tests/data/onnx");
const DIGITS: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../qanten/examples/data/qant_digit_weights.npz"
);

fn app(npus: u32, wait_ms: u64) -> Router {
    app_with(npus, wait_ms, 64)
}

fn app_with(npus: u32, wait_ms: u64, max_batch_rows: usize) -> Router {
    let pool = NpuPool::simulated(npus).unwrap();
    let mut state = AppState::new(
        &pool,
        ServerConfig {
            max_batch_rows,
            max_wait: Duration::from_millis(wait_ms),
            max_body_bytes: 1 << 20,
            ..ServerConfig::default()
        },
    )
    .unwrap();
    state
        .add_model_file(
            "mlp",
            format!("{ONNX}/mlp.onnx").as_ref(),
            ActivationKind::Relu,
        )
        .unwrap();
    state
        .add_model_file(
            "block",
            format!("{ONNX}/block.onnx").as_ref(),
            ActivationKind::Relu,
        )
        .unwrap();
    state
        .add_model_file("digits", DIGITS.as_ref(), ActivationKind::Relu)
        .unwrap();
    router(Arc::new(state))
}

async fn call(
    app: &Router,
    method: &str,
    uri: &str,
    body: Option<serde_json::Value>,
) -> (StatusCode, serde_json::Value) {
    let req = Request::builder()
        .method(method)
        .uri(uri)
        .header("content-type", "application/json")
        .body(body.map_or(Body::empty(), |b| Body::from(b.to_string())))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let json = serde_json::from_slice(&bytes)
        .unwrap_or_else(|_| serde_json::Value::String(String::from_utf8_lossy(&bytes).into()));
    (status, json)
}

#[tokio::test]
async fn health_and_models() {
    let app = app(1, 1);
    let (s, h) = call(&app, "GET", "/health", None).await;
    assert_eq!(
        (s, h["status"].as_str(), h["models"].as_u64()),
        (StatusCode::OK, Some("ok"), Some(3))
    );
    let (_, m) = call(&app, "GET", "/v1/models", None).await;
    let names: Vec<&str> = m["models"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, vec!["block", "digits", "mlp"]);
    let mlp = &m["models"].as_array().unwrap()[2];
    assert_eq!(mlp["dynamic_batching"], true);
    assert_eq!(
        m["models"][0]["dynamic_batching"], false,
        "integer-input model is not batched"
    );
}

#[tokio::test]
async fn infer_matches_direct_run() {
    let app = app(1, 1);
    let inputs = StateDict::load_npz(format!("{ONNX}/mlp.inputs.npz")).unwrap();
    let x = inputs.get("x").unwrap();
    let body = serde_json::json!({ "shape": x.shape(), "data": x.to_f32_vec() });
    let (s, r) = call(&app, "POST", "/v1/models/mlp/infer", Some(body)).await;
    assert_eq!(s, StatusCode::OK, "{r}");
    let r: InferResponse = serde_json::from_value(r).unwrap();
    let got = &r.outputs["probs"];
    let want = OnnxModel::load(&Npu::simulator(), format!("{ONNX}/mlp.onnx"))
        .unwrap()
        .run_single(x)
        .unwrap();
    assert_eq!(got.shape, want.shape());
    let err = got
        .data
        .iter()
        .zip(want.to_f32_vec())
        .map(|(a, b)| (*a as f32 - b).abs())
        .fold(0.0, f32::max);
    assert!(err < 1e-3, "server vs direct: {err}");

    // Named-input form and the MLP weight file.
    let body = serde_json::json!({ "inputs": { "input": { "shape": [2, 784], "data": vec![0.5; 2 * 784] } } });
    let (s, r) = call(&app, "POST", "/v1/models/digits/infer", Some(body)).await;
    assert_eq!(s, StatusCode::OK, "{r}");
    assert_eq!(r["outputs"]["output"]["shape"], serde_json::json!([2, 10]));
}

#[tokio::test]
async fn integer_inputs_on_the_unbatched_path() {
    let app = app(1, 1);
    let body = serde_json::json!({ "inputs": { "ids": { "shape": [1, 3], "data": [1, 4, 2] } } });
    let (s, r) = call(&app, "POST", "/v1/models/block/infer", Some(body)).await;
    assert_eq!(s, StatusCode::OK, "{r}");
    assert_eq!(r["outputs"]["y"]["shape"], serde_json::json!([1, 3, 8]));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_requests_are_batched_across_npus() {
    // Batches of at most 4 rows: 24 requests need several batches, which spread over both NPUs.
    let app = app_with(2, 30, 4);
    let direct = OnnxModel::load(&Npu::simulator(), format!("{ONNX}/mlp.onnx")).unwrap();
    let mut tasks = Vec::new();
    for i in 0..24 {
        let app = app.clone();
        tasks.push(tokio::spawn(async move {
            let x: Vec<f32> = (0..8).map(|k| ((i * 8 + k) % 13) as f32 / 13.0).collect();
            let body = serde_json::json!({ "shape": [1, 8], "data": x });
            let (s, r) = call(&app, "POST", "/v1/models/mlp/infer", Some(body)).await;
            assert_eq!(s, StatusCode::OK);
            (x, serde_json::from_value::<InferResponse>(r).unwrap())
        }));
    }
    let mut max_batch = 0;
    for t in tasks {
        let (x, r) = t.await.unwrap();
        max_batch = max_batch.max(r.batch_rows);
        // Each caller gets its own row back.
        let want = direct
            .run_single(&Tensor::from_f32(&x, [1, 8]).unwrap())
            .unwrap()
            .to_f32_vec();
        let got: Vec<f32> = r.outputs["probs"].data.iter().map(|&v| v as f32).collect();
        let err = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(err < 0.02, "row mismatch {err}");
    }
    assert!(max_batch > 1, "requests were never batched");
    let (_, m) = call(&app, "GET", "/metrics", None).await;
    let text = m.as_str().unwrap();
    assert!(
        text.contains("qanten_requests_total{model=\"mlp\"} 24"),
        "{text}"
    );
    let batches: f64 = text
        .lines()
        .find(|l| l.starts_with("qanten_batches_total{model=\"mlp\"}"))
        .unwrap()
        .rsplit(' ')
        .next()
        .unwrap()
        .parse()
        .unwrap();
    assert!(
        (6.0..24.0).contains(&batches),
        "{batches} batches for 24 requests"
    );
    for npu in ["0", "1"] {
        let macs: f64 = text
            .lines()
            .find(|l| l.starts_with(&format!("qanten_npu_macs_total{{npu=\"{npu}\"}}")))
            .unwrap()
            .rsplit(' ')
            .next()
            .unwrap()
            .parse()
            .unwrap();
        assert!(macs > 0.0, "NPU {npu} did no work");
    }
}

#[tokio::test]
async fn errors() {
    let app = app(1, 1);
    let (s, _) = call(
        &app,
        "POST",
        "/v1/models/nope/infer",
        Some(serde_json::json!({ "shape": [1], "data": [0] })),
    )
    .await;
    assert_eq!(s, StatusCode::NOT_FOUND);
    let (s, r) = call(
        &app,
        "POST",
        "/v1/models/mlp/infer",
        Some(serde_json::json!({ "shape": [2, 8], "data": [1, 2] })),
    )
    .await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    assert!(r["error"].as_str().unwrap().contains("needs 16 values"));
    let (s, _) = call(
        &app,
        "POST",
        "/v1/models/mlp/infer",
        Some(serde_json::json!({ "shape": [1, 5], "data": [1, 2, 3, 4, 5] })),
    )
    .await;
    assert_eq!(
        s,
        StatusCode::BAD_REQUEST,
        "wrong feature count reaches the model and fails cleanly"
    );
    let (_, m) = call(&app, "GET", "/metrics", None).await;
    assert!(
        m.as_str()
            .unwrap()
            .contains("qanten_request_errors_total{model=\"mlp\"} 2")
    );
}

#[tokio::test]
async fn hostile_requests_are_rejected_cleanly() {
    let app = app(1, 1);
    let infer = "/v1/models/mlp/infer";
    // A shape that asks for 2^40 values, and one whose overflow hides behind a 0.
    for shape in [
        serde_json::json!([1 << 20, 1 << 20]),
        serde_json::json!([0, 1u64 << 62, 1u64 << 62]),
    ] {
        let (status, body) = call(
            &app,
            "POST",
            infer,
            Some(serde_json::json!({"shape": shape, "data": []})),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    }
    // Too many dimensions.
    let (status, _) = call(
        &app,
        "POST",
        infer,
        Some(serde_json::json!({"shape": vec![1; 64], "data": [0.0]})),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    // A body over the limit (1 MiB here) is refused before parsing.
    let big = serde_json::json!({"shape": [1, 300_000], "data": vec![0.5; 300_000]});
    let (status, _) = call(&app, "POST", infer, Some(big)).await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
    // Unknown model names are echoed truncated.
    let long = "x".repeat(500);
    let (status, body) = call(
        &app,
        "POST",
        &format!("/v1/models/{long}/infer"),
        Some(serde_json::json!({"shape": [1], "data": [1.0]})),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert!(body["error"].as_str().unwrap().len() < 100);
    // The server still works afterwards.
    let (status, _) = call(&app, "GET", "/health", None).await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn model_names_are_restricted() {
    let pool = NpuPool::simulated(1).unwrap();
    let mut state = AppState::new(&pool, ServerConfig::default()).unwrap();
    for bad in ["", "a\"b", "x\ny", "has space", &"n".repeat(65)] {
        assert!(
            state
                .add_model_file(
                    bad,
                    format!("{ONNX}/mlp.onnx").as_ref(),
                    ActivationKind::Relu
                )
                .is_err(),
            "{bad:?} should be rejected"
        );
    }
    assert!(
        state
            .add_model_file(
                "mlp-v1.2_b",
                format!("{ONNX}/mlp.onnx").as_ref(),
                ActivationKind::Relu
            )
            .is_ok()
    );
}
