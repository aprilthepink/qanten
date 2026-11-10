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

/// A one-node ONNX model `y = Flatten(x, axis=0)`: it merges all rows into
/// one, so batching it would mix callers' data.
fn flatten_axis0_model() -> Vec<u8> {
    fn varint(mut v: u64, out: &mut Vec<u8>) {
        while v >= 0x80 {
            out.push((v as u8) | 0x80);
            v >>= 7;
        }
        out.push(v as u8);
    }
    fn field(num: u64, b: &[u8], out: &mut Vec<u8>) {
        varint((num << 3) | 2, out);
        varint(b.len() as u64, out);
        out.extend_from_slice(b);
    }
    fn int(num: u64, v: u64, out: &mut Vec<u8>) {
        varint(num << 3, out);
        varint(v, out);
    }
    // ValueInfo `name`: float tensor of shape [batch, 2].
    let value_info = |name: &str, dims: &[Option<u64>]| {
        let mut shape = Vec::new();
        for d in dims {
            let mut dim = Vec::new();
            match d {
                Some(v) => int(1, *v, &mut dim),
                None => field(2, b"batch", &mut dim),
            }
            field(1, &dim, &mut shape);
        }
        let mut tensor_type = Vec::new();
        int(1, 1, &mut tensor_type);
        field(2, &shape, &mut tensor_type);
        let mut type_proto = Vec::new();
        field(1, &tensor_type, &mut type_proto);
        let mut v = Vec::new();
        field(1, name.as_bytes(), &mut v);
        field(2, &type_proto, &mut v);
        v
    };
    let mut attr = Vec::new();
    field(1, b"axis", &mut attr);
    int(3, 0, &mut attr);
    let mut node = Vec::new();
    field(1, b"x", &mut node);
    field(2, b"y", &mut node);
    field(4, b"Flatten", &mut node);
    field(5, &attr, &mut node);
    let mut g = Vec::new();
    field(1, &node, &mut g);
    field(11, &value_info("x", &[None, Some(2)]), &mut g);
    field(12, &value_info("y", &[]), &mut g);
    let mut opset = Vec::new();
    int(2, 17, &mut opset);
    let mut m = Vec::new();
    int(1, 8, &mut m);
    field(8, &opset, &mut m);
    field(7, &g, &mut m);
    m
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn row_mixing_models_never_share_a_batch() {
    let path = std::env::temp_dir().join(format!("qanten-flatten0-{}.onnx", std::process::id()));
    std::fs::write(&path, flatten_axis0_model()).unwrap();
    let pool = NpuPool::simulated(1).unwrap();
    let mut state = AppState::new(
        &pool,
        ServerConfig {
            max_wait: Duration::from_millis(50),
            ..ServerConfig::default()
        },
    )
    .unwrap();
    state
        .add_model_file("flat", &path, ActivationKind::Relu)
        .unwrap();
    state
        .add_model_file(
            "mlp",
            format!("{ONNX}/mlp.onnx").as_ref(),
            ActivationKind::Relu,
        )
        .unwrap();
    let _ = std::fs::remove_file(&path);
    let app = router(Arc::new(state));

    let (_, models) = call(&app, "GET", "/v1/models", None).await;
    let batching = |name: &str| {
        models["models"]
            .as_array()
            .unwrap()
            .iter()
            .find(|m| m["name"] == name)
            .unwrap()["dynamic_batching"]
            .as_bool()
            .unwrap()
    };
    assert!(!batching("flat"), "Flatten(axis=0) mixes rows");
    assert!(batching("mlp"), "an MLP is row-wise");

    // Two concurrent callers each get exactly their own rows back.
    let req = |data: [f64; 2]| {
        let app = app.clone();
        async move {
            call(
                &app,
                "POST",
                "/v1/models/flat/infer",
                Some(serde_json::json!({ "shape": [1, 2], "data": data })),
            )
            .await
        }
    };
    let ((s1, a), (s2, b)) = tokio::join!(req([1.0, 2.0]), req([768.0, 896.0]));
    assert_eq!((s1, s2), (StatusCode::OK, StatusCode::OK));
    assert_eq!(a["outputs"]["y"]["data"], serde_json::json!([1.0, 2.0]));
    assert_eq!(b["outputs"]["y"]["data"], serde_json::json!([768.0, 896.0]));
}

#[tokio::test]
async fn busy_server_answers_503() {
    let pool = NpuPool::simulated(1).unwrap();
    // No permits at all: every inference request finds the server busy.
    let mut state = AppState::new(
        &pool,
        ServerConfig {
            max_concurrent_requests: 0,
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
    let app = router(Arc::new(state));
    let (s, _) = call(
        &app,
        "POST",
        "/v1/models/mlp/infer",
        Some(serde_json::json!({ "shape": [1, 4], "data": [0, 0, 0, 0] })),
    )
    .await;
    assert_eq!(s, StatusCode::SERVICE_UNAVAILABLE);
    // Other routes are not limited.
    assert_eq!(call(&app, "GET", "/health", None).await.0, StatusCode::OK);
}
