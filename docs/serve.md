# Inference server (`qanten-serve`)

An HTTP server that serves models on Q.ANT NPUs with dynamic batching, multiple NPUs
and Prometheus metrics.

```sh
cargo install qanten-serve
qanten-serve -m digits=qant_digit_weights.npz -m classifier=model.onnx
# no card? simulate two NPUs:
qanten-serve --sim --sim-npus 2 -m classifier=model.onnx
```

| Option | Default | |
|---|---|---|
| `-m, --model name=path` | (required, repeatable) | `.onnx`, or `.safetensors` / `.npz` MLP weights |
| `--addr` | `127.0.0.1:8080` | also `QANTEN_SERVE_ADDR` |
| `--max-batch` | 64 | maximum rows per NPU batch |
| `--max-wait-ms` | 2 | how long to wait for more requests to batch |
| `--activation` | `relu` | between layers of MLP weight files |
| `--sim`, `--sim-npus N` | off, 1 | simulated NPUs instead of hardware |

Without `--sim` it opens every NPU the driver reports.

## API

| Method | Path | |
|---|---|---|
| GET | `/health` | `{"status":"ok","backend":…,"npus":2,"models":2,"uptime_s":…}` |
| GET | `/v1/models` | name, kind, inputs (name, shape, integer), outputs, whether batched |
| POST | `/v1/models/{name}/infer` | run a model |
| GET | `/metrics` | Prometheus text format |

Request, either form:

```json
{"shape": [2, 8], "data": [0.1, 0.2, ...]}
{"inputs": {"ids": {"shape": [1, 3], "data": [1, 4, 2]}}}
```

Response:

```json
{"model": "mlp",
 "outputs": {"probs": {"shape": [2, 4], "data": [0.0047, 0.9609, ...]}},
 "batch_rows": 2,
 "latency_ms": 4.05}
```

`batch_rows` is the size of the NPU batch the request ended up in. When it's larger
than your own rows, the request was merged with others. Errors return 4xx with
`{"error": "..."}` (unknown model 404, wrong shape or bad values 400).

```sh
curl -s -X POST localhost:8080/v1/models/mlp/infer -H 'content-type: application/json' \
     -d '{"shape":[2,8],"data":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1,1,1,1,0,0,0,0]}'
```

## How requests are scheduled

- **Replicas.** Every model is loaded once per NPU, each copy with its weights resident
  on its own card.
- **Dynamic batching.** For models with a single float input, a batcher collects
  concurrent requests for up to `--max-wait-ms` or `--max-batch` rows, concatenates them
  along the batch dimension, runs one NPU job, and splits the outputs back per request.
  Requests whose shapes don't match the batch run on their own. Models with several
  inputs or integer inputs (e.g. token ids) skip batching.
- **Load balancing.** Each batch is a job on an `NpuExecutor`, which sends it to the NPU
  with the fewest jobs queued or running. Batches for the same model run on different
  cards at the same time.

## Metrics

| Metric | Labels |
|---|---|
| `qanten_requests_total`, `qanten_request_errors_total`, `qanten_request_latency_seconds_sum` | `model` |
| `qanten_batches_total`, `qanten_batched_rows_total` (rows ÷ batches = average batch size) | `model` |
| `qanten_npu_macs_total`, `qanten_npu_busy_seconds_total` | `npu` |
| `qanten_npu_bytes_to_device_total`, `qanten_npu_bytes_from_device_total` | `npu` |
| `qanten_npu_resident_matvec_calls_total`, `qanten_npu_device_memory_bytes` | `npu` |
| `qanten_uptime_seconds` | |

## Embedding the server

The crate is also a library: build an `AppState`, add models (`add_model_file`, or
`add_module` for any `qanten::nn::Module` built per NPU), and mount `router(state)` in
your own axum application.

<!-- doccheck: ignore -->
```rust
let pool = NpuPool::open_all()?;
let mut state = AppState::new(&pool, ServerConfig::default())?;
state.add_model_file("classifier", "model.onnx".as_ref(), ActivationKind::Relu)?;
state.add_module("tiny", 16, |npu| nn::Linear::new(npu, &weights, None))?;
let app = axum::Router::new().nest("/npu", router(Arc::new(state)));
```
