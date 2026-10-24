# qanten-serve

HTTP inference server for Q.ANT photonic NPUs, built on [`qanten`](https://crates.io/crates/qanten):
ONNX and MLP weight files, dynamic batching, multiple NPUs, Prometheus metrics.

```sh
cargo install qanten-serve
qanten-serve -m classifier=model.onnx            # all NPUs the driver sees
qanten-serve --sim --sim-npus 2 -m classifier=model.onnx
curl -s localhost:8080/v1/models
```

See [docs/serve.md](https://github.com/aprilthepink/qanten/blob/main/docs/serve.md) for the API.

Unofficial community project, not affiliated with Q.ANT GmbH.
