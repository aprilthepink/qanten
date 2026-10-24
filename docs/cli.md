# The `qanten` command

```sh
cargo install qanten-cli        # installs the `qanten` binary
```

| Command | Shows |
|---|---|
| `qanten doctor` | Is the driver installed, where it was found, driver package version, device configs, visible NPUs. Explains what's missing. |
| `qanten devices` | NPU ids and serial numbers |
| `qanten info [-n ID]` | backend, driver version, firmware, Zephyr version, board serial, 12 V power |
| `qanten sensors [-n ID] [--watch SECS]` | all temperatures, voltages and the 12 V current; refreshes with `--watch` |
| `qanten perf [-n ID] [--reset]` | elapsed and stalled time, utilisation |
| `qanten bench [-n ID] [--rows R --cols C --batch B --iters I] [--profile]` | matrix-vector throughput (GMAC/s), weights uploaded per call vs resident; `--profile` adds a per-op table |
| `qanten run MODEL [-i name=file.npy]... [-o DIR] [--iters N] [--profile]` | run an `.onnx` model (or `.safetensors` / `.npz` with `--arch mlp`) on the NPU; random inputs if none given; outputs saved as `.npy` |
| `qanten inspect FILE` | tensors in a `.safetensors` / `.npz` / `.npy` file, or inputs, outputs and ops of an `.onnx` model |
| `qanten selftest [-n ID] [--tolerance T]` | runs every op on the NPU and compares with the simulator; exit code 1 on failure |

Global flags:

| Flag | Effect |
|---|---|
| `--sim` (or `QANTEN_SIM=1`) | use the built-in simulator instead of hardware |
| `--json` | machine-readable output |
| `-v`, `-vv`, `-vvv` | more log output (also `RUST_LOG`) |

## Running models

```sh
qanten run model.onnx                           # random input from the model's declared shape
qanten run model.onnx -i input=image.npy -o out # named input, outputs written to out/<name>.npy
qanten run model.onnx --iters 100 --profile     # average time + per-op profile
qanten run weights.npz --arch mlp --activation tanh   # Linear layers + activation
qanten inspect weights.safetensors
```

`--arch mlp` treats every 2-D tensor as a Linear layer `[out, in]`, ordered by the
number in its name (`0.weight`, `fc1.weight`, `l1_w`, …), with the matching bias if
present; it works on PyTorch `nn.Sequential` state dicts, qanten's own training exports,
and Q.ANT's example weights:

```text
$ qanten run qant_digit_weights.npz --arch mlp --batch 4
  layer 0: l1_w [133, 784] + bias
  layer 1: l2_w [10, 133] + bias
loaded 2-layer MLP from qant_digit_weights.npz
  no input given: using random data of shape [4, 784]
ran in 5.663 ms per run (3 runs)
  output: shape [4, 10], min -6.2812, max 4.1875, mean 0.0714
```

## Example session

Against the mock driver (see [Testing](testing.md)), so the NPUs are `MOCK-*` and the
self-test errors are zero; on a real card expect small non-zero errors:

```text
$ qanten doctor
[ ok ] QANTEN_DRIVER_LIB ~/qanten/target/debug/libqant_native_computing_driver.so
[ ok ] driver library   ~/qanten/target/debug/libqant_native_computing_driver.so
[warn] driver package   qant_native_computing_driver not installed via dpkg
[warn] device configs   no *.toml in /etc/qant-native-computing-driver
[ ok ] NPUs             0 (MOCK-0000), 1 (MOCK-0001)

driver found. Try `qanten selftest`.

$ qanten info
NPU            0
backend        qanten driver (~/qanten/target/debug/libqant_native_computing_driver.so)
driver         mock-driver;qanten-simulator;version:0.1.0;commit:none;
firmware       mock-fw-1.0
zephyr         mock-zephyr
board serial   MOCK-0000
12V power      90.0 W

$ qanten selftest
backend: qanten driver (~/qanten/target/debug/libqant_native_computing_driver.so)  (tolerance 0.05)

mul                    0.00000  ok
periodic_nl            0.00000  ok
linear                 0.00000  ok
linear (resident)      0.00000  ok
conv2d                 0.00000  ok
conv_transpose2d       0.00000  ok
batch_norm2d           0.00000  ok
kan_layer              0.00000  ok

all checks passed
```

`selftest` uses an error measure relative to the largest reference value, so its
tolerance reads as a fraction (0.05 = 5 %).
