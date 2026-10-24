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
| `qanten bench [-n ID] [--rows R --cols C --batch B --iters I]` | matrix-vector throughput (GMAC/s), weights uploaded per call vs resident |
| `qanten selftest [-n ID] [--tolerance T]` | runs every op on the NPU and compares with the simulator; exit code 1 on failure |

Global flags:

| Flag | Effect |
|---|---|
| `--sim` (or `QANTEN_SIM=1`) | use the built-in simulator instead of hardware |
| `--json` | machine-readable output |
| `-v`, `-vv`, `-vvv` | more log output (also `RUST_LOG`) |

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
