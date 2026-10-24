# qant-cli

Command-line tool for Q.ANT photonic NPUs, built on the [`qant`](https://crates.io/crates/qant) crate.

```sh
cargo install qant-cli

qant doctor            # is the driver installed? which NPUs are visible?
qant devices           # list NPUs
qant info -n 0         # driver, firmware, board serial, power draw
qant sensors --watch 1 # live temperatures / voltages
qant perf --reset      # utilisation counters
qant bench             # matvec throughput, streamed vs resident weights
qant selftest          # run every op on the NPU and compare with the simulator
```

Add `--sim` to any command to run against the built-in simulator, and `--json`
for machine-readable output.

Unofficial community project, not affiliated with Q.ANT GmbH.
