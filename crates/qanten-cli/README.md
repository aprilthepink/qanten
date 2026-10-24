# qanten-cli

Command-line tool for Q.ANT photonic NPUs, built on the [`qanten`](https://crates.io/crates/qanten) crate.

```sh
cargo install qanten-cli

qanten doctor            # is the driver installed? which NPUs are visible?
qanten devices           # list NPUs
qanten info -n 0         # driver, firmware, board serial, power draw
qanten sensors --watch 1 # live temperatures / voltages
qanten perf --reset      # utilisation counters
qanten bench             # matvec throughput, streamed vs resident weights
qanten selftest          # run every op on the NPU and compare with the simulator
```

Add `--sim` to any command to run against the built-in simulator, and `--json`
for machine-readable output.

Unofficial community project, not affiliated with Q.ANT GmbH.
