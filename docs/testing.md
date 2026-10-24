# Testing without hardware

## Layers of testing

| Layer | What it checks | How |
|---|---|---|
| Unit and op tests | every op against a plain f32 reference, on the simulator | `cargo test --workspace --all-features` |
| Graphics accuracy tests | each NPU formulation against its CPU path | included in the command above |
| FFI path | the real `dlopen` + C ABI, through a mock driver | see below |
| Hardware | the actual card | `qanten selftest`, examples |

## The mock driver

`crates/qanten-mock-driver` builds a `libqant_native_computing_driver.so` that exports
the same 16 C functions as Q.ANT's driver and forwards them to the simulator (with two
simulated NPUs). It lets CI exercise exactly the code path real hardware uses:

```sh
cargo build -p qanten-mock-driver
export QANTEN_DRIVER_LIB=$PWD/target/debug/libqant_native_computing_driver.so
cargo test -p qanten --test driver_ffi   # fails, not skips, if the library won't load
qanten doctor
qanten selftest
```

`QANTEN_MOCK_NOISE=0.05` adds simulated noise inside the mock.

## Q.ANT's simulated driver

Q.ANT's CI uses a separate *simulated* build of their real driver (a `.deb` from their
driver repository) plus fake device configs in `/etc/qant-native-computing-driver/`.
It exports the same interface, so qanten should load it like real hardware:

```sh
sudo apt install ./qant*.deb
sudo mkdir -p /etc/qant-native-computing-driver
sudo cp dep/configs/*.toml /etc/qant-native-computing-driver/   # from Q.ANT's toolkit repo
qanten doctor && qanten selftest
```

This hasn't been tried yet: the package isn't publicly downloadable.

## Checklist for a real card

1. `qanten doctor`: driver found, NPUs listed.
2. `qanten selftest`: all ops within tolerance.
3. `QANTEN_DRIVER_LIB` unset: `cargo test -p qanten --test driver_ffi`.
4. `qanten bench`: compare "weights resident" with "uploaded per call".
5. `cargo run --release -p qanten --example digit_recognition --features safetensors`: expect about 94.6 %.
6. The `qanten-graphics` examples, to see where the NPU beats the CPU.
