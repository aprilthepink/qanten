# qant-sys

Raw FFI bindings to the Q.ANT native computing driver (`libqant_native_computing_driver`).

The driver is loaded **at runtime**, so this crate builds without the driver installed.
Set `QANT_DRIVER_LIB=/path/to/libqant_native_computing_driver.so` to point at a
non-standard install location.

You almost certainly want the safe high-level crate [`qant`](https://crates.io/crates/qant) instead.

Unofficial community project, not affiliated with Q.ANT GmbH.
