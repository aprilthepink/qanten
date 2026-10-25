# qanten-sys

Raw FFI bindings to Q.ANT's driver library, `libqant_native_computing_driver.so`.
Most code should use [`qanten`](qanten/npu.md) instead; it wraps all of this safely.

## What's in it

- The C structs, laid out exactly like the driver's: `QantDriverError`, `SensorInfo`,
  `QantVersionInfo`, `QantDeviceInfo`, `QantDataRole`, `QantDriverHandle`,
  `PerformanceCounterInfo`.
- A function-pointer type for each of the 16 driver entry points (`InitNpuFn`,
  `MulMatrixVecBatchedFn`, …), named after the C symbol `qant_driver_*` they load.
- `DriverLibrary`: the loaded library plus all 16 resolved function pointers.
- `c_buf_to_string`: reads the driver's fixed-size, NUL-terminated string buffers.

The layouts and signatures mirror the ones in Q.ANT's toolkit v2.4.0.

## Loading at run time

The crate never links against the driver when building. `DriverLibrary::load()` opens
it when called:

1. `QANTEN_DRIVER_LIB` (if set),
2. `DEFAULT_LIBRARY_NAMES` via the system loader,
3. `DEFAULT_LIBRARY_PATHS` (`/usr/lib`, `/usr/local/lib`, multiarch dirs).

<!-- doccheck: no_run -->
```rust
use qanten_sys::{DriverLibrary, QantDriverError};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // SAFETY: we trust the library at the standard location to be Q.ANT's driver.
    let lib = unsafe { DriverLibrary::load()? };
    println!("loaded {:?}", lib.path());
    // SAFETY: plain value call.
    let status: QantDriverError = unsafe { lib.init_npu(0) };
    if !status.is_ok() {
        eprintln!("init failed: {} ({})", status.message(), status.code);
    }
    Ok(())
}
```

`LoadError::NotFound` lists every candidate path with the loader's reason;
`LoadError::MissingSymbol` names a missing function (usually an incompatible driver
version).

## Safety

Every function pointer is `unsafe extern "C"`. The driver trusts the caller completely:
it derives buffer lengths from its arguments or from the allocation, and reads or
writes raw pointers accordingly. Wrong lengths or stale handles mean memory corruption.
`qanten`'s `DriverBackend` exists to make this safe: it tracks every allocation and
checks all lengths and handles before calling in.
