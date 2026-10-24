# Errors

Every fallible call returns `qanten::Result<T>`, i.e. `Result<T, qanten::Error>`.
`Error` implements `std::error::Error`, so it works with `?`, `anyhow` and `thiserror`.
It is `#[non_exhaustive]`: match with a `_` arm.

| Variant | When |
|---|---|
| `Driver { npu, code, message }` | The driver (or simulator) rejected a call: bad handle, out of device memory, unknown NPU id, hardware fault. |
| `DriverUnavailable(msg)` | The driver library couldn't be loaded. The message lists every path tried. |
| `DriverFeatureDisabled` | Hardware was requested but the crate was built without the `driver` feature. |
| `Shape(msg)` | Tensor shapes don't fit the operation. The message names both shapes. |
| `InvalidArgument(msg)` | A parameter is out of range (zero stride, too-large output padding, …). |
| `Unsupported(msg)` | Valid but unsupported request, e.g. downloading weights through the raw backend. |
| `DeviceMismatch(msg)` | Operands live on different NPUs. |
| `MissingTensor(name)` | A required tensor isn't in a `StateDict`. |
| `Format(msg)` | A weights file couldn't be parsed. |
| `Io(err)` | File system error. |

## Typical handling

```rust
use qanten::{Error, Npu};

fn open() -> qanten::Result<Npu> {
    match Npu::open(0) {
        Ok(npu) => Ok(npu),
        Err(Error::DriverUnavailable(why)) => {
            eprintln!("no Q.ANT driver ({why}); falling back to the simulator");
            Ok(Npu::simulator())
        }
        Err(e) => Err(e),
    }
}

fn main() -> qanten::Result<()> {
    let npu = open()?;
    println!("{}", npu.backend().name());
    Ok(())
}
```

That is exactly what `Npu::auto` does for you.
