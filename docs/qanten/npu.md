# NPUs and backends

An [`Npu`] is a handle to one initialised device. Every operation takes `&Npu`.

## Opening an NPU

```rust
use qanten::Npu;

fn main() -> qanten::Result<()> {
    let hw = Npu::auto(0)?;        // hardware if the driver loads, else simulator
    let sim = Npu::simulator();    // always the simulator
    println!("{} / {}", hw.backend().name(), sim.backend().name());
    if hw.is_simulated() {
        println!("no Q.ANT driver found, using the simulator");
    }
    Ok(())
}
```

| Constructor | What it does |
|---|---|
| `Npu::open(id)` | Loads Q.ANT's driver and initialises NPU `id`. Errors with `Error::DriverUnavailable` if the driver isn't installed. |
| `Npu::auto(id)` | `open(id)`, falling back to the simulator (with a `tracing` warning) if the driver is missing. |
| `Npu::from_env(id)` | Reads `QANTEN_BACKEND`: `driver`/`hw` → `open`, `sim` → `simulator`, unset/`auto` → `auto`. |
| `Npu::simulator()` | NPU 0 of a shared, exact simulator. |
| `Npu::simulator_with(SimConfig)` | NPU 0 of a fresh simulator with custom settings. |
| `Npu::simulated(&Arc<Simulator>, id)` | NPU `id` of a simulator you share between handles (several simulated NPUs). |
| `Npu::with_backend(Arc<dyn Backend>, id)` | NPU `id` on your own [`Backend`](#custom-backends). |

### Lifetime

- Opening initialises the device (`qant_driver_init_npu`). Dropping the **last**
  handle releases it (`qant_driver_release_npu`).
- `Npu` is cheap to `clone` (it's an `Arc`) and is `Send + Sync`: share it between threads.
- `Npu::open(0)` twice returns the same underlying handle, so the device is only
  initialised once.
- Device memory and layers keep their `Npu` alive, so the device can't be released
  while something still uses it.

## Where the driver is found

`Npu::open` loads `libqant_native_computing_driver.so` from, in order:

1. the path in `QANTEN_DRIVER_LIB`;
2. the system loader path (`LD_LIBRARY_PATH`, `/etc/ld.so.conf`);
3. `/usr/lib`, `/usr/local/lib`, `/usr/lib/x86_64-linux-gnu`, `/usr/lib/aarch64-linux-gnu`.

If nothing loads, the error lists every path tried and why it failed. `qanten doctor`
prints the same information.

```rust
fn main() {
    println!("driver available: {}", qanten::driver_available());
    match qanten::devices() {
        Ok(list) => list.iter().for_each(|d| println!("NPU {} serial {}", d.id, d.serial)),
        Err(e) => println!("no driver: {e}"),
    }
}
```

## The simulator

The simulator runs every NPU primitive on the CPU (in parallel) and is exact by
default. It is a port of the CPU backend in Q.ANT's toolkit, with extras for testing:

```rust
use qanten::{Npu, SimConfig};

fn main() {
    let noisy = Npu::simulator_with(SimConfig {
        noise_std: 0.05,                  // Gaussian noise added to every NPU output
        memory_limit_bytes: Some(1 << 30), // 1 GiB of simulated device memory per NPU
        num_npus: 1,
        seed: 7,
    });
    assert!(noisy.is_simulated());
}
```

| `SimConfig` field | Default | Meaning |
|---|---|---|
| `num_npus` | 1 | Simulated NPU ids `0..num_npus` |
| `noise_std` | 0.0 | Std-dev of noise added to each output value. Not a calibrated hardware model; use it to test robustness. |
| `seed` | fixed | Seed of the noise generator |
| `memory_limit_bytes` | `None` | Per-NPU allocation limit; exceeding it returns an error like real hardware would |

Several NPUs on one simulator:

```rust
use std::sync::Arc;
use qanten::backend::Simulator;
use qanten::{Npu, SimConfig};

fn main() -> qanten::Result<()> {
    let sim = Arc::new(Simulator::new(SimConfig { num_npus: 2, ..SimConfig::default() }));
    let a = Npu::simulated(&sim, 0)?;
    let b = Npu::simulated(&sim, 1)?;
    assert!(!a.same_device(&b));
    println!("live device allocations: {}", sim.live_allocations());
    Ok(())
}
```

## Device information and performance counters

```rust
fn main() -> qanten::Result<()> {
    let npu = qanten::Npu::auto(0)?;
    let info = npu.device_info()?;
    println!("board {} firmware {}", info.version.board_serial, info.version.firmware);
    for (name, value, unit) in info.sensors.rows() {
        println!("{name:<14} {value:8.3} {unit}");
    }
    println!("12 V power: {:.1} W", info.sensors.power_12v_watts());

    npu.reset_perf_counters()?;
    // ... run work ...
    let perf = npu.perf_counters()?;
    println!("busy {:.1} % of {:?}", 100.0 * perf.utilization(), perf.elapsed);
    Ok(())
}
```

- `DeviceInfo { sensors: SensorReadings, version: VersionInfo }`: 5 temperatures (°C),
  11 voltage rails (V) and the 12 V current (A); firmware, Zephyr RTOS version, serial.
- `PerfCounters { elapsed, stalled }`: wall time since the last reset and the idle part
  of it. `utilization()` = `1 − stalled / elapsed`.
- `npu.driver_version()`: the driver's version string.

With the `serde` feature, all of these implement `Serialize`.

### Driver logs

<!-- doccheck: no_run -->
```rust
// Needs the `tracing` crate for the level type.
fn main() -> qanten::Result<()> {
    qanten::setup_driver_logging("/var/log/qant", tracing::Level::INFO)?;
    Ok(())
}
```

## Custom backends

Everything in qanten is built on the [`Backend`] trait: the handful of primitives the
Q.ANT driver offers. Implement it to plug in a mock, a remote NPU, or a recorder:

| Method | Primitive |
|---|---|
| `mul`, `periodic_nl` | elementwise `a·b` and `tcos(u)·v` |
| `matvec_batched` | `out[b, r] = Σ_c matrix[r, c] · vectors[b, c]` |
| `allocate`, `free`, `upload`, `download`, `matvec_resident` | device memory |
| `init_npu`, `release_npu`, `available_npus` | lifecycle |
| `driver_version`, `device_info`, `perf_counters`, `reset_perf_counters` | telemetry |

Implementations must be thread safe and must check slice lengths themselves.
Wrap one with `Npu::with_backend(Arc::new(my_backend), 0)`.

[`Npu`]: https://docs.rs/qanten/latest/qanten/struct.Npu.html
[`Backend`]: https://docs.rs/qanten/latest/qanten/backend/trait.Backend.html
