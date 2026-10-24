# Profiling

`npu.profiled()` returns a second handle to the same device that records every
primitive call it makes, plus a `Profiler` to read the numbers.

```rust
use qanten::nn::{self, Module};
use qanten::profile::OpKind;
use qanten::{Npu, Tensor};

fn main() -> qanten::Result<()> {
    let (npu, profiler) = Npu::auto(0)?.profiled();
    let model = nn::Sequential::new()
        .add(nn::Linear::new(&npu, &Tensor::rand_uniform([256, 784], -0.1, 0.1, 1), None)?)
        .add(nn::ReLU)
        .add(nn::Linear::new(&npu, &Tensor::rand_uniform([10, 256], -0.1, 0.1, 2), None)?);

    profiler.reset(); // ignore the weight uploads above
    for _ in 0..10 {
        model.forward(&Tensor::rand_uniform([64, 784], 0., 1., 3))?;
    }
    let report = profiler.report();
    println!("{report}");
    println!("NPU work: {} MACs", report.op(OpKind::MatVecResident).macs);
    Ok(())
}
```

## What is recorded

Per operation kind (`mul`, `periodic_nl`, `matvec`, `matvec_resident`, `upload`,
`download`, `allocate`, `free`):

| Field | Meaning |
|---|---|
| `calls`, `errors` | how often it ran, how often it failed |
| `time` | time spent inside the backend |
| `bytes_to_device`, `bytes_from_device` | data moved between host and NPU |
| `macs` | multiply-accumulates (elementwise ops count one per element) |

Plus, for the whole handle: wall time since creation or `reset()`, the share of it
spent in the backend, and live / peak device memory.

`report.total()` sums all kinds, `report.op(kind)` reads one, and `OpStats::gmacs_per_sec()`
gives throughput. With the `serde` feature, `Report` implements `Serialize`.

## Example output

`qanten --sim bench --rows 256 --cols 256 --batch 32 --iters 5 --profile`:

```text
qanten profile: qanten simulator (213.8 ms wall)
op                  calls     time ms      to NPU    from NPU         MACs    GMAC/s
matvec                  6     103.439   864.0 KiB    96.0 KiB     12582912      0.12
matvec_resident         6     105.195         0 B         0 B     12582912      0.12
upload                  7       0.016   224.0 KiB         0 B            0      0.00
download                6       0.022         0 B    96.0 KiB            0      0.00
allocate               13       0.812         0 B         0 B            0      0.00
free                   13       0.084         0 B         0 B            0      0.00
total                  51     209.567     1.1 MiB   192.0 KiB     25165824      0.12
backend share of wall time 98.0 %, device memory live 0 B / peak 160.0 KiB
```

The same work streamed (`matvec`) moves 864 KiB to the NPU; with resident weights
(`matvec_resident`) only the inputs travel (the `upload` row). On real hardware this
table shows where time goes: transfers, NPU compute, or host-side work (low backend
share).

## Notes

- Use the profiled handle for everything you want measured. Buffers and layers created
  through it belong to it; mixing them with ones from the original handle returns
  `Error::DeviceMismatch`.
- Dropping the profiled handle does not release the device; the original handle owns it.
- Every call is also a `tracing` event at TRACE level with target `qanten::profile`,
  so `RUST_LOG=qanten::profile=trace` logs each one.
- Overhead is one mutex lock and a timestamp per call.
