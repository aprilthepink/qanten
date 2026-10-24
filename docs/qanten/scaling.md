# Multiple NPUs and CPU fallback

## Several NPUs: `NpuPool`

A pool holds one handle per device. `replicate` builds a copy of your model on each
NPU (every copy keeps its own weights resident) and returns a `Replicated<M>`, which is
itself a `Module`: `forward` splits the batch evenly, runs the parts in parallel and
concatenates the results in order.

```rust
use qanten::nn::{self, Module};
use qanten::pool::NpuPool;
use qanten::{Npu, Tensor};

fn build(npu: &Npu) -> qanten::Result<nn::Sequential> {
    Ok(nn::Sequential::new()
        .add(nn::Linear::new(npu, &Tensor::rand_uniform([128, 64], -0.1, 0.1, 1), None)?)
        .add(nn::ReLU)
        .add(nn::Linear::new(npu, &Tensor::rand_uniform([10, 128], -0.1, 0.1, 2), None)?))
}

fn main() -> qanten::Result<()> {
    // Every card the driver sees; falls back to 2 simulated NPUs without hardware.
    let pool = NpuPool::open_all().or_else(|_| NpuPool::simulated(2))?;
    let model = pool.replicate(build)?;
    let y = model.forward(&Tensor::rand_uniform([256, 64], 0., 1., 3))?;
    assert_eq!(y.shape(), &[256, 10]);
    println!("{} NPUs", pool.len());
    Ok(())
}
```

For one-off work there is `pool.split_batch(&x, |index, npu, part| ...)`, e.g.
`|_, npu, part| ops::linear(npu, part, &w, None)`.

| Constructor | Pool of |
|---|---|
| `NpuPool::open_all()` | every NPU `qanten::devices()` lists |
| `NpuPool::new(vec_of_npus)` | your own handles (e.g. profiled ones) |
| `NpuPool::simulated(n)` | `n` NPUs on one simulator |

Batches are split evenly (sizes differ by at most one row); batches smaller than the
pool use fewer NPUs. `Tensor::split_batch` and `Tensor::concat_batch` are public if you
want to do the same by hand.

## Small jobs on the CPU: `with_cpu_fallback`

Each NPU call has a fixed overhead, so tiny operations finish faster on the host.
A fallback handle sends streamed `mul`, `periodic_nl` and `matvec` calls below a
threshold to an exact CPU implementation and everything else to the NPU:

```rust
use qanten::fallback::FallbackPolicy;
use qanten::{Npu, Tensor, ops};

fn main() -> qanten::Result<()> {
    let npu = Npu::auto(0)?;
    let policy = if npu.is_simulated() {
        FallbackPolicy::min_macs(1 << 16)     // fixed threshold
    } else {
        FallbackPolicy::calibrate(&npu)?      // measure the crossover on this card
    };
    let (npu, monitor) = npu.with_cpu_fallback(policy);
    ops::mul(&npu, &Tensor::ones([16]), &Tensor::ones([16]))?;
    println!("{:?}", monitor.stats()); // FallbackStats { cpu_ops: 1, npu_ops: 0 }
    Ok(())
}
```

- The threshold counts multiply-accumulates: `rows × cols × batch` for `matvec`,
  elements for elementwise ops.
- `FallbackPolicy::calibrate` times batched matrix-vector products of growing size on
  both sides and picks the first size where the NPU wins. It only means something on
  real hardware.
- Work on device-resident data (`DeviceBuffer`, `LinearWeights`, `nn` layers) always
  runs on the NPU, since the data already lives there.
- CPU results match the simulator exactly (f32 accumulation, bf16 output), so models
  behave the same whichever side ran an op.

## Combining wrappers

`profiled()` and `with_cpu_fallback()` both return a new handle to the same device and
stack in either order, and pools accept any handles:

```rust
use qanten::fallback::FallbackPolicy;
use qanten::pool::NpuPool;
use qanten::Npu;

fn main() -> qanten::Result<()> {
    let (profiled, profiler) = Npu::simulator().profiled();
    let (fast, _) = profiled.with_cpu_fallback(FallbackPolicy::min_macs(4096));
    let pool = NpuPool::new(vec![fast])?;
    assert_eq!(pool.len(), 1);
    println!("{}", profiler.report());
    Ok(())
}
```
