# Async and background execution

NPU calls block the calling thread. `NpuExecutor` runs them on dedicated worker
threads and returns a `JobHandle`, which is a standard `Future`, so it works in
tokio, async-std, smol or any other runtime. qanten itself doesn't depend on any of them.

```rust
use std::sync::Arc;
use qanten::executor::NpuExecutor;
use qanten::nn::{self, Module};
use qanten::{Npu, Tensor};

fn main() -> qanten::Result<()> {
    let npu = Npu::auto(0)?;
    let model: Arc<dyn Module> = Arc::new(
        nn::Sequential::new()
            .add(nn::Linear::new(&npu, &Tensor::rand_uniform([16, 8], -0.5, 0.5, 1), None)?)
            .add(nn::Softmax),
    );
    let exec = NpuExecutor::new(npu);

    // Fire off work, keep doing other things, collect later.
    let job = exec.forward(&model, Tensor::ones([4, 8]));
    // ... other work on this thread ...
    let probs = job.wait()?;          // or `job.await?` inside async code
    assert_eq!(probs.shape(), &[4, 16]);
    Ok(())
}
```

With tokio:

<!-- doccheck: ignore -->
```rust
#[tokio::main]
async fn main() -> qanten::Result<()> {
    let exec = NpuExecutor::new(Npu::auto(0)?);
    let (a, b) = tokio::join!(
        exec.run(|npu| ops::mul(npu, &x, &x)),
        exec.forward(&model, batch),
    );
    Ok(())
}
```

## API

| Item | Purpose |
|---|---|
| `NpuExecutor::new(npu)` | one worker thread for one NPU |
| `NpuExecutor::for_pool(&pool)` | one worker per NPU of an [`NpuPool`](scaling.md); each job goes to the worker with the fewest jobs queued or running |
| `NpuExecutor::with_workers(npu, n)` | `n` workers sharing one NPU (overlap host-side work with NPU calls) |
| `exec.run(\|npu\| ...)` | run any closure on a worker with that worker's NPU → `JobHandle<T>` |
| `exec.forward(&Arc<model>, x)` | run `model.forward(x)` in the background |
| `JobHandle::wait()` / `.await` / `is_done()` | collect the result blocking / async / poll |

## Behaviour

- **Errors** from the job come back through the handle unchanged.
- **Panics** inside a job are caught and returned as an error; the worker keeps running.
- **Dropping** the executor finishes every queued job, then stops the workers, so
  handles you still hold always complete.
- **Pools**: with `for_pool`, a job runs on the least busy NPU, so build models
  that live on *every* NPU (e.g. `pool.replicate(...)`) or pick the NPU inside the
  closure via the `npu` argument. A model built on NPU 0 and run by the worker of
  NPU 1 still works but keeps using NPU 0.
