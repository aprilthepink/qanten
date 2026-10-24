# Device memory

[`DeviceBuffer<R>`] is a `rows × cols` block of `bf16` values in NPU memory, freed
when dropped. The type parameter is its **role**:

| Role | Used as | Upload | Download |
|---|---|---|---|
| `Weights` | the matrix of a matrix-vector product | yes | **no** (the hardware can't read it back) |
| `Features` | input vectors and results | yes | yes |

Because the role is part of the type, mixing them up doesn't compile:

<!-- doccheck: compile_fail -->
```rust
let npu = qanten::Npu::simulator();
let w = qanten::DeviceBuffer::<qanten::Weights>::alloc(&npu, 2, 2).unwrap();
w.download(); // error: no method `download` for DeviceBuffer<Weights>
```

## Keeping data on the NPU

```rust
use qanten::{DeviceBuffer, Features, Npu, Tensor, Weights, matvec_resident};

fn main() -> qanten::Result<()> {
    let npu = Npu::simulator();
    let w = DeviceBuffer::<Weights>::from_tensor(&npu, &Tensor::rand_uniform([256, 512], -1., 1., 1))?;
    let x = DeviceBuffer::<Features>::alloc(&npu, 32, 512)?;      // 32 input vectors
    let mut y = DeviceBuffer::<Features>::alloc(&npu, 32, 256)?;  // 32 results

    for step in 0..3 {
        x.upload(&Tensor::rand_uniform([32, 512], -1., 1., step))?;
        matvec_resident(&w, &x, &mut y)?;   // y = x · wᵀ, all on the device
        let out = y.download()?;
        assert_eq!(out.shape(), &[32, 256]);
    }
    println!("weights use {} KiB of device memory", w.size_bytes() / 1024);
    Ok(())
}
```

`matvec_resident` checks shapes (`weights [out, in]`, `vectors [batch, in]`,
`result [batch, out]`) and that all three live on the same NPU (`Error::DeviceMismatch`
otherwise) before anything reaches the driver.

## When to use it

Most code doesn't need `DeviceBuffer` directly: `ops::LinearWeights`, `ops::ConvWeights`
and the `nn` layers already keep their weights resident. Reach for it when you want
control over exactly what stays on the device, e.g. streaming many batches through one
matrix (see the `device_memory` example) or building your own resident ops, as
`qanten-graphics` does.

## Safety

The driver reads and writes raw pointers with lengths derived from the allocation
size. qanten's driver backend records every allocation and checks every slice length
and handle before calling into the driver. Double frees and use-after-free return
errors instead of corrupting memory.

[`DeviceBuffer<R>`]: https://docs.rs/qanten/latest/qanten/struct.DeviceBuffer.html
