# Tensors

[`Tensor`] is an owned, contiguous, row-major array of `bf16` values with any number
of dimensions. `bf16` is the NPU's native format; constructors and accessors convert
from and to `f32` for you.

## Creating

```rust
use qanten::Tensor;

fn main() -> qanten::Result<()> {
    let a = Tensor::from_f32(&[1., 2., 3., 4., 5., 6.], [2, 3])?;  // checks the length
    let v = Tensor::vector(&[0.5, 1.5]);                             // 1-D
    let z = Tensor::zeros([4, 4]);
    let o = Tensor::ones([3]);
    let f = Tensor::full([2, 2], 7.0);
    let r = Tensor::rand_uniform([8, 16], -1.0, 1.0, 42);            // reproducible
    let i = Tensor::from_slice(&[1u8, 2, 3], [3])?;                  // any type → f64 → bf16
    let raw = Tensor::new(vec![qanten::bf16::ONE; 4], [2, 2])?;      // raw bf16
    println!("{a:?} {v:?} {z:?} {o:?} {f:?} {r:?} {i:?} {raw:?}");
    Ok(())
}
```

## Inspecting and converting

| Method | Returns |
|---|---|
| `shape()`, `rank()`, `len()`, `is_empty()` | dimensions, number of dimensions, element count |
| `dims::<D>()` | the shape as `[usize; D]`, or `Error::Shape` if the rank differs |
| `get(&[i, j, ..])` | one element as `f32` (`None` if out of range) |
| `to_f32_vec()` | all data as `f32` |
| `data()`, `data_mut()`, `into_data()`, `into_parts()` | raw `bf16` access |
| `argmax_last()` | index of the maximum along the last dimension, per row |
| `max_abs_diff(&other)` | largest elementwise difference (useful in tests) |

## Reshaping

```rust
use qanten::Tensor;

fn main() -> qanten::Result<()> {
    let t = Tensor::zeros([2, 3, 4]);
    let a = t.reshaped([6, usize::MAX])?;      // usize::MAX infers a dimension (like -1)
    assert_eq!(a.shape(), &[6, 4]);
    let b = t.clone().flatten_from(1)?;        // [2, 12], like torch.flatten(x, 1)
    let c = b.unsqueeze0();                    // [1, 2, 12]
    assert_eq!(c.shape(), &[1, 2, 12]);
    let doubled = t.map(|x| 2.0 * x);          // elementwise, computed in f32
    assert_eq!(doubled.shape(), t.shape());
    Ok(())
}
```

`reshape` consumes the tensor (no copy); `reshaped` clones.

## `ndarray` interop (feature `ndarray`)

<!-- doccheck: ignore -->
```rust
let arr: ndarray::ArrayD<f32> = /* ... */;
let t = qanten::Tensor::from(&arr);   // also From<&ArrayD<bf16>>
let back = t.to_ndarray();            // ArrayD<f32>
let exact = t.into_ndarray_bf16();    // ArrayD<bf16>, no conversion
```

## Layout conventions

Same as PyTorch: images are `NCHW`, linear weights `[out, in]`, conv weights
`[out, in/groups, kh, kw]`, transposed-conv weights `[in, out/groups, kh, kw]`.

[`Tensor`]: https://docs.rs/qanten/latest/qanten/struct.Tensor.html
