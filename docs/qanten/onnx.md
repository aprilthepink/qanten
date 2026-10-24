# ONNX models (feature `onnx`)

Run `.onnx` files exported from PyTorch, TensorFlow, JAX or scikit-learn converters on
the NPU, without writing the model by hand.

```toml
qanten = { version = "0.1", features = ["onnx"] }
```

Export from PyTorch:

```python
torch.onnx.export(model, example_input, "model.onnx", opset_version=17,
                  dynamic_axes={"input": {0: "batch"}})
```

Run in Rust:

<!-- doccheck: no_run -->
```rust
use qanten::onnx::{OnnxModel, OnnxValue};
use qanten::{Npu, Tensor};

fn main() -> qanten::Result<()> {
    let npu = Npu::auto(0)?;
    let model = OnnxModel::load(&npu, "model.onnx")?;
    println!("{}", model.summary());

    // One float input, first output:
    let y = model.run_single(&Tensor::rand_uniform([1, 3, 32, 32], 0., 1., 0))?;

    // Any inputs, including integer token ids; all outputs by name:
    let outputs = model.run(&[(
        "ids",
        OnnxValue::Int { data: vec![1, 5, 9], shape: vec![1, 3] },
    )])?;
    for (name, value) in outputs {
        println!("{name}: {:?}", value.into_tensor()?.shape());
    }
    println!("{y:?}");
    Ok(())
}
```

`OnnxModel` also implements `nn::Module`, so it composes with `Sequential`,
`NpuPool::replicate` and `NpuExecutor::forward`.

## What runs where

| On the NPU | On the host (f32) |
|---|---|
| `Gemm`, `MatMul`, `Conv`, `ConvTranspose` (constant weights uploaded once at load time and kept resident), equal-shape `Mul`, `BatchNormalization` scaling | everything else |

`summary()` reports how many nodes have NPU-resident weights.

## Supported operators

Abs, Add, AveragePool, BatchNormalization, Cast, Ceil, Clip, Concat, Constant,
ConstantOfShape, Conv, ConvTranspose, Cos, Div, Dropout, Elu, Equal, Erf, Exp, Expand,
Flatten, Floor, Gather, Gelu, Gemm, GlobalAveragePool, GlobalMaxPool, Greater,
HardSigmoid, Identity, LayerNormalization, LeakyRelu, Less, Log, LogSoftmax, MatMul,
Max, MaxPool, Mean, Min, Mul, Neg, Not, Pow, Range, Reciprocal, ReduceMax, ReduceMean,
ReduceMin, ReduceSum, Relu, Reshape, Shape, Sigmoid, Sin, Slice, Softmax, Softplus,
Split, Sqrt, Squeeze, Sub, Sum, Tanh, Transpose, Unsqueeze, Where.

That covers typical exports of MLPs, CNNs (including grouped and depthwise
convolutions, padding modes, pooling), and transformer blocks (LayerNorm, GELU via
`Erf`, attention reshapes and transposes, embedding `Gather`, `Split`, masks via
`Where`). The list is also in code as `qanten::onnx::SUPPORTED_OPS`.

## Limitations

- Unknown operators and custom domains fail at **load** time, naming the node.
- Convolutions and pooling: 1-D and 2-D. `ConvTranspose` needs symmetric pads and no
  `output_shape` attribute; pooling with `ceil_mode=1` or dilations isn't supported.
- `Conv` / `ConvTranspose` weights must be constants in the file (they always are in
  normal exports).
- Models must be self-contained: no external-data files (models > 2 GB).
- No control flow (`If`, `Loop`, `Scan`).

## Accuracy

The tests build models with `onnx.helper`, compute reference outputs with
`onnxruntime`, and compare (`tests/data/onnx/make_fixtures.py`, `tests/onnx.rs`):

| Model | Relative error vs onnxruntime |
|---|---|
| MLP (Gemm, Relu, Softmax) | 0.6 % |
| CNN (Conv, BN, pools, grouped/strided Conv, ConvTranspose, Gemm) | 1.1 % |
| transformer-style block (Gather, LayerNorm, 3-D MatMul, attention, GELU via Erf) | 0.7 % |
| assorted ops (broadcasting, Clip, Concat, Shape/Gather on ints, Cast, reductions, …) | 0.4 % |

The error comes from bf16 arithmetic on the NPU path, as with every other qanten op.
