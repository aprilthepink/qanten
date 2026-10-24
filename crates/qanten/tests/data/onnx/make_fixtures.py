"""Builds the ONNX test models and their reference outputs.

Run with a Python that has `onnx`, `onnxruntime` and `numpy`:
    python make_fixtures.py
Writes <name>.onnx, <name>.inputs.npz and <name>.outputs.npz (onnxruntime results).
"""
import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

rng = np.random.default_rng(0)
F = TensorProto.FLOAT


def init(name, arr):
    return numpy_helper.from_array(np.asarray(arr), name)


def rand(*shape, scale=0.5):
    return (rng.standard_normal(shape) * scale).astype(np.float32)


def save(name, nodes, inputs, outputs, inits, feeds, opset=17):
    graph = helper.make_graph(nodes, name, inputs, outputs, inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, f"{name}.onnx")
    sess = ort.InferenceSession(f"{name}.onnx", providers=["CPUExecutionProvider"])
    out = sess.run(None, feeds)
    np.savez(f"{name}.inputs.npz", **feeds)
    np.savez(f"{name}.outputs.npz", **{o.name: v for o, v in zip(outputs, out)})
    print(name, [v.shape for v in out])


def vi(name, shape, dtype=F):
    return helper.make_tensor_value_info(name, dtype, shape)


# 1. MLP as exported from torch: Gemm(transB) + Relu + Gemm + Softmax
save(
    "mlp",
    [
        helper.make_node("Gemm", ["x", "w1", "b1"], ["h"], transB=1),
        helper.make_node("Relu", ["h"], ["hr"]),
        helper.make_node("Gemm", ["hr", "w2", "b2"], ["logits"], transB=1),
        helper.make_node("Softmax", ["logits"], ["probs"], axis=-1),
    ],
    [vi("x", ["batch", 8])],
    [vi("probs", ["batch", 4])],
    [init("w1", rand(16, 8)), init("b1", rand(16)), init("w2", rand(4, 16)), init("b2", rand(4))],
    {"x": rand(5, 8, scale=1.0)},
)

# 2. CNN: Conv + BN + Relu + MaxPool + grouped/strided Conv + AveragePool + ConvTranspose + GAP + Flatten + Gemm
save(
    "cnn",
    [
        helper.make_node("Conv", ["x", "c1w", "c1b"], ["c1"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        helper.make_node("BatchNormalization", ["c1", "bn_s", "bn_b", "bn_m", "bn_v"], ["bn"], epsilon=1e-5),
        helper.make_node("Relu", ["bn"], ["r1"]),
        helper.make_node("MaxPool", ["r1"], ["p1"], kernel_shape=[2, 2], strides=[2, 2]),
        helper.make_node("Conv", ["p1", "c2w"], ["c2"], kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1], group=2),
        helper.make_node("AveragePool", ["c2"], ["p2"], kernel_shape=[2, 2], strides=[1, 1], pads=[0, 0, 1, 1], count_include_pad=0),
        helper.make_node("ConvTranspose", ["p2", "ctw", "ctb"], ["ct"], kernel_shape=[2, 2], strides=[2, 2]),
        helper.make_node("GlobalAveragePool", ["ct"], ["gap"]),
        helper.make_node("Flatten", ["gap"], ["flat"], axis=1),
        helper.make_node("Gemm", ["flat", "fw", "fb"], ["out"], transB=1),
    ],
    [vi("x", [2, 3, 12, 12])],
    [vi("out", [2, 5])],
    [
        init("c1w", rand(8, 3, 3, 3)), init("c1b", rand(8)),
        init("bn_s", 1 + rand(8, scale=0.1)), init("bn_b", rand(8, scale=0.1)),
        init("bn_m", rand(8, scale=0.1)), init("bn_v", np.abs(1 + rand(8, scale=0.1))),
        init("c2w", rand(6, 4, 3, 3)),
        init("ctw", rand(6, 4, 2, 2)), init("ctb", rand(4)),
        init("fw", rand(5, 4)), init("fb", rand(5)),
    ],
    {"x": rand(2, 3, 12, 12, scale=1.0)},
)

# 3. Transformer-style block: embedding Gather, MatMul on 3-D, LayerNormalization,
#    GELU via Erf, attention reshapes/transposes, residual adds.
d, h = 8, 2
save(
    "block",
    [
        helper.make_node("Gather", ["emb", "ids"], ["x"], axis=0),
        helper.make_node("LayerNormalization", ["x", "ln_g", "ln_b"], ["ln"], axis=-1, epsilon=1e-5),
        helper.make_node("MatMul", ["ln", "wq"], ["q"]),
        helper.make_node("MatMul", ["ln", "wk"], ["k"]),
        helper.make_node("Reshape", ["q", "heads_shape"], ["q4"]),
        helper.make_node("Reshape", ["k", "heads_shape"], ["k4"]),
        helper.make_node("Transpose", ["q4"], ["qt"], perm=[0, 2, 1, 3]),
        helper.make_node("Transpose", ["k4"], ["kt"], perm=[0, 2, 3, 1]),
        helper.make_node("MatMul", ["qt", "kt"], ["scores"]),
        helper.make_node("Div", ["scores", "sqrt_hd"], ["scaled"]),
        helper.make_node("Softmax", ["scaled"], ["attn"], axis=-1),
        helper.make_node("ReduceMean", ["attn"], ["attn_mean"], axes=[1], keepdims=0),
        helper.make_node("MatMul", ["attn_mean", "ln"], ["ctx"]),
        helper.make_node("Add", ["x", "ctx"], ["res1"]),
        helper.make_node("MatMul", ["res1", "w1"], ["m1"]),
        helper.make_node("Add", ["m1", "b1"], ["m1b"]),
        # GELU (exact) the way torch exports it: x * 0.5 * (1 + erf(x / sqrt(2)))
        helper.make_node("Div", ["m1b", "sqrt2"], ["g_div"]),
        helper.make_node("Erf", ["g_div"], ["g_erf"]),
        helper.make_node("Add", ["g_erf", "one"], ["g_add"]),
        helper.make_node("Mul", ["m1b", "g_add"], ["g_mul"]),
        helper.make_node("Mul", ["g_mul", "half"], ["gelu"]),
        helper.make_node("MatMul", ["gelu", "w2"], ["m2"]),
        helper.make_node("Add", ["res1", "m2"], ["y"]),
    ],
    [vi("ids", ["batch", "seq"], TensorProto.INT64)],
    [vi("y", ["batch", "seq", d])],
    [
        init("emb", rand(10, d)), init("ln_g", 1 + rand(d, scale=0.1)), init("ln_b", rand(d, scale=0.1)),
        init("wq", rand(d, d)), init("wk", rand(d, d)),
        init("heads_shape", np.array([0, 0, h, d // h], dtype=np.int64)),
        init("sqrt_hd", np.array(np.sqrt(d // h), dtype=np.float32)),
        init("w1", rand(d, 2 * d)), init("b1", rand(2 * d)), init("w2", rand(2 * d, d)),
        init("sqrt2", np.array(np.sqrt(2), dtype=np.float32)),
        init("one", np.array(1.0, dtype=np.float32)), init("half", np.array(0.5, dtype=np.float32)),
    ],
    {"ids": np.array([[1, 4, 2, 9, 0], [3, 3, 7, 1, 5]], dtype=np.int64)},
)

# 4. Assorted ops: broadcasting arithmetic, Sub, Sigmoid, Tanh, LeakyRelu, Clip,
#    Sqrt/Pow, Concat, Unsqueeze/Squeeze, Shape + Gather on ints, Identity, Dropout,
#    Constant, Cast, ReduceSum, LogSoftmax, Neg, Exp, Abs, Max/Min.
save(
    "misc",
    [
        helper.make_node("Constant", [], ["c"], value=init("c_val", rand(1, 6))),
        helper.make_node("Sub", ["a", "c"], ["s"]),
        helper.make_node("Sigmoid", ["s"], ["sg"]),
        helper.make_node("Tanh", ["a"], ["th"]),
        helper.make_node("LeakyRelu", ["a"], ["lr"], alpha=0.1),
        helper.make_node("Clip", ["a", "lo", "hi"], ["cl"]),
        helper.make_node("Abs", ["a"], ["ab"]),
        helper.make_node("Sqrt", ["ab"], ["sq"]),
        helper.make_node("Pow", ["th", "two"], ["pw"]),
        helper.make_node("Concat", ["sg", "lr", "cl", "sq", "pw"], ["cat"], axis=1),
        helper.make_node("Unsqueeze", ["cat", "ax0"], ["us"]),
        helper.make_node("Squeeze", ["us", "ax0"], ["sq2"]),
        helper.make_node("Identity", ["sq2"], ["id"]),
        helper.make_node("Dropout", ["id"], ["dr"]),
        helper.make_node("Shape", ["dr"], ["shp"]),
        helper.make_node("Gather", ["shp", "idx1"], ["ncols"], axis=0),
        helper.make_node("Cast", ["ncols"], ["ncols_f"], to=TensorProto.FLOAT),
        helper.make_node("Div", ["dr", "ncols_f"], ["norm"]),
        helper.make_node("ReduceSum", ["norm", "ax1"], ["rs"], keepdims=1),
        helper.make_node("Exp", ["rs"], ["ex"]),
        helper.make_node("Neg", ["ex"], ["ng"]),
        helper.make_node("Max", ["ng", "dr"], ["mx"]),
        helper.make_node("Min", ["mx", "b"], ["mn"]),
        helper.make_node("LogSoftmax", ["mn"], ["out"], axis=1),
    ],
    [vi("a", [3, 6]), vi("b", [3, 1])],
    [vi("out", [3, 30])],
    [
        init("lo", np.array(-0.3, dtype=np.float32)), init("hi", np.array(0.4, dtype=np.float32)),
        init("two", np.array(2.0, dtype=np.float32)),
        init("ax0", np.array([0], dtype=np.int64)), init("ax1", np.array([1], dtype=np.int64)),
        init("idx1", np.array(1, dtype=np.int64)),
    ],
    {"a": rand(3, 6, scale=1.0), "b": rand(3, 1, scale=1.0)},
)
