//! Malformed and hostile ONNX models must produce errors, never panics or
//! huge allocations.
#![cfg(feature = "onnx")]

use std::panic::{AssertUnwindSafe, catch_unwind};

use qanten::Npu;
use qanten::Tensor;
use qanten::onnx::{OnnxModel, OnnxValue};
use qanten::rng::SplitMix64;

// ---- a minimal protobuf encoder ---------------------------------------------

fn varint(mut v: u64, out: &mut Vec<u8>) {
    loop {
        let byte = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

fn int_field(num: u64, v: i64, out: &mut Vec<u8>) {
    varint(num << 3, out);
    varint(v as u64, out);
}

fn bytes_field(num: u64, b: &[u8], out: &mut Vec<u8>) {
    varint((num << 3) | 2, out);
    varint(b.len() as u64, out);
    out.extend_from_slice(b);
}

const FLOAT: i64 = 1;
const INT64: i64 = 7;

/// A `TensorProto` with raw little-endian data.
fn tensor(name: &str, dtype: i64, dims: &[i64], raw: &[u8]) -> Vec<u8> {
    let mut t = Vec::new();
    for &d in dims {
        int_field(1, d, &mut t);
    }
    int_field(2, dtype, &mut t);
    bytes_field(8, name.as_bytes(), &mut t);
    bytes_field(9, raw, &mut t);
    t
}

fn ints(name: &str, values: &[i64]) -> Vec<u8> {
    let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    tensor(name, INT64, &[values.len() as i64], &raw)
}

fn floats(name: &str, values: &[f32]) -> Vec<u8> {
    let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    tensor(name, FLOAT, &[values.len() as i64], &raw)
}

enum Attr<'a> {
    Int(&'a str, i64),
    Ints(&'a str, &'a [i64]),
}

fn node(op: &str, inputs: &[&str], outputs: &[&str], attrs: &[Attr]) -> Vec<u8> {
    let mut n = Vec::new();
    for i in inputs {
        bytes_field(1, i.as_bytes(), &mut n);
    }
    for o in outputs {
        bytes_field(2, o.as_bytes(), &mut n);
    }
    bytes_field(4, op.as_bytes(), &mut n);
    for a in attrs {
        let mut b = Vec::new();
        match a {
            Attr::Int(name, v) => {
                bytes_field(1, name.as_bytes(), &mut b);
                int_field(3, *v, &mut b);
            }
            Attr::Ints(name, vs) => {
                bytes_field(1, name.as_bytes(), &mut b);
                for v in *vs {
                    int_field(8, *v, &mut b);
                }
            }
        }
        bytes_field(5, &b, &mut n);
    }
    n
}

/// A graph input of the given element type.
fn value_info(name: &str, elem_type: i64) -> Vec<u8> {
    let mut tensor_type = Vec::new();
    int_field(1, elem_type, &mut tensor_type);
    let mut type_proto = Vec::new();
    bytes_field(1, &tensor_type, &mut type_proto);
    let mut v = Vec::new();
    bytes_field(1, name.as_bytes(), &mut v);
    bytes_field(2, &type_proto, &mut v);
    v
}

/// A model with one node, its initializers, graph inputs and output `y`.
fn model(op_node: Vec<u8>, initializers: &[Vec<u8>], inputs: &[(&str, i64)]) -> Vec<u8> {
    let mut g = Vec::new();
    bytes_field(1, &op_node, &mut g);
    for t in initializers {
        bytes_field(5, t, &mut g);
    }
    for (name, ty) in inputs {
        bytes_field(11, &value_info(name, *ty), &mut g);
    }
    bytes_field(12, &value_info("y", FLOAT), &mut g);
    let mut opset = Vec::new();
    int_field(2, 17, &mut opset);
    let mut m = Vec::new();
    int_field(1, 8, &mut m);
    bytes_field(8, &opset, &mut m);
    bytes_field(7, &g, &mut m);
    m
}

/// Loads and runs `bytes` with `feeds`; panics fail the test, errors are returned.
fn run(bytes: &[u8], feeds: &[(&str, OnnxValue)]) -> Result<Vec<(String, OnnxValue)>, String> {
    let npu = Npu::simulator();
    match catch_unwind(AssertUnwindSafe(|| {
        OnnxModel::from_bytes(&npu, bytes).and_then(|m| m.run(feeds))
    })) {
        Ok(r) => r.map_err(|e| e.to_string()),
        Err(_) => panic!("the model made the interpreter panic"),
    }
}

fn expect_err(bytes: &[u8], feeds: &[(&str, OnnxValue)], what: &str) {
    assert!(run(bytes, feeds).is_err(), "{what}: expected an error");
}

fn float_input(shape: &[usize]) -> OnnxValue {
    OnnxValue::Float(Tensor::zeros(shape.to_vec()))
}

// ---- hostile single-node models -------------------------------------------

#[test]
fn huge_shapes_are_refused_before_allocating() {
    let huge = node("ConstantOfShape", &["s"], &["y"], &[]);
    expect_err(
        &model(huge, &[ints("s", &[1 << 40])], &[]),
        &[],
        "ConstantOfShape 2^40",
    );
    let overflow = node("ConstantOfShape", &["s"], &["y"], &[]);
    expect_err(
        &model(overflow, &[ints("s", &[1 << 32, 1 << 32, 0])], &[]),
        &[],
        "ConstantOfShape with a hidden overflow",
    );
    let negative = node("ConstantOfShape", &["s"], &["y"], &[]);
    expect_err(
        &model(negative, &[ints("s", &[-1])], &[]),
        &[],
        "negative dim",
    );

    let expand = node("Expand", &["x", "s"], &["y"], &[]);
    expect_err(
        &model(expand, &[ints("s", &[1 << 20, 1 << 20])], &[("x", FLOAT)]),
        &[("x", float_input(&[1, 1]))],
        "Expand to 2^40",
    );

    let range = node("Range", &["a", "b", "c"], &["y"], &[]);
    expect_err(
        &model(
            range,
            &[ints("a", &[0]), ints("b", &[i64::MAX]), ints("c", &[1])],
            &[],
        ),
        &[],
        "integer Range to i64::MAX",
    );
    let frange = node("Range", &["a", "b", "c"], &["y"], &[]);
    expect_err(
        &model(
            frange,
            &[
                floats("a", &[0.0]),
                floats("b", &[1e30]),
                floats("c", &[1e-30]),
            ],
            &[],
        ),
        &[],
        "float Range with 1e60 steps",
    );

    let initializer = node("Identity", &["t"], &["y"], &[]);
    expect_err(
        &model(
            initializer,
            &[tensor("t", FLOAT, &[1 << 31, 1 << 31, 1 << 31], &[])],
            &[],
        ),
        &[],
        "initializer whose dims overflow",
    );
}

#[test]
fn empty_and_malformed_operands_are_errors() {
    let clip = node("Clip", &["x", "lo"], &["y"], &[]);
    expect_err(
        &model(clip, &[floats("lo", &[])], &[("x", FLOAT)]),
        &[("x", float_input(&[2]))],
        "Clip with an empty min",
    );
    let range = node("Range", &["a", "b", "c"], &["y"], &[]);
    expect_err(
        &model(
            range,
            &[ints("a", &[]), ints("b", &[3]), ints("c", &[1])],
            &[],
        ),
        &[],
        "Range with an empty start",
    );
    let softmax = node("Softmax", &["x"], &["y"], &[]);
    expect_err(
        &model(softmax, &[], &[("x", FLOAT)]),
        &[("x", float_input(&[]))],
        "Softmax of a scalar",
    );
    let slice = node("Slice", &["x", "st", "en", "ax"], &["y"], &[]);
    expect_err(
        &model(
            slice,
            &[ints("st", &[0, 0]), ints("en", &[1]), ints("ax", &[0])],
            &[("x", FLOAT)],
        ),
        &[("x", float_input(&[4, 4]))],
        "Slice with mismatched starts/ends",
    );
    let split = node("Split", &["x", "sz"], &["y", "z"], &[]);
    expect_err(
        &model(split, &[ints("sz", &[-1, 5])], &[("x", FLOAT)]),
        &[("x", float_input(&[4]))],
        "Split with a negative size",
    );
    let transpose = node("Transpose", &["x"], &["y"], &[Attr::Ints("perm", &[0, 0])]);
    expect_err(
        &model(transpose, &[], &[("x", FLOAT)]),
        &[("x", float_input(&[2, 3]))],
        "Transpose with a repeated axis",
    );
    let squeeze = node("Squeeze", &["x", "ax"], &["y"], &[]);
    expect_err(
        &model(squeeze, &[ints("ax", &[0])], &[("x", FLOAT)]),
        &[("x", float_input(&[3, 1]))],
        "Squeeze of a dimension of size 3",
    );
    let matmul = node("MatMul", &["x", "x"], &["y"], &[]);
    expect_err(
        &model(matmul, &[], &[("x", FLOAT)]),
        &[("x", float_input(&[]))],
        "MatMul of scalars",
    );
    let gather = node("Gather", &["x", "i"], &["y"], &[Attr::Int("axis", 0)]);
    expect_err(
        &model(gather, &[ints("i", &[7])], &[("x", FLOAT)]),
        &[("x", float_input(&[3]))],
        "Gather out of range",
    );
}

#[test]
fn bugs_found_by_fuzzing_stay_fixed() {
    // Stride 0 used to divide by zero in pooling.
    let pool = node(
        "MaxPool",
        &["x"],
        &["y"],
        &[
            Attr::Ints("kernel_shape", &[2, 2]),
            Attr::Ints("strides", &[0, 0]),
        ],
    );
    expect_err(
        &model(pool, &[], &[("x", FLOAT)]),
        &[("x", float_input(&[1, 1, 4, 4]))],
        "stride 0",
    );
    let pads = node(
        "MaxPool",
        &["x"],
        &["y"],
        &[
            Attr::Ints("kernel_shape", &[2, 2]),
            Attr::Ints("pads", &[-1, 0, 0, 0]),
        ],
    );
    expect_err(
        &model(pads, &[], &[("x", FLOAT)]),
        &[("x", float_input(&[1, 1, 4, 4]))],
        "negative pad",
    );
    // Clip with min > max used to panic in f32::clamp; ONNX gives min(max(x, lo), hi).
    let clip = node("Clip", &["x", "lo", "hi"], &["y"], &[]);
    let out = run(
        &model(
            clip,
            &[floats("lo", &[5.0]), floats("hi", &[1.0])],
            &[("x", FLOAT)],
        ),
        &[("x", float_input(&[3]))],
    )
    .unwrap();
    let OnnxValue::Float(t) = &out[0].1 else {
        panic!("float output")
    };
    assert_eq!(t.to_f32_vec(), vec![1.0; 3]);
}

#[test]
fn integer_arithmetic_never_panics() {
    let div = node("Div", &["a", "b"], &["y"], &[]);
    let out = run(
        &model(div, &[ints("a", &[i64::MIN, 7]), ints("b", &[-1, 0])], &[]),
        &[],
    )
    .unwrap();
    assert!(matches!(&out[0].1, OnnxValue::Int { data, .. } if data == &[0, 0]));
    for op in ["Add", "Sub", "Mul", "Pow"] {
        let n = node(op, &["a", "b"], &["y"], &[]);
        assert!(
            run(
                &model(n, &[ints("a", &[i64::MAX]), ints("b", &[i64::MAX])], &[]),
                &[]
            )
            .is_ok()
        );
    }
}

#[test]
fn integer_inputs_must_match_their_shape() {
    let id = node("Identity", &["x"], &["y"], &[]);
    let bytes = model(id, &[], &[("x", INT64)]);
    let lying = OnnxValue::Int {
        data: vec![1, 2],
        shape: vec![1 << 20, 1 << 20],
    };
    assert!(run(&bytes, &[("x", lying)]).is_err());
}

// ---- mutation fuzzing of real models ------------------------------------------

#[test]
fn corrupted_fixture_models_never_panic() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/onnx");
    let mut rng = SplitMix64::new(0xF022);
    for name in ["misc.onnx", "block.onnx", "cnn.onnx"] {
        let original = std::fs::read(dir.join(name)).unwrap();
        for _ in 0..300 {
            let mut bytes = original.clone();
            for _ in 0..1 + rng.below(8) {
                if bytes.is_empty() {
                    break;
                }
                let at = rng.below(bytes.len());
                match rng.below(3) {
                    0 => bytes[at] = rng.next_u64() as u8,
                    1 => bytes[at] ^= 1 << rng.below(8),
                    _ => bytes.truncate(at),
                }
            }
            // Load errors are fine; whatever loads must also run without panicking.
            let npu = Npu::simulator();
            let outcome = catch_unwind(AssertUnwindSafe(|| {
                let Ok(model) = OnnxModel::from_bytes(&npu, &bytes) else {
                    return;
                };
                let feeds: Vec<(String, OnnxValue)> = model
                    .inputs()
                    .iter()
                    .map(|i| {
                        let shape: Vec<usize> =
                            i.shape.iter().map(|d| d.unwrap_or(1).min(64)).collect();
                        (i.name.clone(), float_input(&shape))
                    })
                    .collect();
                let refs: Vec<(&str, OnnxValue)> =
                    feeds.iter().map(|(n, v)| (n.as_str(), v.clone())).collect();
                let _ = model.run(&refs);
            }));
            assert!(
                outcome.is_ok(),
                "{name}: a corrupted model made the interpreter panic"
            );
        }
    }
}
