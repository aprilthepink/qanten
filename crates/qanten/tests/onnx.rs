//! ONNX models built with onnx.helper, run in qanten and compared with
//! onnxruntime's outputs (tests/data/onnx/make_fixtures.py).
#![cfg(all(feature = "onnx", feature = "numpy"))]

use qanten::Npu;
use qanten::nn::StateDict;
use qanten::onnx::{OnnxModel, OnnxValue};

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/onnx");

fn check(name: &str, rel_tol: f32) {
    let npu = Npu::simulator();
    let model = OnnxModel::load(&npu, format!("{DIR}/{name}.onnx")).unwrap();
    let inputs = StateDict::load_npz(format!("{DIR}/{name}.inputs.npz")).unwrap();
    let expected = StateDict::load_npz(format!("{DIR}/{name}.outputs.npz")).unwrap();
    let feeds: Vec<(String, OnnxValue)> = model
        .inputs()
        .iter()
        .map(|i| {
            let t = inputs.get(&i.name).unwrap().clone();
            let v = if i.is_integer {
                OnnxValue::Int {
                    data: t.to_f32_vec().iter().map(|&v| v as i64).collect(),
                    shape: t.shape().to_vec(),
                }
            } else {
                OnnxValue::Float(t)
            };
            (i.name.clone(), v)
        })
        .collect();
    let refs: Vec<(&str, OnnxValue)> = feeds.iter().map(|(n, v)| (n.as_str(), v.clone())).collect();
    let outputs = model.run(&refs).unwrap();
    for (out_name, value) in outputs {
        let got = value.into_tensor().unwrap();
        let want = expected.get(&out_name).unwrap();
        assert_eq!(got.shape(), want.shape(), "{name}/{out_name} shape");
        let scale = want
            .to_f32_vec()
            .iter()
            .fold(1e-3f32, |m, v| m.max(v.abs()));
        let err = got.max_abs_diff(want).unwrap() / scale;
        println!("{name}/{out_name}: relative error {err:.4}");
        assert!(
            err < rel_tol,
            "{name}/{out_name}: relative error {err} >= {rel_tol}"
        );
    }
}

#[test]
fn mlp() {
    check("mlp", 0.03);
}

#[test]
fn cnn() {
    check("cnn", 0.05);
}

#[test]
fn transformer_style_block() {
    check("block", 0.05);
}

#[test]
fn misc_ops() {
    check("misc", 0.05);
}

#[test]
fn summary_and_errors() {
    let npu = Npu::simulator();
    let model = OnnxModel::load(&npu, format!("{DIR}/cnn.onnx")).unwrap();
    let s = model.summary();
    assert!(
        s.contains("Conv×2") && s.contains("4 with NPU-resident weights"),
        "{s}"
    );
    assert_eq!(
        model.inputs()[0].shape,
        vec![Some(2), Some(3), Some(12), Some(12)]
    );
    assert!(model.run(&[]).is_err(), "missing input");
    assert!(OnnxModel::from_bytes(&npu, b"definitely not onnx").is_err());
}
