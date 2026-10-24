//! Runs the real `qanten` binary.

use std::process::Command;

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../qanten");

fn qanten(args: &[&str]) -> (bool, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_qanten"))
        .args(args)
        .env_remove("QANTEN_DRIVER_LIB")
        .output()
        .expect("binary runs");
    let text =
        String::from_utf8_lossy(&out.stdout).into_owned() + &String::from_utf8_lossy(&out.stderr);
    (out.status.success(), text)
}

#[test]
fn inspect_files() {
    let (ok, out) = qanten(&["inspect", &format!("{ROOT}/tests/data/onnx/cnn.onnx")]);
    assert!(ok && out.contains("4 with NPU-resident weights"), "{out}");
    let (ok, out) = qanten(&[
        "inspect",
        &format!("{ROOT}/examples/data/qant_digit_weights.npz"),
    ]);
    assert!(
        ok && out.contains("l1_w") && out.contains("[133, 784]"),
        "{out}"
    );
}

#[test]
fn run_onnx_with_output_files() {
    let dir = std::env::temp_dir().join(format!("qanten-cli-test-{}", std::process::id()));
    let (ok, out) = qanten(&[
        "--sim",
        "run",
        &format!("{ROOT}/tests/data/onnx/mlp.onnx"),
        "--batch",
        "3",
        "-o",
        dir.to_str().unwrap(),
    ]);
    assert!(ok, "{out}");
    assert!(out.contains("probs: shape [3, 4]"), "{out}");
    let t = qanten::numpy::load_npy(dir.join("probs.npy")).unwrap();
    assert_eq!(t.shape(), &[3, 4]);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn run_weights_as_mlp() {
    let (ok, out) = qanten(&[
        "--sim",
        "run",
        &format!("{ROOT}/examples/data/qant_digit_weights.npz"),
        "--arch",
        "mlp",
        "--profile",
    ]);
    assert!(ok, "{out}");
    assert!(
        out.contains("output: shape [1, 10]") && out.contains("matvec_resident"),
        "{out}"
    );
    let (ok, out) = qanten(&[
        "--sim",
        "run",
        &format!("{ROOT}/examples/data/qant_digit_weights.npz"),
    ]);
    assert!(!ok && out.contains("--arch mlp"), "{out}");
}

#[test]
fn selftest_on_simulator() {
    let (ok, out) = qanten(&["--sim", "selftest"]);
    assert!(ok && out.contains("all checks passed"), "{out}");
}
