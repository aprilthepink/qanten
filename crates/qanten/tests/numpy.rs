//! NumPy interop, against files written by real NumPy (tests/data, see make_fixtures note below).
//!
//! Fixtures were generated with:
//! ```python
//! a = np.arange(12, dtype=np.float32).reshape(3, 4) / 4
//! np.save('c_f32.npy', a); np.save('fortran_f64.npy', np.asfortranarray(a.astype(np.float64)))
//! np.save('be_f32.npy', a.astype('>f4')); np.save('i64.npy', np.array([[1, -2, 3]], dtype=np.int64))
//! np.save('f16.npy', np.array([0.5, 1.5, -2.0], dtype=np.float16)); np.save('scalar.npy', np.float32(7.25))
//! np.savez('stored.npz', weight=a, bias=np.array([1, 2, 3], dtype=np.float32))
//! np.savez_compressed('compressed.npz', weight=a, bias=np.array([1, 2, 3], dtype=np.float32))
//! ```
#![cfg(feature = "numpy")]

use qanten::nn::StateDict;
use qanten::numpy::{load_npy, read_npy, write_npy};
use qanten::{Npu, Tensor};

const DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data");

fn expected() -> Vec<f32> {
    (0..12).map(|v| v as f32 / 4.0).collect()
}

#[test]
fn npy_layouts_and_dtypes() {
    for f in ["c_f32.npy", "fortran_f64.npy", "be_f32.npy"] {
        let t = load_npy(format!("{DATA}/{f}")).unwrap();
        assert_eq!(t.shape(), &[3, 4], "{f}");
        assert_eq!(t.to_f32_vec(), expected(), "{f}");
    }
    assert_eq!(
        load_npy(format!("{DATA}/i64.npy")).unwrap().to_f32_vec(),
        vec![1., -2., 3.]
    );
    assert_eq!(
        load_npy(format!("{DATA}/f16.npy")).unwrap().to_f32_vec(),
        vec![0.5, 1.5, -2.0]
    );
    let s = load_npy(format!("{DATA}/scalar.npy")).unwrap();
    assert_eq!((s.shape(), s.to_f32_vec()), (&[][..], vec![7.25]));
}

#[test]
fn npz_stored_and_compressed() {
    for f in ["stored.npz", "compressed.npz"] {
        let sd = StateDict::load_npz(format!("{DATA}/{f}")).unwrap();
        assert_eq!(sd.len(), 2, "{f}");
        assert_eq!(sd.get("weight").unwrap().to_f32_vec(), expected());
        assert_eq!(sd.get("bias").unwrap().to_f32_vec(), vec![1., 2., 3.]);
    }
}

#[test]
fn roundtrip_through_our_writer() {
    let t = Tensor::rand_uniform([5, 2, 3], -1., 1., 9);
    assert_eq!(read_npy(&write_npy(&t)).unwrap(), t);
    let mut sd = StateDict::new();
    sd.insert("a.weight", t.clone());
    sd.insert("a.bias", Tensor::vector(&[1.0]));
    sd.insert("scalar", Tensor::zeros([]));
    let back = StateDict::from_npz_bytes(&sd.to_npz_bytes().unwrap()).unwrap();
    assert_eq!(back.get("a.weight").unwrap(), &t);
    assert_eq!(back.len(), 3);
}

#[test]
fn rejects_garbage() {
    assert!(read_npy(b"not numpy at all").is_err());
    assert!(StateDict::from_npz_bytes(b"PK nope").is_err());
}

#[test]
fn qant_digit_weights_from_original_npz() {
    // Q.ANT's own file from their examples repo: l1_w [133, 784], l1_b, l2_w [10, 133], l2_b.
    let sd = StateDict::load_npz(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/data/qant_digit_weights.npz"
    ))
    .unwrap();
    assert_eq!(sd.get("l1_w").unwrap().shape(), &[133, 784]);
    let npu = Npu::simulator();
    let x = Tensor::rand_uniform([2, 784], 0., 1., 1);
    let h = qanten::ops::relu(
        &qanten::ops::linear(&npu, &x, sd.get("l1_w").unwrap(), sd.get_opt("l1_b")).unwrap(),
    );
    let y = qanten::ops::linear(&npu, &h, sd.get("l2_w").unwrap(), sd.get_opt("l2_b")).unwrap();
    assert_eq!(y.shape(), &[2, 10]);
}
