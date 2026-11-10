//! Corrupted .npy / .npz / .safetensors files must produce errors, never panics.
#![cfg(all(feature = "numpy", feature = "safetensors"))]

use std::panic::{AssertUnwindSafe, catch_unwind};

use qanten::nn::StateDict;
use qanten::numpy::read_npy;
use qanten::rng::SplitMix64;
use qanten::{Tensor, limits};

fn mutate(rng: &mut SplitMix64, original: &[u8]) -> Vec<u8> {
    let mut bytes = original.to_vec();
    for _ in 0..1 + rng.below(6) {
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
    bytes
}

fn never_panics(what: &str, f: impl FnOnce()) {
    assert!(
        catch_unwind(AssertUnwindSafe(f)).is_ok(),
        "{what}: parser panicked"
    );
}

#[test]
fn corrupted_numpy_files_never_panic() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
    let mut rng = SplitMix64::new(0x9F);
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        if !matches!(ext, "npy" | "npz") {
            continue;
        }
        let original = std::fs::read(&path).unwrap();
        for _ in 0..500 {
            let bytes = mutate(&mut rng, &original);
            never_panics(&path.display().to_string(), || {
                let _ = if ext == "npy" {
                    read_npy(&bytes).map(|_| ())
                } else {
                    StateDict::from_npz_bytes(&bytes).map(|_| ())
                };
            });
        }
    }
}

#[test]
fn corrupted_safetensors_never_panic() {
    let mut sd = StateDict::new();
    sd.insert("w", Tensor::rand_uniform([4, 8], -1.0, 1.0, 1));
    sd.insert("b", Tensor::vector(&[1.0, 2.0, 3.0]));
    let original = sd.to_safetensors_bytes().unwrap();
    let mut rng = SplitMix64::new(0x5A);
    for _ in 0..2000 {
        let bytes = mutate(&mut rng, &original);
        never_panics("safetensors", || {
            let _ = StateDict::from_safetensors_bytes(&bytes);
        });
    }
}

#[test]
fn npy_headers_cannot_request_huge_arrays() {
    for shape in [
        "(4294967296, 4294967296)",
        "(0, 4294967296, 4294967296)",
        "(536870912,)",
    ] {
        let header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': {shape}, }}\n");
        let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
        bytes.extend((header.len() as u16).to_le_bytes());
        bytes.extend(header.as_bytes());
        assert!(read_npy(&bytes).is_err(), "{shape}");
    }
    assert!(limits::elements(&[1 << 20, 1 << 20]).is_err());
    assert_eq!(limits::elements(&[3, 0, 5]).unwrap(), 0);
    assert!(
        Tensor::checked_numel(&[0, usize::MAX, 2]).is_none(),
        "zeros don't hide overflow"
    );
    assert!(Tensor::new(vec![], [0, usize::MAX, 2]).is_err());
}

#[test]
fn npz_entries_must_not_share_data() {
    let mut sd = StateDict::new();
    sd.insert("a", Tensor::vector(&[1.0, 2.0]));
    sd.insert("b", Tensor::vector(&[3.0, 4.0]));
    let mut bytes = sd.to_npz_bytes().unwrap();
    assert!(StateDict::from_npz_bytes(&bytes).is_ok());
    // Point the second central directory entry at the first local header:
    // the classic overlapping-entry zip bomb.
    let central: Vec<usize> = bytes
        .windows(4)
        .enumerate()
        .filter(|(_, w)| *w == b"PK\x01\x02")
        .map(|(i, _)| i)
        .collect();
    assert_eq!(central.len(), 2);
    let first_local: [u8; 4] = bytes[central[0] + 42..central[0] + 46].try_into().unwrap();
    bytes[central[1] + 42..central[1] + 46].copy_from_slice(&first_local);
    let err = StateDict::from_npz_bytes(&bytes).unwrap_err().to_string();
    assert!(err.contains("overlaps"), "{err}");
}

#[test]
fn safetensors_shapes_are_limited() {
    let header = r#"{"w":{"dtype":"F32","shape":[0,1099511627776],"data_offsets":[0,0]}}"#;
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header.as_bytes());
    let err = StateDict::from_safetensors_bytes(&bytes).expect_err("a [0, 2^40] tensor is refused");
    assert!(err.to_string().contains("elements"), "{err}");
}

#[test]
fn matrix_sizes_cannot_overflow() {
    use qanten::compute::Matrix;
    assert!(Matrix::new(1 << 32, 1 << 32, vec![]).is_err());
    assert!(catch_unwind(|| Matrix::zeros(1 << 33, 1 << 31)).is_err());
}
