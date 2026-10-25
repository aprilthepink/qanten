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
