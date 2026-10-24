//! NumPy `.npy` and `.npz` files (feature `numpy`).
//!
//! Reads what `np.save`, `np.savez` and `np.savez_compressed` write: float
//! (`f2`, `f4`, `f8`), integer and bool arrays, little- or big-endian, C or
//! Fortran order. Everything converts to `bf16` tensors. Writing produces
//! little-endian `f4` arrays in an uncompressed `.npz`, which NumPy reads back
//! with `np.load`.
//!
//! ```
//! # fn main() -> qanten::Result<()> {
//! use qanten::{Tensor, nn::StateDict};
//! let mut sd = StateDict::new();
//! sd.insert("fc.weight", Tensor::from_f32(&[1., 2., 3., 4.], [2, 2])?);
//! let bytes = sd.to_npz_bytes()?;
//! let back = StateDict::from_npz_bytes(&bytes)?;
//! assert_eq!(back.get("fc.weight")?.to_f32_vec(), vec![1., 2., 3., 4.]);
//! # Ok(()) }
//! ```

use half::{bf16, f16};

use crate::error::{Error, Result};
use crate::nn::StateDict;
use crate::tensor::Tensor;

fn fmt_err(msg: impl Into<String>) -> Error {
    Error::Format(msg.into())
}

/// Parsed `.npy` header.
struct Header {
    descr: String,
    fortran: bool,
    shape: Vec<usize>,
}

fn parse_header(text: &str) -> Result<Header> {
    let value_of = |key: &str| -> Result<&str> {
        let k = format!("'{key}':");
        let start = text
            .find(&k)
            .ok_or_else(|| fmt_err(format!("npy header has no '{key}'")))?
            + k.len();
        Ok(text[start..].trim_start())
    };
    let descr_raw = value_of("descr")?;
    let descr = descr_raw
        .strip_prefix('\'')
        .and_then(|s| s.split('\'').next())
        .ok_or_else(|| fmt_err("npy header: bad 'descr' (structured dtypes are not supported)"))?
        .to_owned();
    let fortran = value_of("fortran_order")?.starts_with("True");
    let shape_raw = value_of("shape")?;
    let inner = shape_raw
        .strip_prefix('(')
        .and_then(|s| s.split(')').next())
        .ok_or_else(|| fmt_err("npy header: bad 'shape'"))?;
    let shape = inner
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<usize>()
                .map_err(|_| fmt_err(format!("npy header: bad dimension {s:?}")))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Header {
        descr,
        fortran,
        shape,
    })
}

fn decode(descr: &str, bytes: &[u8], count: usize) -> Result<Vec<bf16>> {
    let (endian, kind) = match descr.chars().next() {
        Some(c @ ('<' | '>' | '|' | '=')) => (c, &descr[1..]),
        _ => ('=', descr),
    };
    let big = endian == '>' || (endian == '=' && cfg!(target_endian = "big"));
    let size: usize = kind
        .get(1..)
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| fmt_err(format!("unsupported dtype {descr:?}")))?;
    let needed = count
        .checked_mul(size)
        .ok_or_else(|| fmt_err("array size overflows"))?;
    if bytes.len() < needed {
        return Err(fmt_err(format!(
            "npy data truncated: need {needed} bytes, have {}",
            bytes.len()
        )));
    }
    macro_rules! num {
        ($ty:ty, $n:expr, $conv:expr) => {
            bytes[..needed]
                .chunks_exact($n)
                .map(|c| {
                    let mut a = [0u8; $n];
                    a.copy_from_slice(c);
                    let v = if big {
                        <$ty>::from_be_bytes(a)
                    } else {
                        <$ty>::from_le_bytes(a)
                    };
                    #[allow(clippy::redundant_closure_call)]
                    ($conv)(v)
                })
                .collect()
        };
    }
    Ok(match (&kind[..1], size) {
        ("f", 2) => num!(u16, 2, |b| bf16::from_f32(f16::from_bits(b).to_f32())),
        ("f", 4) => num!(f32, 4, bf16::from_f32),
        ("f", 8) => num!(f64, 8, bf16::from_f64),
        ("i", 1) => num!(i8, 1, |v| bf16::from_f32(v as f32)),
        ("i", 2) => num!(i16, 2, |v| bf16::from_f32(v as f32)),
        ("i", 4) => num!(i32, 4, |v| bf16::from_f64(v as f64)),
        ("i", 8) => num!(i64, 8, |v| bf16::from_f64(v as f64)),
        ("u", 1) => num!(u8, 1, |v| bf16::from_f32(v as f32)),
        ("u", 2) => num!(u16, 2, |v| bf16::from_f32(v as f32)),
        ("u", 4) => num!(u32, 4, |v| bf16::from_f64(v as f64)),
        ("u", 8) => num!(u64, 8, |v| bf16::from_f64(v as f64)),
        ("b", 1) => num!(u8, 1, |v| if v != 0 { bf16::ONE } else { bf16::ZERO }),
        _ => return Err(fmt_err(format!("unsupported dtype {descr:?}"))),
    })
}

/// Reorders column-major data to row-major.
fn fortran_to_c(data: Vec<bf16>, shape: &[usize]) -> Vec<bf16> {
    if shape.len() < 2 {
        return data;
    }
    let mut out = vec![bf16::ZERO; data.len()];
    let mut idx = vec![0usize; shape.len()];
    for (c_pos, slot) in out.iter_mut().enumerate() {
        // idx = multi-index of c_pos in row-major order.
        let mut rem = c_pos;
        for d in (0..shape.len()).rev() {
            idx[d] = rem % shape[d];
            rem /= shape[d];
        }
        // Column-major offset: first index varies fastest.
        let mut f_pos = 0;
        for d in (0..shape.len()).rev() {
            f_pos = f_pos * shape[d] + idx[d];
        }
        *slot = data[f_pos];
    }
    out
}

/// Parses the contents of a `.npy` file.
pub fn read_npy(bytes: &[u8]) -> Result<Tensor> {
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err(fmt_err("not a .npy file (bad magic)"));
    }
    let (header_len, start) = match bytes[6] {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        2 | 3 if bytes.len() >= 12 => (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12,
        ),
        v => return Err(fmt_err(format!("unsupported .npy version {v}"))),
    };
    let header_bytes = bytes
        .get(start..start + header_len)
        .ok_or_else(|| fmt_err("npy header truncated"))?;
    let header = parse_header(&String::from_utf8_lossy(header_bytes))?;
    let count: usize = header.shape.iter().product();
    let mut data = decode(&header.descr, &bytes[start + header_len..], count)?;
    if header.fortran {
        data = fortran_to_c(data, &header.shape);
    }
    Tensor::new(data, header.shape)
}

/// Reads a `.npy` file from disk.
pub fn load_npy(path: impl AsRef<std::path::Path>) -> Result<Tensor> {
    read_npy(&std::fs::read(path)?)
}

/// Serialises a tensor as a version-1 `.npy` file with little-endian `f4` data.
pub fn write_npy(tensor: &Tensor) -> Vec<u8> {
    let shape = match tensor.shape() {
        [] => "()".to_owned(),
        [d] => format!("({d},)"),
        dims => format!(
            "({})",
            dims.iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ),
    };
    let mut header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': {shape}, }}");
    // Pad so the data starts on a 64-byte boundary, ending with a newline.
    let total = 10 + header.len() + 1;
    header.push_str(&" ".repeat((64 - total % 64) % 64));
    header.push('\n');
    let mut out = Vec::with_capacity(10 + header.len() + tensor.len() * 4);
    out.extend_from_slice(b"\x93NUMPY\x01\x00");
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    for v in tensor.data() {
        out.extend_from_slice(&v.to_f32().to_le_bytes());
    }
    out
}

/// Writes a tensor to a `.npy` file.
pub fn save_npy(path: impl AsRef<std::path::Path>, tensor: &Tensor) -> Result<()> {
    std::fs::write(path, write_npy(tensor))?;
    Ok(())
}

// ---- minimal ZIP container (what np.savez / np.savez_compressed produce) ----

fn le16(b: &[u8], at: usize) -> Result<usize> {
    b.get(at..at + 2)
        .map(|s| u16::from_le_bytes([s[0], s[1]]) as usize)
        .ok_or_else(|| fmt_err("zip structure truncated"))
}

fn le32(b: &[u8], at: usize) -> Result<usize> {
    b.get(at..at + 4)
        .map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]) as usize)
        .ok_or_else(|| fmt_err("zip structure truncated"))
}

/// `(name, contents)` of every file in a ZIP archive (stored or deflated).
fn unzip(bytes: &[u8]) -> Result<Vec<(String, Vec<u8>)>> {
    // End of central directory record: search backwards for its signature.
    let eocd = (0..bytes.len().saturating_sub(21))
        .rev()
        .find(|&i| bytes[i..].starts_with(b"PK\x05\x06"))
        .ok_or_else(|| fmt_err("not a .npz/.zip file (no end of central directory)"))?;
    let entries = le16(bytes, eocd + 10)?;
    let mut at = le32(bytes, eocd + 16)?;
    if at == 0xFFFF_FFFF {
        return Err(fmt_err("ZIP64 archives (> 4 GiB) are not supported"));
    }
    let mut files = Vec::with_capacity(entries);
    for _ in 0..entries {
        if !bytes
            .get(at..)
            .is_some_and(|b| b.starts_with(b"PK\x01\x02"))
        {
            return Err(fmt_err("corrupt zip central directory"));
        }
        let method = le16(bytes, at + 10)?;
        let comp_size = le32(bytes, at + 20)?;
        let size = le32(bytes, at + 24)?;
        let (name_len, extra_len, comment_len) = (
            le16(bytes, at + 28)?,
            le16(bytes, at + 30)?,
            le16(bytes, at + 32)?,
        );
        let local = le32(bytes, at + 42)?;
        let name = String::from_utf8_lossy(
            bytes
                .get(at + 46..at + 46 + name_len)
                .ok_or_else(|| fmt_err("zip name truncated"))?,
        )
        .into_owned();
        if comp_size == 0xFFFF_FFFF || size == 0xFFFF_FFFF || local == 0xFFFF_FFFF {
            return Err(fmt_err(format!("{name}: ZIP64 entries are not supported")));
        }
        at += 46 + name_len + extra_len + comment_len;

        let data_start = local + 30 + le16(bytes, local + 26)? + le16(bytes, local + 28)?;
        let raw = bytes
            .get(data_start..data_start + comp_size)
            .ok_or_else(|| fmt_err(format!("{name}: data truncated")))?;
        let data = match method {
            0 => raw.to_vec(),
            8 => miniz_oxide::inflate::decompress_to_vec(raw)
                .map_err(|e| fmt_err(format!("{name}: deflate error {e:?}")))?,
            m => return Err(fmt_err(format!("{name}: unsupported zip compression {m}"))),
        };
        if data.len() != size {
            return Err(fmt_err(format!(
                "{name}: size mismatch after decompression"
            )));
        }
        files.push((name, data));
    }
    Ok(files)
}

fn crc32(data: &[u8]) -> u32 {
    let mut table = [0u32; 256];
    for (i, t) in table.iter_mut().enumerate() {
        let mut c = i as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 {
                0xEDB8_8320 ^ (c >> 1)
            } else {
                c >> 1
            };
        }
        *t = c;
    }
    !data.iter().fold(!0u32, |c, &b| {
        table[((c ^ b as u32) & 0xFF) as usize] ^ (c >> 8)
    })
}

/// An uncompressed ZIP archive.
fn zip_stored(files: &[(String, Vec<u8>)]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    let mut central = Vec::new();
    for (name, data) in files {
        let offset = u32::try_from(out.len()).map_err(|_| fmt_err("archive larger than 4 GiB"))?;
        let size = u32::try_from(data.len()).map_err(|_| fmt_err("entry larger than 4 GiB"))?;
        let crc = crc32(data);
        let n = name.len() as u16;
        out.extend_from_slice(b"PK\x03\x04");
        out.extend_from_slice(&[20, 0, 0, 0, 0, 0, 0, 0, 0x21, 0]); // version, flags, method 0, time, date
        out.extend_from_slice(&crc.to_le_bytes());
        out.extend_from_slice(&size.to_le_bytes());
        out.extend_from_slice(&size.to_le_bytes());
        out.extend_from_slice(&n.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(data);

        central.extend_from_slice(b"PK\x01\x02");
        central.extend_from_slice(&[20, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0x21, 0]);
        central.extend_from_slice(&crc.to_le_bytes());
        central.extend_from_slice(&size.to_le_bytes());
        central.extend_from_slice(&size.to_le_bytes());
        central.extend_from_slice(&n.to_le_bytes());
        central.extend_from_slice(&[0; 12]); // extra, comment, disk, internal & external attrs
        central.extend_from_slice(&offset.to_le_bytes());
        central.extend_from_slice(name.as_bytes());
    }
    let cd_offset = u32::try_from(out.len()).map_err(|_| fmt_err("archive larger than 4 GiB"))?;
    let count = u16::try_from(files.len()).map_err(|_| fmt_err("too many arrays for one .npz"))?;
    out.extend_from_slice(&central);
    out.extend_from_slice(b"PK\x05\x06\0\0\0\0");
    out.extend_from_slice(&count.to_le_bytes());
    out.extend_from_slice(&count.to_le_bytes());
    out.extend_from_slice(&(central.len() as u32).to_le_bytes());
    out.extend_from_slice(&cd_offset.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    Ok(out)
}

impl StateDict {
    /// Parses a `.npz` archive (`np.savez` or `np.savez_compressed`). Array
    /// names become tensor names (the `.npy` suffix is dropped).
    pub fn from_npz_bytes(bytes: &[u8]) -> Result<Self> {
        unzip(bytes)?
            .into_iter()
            .filter(|(name, _)| name.ends_with(".npy"))
            .map(|(name, data)| {
                let t = read_npy(&data).map_err(|e| fmt_err(format!("{name}: {e}")))?;
                Ok((name.trim_end_matches(".npy").to_owned(), t))
            })
            .collect()
    }

    /// Reads a `.npz` file from disk.
    pub fn load_npz(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::from_npz_bytes(&std::fs::read(path)?)
    }

    /// Serialises all tensors as an uncompressed `.npz` of `f4` arrays.
    pub fn to_npz_bytes(&self) -> Result<Vec<u8>> {
        let files: Vec<(String, Vec<u8>)> = self
            .iter()
            .map(|(k, t)| (format!("{k}.npy"), write_npy(t)))
            .collect();
        zip_stored(&files)
    }

    /// Writes all tensors to a `.npz` file (`np.load` reads it).
    pub fn save_npz(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        std::fs::write(path, self.to_npz_bytes()?)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc_matches_reference() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn fortran_reorder() {
        // Column-major [[0, 1, 2], [3, 4, 5]] is stored as 0 3 1 4 2 5.
        let f: Vec<bf16> = [0., 3., 1., 4., 2., 5.]
            .iter()
            .map(|&v| bf16::from_f32(v))
            .collect();
        let c = fortran_to_c(f, &[2, 3]);
        assert_eq!(
            c.iter().map(|v| v.to_f32()).collect::<Vec<_>>(),
            vec![0., 1., 2., 3., 4., 5.]
        );
    }
}
