//! Minimal protobuf decoding for the parts of `onnx.proto3` the importer uses.
//!
//! Field numbers are from <https://github.com/onnx/onnx/blob/main/onnx/onnx.proto>.

use crate::error::{Error, Result};

fn err(msg: impl Into<String>) -> Error {
    Error::Format(format!("onnx: {}", msg.into()))
}

/// One decoded field: number + payload.
enum Field<'a> {
    Varint(u64),
    Fixed64(u64),
    Bytes(&'a [u8]),
    Fixed32(u32),
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn varint(&mut self) -> Result<u64> {
        let mut v = 0u64;
        for shift in (0..64).step_by(7) {
            let b = *self
                .buf
                .get(self.pos)
                .ok_or_else(|| err("truncated varint"))?;
            self.pos += 1;
            v |= u64::from(b & 0x7F) << shift;
            if b & 0x80 == 0 {
                return Ok(v);
            }
        }
        Err(err("varint too long"))
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&e| e <= self.buf.len())
            .ok_or_else(|| err("truncated field"))?;
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn next(&mut self) -> Result<Option<(u32, Field<'a>)>> {
        if self.pos >= self.buf.len() {
            return Ok(None);
        }
        let key = self.varint()?;
        let number = (key >> 3) as u32;
        let field = match key & 7 {
            0 => Field::Varint(self.varint()?),
            1 => Field::Fixed64(u64::from_le_bytes(
                self.take(8)?.try_into().map_err(|_| err("bad fixed64"))?,
            )),
            2 => {
                let n = self.varint()? as usize;
                Field::Bytes(self.take(n)?)
            }
            5 => Field::Fixed32(u32::from_le_bytes(
                self.take(4)?.try_into().map_err(|_| err("bad fixed32"))?,
            )),
            w => return Err(err(format!("unsupported wire type {w}"))),
        };
        Ok(Some((number, field)))
    }
}

fn string(b: &[u8]) -> String {
    String::from_utf8_lossy(b).into_owned()
}

/// Repeated int64 that may be packed (bytes) or not (varint).
fn push_ints(out: &mut Vec<i64>, f: &Field<'_>) -> Result<()> {
    match f {
        Field::Varint(v) => out.push(*v as i64),
        Field::Bytes(b) => {
            let mut r = Reader::new(b);
            while r.pos < b.len() {
                out.push(r.varint()? as i64);
            }
        }
        _ => return Err(err("expected int64 values")),
    }
    Ok(())
}

fn push_floats(out: &mut Vec<f32>, f: &Field<'_>) -> Result<()> {
    match f {
        Field::Fixed32(v) => out.push(f32::from_bits(*v)),
        Field::Bytes(b) => out.extend(
            b.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
        ),
        _ => return Err(err("expected float values")),
    }
    Ok(())
}

fn push_doubles(out: &mut Vec<f64>, f: &Field<'_>) -> Result<()> {
    match f {
        Field::Fixed64(v) => out.push(f64::from_bits(*v)),
        Field::Bytes(b) => out.extend(b.chunks_exact(8).map(|c| {
            let mut a = [0u8; 8];
            a.copy_from_slice(c);
            f64::from_le_bytes(a)
        })),
        _ => return Err(err("expected double values")),
    }
    Ok(())
}

/// A tensor's raw contents, before conversion.
#[derive(Debug, Clone, Default)]
pub(crate) struct TensorProto {
    pub name: String,
    pub dims: Vec<i64>,
    pub data_type: i32,
    pub raw: Vec<u8>,
    pub floats: Vec<f32>,
    pub ints: Vec<i64>,
    pub doubles: Vec<f64>,
    pub external: bool,
}

/// ONNX `TensorProto.DataType` values used here.
pub(crate) mod dtype {
    pub const FLOAT: i32 = 1;
    pub const UINT8: i32 = 2;
    pub const INT8: i32 = 3;
    pub const INT16: i32 = 5;
    pub const INT32: i32 = 6;
    pub const INT64: i32 = 7;
    pub const BOOL: i32 = 9;
    pub const FLOAT16: i32 = 10;
    pub const DOUBLE: i32 = 11;
    pub const BFLOAT16: i32 = 16;
}

impl TensorProto {
    fn decode(b: &[u8]) -> Result<Self> {
        let mut t = Self::default();
        let mut r = Reader::new(b);
        while let Some((n, f)) = r.next()? {
            match (n, &f) {
                (1, _) => push_ints(&mut t.dims, &f)?,
                (2, Field::Varint(v)) => t.data_type = *v as i32,
                (4, _) => push_floats(&mut t.floats, &f)?,
                (5 | 7, _) => push_ints(&mut t.ints, &f)?, // int32_data / int64_data
                (8, Field::Bytes(s)) => t.name = string(s),
                (9, Field::Bytes(s)) => t.raw = s.to_vec(),
                (10, _) => push_doubles(&mut t.doubles, &f)?,
                (13, _) => t.external = true,
                (14, Field::Varint(v)) => t.external |= *v == 1,
                _ => {}
            }
        }
        Ok(t)
    }

    pub fn shape(&self) -> Result<Vec<usize>> {
        self.dims
            .iter()
            .map(|&d| {
                usize::try_from(d)
                    .map_err(|_| err(format!("tensor `{}` has negative dim", self.name)))
            })
            .collect()
    }

    /// Values as f64 (float and integer types).
    pub fn to_f64(&self) -> Result<Vec<f64>> {
        if self.external {
            return Err(err(format!(
                "tensor `{}` uses external data files, which are not supported (re-export with all data inside the .onnx)",
                self.name
            )));
        }
        let n = crate::limits::elements(&self.shape()?)?;
        let raw = &self.raw;
        let from_raw = |size: usize, f: &dyn Fn(&[u8]) -> f64| -> Vec<f64> {
            raw.chunks_exact(size).map(f).collect()
        };
        let v = match self.data_type {
            dtype::FLOAT if !raw.is_empty() => {
                from_raw(4, &|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
            }
            dtype::FLOAT => self.floats.iter().map(|&x| x as f64).collect(),
            dtype::DOUBLE if !raw.is_empty() => from_raw(8, &|c| {
                let mut a = [0u8; 8];
                a.copy_from_slice(c);
                f64::from_le_bytes(a)
            }),
            dtype::DOUBLE => self.doubles.clone(),
            dtype::FLOAT16 if !raw.is_empty() => {
                from_raw(2, &|c| half::f16::from_le_bytes([c[0], c[1]]).to_f64())
            }
            dtype::BFLOAT16 if !raw.is_empty() => {
                from_raw(2, &|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f64())
            }
            // f16/bf16 stored in int32_data hold the raw bits.
            dtype::FLOAT16 => self
                .ints
                .iter()
                .map(|&b| half::f16::from_bits(b as u16).to_f64())
                .collect(),
            dtype::BFLOAT16 => self
                .ints
                .iter()
                .map(|&b| half::bf16::from_bits(b as u16).to_f64())
                .collect(),
            dtype::INT64 if !raw.is_empty() => from_raw(8, &|c| {
                let mut a = [0u8; 8];
                a.copy_from_slice(c);
                i64::from_le_bytes(a) as f64
            }),
            dtype::INT32 if !raw.is_empty() => {
                from_raw(4, &|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
            }
            dtype::INT16 if !raw.is_empty() => {
                from_raw(2, &|c| i16::from_le_bytes([c[0], c[1]]) as f64)
            }
            dtype::INT8 if !raw.is_empty() => from_raw(1, &|c| c[0] as i8 as f64),
            dtype::UINT8 | dtype::BOOL if !raw.is_empty() => from_raw(1, &|c| c[0] as f64),
            dtype::INT64
            | dtype::INT32
            | dtype::INT16
            | dtype::INT8
            | dtype::UINT8
            | dtype::BOOL => self.ints.iter().map(|&x| x as f64).collect(),
            t => {
                return Err(err(format!(
                    "tensor `{}` has unsupported data type {t}",
                    self.name
                )));
            }
        };
        if v.len() != n {
            return Err(err(format!(
                "tensor `{}`: {} values for shape {:?}",
                self.name,
                v.len(),
                self.dims
            )));
        }
        Ok(v)
    }

    pub fn is_integer(&self) -> bool {
        matches!(
            self.data_type,
            dtype::INT64 | dtype::INT32 | dtype::INT16 | dtype::INT8 | dtype::UINT8 | dtype::BOOL
        )
    }
}

/// A node attribute.
#[derive(Debug, Clone, Default)]
pub(crate) struct Attribute {
    pub name: String,
    pub f: Option<f32>,
    pub i: Option<i64>,
    pub s: Option<String>,
    pub t: Option<TensorProto>,
    pub floats: Vec<f32>,
    pub ints: Vec<i64>,
}

impl Attribute {
    fn decode(b: &[u8]) -> Result<Self> {
        let mut a = Self::default();
        let mut r = Reader::new(b);
        while let Some((n, f)) = r.next()? {
            match (n, &f) {
                (1, Field::Bytes(s)) => a.name = string(s),
                (2, Field::Fixed32(v)) => a.f = Some(f32::from_bits(*v)),
                (3, Field::Varint(v)) => a.i = Some(*v as i64),
                (4, Field::Bytes(s)) => a.s = Some(string(s)),
                (5, Field::Bytes(s)) => a.t = Some(TensorProto::decode(s)?),
                (7, _) => push_floats(&mut a.floats, &f)?,
                (8, _) => push_ints(&mut a.ints, &f)?,
                _ => {}
            }
        }
        Ok(a)
    }
}

/// A graph node.
#[derive(Debug, Clone, Default)]
pub(crate) struct Node {
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub name: String,
    pub op_type: String,
    pub domain: String,
    pub attributes: Vec<Attribute>,
}

impl Node {
    fn decode(b: &[u8]) -> Result<Self> {
        let mut nd = Self::default();
        let mut r = Reader::new(b);
        while let Some((n, f)) = r.next()? {
            match (n, &f) {
                (1, Field::Bytes(s)) => nd.inputs.push(string(s)),
                (2, Field::Bytes(s)) => nd.outputs.push(string(s)),
                (3, Field::Bytes(s)) => nd.name = string(s),
                (4, Field::Bytes(s)) => nd.op_type = string(s),
                (5, Field::Bytes(s)) => nd.attributes.push(Attribute::decode(s)?),
                (7, Field::Bytes(s)) => nd.domain = string(s),
                _ => {}
            }
        }
        Ok(nd)
    }

    pub fn attr(&self, name: &str) -> Option<&Attribute> {
        self.attributes.iter().find(|a| a.name == name)
    }

    pub fn attr_i(&self, name: &str, default: i64) -> i64 {
        self.attr(name).and_then(|a| a.i).unwrap_or(default)
    }

    pub fn attr_f(&self, name: &str, default: f32) -> f32 {
        self.attr(name).and_then(|a| a.f).unwrap_or(default)
    }

    pub fn attr_ints(&self, name: &str) -> Option<&[i64]> {
        self.attr(name).map(|a| a.ints.as_slice())
    }

    pub fn attr_s(&self, name: &str) -> Option<&str> {
        self.attr(name).and_then(|a| a.s.as_deref())
    }
}

/// A graph input or output: name and (possibly symbolic) shape.
#[derive(Debug, Clone, Default)]
pub(crate) struct ValueInfo {
    pub name: String,
    pub elem_type: i32,
    /// `None` for symbolic / unknown dimensions.
    pub shape: Vec<Option<usize>>,
}

impl ValueInfo {
    fn decode(b: &[u8]) -> Result<Self> {
        let mut v = Self::default();
        let mut r = Reader::new(b);
        while let Some((n, f)) = r.next()? {
            match (n, f) {
                (1, Field::Bytes(s)) => v.name = string(s),
                (2, Field::Bytes(type_proto)) => {
                    // TypeProto.tensor_type (1) → elem_type (1), shape (2) → dim (1) → dim_value (1)
                    let mut tr = Reader::new(type_proto);
                    while let Some((tn, tf)) = tr.next()? {
                        if let (1, Field::Bytes(tensor)) = (tn, tf) {
                            let mut ttr = Reader::new(tensor);
                            while let Some((k, kf)) = ttr.next()? {
                                match (k, kf) {
                                    (1, Field::Varint(e)) => v.elem_type = e as i32,
                                    (2, Field::Bytes(shape)) => {
                                        let mut sr = Reader::new(shape);
                                        while let Some((_, Field::Bytes(dim))) = sr.next()? {
                                            let mut dr = Reader::new(dim);
                                            let mut value = None;
                                            while let Some((dn, df)) = dr.next()? {
                                                if let (1, Field::Varint(d)) = (dn, df) {
                                                    value = Some(d as usize);
                                                }
                                            }
                                            v.shape.push(value);
                                        }
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        Ok(v)
    }
}

/// A decoded model.
#[derive(Debug, Clone, Default)]
pub(crate) struct Model {
    pub opset: i64,
    pub producer: String,
    pub nodes: Vec<Node>,
    pub initializers: Vec<TensorProto>,
    pub inputs: Vec<ValueInfo>,
    pub outputs: Vec<ValueInfo>,
}

impl Model {
    pub fn decode(b: &[u8]) -> Result<Self> {
        let mut m = Self {
            opset: 1,
            ..Self::default()
        };
        let mut r = Reader::new(b);
        let mut graph = None;
        while let Some((n, f)) = r.next()? {
            match (n, f) {
                (2, Field::Bytes(s)) => m.producer = string(s),
                (7, Field::Bytes(g)) => graph = Some(g),
                (8, Field::Bytes(op)) => {
                    // OperatorSetIdProto: domain (1), version (2)
                    let (mut domain, mut version) = (String::new(), 0);
                    let mut or = Reader::new(op);
                    while let Some((on, of)) = or.next()? {
                        match (on, of) {
                            (1, Field::Bytes(s)) => domain = string(s),
                            (2, Field::Varint(v)) => version = v as i64,
                            _ => {}
                        }
                    }
                    if domain.is_empty() || domain == "ai.onnx" {
                        m.opset = version;
                    }
                }
                _ => {}
            }
        }
        let graph = graph.ok_or_else(|| err("file has no graph (is it an ONNX model?)"))?;
        let mut gr = Reader::new(graph);
        while let Some((n, f)) = gr.next()? {
            match (n, f) {
                (1, Field::Bytes(s)) => m.nodes.push(Node::decode(s)?),
                (5, Field::Bytes(s)) => m.initializers.push(TensorProto::decode(s)?),
                (11, Field::Bytes(s)) => m.inputs.push(ValueInfo::decode(s)?),
                (12, Field::Bytes(s)) => m.outputs.push(ValueInfo::decode(s)?),
                _ => {}
            }
        }
        Ok(m)
    }
}
