//! Running ONNX models (feature `onnx`).
//!
//! Load a `.onnx` file exported from PyTorch, TensorFlow or anything else and
//! run it on the NPU:
//!
//! ```no_run
//! # fn main() -> qanten::Result<()> {
//! use qanten::onnx::OnnxModel;
//! use qanten::{Npu, Tensor};
//! let npu = Npu::auto(0)?;
//! let model = OnnxModel::load(&npu, "model.onnx")?;
//! println!("{}", model.summary());
//! let y = model.run_single(&Tensor::rand_uniform([1, 3, 224, 224], 0., 1., 0))?;
//! # Ok(()) }
//! ```
//!
//! **What runs on the NPU:** `Gemm`, `MatMul`, `Conv` and `ConvTranspose`
//! (weights that are constants in the file are uploaded once at load time and
//! stay resident), elementwise `Mul` of equal shapes, and `BatchNormalization`
//! scaling. Everything else runs on the host in f32.
//!
//! Supported operators (default domain): see [`SUPPORTED_OPS`]. Unsupported
//! operators fail at load time with a clear message. Models must be
//! self-contained (no external data files).

mod proto;

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::path::Path;

use crate::device::Npu;
use crate::error::{Error, Result};
use crate::nn::Module;
use crate::ops::{
    self, Conv2dParams, ConvTranspose2dParams, ConvTransposeWeights, ConvWeights, LinearWeights,
};
use crate::tensor::Tensor;
use proto::{Node, TensorProto, dtype};

/// Every operator the importer understands.
pub const SUPPORTED_OPS: &[&str] = &[
    "Abs",
    "Add",
    "AveragePool",
    "BatchNormalization",
    "Cast",
    "Ceil",
    "Clip",
    "Concat",
    "Constant",
    "ConstantOfShape",
    "Conv",
    "ConvTranspose",
    "Cos",
    "Div",
    "Dropout",
    "Elu",
    "Equal",
    "Erf",
    "Exp",
    "Expand",
    "Flatten",
    "Floor",
    "Gather",
    "Gelu",
    "Gemm",
    "GlobalAveragePool",
    "GlobalMaxPool",
    "Greater",
    "HardSigmoid",
    "Identity",
    "LayerNormalization",
    "LeakyRelu",
    "Less",
    "Log",
    "LogSoftmax",
    "MatMul",
    "Max",
    "MaxPool",
    "Mean",
    "Min",
    "Mul",
    "Neg",
    "Not",
    "Pow",
    "Range",
    "Reciprocal",
    "ReduceMax",
    "ReduceMean",
    "ReduceMin",
    "ReduceSum",
    "Relu",
    "Reshape",
    "Shape",
    "Sigmoid",
    "Sin",
    "Slice",
    "Softmax",
    "Softplus",
    "Split",
    "Sqrt",
    "Squeeze",
    "Sub",
    "Sum",
    "Tanh",
    "Transpose",
    "Unsqueeze",
    "Where",
];

fn unsupported(node: &Node, what: impl fmt::Display) -> Error {
    Error::Unsupported(format!(
        "ONNX {} node `{}`: {what}",
        node.op_type, node.name
    ))
}

fn bad(node: &Node, what: impl fmt::Display) -> Error {
    Error::Format(format!(
        "ONNX {} node `{}`: {what}",
        node.op_type, node.name
    ))
}

// ---------------------------------------------------------------- values

/// An f32 array used for host-side computation.
#[derive(Debug, Clone, PartialEq)]
struct Arr {
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl Arr {
    fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self { data, shape }
    }
    fn tensor(&self) -> Result<Tensor> {
        Tensor::from_f32(&self.data, self.shape.clone())
    }
    fn from_tensor(t: &Tensor) -> Self {
        Self::new(t.to_f32_vec(), t.shape().to_vec())
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Val {
    F(Arr),
    I(Vec<i64>, Vec<usize>),
}

impl Val {
    fn shape(&self) -> &[usize] {
        match self {
            Val::F(a) => &a.shape,
            Val::I(_, s) => s,
        }
    }
    fn to_f(&self) -> Arr {
        match self {
            Val::F(a) => a.clone(),
            Val::I(d, s) => Arr::new(d.iter().map(|&v| v as f32).collect(), s.clone()),
        }
    }
    fn to_i(&self) -> (Vec<i64>, Vec<usize>) {
        match self {
            Val::I(d, s) => (d.clone(), s.clone()),
            Val::F(a) => (a.data.iter().map(|&v| v as i64).collect(), a.shape.clone()),
        }
    }
}

/// An input or output value of an ONNX model.
#[derive(Debug, Clone, PartialEq)]
pub enum OnnxValue {
    /// Floating point data (any float type in the model).
    Float(Tensor),
    /// Integer or boolean data (token ids, masks, shapes).
    Int {
        /// Values, row-major.
        data: Vec<i64>,
        /// Shape.
        shape: Vec<usize>,
    },
}

impl OnnxValue {
    /// The float tensor, or an error for integer values.
    pub fn into_tensor(self) -> Result<Tensor> {
        match self {
            OnnxValue::Float(t) => Ok(t),
            OnnxValue::Int { data, shape } => Tensor::new(
                data.iter()
                    .map(|&v| half::bf16::from_f64(v as f64))
                    .collect(),
                shape,
            ),
        }
    }
}

impl From<Tensor> for OnnxValue {
    fn from(t: Tensor) -> Self {
        OnnxValue::Float(t)
    }
}

/// Description of a model input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OnnxInput {
    /// Input name.
    pub name: String,
    /// Shape; `None` for dynamic dimensions (e.g. batch).
    pub shape: Vec<Option<usize>>,
    /// `true` if the model expects integers (e.g. token ids).
    pub is_integer: bool,
}

fn strides(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * shape[i + 1];
    }
    s
}

fn axis(a: i64, rank: usize) -> Result<usize> {
    let r = rank as i64;
    let v = if a < 0 { a + r } else { a };
    if (0..r.max(1)).contains(&v) {
        Ok(v as usize)
    } else {
        Err(Error::Shape(format!(
            "axis {a} out of range for rank {rank}"
        )))
    }
}

fn broadcast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    let n = a.len().max(b.len());
    (0..n)
        .map(|i| {
            let da = if i + a.len() >= n {
                a[i + a.len() - n]
            } else {
                1
            };
            let db = if i + b.len() >= n {
                b[i + b.len() - n]
            } else {
                1
            };
            match (da, db) {
                (x, y) if x == y => Ok(x),
                (1, y) => Ok(y),
                (x, 1) => Ok(x),
                _ => Err(Error::Shape(format!("cannot broadcast {a:?} with {b:?}"))),
            }
        })
        .collect()
}

/// Index of `out_index` in an array of `shape` broadcast to `out`.
fn broadcast_index(out: &[usize], shape: &[usize]) -> impl Fn(usize) -> usize + use<> {
    let n = out.len();
    let src_strides = strides(shape);
    let out_strides = strides(out);
    let map: Vec<(usize, usize)> = (0..n)
        .map(|i| {
            let k = i as isize - (n - shape.len()) as isize;
            if k < 0 || shape[k as usize] == 1 {
                (out_strides[i], 0)
            } else {
                (out_strides[i], src_strides[k as usize])
            }
        })
        .collect();
    move |mut idx| {
        let mut src = 0;
        for &(os, ss) in &map {
            let coord = idx / os;
            idx %= os;
            src += coord * ss;
        }
        src
    }
}

fn binary<T: Copy, U>(
    a: &[T],
    sa: &[usize],
    b: &[T],
    sb: &[usize],
    f: impl Fn(T, T) -> U,
) -> Result<(Vec<U>, Vec<usize>)> {
    let shape = broadcast_shape(sa, sb)?;
    let n: usize = shape.iter().product();
    if sa == sb {
        return Ok((a.iter().zip(b).map(|(&x, &y)| f(x, y)).collect(), shape));
    }
    let ia = broadcast_index(&shape, sa);
    let ib = broadcast_index(&shape, sb);
    Ok(((0..n).map(|i| f(a[ia(i)], b[ib(i)])).collect(), shape))
}

fn transpose_any<T: Copy>(data: &[T], shape: &[usize], perm: &[usize]) -> (Vec<T>, Vec<usize>) {
    let out_shape: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();
    let in_strides = strides(shape);
    let out_strides = strides(&out_shape);
    let n = data.len();
    let mut out = Vec::with_capacity(n);
    for o in 0..n {
        let mut rem = o;
        let mut src = 0;
        for (d, &p) in perm.iter().enumerate() {
            let c = rem / out_strides[d];
            rem %= out_strides[d];
            src += c * in_strides[p];
        }
        out.push(data[src]);
    }
    (out, out_shape)
}

fn erf(x: f32) -> f32 {
    // Abramowitz & Stegun 7.1.26, |error| < 1.5e-7.
    let t = 1.0 / (1.0 + 0.327_591_1 * x.abs());
    let y = 1.0
        - (((((1.061_405_4 * t - 1.453_152_1) * t) + 1.421_413_8) * t - 0.284_496_74) * t
            + 0.254_829_6)
            * t
            * (-x * x).exp();
    y.copysign(x)
}

// ---------------------------------------------------------------- model

#[derive(Debug)]
enum Resident {
    /// Matrix `[N, K]` such that the op is `x · matrixᵀ`.
    Linear(LinearWeights),
    Conv(ConvWeights),
    ConvT(ConvTransposeWeights),
}

/// A loaded ONNX model, ready to run on an NPU.
#[derive(Debug)]
pub struct OnnxModel {
    npu: Npu,
    opset: i64,
    producer: String,
    nodes: Vec<Node>,
    consts: HashMap<String, Val>,
    inputs: Vec<OnnxInput>,
    outputs: Vec<String>,
    resident: HashMap<usize, Resident>,
}

fn tensor_to_val(t: &TensorProto) -> Result<Val> {
    let data = t.to_f64()?;
    let shape = t.shape()?;
    Ok(if t.is_integer() {
        Val::I(data.iter().map(|&v| v as i64).collect(), shape)
    } else {
        Val::F(Arr::new(data.iter().map(|&v| v as f32).collect(), shape))
    })
}

impl OnnxModel {
    /// Loads a `.onnx` file.
    pub fn load(npu: &Npu, path: impl AsRef<Path>) -> Result<Self> {
        Self::from_bytes(npu, &std::fs::read(path)?)
    }

    /// Parses an ONNX model and uploads its constant weights to `npu`.
    pub fn from_bytes(npu: &Npu, bytes: &[u8]) -> Result<Self> {
        let m = proto::Model::decode(bytes)?;
        for n in &m.nodes {
            if !(n.domain.is_empty() || n.domain == "ai.onnx") {
                return Err(unsupported(n, format!("custom domain `{}`", n.domain)));
            }
            if !SUPPORTED_OPS.contains(&n.op_type.as_str()) {
                return Err(unsupported(
                    n,
                    "operator not supported (see qanten::onnx::SUPPORTED_OPS)",
                ));
            }
        }
        let mut consts = HashMap::new();
        for t in &m.initializers {
            consts.insert(t.name.clone(), tensor_to_val(t)?);
        }
        let inputs = m
            .inputs
            .iter()
            .filter(|v| !consts.contains_key(&v.name))
            .map(|v| OnnxInput {
                name: v.name.clone(),
                shape: v.shape.clone(),
                is_integer: matches!(
                    v.elem_type,
                    dtype::INT64
                        | dtype::INT32
                        | dtype::INT8
                        | dtype::UINT8
                        | dtype::BOOL
                        | dtype::INT16
                ),
            })
            .collect();
        let mut model = Self {
            npu: npu.clone(),
            opset: m.opset,
            producer: m.producer,
            outputs: m.outputs.iter().map(|o| o.name.clone()).collect(),
            nodes: m.nodes,
            consts,
            inputs,
            resident: HashMap::new(),
        };
        model.prepare()?;
        Ok(model)
    }

    /// Uploads constant weight matrices / kernels once.
    fn prepare(&mut self) -> Result<()> {
        for (i, n) in self.nodes.iter().enumerate() {
            let c = |k: usize| {
                n.inputs
                    .get(k)
                    .and_then(|name| self.consts.get(name))
                    .map(Val::to_f)
            };
            let r = match n.op_type.as_str() {
                "Gemm" => match c(1) {
                    Some(b) if b.shape.len() == 2 => {
                        let m = if n.attr_i("transB", 0) == 1 {
                            b
                        } else {
                            let (d, s) = transpose_any(&b.data, &b.shape, &[1, 0]);
                            Arr::new(d, s)
                        };
                        Some(Resident::Linear(LinearWeights::upload(
                            &self.npu,
                            &m.tensor()?,
                        )?))
                    }
                    _ => None,
                },
                "MatMul" => match c(1) {
                    Some(b) if b.shape.len() == 2 => {
                        let (d, s) = transpose_any(&b.data, &b.shape, &[1, 0]);
                        Some(Resident::Linear(LinearWeights::upload(
                            &self.npu,
                            &Arr::new(d, s).tensor()?,
                        )?))
                    }
                    _ => None,
                },
                "Conv" => match c(1) {
                    Some(w) => {
                        let w4 = to_4d_kernel(&w)?;
                        Some(Resident::Conv(ConvWeights::upload(
                            &self.npu,
                            &w4.tensor()?,
                            n.attr_i("group", 1) as usize,
                        )?))
                    }
                    None => None,
                },
                "ConvTranspose" => match c(1) {
                    Some(w) => {
                        let w4 = to_4d_kernel(&w)?;
                        Some(Resident::ConvT(ConvTransposeWeights::upload(
                            &self.npu,
                            &w4.tensor()?,
                            n.attr_i("group", 1) as usize,
                        )?))
                    }
                    None => None,
                },
                _ => None,
            };
            if let Some(r) = r {
                self.resident.insert(i, r);
            }
        }
        Ok(())
    }

    /// The inputs the caller must provide (constants are excluded).
    pub fn inputs(&self) -> &[OnnxInput] {
        &self.inputs
    }

    /// Names of the model outputs.
    pub fn outputs(&self) -> &[String] {
        &self.outputs
    }

    /// ONNX operator set version the model was exported with.
    pub fn opset(&self) -> i64 {
        self.opset
    }

    /// Operator counts, NPU-resident weights and inputs/outputs, for printing.
    pub fn summary(&self) -> String {
        let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
        for n in &self.nodes {
            *counts.entry(n.op_type.as_str()).or_default() += 1;
        }
        let ops: Vec<String> = counts.iter().map(|(k, v)| format!("{k}×{v}")).collect();
        let fmt_shape = |s: &[Option<usize>]| {
            s.iter()
                .map(|d| d.map_or("?".to_owned(), |v| v.to_string()))
                .collect::<Vec<_>>()
                .join("×")
        };
        format!(
            "ONNX model (opset {}, producer `{}`): {} nodes, {} with NPU-resident weights\n  inputs: {}\n  outputs: {}\n  ops: {}",
            self.opset,
            self.producer,
            self.nodes.len(),
            self.resident.len(),
            self.inputs
                .iter()
                .map(|i| format!(
                    "{} [{}]{}",
                    i.name,
                    fmt_shape(&i.shape),
                    if i.is_integer { " int" } else { "" }
                ))
                .collect::<Vec<_>>()
                .join(", "),
            self.outputs.join(", "),
            ops.join(" ")
        )
    }

    /// Runs the model on a single float input and returns the first output.
    pub fn run_single(&self, x: &Tensor) -> Result<Tensor> {
        let name = self
            .inputs
            .first()
            .ok_or_else(|| Error::InvalidArgument("model has no inputs".into()))?
            .name
            .clone();
        let out = self.run(&[(name.as_str(), OnnxValue::Float(x.clone()))])?;
        out.into_iter()
            .next()
            .ok_or_else(|| Error::Format("model produced no output".into()))?
            .1
            .into_tensor()
    }

    /// Runs the model. `inputs` maps input names to values; returns every
    /// graph output in order.
    pub fn run(&self, inputs: &[(&str, OnnxValue)]) -> Result<Vec<(String, OnnxValue)>> {
        let mut vals: HashMap<String, Val> = self.consts.clone();
        for spec in &self.inputs {
            let (_, v) = inputs
                .iter()
                .find(|(n, _)| *n == spec.name)
                .ok_or_else(|| {
                    Error::InvalidArgument(format!("missing model input `{}`", spec.name))
                })?;
            let v = match v {
                OnnxValue::Float(t) if spec.is_integer => Val::I(
                    t.to_f32_vec().iter().map(|&x| x.round() as i64).collect(),
                    t.shape().to_vec(),
                ),
                OnnxValue::Float(t) => Val::F(Arr::from_tensor(t)),
                OnnxValue::Int { data, shape } => Val::I(data.clone(), shape.clone()),
            };
            vals.insert(spec.name.clone(), v);
        }
        for (i, node) in self.nodes.iter().enumerate() {
            let outs = self.eval(i, node, &vals)?;
            for (name, v) in node.outputs.iter().zip(outs) {
                if !name.is_empty() {
                    vals.insert(name.clone(), v);
                }
            }
        }
        self.outputs
            .iter()
            .map(|o| {
                let v = vals
                    .get(o)
                    .ok_or_else(|| Error::Format(format!("output `{o}` was never computed")))?;
                Ok((
                    o.clone(),
                    match v {
                        Val::F(a) => OnnxValue::Float(a.tensor()?),
                        Val::I(d, s) => OnnxValue::Int {
                            data: d.clone(),
                            shape: s.clone(),
                        },
                    },
                ))
            })
            .collect()
    }

    fn eval(&self, idx: usize, n: &Node, vals: &HashMap<String, Val>) -> Result<Vec<Val>> {
        let input = |k: usize| -> Result<&Val> {
            let name = n
                .inputs
                .get(k)
                .filter(|s| !s.is_empty())
                .ok_or_else(|| bad(n, format!("missing input {k}")))?;
            vals.get(name)
                .ok_or_else(|| bad(n, format!("input `{name}` not available")))
        };
        let opt = |k: usize| -> Option<&Val> {
            n.inputs
                .get(k)
                .filter(|s| !s.is_empty())
                .and_then(|s| vals.get(s))
        };
        let f = |k: usize| input(k).map(Val::to_f);
        let ints = |k: usize| input(k).map(|v| v.to_i().0);
        let unary = |g: &dyn Fn(f32) -> f32| -> Result<Vec<Val>> {
            let a = f(0)?;
            Ok(vec![Val::F(Arr::new(
                a.data.iter().map(|&x| g(x)).collect(),
                a.shape,
            ))])
        };
        let op = n.op_type.as_str();
        let one = |v: Val| Ok(vec![v]);
        match op {
            // ---- NPU-heavy ops
            "Gemm" => {
                let mut a = f(0)?;
                if n.attr_i("transA", 0) == 1 {
                    let (d, s) = transpose_any(&a.data, &a.shape, &[1, 0]);
                    a = Arr::new(d, s);
                }
                let y = match self.resident.get(&idx) {
                    Some(Resident::Linear(lw)) => lw.forward(&a.tensor()?)?,
                    _ => {
                        let b = f(1)?;
                        let m = if n.attr_i("transB", 0) == 1 {
                            b
                        } else {
                            let (d, s) = transpose_any(&b.data, &b.shape, &[1, 0]);
                            Arr::new(d, s)
                        };
                        ops::matvec(&self.npu, &m.tensor()?, &a.tensor()?)?
                    }
                };
                let alpha = n.attr_f("alpha", 1.0);
                let beta = n.attr_f("beta", 1.0);
                let mut y = Arr::from_tensor(&y);
                if alpha != 1.0 {
                    y.data.iter_mut().for_each(|v| *v *= alpha);
                }
                if let Some(c) = opt(2) {
                    let c = c.to_f();
                    let (d, s) = binary(&y.data, &y.shape, &c.data, &c.shape, |a, b| a + beta * b)?;
                    y = Arr::new(d, s);
                }
                one(Val::F(y))
            }
            "MatMul" => {
                let a = f(0)?;
                if let Some(Resident::Linear(lw)) = self.resident.get(&idx) {
                    return one(Val::F(Arr::from_tensor(&lw.forward(&a.tensor()?)?)));
                }
                one(Val::F(self.matmul_general(&a, &f(1)?)?))
            }
            "Conv" => one(Val::F(self.conv(idx, n, &f(0)?, opt(2).map(Val::to_f))?)),
            "ConvTranspose" => one(Val::F(self.conv_transpose(
                idx,
                n,
                &f(0)?,
                opt(2).map(Val::to_f),
            )?)),
            "BatchNormalization" => {
                let x = f(0)?;
                if x.shape.len() < 2 {
                    return Err(bad(n, "input must have a channel dimension"));
                }
                let (nb, c) = (x.shape[0], x.shape[1]);
                let rest: usize = x.shape[2..].iter().product();
                let t = Tensor::from_f32(&x.data, [nb, c, rest.max(1), 1])?;
                let p = |k| f(k).and_then(|a| a.tensor());
                let y = ops::batch_norm2d(
                    &self.npu,
                    &t,
                    &p(3)?,
                    &p(4)?,
                    Some(&p(1)?),
                    Some(&p(2)?),
                    n.attr_f("epsilon", 1e-5),
                )?;
                one(Val::F(Arr::new(y.to_f32_vec(), x.shape)))
            }
            "Mul" => match (input(0)?, input(1)?) {
                (Val::I(a, sa), Val::I(b, sb)) => {
                    let (d, s) = binary(a, sa, b, sb, |x, y| x * y)?;
                    one(Val::I(d, s))
                }
                (x, y) if x.shape() == y.shape() && x.shape().iter().product::<usize>() >= 64 => {
                    let (a, b) = (x.to_f(), y.to_f());
                    let r = ops::mul(&self.npu, &a.tensor()?, &b.tensor()?)?;
                    one(Val::F(Arr::from_tensor(&r)))
                }
                (x, y) => {
                    let (a, b) = (x.to_f(), y.to_f());
                    let (d, s) = binary(&a.data, &a.shape, &b.data, &b.shape, |p, q| p * q)?;
                    one(Val::F(Arr::new(d, s)))
                }
            },
            // ---- arithmetic
            "Add" | "Sub" | "Div" | "Pow" => {
                let (x, y) = (input(0)?, input(1)?);
                if let (Val::I(a, sa), Val::I(b, sb)) = (x, y) {
                    let (d, s) = binary(a, sa, b, sb, |p, q| match op {
                        "Add" => p + q,
                        "Sub" => p - q,
                        "Div" => {
                            if q == 0 {
                                0
                            } else {
                                p / q
                            }
                        }
                        _ => p.pow(q.max(0) as u32),
                    })?;
                    return one(Val::I(d, s));
                }
                let (a, b) = (x.to_f(), y.to_f());
                let (d, s) = binary(&a.data, &a.shape, &b.data, &b.shape, |p, q| match op {
                    "Add" => p + q,
                    "Sub" => p - q,
                    "Div" => p / q,
                    _ => p.powf(q),
                })?;
                one(Val::F(Arr::new(d, s)))
            }
            "Max" | "Min" | "Sum" | "Mean" => {
                let mut acc = f(0)?;
                let count = n.inputs.iter().filter(|s| !s.is_empty()).count();
                for k in 1..count {
                    let b = f(k)?;
                    let (d, s) =
                        binary(&acc.data, &acc.shape, &b.data, &b.shape, |p, q| match op {
                            "Max" => p.max(q),
                            "Min" => p.min(q),
                            _ => p + q,
                        })?;
                    acc = Arr::new(d, s);
                }
                if op == "Mean" {
                    acc.data.iter_mut().for_each(|v| *v /= count as f32);
                }
                one(Val::F(acc))
            }
            "Equal" | "Less" | "Greater" => {
                let (a, b) = (f(0)?, f(1)?);
                let (d, s) = binary(&a.data, &a.shape, &b.data, &b.shape, |p, q| {
                    i64::from(match op {
                        "Equal" => p == q,
                        "Less" => p < q,
                        _ => p > q,
                    })
                })?;
                one(Val::I(d, s))
            }
            "Not" => {
                let (d, s) = input(0)?.to_i();
                one(Val::I(d.iter().map(|&v| i64::from(v == 0)).collect(), s))
            }
            "Where" => {
                let (c, sc) = input(0)?.to_i();
                let (x, y) = (f(1)?, f(2)?);
                let (xy, sxy) = binary(&x.data, &x.shape, &y.data, &y.shape, |p, q| (p, q))?;
                let (d, s) = binary(
                    &c,
                    &sc,
                    &(0..xy.len() as i64).collect::<Vec<_>>(),
                    &sxy,
                    |cond, k| {
                        if cond != 0 {
                            xy[k as usize].0
                        } else {
                            xy[k as usize].1
                        }
                    },
                )?;
                one(Val::F(Arr::new(d, s)))
            }
            // ---- elementwise
            "Relu" => unary(&|x| x.max(0.0)),
            "Sigmoid" => unary(&|x| 1.0 / (1.0 + (-x).exp())),
            "Tanh" => unary(&f32::tanh),
            "Exp" => unary(&f32::exp),
            "Log" => unary(&f32::ln),
            "Neg" => unary(&|x| -x),
            "Abs" => unary(&f32::abs),
            "Sqrt" => unary(&f32::sqrt),
            "Reciprocal" => unary(&|x| 1.0 / x),
            "Floor" => unary(&f32::floor),
            "Ceil" => unary(&f32::ceil),
            "Sin" => unary(&f32::sin),
            "Cos" => unary(&f32::cos),
            "Erf" => unary(&erf),
            "Softplus" => unary(&|x| if x > 20.0 { x } else { x.exp().ln_1p() }),
            "LeakyRelu" => {
                let a = n.attr_f("alpha", 0.01);
                unary(&|x| if x >= 0.0 { x } else { a * x })
            }
            "Elu" => {
                let a = n.attr_f("alpha", 1.0);
                unary(&|x| if x >= 0.0 { x } else { a * (x.exp() - 1.0) })
            }
            "HardSigmoid" => {
                let (a, b) = (n.attr_f("alpha", 0.2), n.attr_f("beta", 0.5));
                unary(&|x| (a * x + b).clamp(0.0, 1.0))
            }
            "Gelu" => {
                if n.attr_s("approximate") == Some("tanh") {
                    unary(&|x| 0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044_715 * x * x * x)).tanh()))
                } else {
                    unary(&|x| 0.5 * x * (1.0 + erf(x / std::f32::consts::SQRT_2)))
                }
            }
            "Clip" => {
                let (lo, hi) = if self.opset >= 11 {
                    (
                        opt(1)
                            .map(|v| v.to_f().data[0])
                            .unwrap_or(f32::NEG_INFINITY),
                        opt(2).map(|v| v.to_f().data[0]).unwrap_or(f32::INFINITY),
                    )
                } else {
                    (
                        n.attr_f("min", f32::NEG_INFINITY),
                        n.attr_f("max", f32::INFINITY),
                    )
                };
                unary(&|x| x.clamp(lo, hi))
            }
            "Identity" | "Dropout" => one(input(0)?.clone()),
            // ---- normalisation
            "Softmax" | "LogSoftmax" => {
                let x = f(0)?;
                let r = x.shape.len();
                let ax = axis(n.attr_i("axis", if self.opset >= 13 { -1 } else { 1 }), r)?;
                let (outer, len, inner) = if self.opset >= 13 {
                    (
                        x.shape[..ax].iter().product::<usize>(),
                        x.shape[ax],
                        x.shape[ax + 1..].iter().product::<usize>(),
                    )
                } else {
                    (
                        x.shape[..ax].iter().product::<usize>(),
                        x.shape[ax..].iter().product::<usize>(),
                        1,
                    )
                };
                let mut y = x.data.clone();
                for o in 0..outer {
                    for i in 0..inner {
                        let at = |k: usize| o * len * inner + k * inner + i;
                        let m = (0..len)
                            .map(|k| x.data[at(k)])
                            .fold(f32::NEG_INFINITY, f32::max);
                        let s: f32 = (0..len).map(|k| (x.data[at(k)] - m).exp()).sum();
                        for k in 0..len {
                            y[at(k)] = if op == "Softmax" {
                                (x.data[at(k)] - m).exp() / s
                            } else {
                                x.data[at(k)] - m - s.ln()
                            };
                        }
                    }
                }
                one(Val::F(Arr::new(y, x.shape)))
            }
            "LayerNormalization" => {
                let x = f(0)?;
                let ax = axis(n.attr_i("axis", -1), x.shape.len())?;
                let len: usize = x.shape[ax..].iter().product();
                let eps = n.attr_f("epsilon", 1e-5);
                let scale = f(1)?;
                let bias = opt(2).map(Val::to_f);
                let mut y = x.data.clone();
                for row in y.chunks_mut(len.max(1)) {
                    let m = row.iter().sum::<f32>() / len as f32;
                    let v = row.iter().map(|x| (x - m).powi(2)).sum::<f32>() / len as f32;
                    let inv = 1.0 / (v + eps).sqrt();
                    for (k, val) in row.iter_mut().enumerate() {
                        let sc = scale.data[k % scale.data.len().max(1)];
                        let b = bias
                            .as_ref()
                            .map_or(0.0, |b| b.data[k % b.data.len().max(1)]);
                        *val = (*val - m) * inv * sc + b;
                    }
                }
                one(Val::F(Arr::new(y, x.shape)))
            }
            // ---- pooling
            "MaxPool" | "AveragePool" => {
                if n.outputs.len() > 1 && !n.outputs[1].is_empty() {
                    return Err(unsupported(n, "the Indices output"));
                }
                if n.attr_i("ceil_mode", 0) != 0 {
                    return Err(unsupported(n, "ceil_mode=1"));
                }
                if n.attr_ints("dilations")
                    .is_some_and(|d| d.iter().any(|&v| v != 1))
                {
                    return Err(unsupported(n, "dilations"));
                }
                one(Val::F(pool(n, &f(0)?, op == "MaxPool")?))
            }
            "GlobalAveragePool" | "GlobalMaxPool" => {
                let x = f(0)?;
                if x.shape.len() < 3 {
                    return Err(bad(n, "needs [N, C, spatial...]"));
                }
                let plane: usize = x.shape[2..].iter().product();
                let data = x
                    .data
                    .chunks(plane.max(1))
                    .map(|c| {
                        if op == "GlobalMaxPool" {
                            c.iter().copied().fold(f32::NEG_INFINITY, f32::max)
                        } else {
                            c.iter().sum::<f32>() / plane as f32
                        }
                    })
                    .collect();
                let mut shape = x.shape.clone();
                shape[2..].iter_mut().for_each(|d| *d = 1);
                one(Val::F(Arr::new(data, shape)))
            }
            // ---- shape manipulation
            "Flatten" => {
                let v = input(0)?;
                let s = v.shape();
                let ax = if n.attr_i("axis", 1) == s.len() as i64 {
                    s.len()
                } else {
                    axis(n.attr_i("axis", 1), s.len())?
                };
                let shape = vec![s[..ax].iter().product(), s[ax..].iter().product()];
                one(reshape_val(v, shape))
            }
            "Reshape" => {
                let v = input(0)?;
                let target = ints(1)?;
                let allow_zero = n.attr_i("allowzero", 0) == 1;
                let total: usize = v.shape().iter().product();
                let mut shape: Vec<usize> = Vec::with_capacity(target.len());
                let mut infer = None;
                for (k, &d) in target.iter().enumerate() {
                    match d {
                        -1 => {
                            infer = Some(k);
                            shape.push(1);
                        }
                        0 if !allow_zero => shape.push(
                            *v.shape()
                                .get(k)
                                .ok_or_else(|| bad(n, "0 refers past input rank"))?,
                        ),
                        d if d >= 0 => shape.push(d as usize),
                        _ => return Err(bad(n, format!("invalid dim {d}"))),
                    }
                }
                if let Some(k) = infer {
                    let known: usize = shape.iter().product();
                    shape[k] = total.checked_div(known).unwrap_or(0);
                }
                if shape.iter().product::<usize>() != total {
                    return Err(bad(
                        n,
                        format!("cannot reshape {:?} to {shape:?}", v.shape()),
                    ));
                }
                one(reshape_val(v, shape))
            }
            "Transpose" => {
                let v = input(0)?;
                let r = v.shape().len();
                let perm: Vec<usize> = match n.attr_ints("perm") {
                    Some(p) => p.iter().map(|&a| axis(a, r)).collect::<Result<_>>()?,
                    None => (0..r).rev().collect(),
                };
                one(match v {
                    Val::F(a) => {
                        let (d, s) = transpose_any(&a.data, &a.shape, &perm);
                        Val::F(Arr::new(d, s))
                    }
                    Val::I(d, s) => {
                        let (d, s) = transpose_any(d, s, &perm);
                        Val::I(d, s)
                    }
                })
            }
            "Squeeze" | "Unsqueeze" => {
                let v = input(0)?;
                let axes: Vec<i64> = if self.opset >= 13 {
                    opt(1).map(|a| a.to_i().0).unwrap_or_default()
                } else {
                    n.attr_ints("axes").map(<[i64]>::to_vec).unwrap_or_default()
                };
                let mut shape = v.shape().to_vec();
                if op == "Squeeze" {
                    let r = shape.len();
                    let drop: Vec<usize> = if axes.is_empty() {
                        (0..r).filter(|&k| shape[k] == 1).collect()
                    } else {
                        axes.iter().map(|&a| axis(a, r)).collect::<Result<_>>()?
                    };
                    shape = shape
                        .iter()
                        .enumerate()
                        .filter(|(k, _)| !drop.contains(k))
                        .map(|(_, &d)| d)
                        .collect();
                } else {
                    let out_rank = shape.len() + axes.len();
                    let mut ax: Vec<usize> = axes
                        .iter()
                        .map(|&a| axis(a, out_rank))
                        .collect::<Result<_>>()?;
                    ax.sort_unstable();
                    for a in ax {
                        shape.insert(a, 1);
                    }
                }
                one(reshape_val(v, shape))
            }
            "Concat" => {
                let parts: Vec<&Val> = (0..n.inputs.len())
                    .filter(|&k| !n.inputs[k].is_empty())
                    .map(input)
                    .collect::<Result<_>>()?;
                let r = parts.first().map_or(0, |p| p.shape().len());
                let ax = axis(n.attr_i("axis", 0), r)?;
                if parts.iter().all(|p| matches!(p, Val::I(..))) {
                    let arrs: Vec<(Vec<i64>, Vec<usize>)> =
                        parts.iter().map(|p| p.to_i()).collect();
                    let (d, s) = concat(&arrs, ax)?;
                    one(Val::I(d, s))
                } else {
                    let arrs: Vec<(Vec<f32>, Vec<usize>)> = parts
                        .iter()
                        .map(|p| {
                            let a = p.to_f();
                            (a.data, a.shape)
                        })
                        .collect();
                    let (d, s) = concat(&arrs, ax)?;
                    one(Val::F(Arr::new(d, s)))
                }
            }
            "Split" => {
                let v = input(0)?;
                let s = v.shape().to_vec();
                let ax = axis(n.attr_i("axis", 0), s.len())?;
                let outs = n.outputs.len();
                let sizes: Vec<usize> = match opt(1)
                    .map(|x| x.to_i().0)
                    .or_else(|| n.attr_ints("split").map(<[i64]>::to_vec))
                {
                    Some(sz) => sz.iter().map(|&x| x as usize).collect(),
                    None => {
                        let base = s[ax].div_ceil(outs.max(1));
                        (0..outs)
                            .map(|k| base.min(s[ax].saturating_sub(k * base)))
                            .collect()
                    }
                };
                let mut start = 0;
                sizes
                    .iter()
                    .map(|&len| {
                        let r = slice_val(v, ax, start, start + len, 1);
                        start += len;
                        r
                    })
                    .collect()
            }
            "Slice" => {
                let v = input(0)?;
                let s = v.shape().to_vec();
                let (starts, ends, axes, steps) = if self.opset >= 10 {
                    (
                        ints(1)?,
                        ints(2)?,
                        opt(3).map(|a| a.to_i().0),
                        opt(4).map(|a| a.to_i().0),
                    )
                } else {
                    (
                        n.attr_ints("starts")
                            .map(<[i64]>::to_vec)
                            .unwrap_or_default(),
                        n.attr_ints("ends").map(<[i64]>::to_vec).unwrap_or_default(),
                        n.attr_ints("axes").map(<[i64]>::to_vec),
                        None,
                    )
                };
                let axes = axes.unwrap_or_else(|| (0..starts.len() as i64).collect());
                let mut cur = v.clone();
                for k in 0..starts.len() {
                    let ax = axis(axes[k], s.len())?;
                    let step = steps.as_ref().map_or(1, |st| st[k]);
                    if step <= 0 {
                        return Err(unsupported(n, "non-positive steps"));
                    }
                    let dim = s[ax] as i64;
                    let clamp = |x: i64| (if x < 0 { x + dim } else { x }).clamp(0, dim) as usize;
                    cur = slice_val(&cur, ax, clamp(starts[k]), clamp(ends[k]), step as usize)?;
                }
                one(cur)
            }
            "Gather" => {
                let data = input(0)?;
                let (idx_v, idx_s) = input(1)?.to_i();
                let s = data.shape().to_vec();
                let ax = axis(n.attr_i("axis", 0), s.len())?;
                let outer: usize = s[..ax].iter().product();
                let inner: usize = s[ax + 1..].iter().product();
                let dim = s[ax] as i64;
                let mut out_shape = s[..ax].to_vec();
                out_shape.extend(&idx_s);
                out_shape.extend(&s[ax + 1..]);
                let positions: Vec<usize> = idx_v
                    .iter()
                    .map(|&i| {
                        let j = if i < 0 { i + dim } else { i };
                        if (0..dim).contains(&j) {
                            Ok(j as usize)
                        } else {
                            Err(bad(n, format!("index {i} out of range")))
                        }
                    })
                    .collect::<Result<_>>()?;
                let gather = || -> Vec<usize> {
                    let mut order = Vec::with_capacity(outer * positions.len() * inner);
                    for o in 0..outer {
                        for &p in &positions {
                            for i in 0..inner {
                                order.push((o * s[ax] + p) * inner + i);
                            }
                        }
                    }
                    order
                };
                one(match data {
                    Val::F(a) => Val::F(Arr::new(
                        gather().into_iter().map(|k| a.data[k]).collect(),
                        out_shape,
                    )),
                    Val::I(d, _) => Val::I(gather().into_iter().map(|k| d[k]).collect(), out_shape),
                })
            }
            "Expand" => {
                let v = input(0)?;
                let target: Vec<usize> = ints(1)?.iter().map(|&d| d.max(0) as usize).collect();
                let shape = broadcast_shape(v.shape(), &target)?;
                let map = broadcast_index(&shape, v.shape());
                let total: usize = shape.iter().product();
                one(match v {
                    Val::F(a) => Val::F(Arr::new(
                        (0..total).map(|i| a.data[map(i)]).collect(),
                        shape,
                    )),
                    Val::I(d, _) => Val::I((0..total).map(|i| d[map(i)]).collect(), shape),
                })
            }
            "Shape" => {
                let s = input(0)?.shape();
                let r = s.len() as i64;
                let norm = |x: i64| (if x < 0 { x + r } else { x }).clamp(0, r) as usize;
                let start = norm(n.attr_i("start", 0));
                let end = norm(n.attr_i("end", r));
                let d: Vec<i64> = s[start..end.max(start)].iter().map(|&x| x as i64).collect();
                let len = d.len();
                one(Val::I(d, vec![len]))
            }
            "Constant" => {
                if let Some(t) = n.attr("value").and_then(|a| a.t.as_ref()) {
                    return one(tensor_to_val(t)?);
                }
                if let Some(v) = n.attr("value_float").and_then(|a| a.f) {
                    return one(Val::F(Arr::new(vec![v], vec![])));
                }
                if let Some(v) = n.attr("value_floats") {
                    return one(Val::F(Arr::new(v.floats.clone(), vec![v.floats.len()])));
                }
                if let Some(v) = n.attr("value_int").and_then(|a| a.i) {
                    return one(Val::I(vec![v], vec![]));
                }
                if let Some(v) = n.attr("value_ints") {
                    return one(Val::I(v.ints.clone(), vec![v.ints.len()]));
                }
                Err(unsupported(n, "this kind of constant"))
            }
            "ConstantOfShape" => {
                let shape: Vec<usize> = ints(0)?.iter().map(|&d| d.max(0) as usize).collect();
                let total = shape.iter().product();
                match n.attr("value").and_then(|a| a.t.as_ref()) {
                    Some(t) if t.is_integer() => {
                        one(Val::I(vec![t.to_f64()?[0] as i64; total], shape))
                    }
                    Some(t) => one(Val::F(Arr::new(vec![t.to_f64()?[0] as f32; total], shape))),
                    None => one(Val::F(Arr::new(vec![0.0; total], shape))),
                }
            }
            "Range" => {
                let (st, li, de) = (input(0)?, input(1)?, input(2)?);
                if let (Val::I(a, _), Val::I(b, _), Val::I(c, _)) = (st, li, de) {
                    let (a, b, c) = (a[0], b[0], c[0]);
                    if c == 0 {
                        return Err(bad(n, "delta is 0"));
                    }
                    let d: Vec<i64> = std::iter::successors(Some(a), |&x| Some(x + c))
                        .take_while(|&x| if c > 0 { x < b } else { x > b })
                        .collect();
                    let len = d.len();
                    return one(Val::I(d, vec![len]));
                }
                let (a, b, c) = (st.to_f().data[0], li.to_f().data[0], de.to_f().data[0]);
                let count = ((b - a) / c).ceil().max(0.0) as usize;
                one(Val::F(Arr::new(
                    (0..count).map(|k| a + k as f32 * c).collect(),
                    vec![count],
                )))
            }
            "Cast" => {
                let to = n.attr_i("to", dtype::FLOAT as i64) as i32;
                one(match to {
                    dtype::FLOAT | dtype::DOUBLE | dtype::FLOAT16 | dtype::BFLOAT16 => {
                        Val::F(input(0)?.to_f())
                    }
                    dtype::BOOL => {
                        let a = input(0)?.to_f();
                        Val::I(
                            a.data.iter().map(|&v| i64::from(v != 0.0)).collect(),
                            a.shape,
                        )
                    }
                    _ => {
                        let a = input(0)?.to_f();
                        Val::I(a.data.iter().map(|&v| v.trunc() as i64).collect(), a.shape)
                    }
                })
            }
            "ReduceMean" | "ReduceSum" | "ReduceMax" | "ReduceMin" => {
                let x = f(0)?;
                let axes_from_input = (op == "ReduceSum" && self.opset >= 13) || self.opset >= 18;
                let axes: Vec<i64> = if axes_from_input {
                    opt(1).map(|a| a.to_i().0).unwrap_or_default()
                } else {
                    n.attr_ints("axes").map(<[i64]>::to_vec).unwrap_or_default()
                };
                let r = x.shape.len();
                let axes: Vec<usize> = if axes.is_empty() {
                    if n.attr_i("noop_with_empty_axes", 0) == 1 {
                        return one(Val::F(x));
                    }
                    (0..r).collect()
                } else {
                    axes.iter().map(|&a| axis(a, r)).collect::<Result<_>>()?
                };
                one(Val::F(reduce(&x, &axes, n.attr_i("keepdims", 1) == 1, op)))
            }
            other => Err(unsupported(
                n,
                format!("{other} is listed but not implemented"),
            )),
        }
    }

    /// Batched matmul with numpy broadcasting; every product runs on the NPU.
    fn matmul_general(&self, a: &Arr, b: &Arr) -> Result<Arr> {
        let a_vec = a.shape.len() == 1;
        let b_vec = b.shape.len() == 1;
        let sa = if a_vec {
            vec![1, a.shape[0]]
        } else {
            a.shape.clone()
        };
        let sb = if b_vec {
            vec![b.shape[0], 1]
        } else {
            b.shape.clone()
        };
        let (m, k) = (sa[sa.len() - 2], sa[sa.len() - 1]);
        let (k2, nn) = (sb[sb.len() - 2], sb[sb.len() - 1]);
        if k != k2 {
            return Err(Error::Shape(format!(
                "MatMul: {:?} @ {:?}",
                a.shape, b.shape
            )));
        }
        let batch = broadcast_shape(&sa[..sa.len() - 2], &sb[..sb.len() - 2])?;
        let nb: usize = batch.iter().product();
        let ia = broadcast_index(&batch, &sa[..sa.len() - 2]);
        let ib = broadcast_index(&batch, &sb[..sb.len() - 2]);
        let mut out = Vec::with_capacity(nb * m * nn);
        for bi in 0..nb {
            let am = &a.data[ia(bi) * m * k..][..m * k];
            let bm = &b.data[ib(bi) * k * nn..][..k * nn];
            let (bt, _) = transpose_any(bm, &[k, nn], &[1, 0]);
            let y = ops::matvec(
                &self.npu,
                &Tensor::from_f32(&bt, [nn, k])?,
                &Tensor::from_f32(am, [m, k])?,
            )?;
            out.extend(y.to_f32_vec());
        }
        let mut shape = batch;
        if !a_vec {
            shape.push(m);
        }
        if !b_vec {
            shape.push(nn);
        }
        Ok(Arr::new(out, shape))
    }

    fn conv(&self, idx: usize, n: &Node, x: &Arr, bias: Option<Arr>) -> Result<Arr> {
        let one_d = x.shape.len() == 3;
        let x4 = to_4d_input(x, n)?;
        let w_shape = match self.resident.get(&idx) {
            Some(Resident::Conv(w)) => w.shape().to_vec(),
            _ => return Err(unsupported(n, "weights that are not constants in the file")),
        };
        let (kh, kw) = (w_shape[2], w_shape[3]);
        let (strides, dil) = (
            attr_pair(n, "strides", one_d)?,
            attr_pair(n, "dilations", one_d)?,
        );
        let (h, w) = (x4.shape[2], x4.shape[3]);
        let pads = conv_pads(n, one_d, [h, w], [kh, kw], strides, dil)?;
        let (input, padding) = apply_pads(&x4, pads)?;
        let params = Conv2dParams::new()
            .stride((strides[0], strides[1]))
            .dilation((dil[0], dil[1]))
            .padding(padding)
            .groups(n.attr_i("group", 1) as usize);
        let Some(Resident::Conv(weights)) = self.resident.get(&idx) else {
            return Err(unsupported(n, "weights that are not constants in the file"));
        };
        let mut y = weights.forward(&input.tensor()?, &params)?;
        if let Some(b) = bias {
            y = ops::add_channel_bias(&y, &b.tensor()?)?;
        }
        let mut out = Arr::from_tensor(&y);
        if one_d {
            out.shape.remove(2);
        }
        Ok(out)
    }

    fn conv_transpose(&self, idx: usize, n: &Node, x: &Arr, bias: Option<Arr>) -> Result<Arr> {
        if n.attr("output_shape").is_some() {
            return Err(unsupported(n, "the output_shape attribute"));
        }
        let one_d = x.shape.len() == 3;
        let x4 = to_4d_input(x, n)?;
        let Some(Resident::ConvT(weights)) = self.resident.get(&idx) else {
            return Err(unsupported(n, "weights that are not constants in the file"));
        };
        let pads = n.attr_ints("pads").map(<[i64]>::to_vec).unwrap_or_default();
        let pads: Vec<usize> = if one_d && pads.len() == 2 {
            vec![0, pads[0] as usize, 0, pads[1] as usize]
        } else {
            pads.iter().map(|&p| p as usize).collect()
        };
        if pads.len() == 4 && (pads[0] != pads[2] || pads[1] != pads[3]) {
            return Err(unsupported(n, "asymmetric pads"));
        }
        let pad = if pads.len() == 4 {
            (pads[0], pads[1])
        } else {
            (0, 0)
        };
        let op = attr_pair(n, "output_padding", one_d).unwrap_or([0, 0]);
        let op = if n.attr("output_padding").is_none() {
            [0, 0]
        } else {
            op
        };
        let s = attr_pair(n, "strides", one_d)?;
        let d = attr_pair(n, "dilations", one_d)?;
        let params = ConvTranspose2dParams::new()
            .stride((s[0], s[1]))
            .padding(pad)
            .output_padding((op[0], op[1]))
            .dilation((d[0], d[1]))
            .groups(n.attr_i("group", 1) as usize);
        let mut y = weights.forward(&x4.tensor()?, &params)?;
        if let Some(b) = bias {
            y = ops::add_channel_bias(&y, &b.tensor()?)?;
        }
        let mut out = Arr::from_tensor(&y);
        if one_d {
            out.shape.remove(2);
        }
        Ok(out)
    }
}

impl Module for OnnxModel {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.run_single(x)
    }
    fn name(&self) -> &'static str {
        "OnnxModel"
    }
}

fn reshape_val(v: &Val, shape: Vec<usize>) -> Val {
    match v {
        Val::F(a) => Val::F(Arr::new(a.data.clone(), shape)),
        Val::I(d, _) => Val::I(d.clone(), shape),
    }
}

fn concat<T: Copy>(parts: &[(Vec<T>, Vec<usize>)], ax: usize) -> Result<(Vec<T>, Vec<usize>)> {
    let first = &parts[0].1;
    let outer: usize = first[..ax].iter().product();
    let mut shape = first.clone();
    shape[ax] = parts.iter().map(|p| p.1[ax]).sum();
    for p in parts {
        if p.1.len() != first.len() || p.1[..ax] != first[..ax] || p.1[ax + 1..] != first[ax + 1..]
        {
            return Err(Error::Shape(format!("Concat: {:?} vs {first:?}", p.1)));
        }
    }
    let mut out = Vec::with_capacity(shape.iter().product());
    for o in 0..outer {
        for (data, s) in parts {
            let chunk: usize = s[ax..].iter().product();
            out.extend_from_slice(&data[o * chunk..(o + 1) * chunk]);
        }
    }
    Ok((out, shape))
}

fn slice_val(v: &Val, ax: usize, start: usize, end: usize, step: usize) -> Result<Val> {
    let s = v.shape().to_vec();
    let outer: usize = s[..ax].iter().product();
    let inner: usize = s[ax + 1..].iter().product();
    let picks: Vec<usize> = (start..end.max(start)).step_by(step.max(1)).collect();
    let mut shape = s.clone();
    shape[ax] = picks.len();
    let dim = s[ax];
    let order: Vec<usize> = (0..outer)
        .flat_map(|o| {
            picks
                .iter()
                .flat_map(move |&p| (0..inner).map(move |i| (o * dim + p) * inner + i))
        })
        .collect();
    Ok(match v {
        Val::F(a) => Val::F(Arr::new(order.iter().map(|&k| a.data[k]).collect(), shape)),
        Val::I(d, _) => Val::I(order.iter().map(|&k| d[k]).collect(), shape),
    })
}

fn reduce(x: &Arr, axes: &[usize], keep: bool, op: &str) -> Arr {
    let r = x.shape.len();
    let out_full: Vec<usize> = (0..r)
        .map(|d| if axes.contains(&d) { 1 } else { x.shape[d] })
        .collect();
    let total: usize = out_full.iter().product();
    let init = match op {
        "ReduceMax" => f32::NEG_INFINITY,
        "ReduceMin" => f32::INFINITY,
        _ => 0.0,
    };
    let mut acc = vec![init; total];
    let map = broadcast_index(&x.shape, &out_full);
    for (i, &v) in x.data.iter().enumerate() {
        let o = map(i);
        acc[o] = match op {
            "ReduceMax" => acc[o].max(v),
            "ReduceMin" => acc[o].min(v),
            _ => acc[o] + v,
        };
    }
    if op == "ReduceMean" {
        let count = (x.data.len() / total.max(1)) as f32;
        acc.iter_mut().for_each(|v| *v /= count);
    }
    let shape = if keep {
        out_full
    } else {
        (0..r)
            .filter(|d| !axes.contains(d))
            .map(|d| x.shape[d])
            .collect()
    };
    Arr::new(acc, shape)
}

/// `[OC, C, K]` → `[OC, C, 1, K]`; 4-D kernels unchanged.
fn to_4d_kernel(w: &Arr) -> Result<Arr> {
    match w.shape.len() {
        4 => Ok(w.clone()),
        3 => Ok(Arr::new(
            w.data.clone(),
            vec![w.shape[0], w.shape[1], 1, w.shape[2]],
        )),
        _ => Err(Error::Unsupported(format!(
            "convolution kernels of rank {} (only 1-D and 2-D)",
            w.shape.len()
        ))),
    }
}

fn to_4d_input(x: &Arr, n: &Node) -> Result<Arr> {
    match x.shape.len() {
        4 => Ok(x.clone()),
        3 => Ok(Arr::new(
            x.data.clone(),
            vec![x.shape[0], x.shape[1], 1, x.shape[2]],
        )),
        r => Err(unsupported(
            n,
            format!("inputs of rank {r} (only 1-D and 2-D convolutions)"),
        )),
    }
}

/// `(h, w)` from a 1- or 2-element attribute (default 1).
fn attr_pair(n: &Node, name: &str, one_d: bool) -> Result<[usize; 2]> {
    match n.attr_ints(name) {
        None => Ok([1, 1]),
        Some([v]) if one_d => Ok([1, *v as usize]),
        Some([a, b]) => Ok([*a as usize, *b as usize]),
        Some(v) => Err(unsupported(n, format!("{name}={v:?}"))),
    }
}

/// `[top, left, bottom, right]`, resolving `auto_pad`.
fn conv_pads(
    n: &Node,
    one_d: bool,
    input: [usize; 2],
    kernel: [usize; 2],
    s: [usize; 2],
    d: [usize; 2],
) -> Result<[usize; 4]> {
    match n.attr_s("auto_pad").unwrap_or("NOTSET") {
        "NOTSET" | "" => match n.attr_ints("pads") {
            None => Ok([0; 4]),
            Some([b, e]) if one_d => Ok([0, *b as usize, 0, *e as usize]),
            Some([t, l, b, r]) => Ok([*t as usize, *l as usize, *b as usize, *r as usize]),
            Some(p) => Err(unsupported(n, format!("pads={p:?}"))),
        },
        "VALID" => Ok([0; 4]),
        mode @ ("SAME_UPPER" | "SAME_LOWER") => {
            let total = |i: usize| {
                let out = input[i].div_ceil(s[i]);
                ((out - 1) * s[i] + (kernel[i] - 1) * d[i] + 1).saturating_sub(input[i])
            };
            let split = |t: usize| {
                if mode == "SAME_UPPER" {
                    (t / 2, t - t / 2)
                } else {
                    (t - t / 2, t / 2)
                }
            };
            let (t, b) = split(total(0));
            let (l, r) = split(total(1));
            Ok([t, l, b, r])
        }
        other => Err(unsupported(n, format!("auto_pad={other}"))),
    }
}

/// Symmetric pads become conv padding; asymmetric ones are applied to the input.
fn apply_pads(x: &Arr, p: [usize; 4]) -> Result<(Arr, (usize, usize))> {
    if p[0] == p[2] && p[1] == p[3] {
        return Ok((x.clone(), (p[0], p[1])));
    }
    let [nb, c, h, w] = [x.shape[0], x.shape[1], x.shape[2], x.shape[3]];
    let (nh, nw) = (h + p[0] + p[2], w + p[1] + p[3]);
    let mut out = vec![0.0f32; nb * c * nh * nw];
    for plane in 0..nb * c {
        for y in 0..h {
            let src = &x.data[(plane * h + y) * w..][..w];
            out[(plane * nh + y + p[0]) * nw + p[1]..][..w].copy_from_slice(src);
        }
    }
    Ok((Arr::new(out, vec![nb, c, nh, nw]), (0, 0)))
}

/// ONNX MaxPool / AveragePool with (possibly asymmetric) padding.
fn pool(n: &Node, x: &Arr, max: bool) -> Result<Arr> {
    let one_d = x.shape.len() == 3;
    let x4 = to_4d_input(x, n)?;
    let k = attr_pair(n, "kernel_shape", one_d)?;
    let s = attr_pair(n, "strides", one_d)?;
    let [nb, c, h, w] = [x4.shape[0], x4.shape[1], x4.shape[2], x4.shape[3]];
    let p = conv_pads(n, one_d, [h, w], k, s, [1, 1])?;
    let include_pad = n.attr_i("count_include_pad", 0) == 1;
    let oh = (h + p[0] + p[2]).saturating_sub(k[0]) / s[0] + 1;
    let ow = (w + p[1] + p[3]).saturating_sub(k[1]) / s[1] + 1;
    let mut out = Vec::with_capacity(nb * c * oh * ow);
    for plane in 0..nb * c {
        let src = &x4.data[plane * h * w..][..h * w];
        for oy in 0..oh {
            for ox in 0..ow {
                let y0 = (oy * s[0]) as isize - p[0] as isize;
                let x0 = (ox * s[1]) as isize - p[1] as isize;
                let (mut acc, mut cnt) = (if max { f32::NEG_INFINITY } else { 0.0 }, 0usize);
                let mut window = 0usize;
                for ky in 0..k[0] as isize {
                    for kx in 0..k[1] as isize {
                        let (yy, xx) = (y0 + ky, x0 + kx);
                        // Count positions inside the padded input for count_include_pad.
                        if yy < (h + p[2]) as isize && xx < (w + p[3]) as isize {
                            window += 1;
                        }
                        if yy >= 0 && xx >= 0 && (yy as usize) < h && (xx as usize) < w {
                            let v = src[yy as usize * w + xx as usize];
                            acc = if max { acc.max(v) } else { acc + v };
                            cnt += 1;
                        }
                    }
                }
                out.push(if max {
                    acc
                } else {
                    acc / (if include_pad { window } else { cnt }).max(1) as f32
                });
            }
        }
    }
    let shape = if one_d {
        vec![nb, c, ow]
    } else {
        vec![nb, c, oh, ow]
    };
    Ok(Arr::new(out, shape))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn broadcasting() {
        let (d, s) = binary(&[1.0f32, 2.0, 3.0], &[3], &[10.0, 20.0], &[2, 1], |a, b| {
            a + b
        })
        .unwrap();
        assert_eq!(s, vec![2, 3]);
        assert_eq!(d, vec![11.0, 12.0, 13.0, 21.0, 22.0, 23.0]);
        assert!(broadcast_shape(&[2, 3], &[4]).is_err());
    }

    #[test]
    fn transpose_3d() {
        let data: Vec<i32> = (0..24).collect();
        let (t, s) = transpose_any(&data, &[2, 3, 4], &[2, 0, 1]);
        assert_eq!(s, vec![4, 2, 3]);
        assert_eq!(t[1], 4); // element [0][0][1] of the output = input [0][1][0]
    }

    #[test]
    fn erf_accuracy() {
        for (x, want) in [
            (0.0f32, 0.0f32),
            (0.5, 0.520_499_9),
            (1.0, 0.842_700_8),
            (-2.0, -0.995_322_3),
        ] {
            assert!((erf(x) - want).abs() < 1e-6);
        }
    }
}
