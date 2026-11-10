//! Static check that a graph treats the rows of its input independently,
//! which is what makes it safe to batch requests from different callers.

use std::collections::HashMap;

use super::proto::Node;
use super::{OnnxModel, Val, constant_value};

/// What the check knows about a value.
#[derive(Clone, Copy)]
enum Kind<'a> {
    /// Does not depend on the input. The value is known for initializers and
    /// `Constant` nodes, unknown for values computed from them.
    Const(Option<&'a Val>),
    /// Row `i` along axis 0 depends only on row `i` of the input; the rank.
    Batched(usize),
}

impl OnnxModel {
    /// Whether every output row depends only on the matching row of the
    /// (single, float) input, so that concatenating inputs from different
    /// callers along axis 0 and splitting the outputs gives each caller
    /// exactly what a run on its own input would.
    ///
    /// The check is conservative: it accepts the row-wise ops (elementwise
    /// math with constants that do not vary along the batch axis, `Gemm`,
    /// `MatMul` by a constant, convolutions, pooling, normalisation and
    /// softmax over non-batch axes, and reshapes that keep axis 0) and
    /// rejects anything that could mix rows, such as `Shape`, reductions or
    /// transposes over axis 0, and reshapes that move the batch axis.
    pub fn rows_are_independent(&self) -> bool {
        let [input] = self.inputs.as_slice() else {
            return false;
        };
        let rank = input.shape.len();
        if rank == 0 || input.is_integer {
            return false;
        }
        let node_consts: HashMap<&str, Val> = self
            .nodes
            .iter()
            .filter(|n| n.op_type == "Constant")
            .filter_map(|n| Some((n.outputs.first()?.as_str(), constant_value(n).ok()?)))
            .collect();
        let mut kinds: HashMap<&str, Kind<'_>> = self
            .consts
            .iter()
            .map(|(k, v)| (k.as_str(), v))
            .chain(node_consts.iter().map(|(&k, v)| (k, v)))
            .map(|(k, v)| (k, Kind::Const(Some(v))))
            .collect();
        kinds.insert(&input.name, Kind::Batched(rank));

        for n in self.nodes.iter().filter(|n| n.op_type != "Constant") {
            let mut ins = Vec::with_capacity(n.inputs.len());
            for name in &n.inputs {
                ins.push(match name.as_str() {
                    "" => None,
                    name => match kinds.get(name) {
                        Some(&k) => Some(k),
                        None => return false, // used before it is computed
                    },
                });
            }
            let outs: Vec<Kind<'_>> = if ins.iter().flatten().all(|k| matches!(k, Kind::Const(_))) {
                vec![Kind::Const(None); n.outputs.len()]
            } else {
                match RowWise::new(self.opset, n, &ins).out_ranks() {
                    Some(ranks) => ranks.into_iter().map(Kind::Batched).collect(),
                    None => return false,
                }
            };
            for (name, k) in n.outputs.iter().zip(outs) {
                kinds.insert(name, k);
            }
        }
        self.outputs
            .iter()
            .all(|o| matches!(kinds.get(o.as_str()), Some(Kind::Batched(_))))
    }
}

/// One node whose inputs include a batched value.
struct RowWise<'n, 'a> {
    opset: i64,
    node: &'n Node,
    ins: &'n [Option<Kind<'a>>],
}

impl<'n, 'a> RowWise<'n, 'a> {
    fn new(opset: i64, node: &'n Node, ins: &'n [Option<Kind<'a>>]) -> Self {
        Self { opset, node, ins }
    }

    fn input(&self, k: usize) -> Option<Kind<'a>> {
        self.ins.get(k).copied().flatten()
    }

    /// Rank of input `k` if it is batched.
    fn batched(&self, k: usize) -> Option<usize> {
        match self.input(k)? {
            Kind::Batched(r) => Some(r),
            Kind::Const(_) => None,
        }
    }

    /// Integer contents of constant input `k`.
    fn const_ints(&self, k: usize) -> Option<Vec<i64>> {
        match self.input(k)? {
            Kind::Const(Some(v)) => Some(v.to_i().0),
            _ => None,
        }
    }

    /// Whether every input other than the first is a constant.
    fn rest_const(&self) -> bool {
        self.ins
            .iter()
            .skip(1)
            .flatten()
            .all(|k| matches!(k, Kind::Const(_)))
    }

    /// The axes from input `k` (newer opsets) or the `axes` attribute.
    fn axes(&self, k: usize, from_opset: i64) -> Option<Vec<i64>> {
        if self.opset >= from_opset {
            self.const_ints(k)
        } else {
            self.node.attr_ints("axes").map(<[i64]>::to_vec)
        }
    }

    /// Whether none of `axes` (for a value of rank `rank`) is the batch axis.
    fn avoids_batch_axis(axes: &[i64], rank: usize) -> bool {
        !axes.is_empty() && axes.iter().all(|&a| a != 0 && a != -(rank as i64))
    }

    /// Ranks of the outputs if the node is row-wise, `None` if it might mix rows.
    fn out_ranks(&self) -> Option<Vec<usize>> {
        let n = self.node;
        let outs = n.outputs.len();
        let axis_ok = |a: i64, rank: usize| Self::avoids_batch_axis(&[a], rank);
        let r = self.batched(0);
        match n.op_type.as_str() {
            "Abs" | "Add" | "Cast" | "Ceil" | "Clip" | "Cos" | "Div" | "Dropout" | "Elu"
            | "Equal" | "Erf" | "Exp" | "Floor" | "Gelu" | "Greater" | "HardSigmoid"
            | "Identity" | "LeakyRelu" | "Less" | "Log" | "Max" | "Mean" | "Min" | "Mul"
            | "Neg" | "Not" | "Pow" | "Reciprocal" | "Relu" | "Sigmoid" | "Sin" | "Softplus"
            | "Sqrt" | "Sub" | "Sum" | "Tanh" | "Where" => {
                self.elementwise().map(|r| vec![r; outs])
            }
            "Gemm" => {
                let bias_ok = match self.input(2) {
                    None => true,
                    Some(Kind::Const(Some(c))) => c.shape().len() < 2 || c.shape()[0] == 1,
                    Some(_) => false,
                };
                (r == Some(2)
                    && n.attr_i("transA", 0) == 0
                    && matches!(self.input(1), Some(Kind::Const(_)))
                    && bias_ok)
                    .then(|| vec![2])
            }
            "MatMul" => {
                let b_is_matrix =
                    matches!(self.input(1), Some(Kind::Const(Some(b))) if b.shape().len() == 2);
                r.filter(|&r| r >= 2 && b_is_matrix).map(|r| vec![r])
            }
            "Conv" | "ConvTranspose" | "MaxPool" | "AveragePool" | "GlobalAveragePool"
            | "GlobalMaxPool" | "BatchNormalization" => {
                r.filter(|_| self.rest_const()).map(|r| vec![r; outs])
            }
            "LayerNormalization" => r
                .filter(|&r| self.rest_const() && axis_ok(n.attr_i("axis", -1), r))
                .map(|r| vec![r; outs]),
            "Softmax" | "LogSoftmax" => {
                let default = if self.opset >= 13 { -1 } else { 1 };
                r.filter(|&r| axis_ok(n.attr_i("axis", default), r))
                    .map(|r| vec![r])
            }
            "Flatten" => r
                .filter(|&r| {
                    let a = n.attr_i("axis", 1);
                    a == r as i64 || axis_ok(a, r)
                })
                .map(|_| vec![2]),
            "Transpose" => {
                let perm_keeps_batch = |r: usize| match n.attr_ints("perm") {
                    Some(p) => p.first() == Some(&0),
                    None => r == 1,
                };
                r.filter(|&r| perm_keeps_batch(r)).map(|r| vec![r])
            }
            "Reshape" => {
                let t = self.const_ints(1)?;
                // Keep axis 0 (`0` copies it), or infer it from the rest; a
                // different row size changes the first dimension, which the
                // caller checks at run time.
                let keeps_batch = match t.first() {
                    Some(0) => n.attr_i("allowzero", 0) == 0,
                    Some(-1) => t[1..].iter().all(|&d| d > 0),
                    _ => false,
                };
                (r.is_some() && keeps_batch).then(|| vec![t.len()])
            }
            "Squeeze" => {
                let (r, axes) = (r?, self.axes(1, 13)?);
                let out = r.checked_sub(axes.len())?;
                Self::avoids_batch_axis(&axes, r).then(|| vec![out])
            }
            "Unsqueeze" => {
                let (r, axes) = (r?, self.axes(1, 13)?);
                let out = r + axes.len();
                Self::avoids_batch_axis(&axes, out).then(|| vec![out])
            }
            "ReduceMax" | "ReduceMean" | "ReduceMin" | "ReduceSum" => {
                let from = if n.op_type == "ReduceSum" { 13 } else { 18 };
                let (r, axes) = (r?, self.axes(1, from)?);
                let keep = n.attr_i("keepdims", 1) == 1;
                Self::avoids_batch_axis(&axes, r).then(|| {
                    vec![if keep {
                        r
                    } else {
                        r.saturating_sub(axes.len())
                    }]
                })
            }
            "Slice" => {
                let r = r?;
                let axes = if self.opset >= 10 {
                    self.const_ints(3)?
                } else {
                    n.attr_ints("axes")?.to_vec()
                };
                (self.rest_const() && Self::avoids_batch_axis(&axes, r)).then(|| vec![r])
            }
            "Split" => r
                .filter(|&r| self.rest_const() && axis_ok(n.attr_i("axis", 0), r))
                .map(|r| vec![r; outs]),
            "Concat" => {
                let ranks: Vec<usize> = (0..self.ins.len())
                    .map(|k| self.batched(k))
                    .collect::<Option<_>>()?;
                let r = *ranks.first()?;
                (ranks.iter().all(|&x| x == r) && axis_ok(n.attr_i("axis", 0), r)).then(|| vec![r])
            }
            "Gather" => {
                let idx_rank = match self.input(1)? {
                    Kind::Const(Some(i)) => i.shape().len(),
                    _ => return None,
                };
                r.filter(|&r| axis_ok(n.attr_i("axis", 0), r))
                    .map(|r| vec![r - 1 + idx_rank])
            }
            _ => None,
        }
    }

    /// Elementwise ops with broadcasting: all batched operands must have the
    /// same rank, and constants must not vary along the batch axis.
    fn elementwise(&self) -> Option<usize> {
        let r = (0..self.ins.len()).find_map(|k| self.batched(k))?;
        self.ins
            .iter()
            .flatten()
            .all(|k| match *k {
                Kind::Batched(x) => x == r,
                Kind::Const(Some(c)) => c.shape().len() < r || c.shape().first() == Some(&1),
                Kind::Const(None) => false,
            })
            .then_some(r)
    }
}
