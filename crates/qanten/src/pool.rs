//! Several NPUs working on one batch.
//!
//! An [`NpuPool`] holds one handle per device. [`NpuPool::replicate`] builds a
//! copy of a model on every NPU (each with its own resident weights); the
//! result is itself a [`Module`] that splits each batch evenly across the
//! devices, runs the parts in parallel and concatenates the outputs:
//!
//! ```
//! # fn main() -> qanten::Result<()> {
//! use qanten::nn::{self, Module};
//! use qanten::pool::NpuPool;
//! use qanten::Tensor;
//!
//! let pool = NpuPool::simulated(2)?; // or NpuPool::open_all()? on a multi-card machine
//! let w = Tensor::rand_uniform([32, 64], -0.1, 0.1, 1);
//! let model = pool.replicate(|npu| nn::Linear::new(npu, &w, None))?;
//! let y = model.forward(&Tensor::rand_uniform([100, 64], 0., 1., 2))?; // 50 rows per NPU
//! assert_eq!(y.shape(), &[100, 32]);
//! # Ok(()) }
//! ```

use std::sync::Arc;

use crate::backend::{SimConfig, Simulator};
use crate::device::Npu;
use crate::error::{Error, Result};
use crate::nn::Module;
use crate::tensor::Tensor;

/// A set of NPUs used together.
#[derive(Debug, Clone)]
pub struct NpuPool {
    npus: Vec<Npu>,
}

impl NpuPool {
    /// A pool of the given handles (at least one).
    pub fn new(npus: Vec<Npu>) -> Result<Self> {
        if npus.is_empty() {
            return Err(Error::InvalidArgument(
                "an NPU pool needs at least one NPU".into(),
            ));
        }
        Ok(Self { npus })
    }

    /// Opens every NPU the Q.ANT driver reports.
    pub fn open_all() -> Result<Self> {
        let npus = crate::devices()?
            .into_iter()
            .map(|d| Npu::open(d.id))
            .collect::<Result<Vec<_>>>()?;
        Self::new(npus)
    }

    /// `n` NPUs of one shared simulator, for development and tests.
    pub fn simulated(n: u32) -> Result<Self> {
        let sim = Arc::new(Simulator::new(SimConfig {
            num_npus: n.max(1),
            ..SimConfig::default()
        }));
        Self::new(
            (0..n.max(1))
                .map(|id| Npu::simulated(&sim, id))
                .collect::<Result<_>>()?,
        )
    }

    /// Number of NPUs.
    pub fn len(&self) -> usize {
        self.npus.len()
    }

    /// Always `false`: a pool has at least one NPU.
    pub fn is_empty(&self) -> bool {
        self.npus.is_empty()
    }

    /// The NPU handles.
    pub fn npus(&self) -> &[Npu] {
        &self.npus
    }

    /// Splits `x` along its first dimension into one part per NPU, runs
    /// `f(npu, part)` for all parts in parallel, and concatenates the results.
    /// Batches smaller than the pool use fewer NPUs.
    pub fn split_batch<F>(&self, x: &Tensor, f: F) -> Result<Tensor>
    where
        F: Fn(usize, &Npu, &Tensor) -> Result<Tensor> + Sync,
    {
        let parts = x.split_batch(self.npus.len());
        if parts.len() == 1 {
            return f(0, &self.npus[0], &parts[0]);
        }
        let results: Vec<Result<Tensor>> = std::thread::scope(|s| {
            let handles: Vec<_> = parts
                .iter()
                .enumerate()
                .map(|(i, part)| {
                    let npu = &self.npus[i];
                    let f = &f;
                    s.spawn(move || f(i, npu, part))
                })
                .collect();
            handles
                .into_iter()
                .map(|h| {
                    h.join().unwrap_or_else(|_| {
                        Err(Error::Unsupported("worker thread panicked".into()))
                    })
                })
                .collect()
        });
        Tensor::concat_batch(&results.into_iter().collect::<Result<Vec<_>>>()?)
    }

    /// Builds one copy of a model per NPU with `build(npu)`.
    pub fn replicate<M, F>(&self, build: F) -> Result<Replicated<M>>
    where
        M: Module,
        F: Fn(&Npu) -> Result<M>,
    {
        Ok(Replicated {
            pool: self.clone(),
            copies: self.npus.iter().map(build).collect::<Result<_>>()?,
        })
    }
}

/// One copy of a model per NPU; `forward` splits the batch across them.
#[derive(Debug)]
pub struct Replicated<M> {
    pool: NpuPool,
    copies: Vec<M>,
}

impl<M> Replicated<M> {
    /// The per-NPU copies, in pool order.
    pub fn copies(&self) -> &[M] {
        &self.copies
    }

    /// The pool the copies live on.
    pub fn pool(&self) -> &NpuPool {
        &self.pool
    }
}

impl<M: Module> Module for Replicated<M> {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.pool
            .split_batch(x, |i, _, part| self.copies[i].forward(part))
    }

    fn name(&self) -> &'static str {
        "Replicated"
    }
}
