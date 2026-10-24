//! Automatic CPU fallback for small operations.
//!
//! Sending work to the NPU has a fixed cost (transfer setup, driver call).
//! For tiny streamed operations the host CPU is faster. A fallback handle
//! runs every streamed `mul`, `periodic_nl` and `matvec` below a size
//! threshold on the CPU (exactly like the simulator: f32 accumulation, bf16
//! results) and everything else on the NPU:
//!
//! ```
//! # fn main() -> qanten::Result<()> {
//! use qanten::fallback::FallbackPolicy;
//! use qanten::{Npu, Tensor, ops};
//! let (npu, monitor) = Npu::auto(0)?.with_cpu_fallback(FallbackPolicy::min_macs(1 << 16));
//! let small = ops::mul(&npu, &Tensor::ones([8]), &Tensor::ones([8]))?; // CPU
//! assert_eq!(small.to_f32_vec(), vec![1.0; 8]);
//! assert_eq!(monitor.stats().cpu_ops, 1);
//! # Ok(()) }
//! ```
//!
//! Operations on device-resident memory (`DeviceBuffer`, `LinearWeights`,
//! `nn` layers) always run on the NPU, since their data already lives there.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use half::bf16;

use crate::backend::{Backend, BackendKind, DataRole, NpuEntry, RawHandle, Simulator};
use crate::device::Npu;
use crate::error::Result;
use crate::info::{DeviceInfo, PerfCounters};

/// When to run a streamed operation on the CPU instead of the NPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FallbackPolicy {
    /// Operations with fewer multiply-accumulates than this run on the CPU.
    /// (Elementwise ops count one per element; `matvec` counts rows × cols × batch.)
    pub min_macs: u64,
}

impl FallbackPolicy {
    /// Run operations below `min_macs` multiply-accumulates on the CPU.
    pub fn min_macs(min_macs: u64) -> Self {
        Self { min_macs }
    }

    /// Never fall back (every op goes to the NPU).
    pub fn never() -> Self {
        Self { min_macs: 0 }
    }

    /// Measures where the NPU starts beating the CPU for batched
    /// matrix-vector products on `npu`, and returns a policy with that
    /// threshold. Takes a fraction of a second; run it once at startup.
    ///
    /// On the simulator both sides do the same work, so the result is
    /// meaningless there; it only makes sense on real hardware.
    pub fn calibrate(npu: &Npu) -> Result<Self> {
        let cpu = Simulator::default();
        let mut threshold = u64::MAX;
        // Square-ish problems from 4 K to 16 M MACs.
        for &(n, batch) in &[
            (16usize, 16usize),
            (32, 32),
            (64, 32),
            (128, 32),
            (256, 64),
            (512, 64),
        ] {
            let m = vec![bf16::ONE; n * n];
            let v = vec![bf16::ONE; batch * n];
            let mut out = vec![bf16::ZERO; batch * n];
            let time = |f: &mut dyn FnMut() -> Result<()>| -> Result<f64> {
                f()?; // warm-up
                let start = Instant::now();
                for _ in 0..5 {
                    f()?;
                }
                Ok(start.elapsed().as_secs_f64())
            };
            let t_npu = time(&mut || {
                npu.backend()
                    .matvec_batched(npu.id(), &m, &v, &mut out, n, n)
            })?;
            let t_cpu = time(&mut || cpu.matvec_batched(0, &m, &v, &mut out, n, n))?;
            if t_npu < t_cpu {
                threshold = (n * n * batch) as u64;
                break;
            }
        }
        tracing::info!(threshold, "CPU fallback calibrated");
        Ok(Self {
            min_macs: threshold,
        })
    }
}

/// How many operations went where (see [`FallbackMonitor`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FallbackStats {
    /// Streamed operations run on the CPU.
    pub cpu_ops: u64,
    /// Streamed operations run on the NPU.
    pub npu_ops: u64,
}

#[derive(Debug, Default)]
struct Counters {
    cpu: AtomicU64,
    npu: AtomicU64,
}

/// Reads how many streamed operations a fallback handle sent to the CPU vs the NPU.
#[derive(Debug, Clone)]
pub struct FallbackMonitor {
    counters: Arc<Counters>,
}

impl FallbackMonitor {
    /// Counts so far.
    pub fn stats(&self) -> FallbackStats {
        FallbackStats {
            cpu_ops: self.counters.cpu.load(Ordering::Relaxed),
            npu_ops: self.counters.npu.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug)]
pub(crate) struct FallbackBackend {
    npu: Arc<dyn Backend>,
    cpu: Simulator,
    policy: FallbackPolicy,
    counters: Arc<Counters>,
}

pub(crate) fn wrap(
    npu: Arc<dyn Backend>,
    policy: FallbackPolicy,
) -> (Arc<dyn Backend>, FallbackMonitor) {
    let counters = Arc::new(Counters::default());
    let backend = Arc::new(FallbackBackend {
        npu,
        cpu: Simulator::default(),
        policy,
        counters: Arc::clone(&counters),
    });
    (backend, FallbackMonitor { counters })
}

impl FallbackBackend {
    fn on_cpu(&self, macs: u64) -> bool {
        let cpu = macs < self.policy.min_macs;
        let counter = if cpu {
            &self.counters.cpu
        } else {
            &self.counters.npu
        };
        counter.fetch_add(1, Ordering::Relaxed);
        cpu
    }
}

impl Backend for FallbackBackend {
    fn kind(&self) -> BackendKind {
        self.npu.kind()
    }

    fn name(&self) -> String {
        format!(
            "{} [cpu fallback < {} MAC]",
            self.npu.name(),
            self.policy.min_macs
        )
    }

    fn init_npu(&self, npu: u32) -> Result<()> {
        self.npu.init_npu(npu)
    }

    fn release_npu(&self, npu: u32) -> Result<()> {
        self.npu.release_npu(npu)
    }

    fn available_npus(&self) -> Result<Vec<NpuEntry>> {
        self.npu.available_npus()
    }

    fn mul(&self, npu: u32, a: &[bf16], b: &[bf16], out: &mut [bf16]) -> Result<()> {
        if self.on_cpu(a.len() as u64) {
            self.cpu.mul(0, a, b, out)
        } else {
            self.npu.mul(npu, a, b, out)
        }
    }

    fn periodic_nl(&self, npu: u32, u: &[bf16], v: &[bf16], out: &mut [bf16]) -> Result<()> {
        if self.on_cpu(u.len() as u64) {
            self.cpu.periodic_nl(0, u, v, out)
        } else {
            self.npu.periodic_nl(npu, u, v, out)
        }
    }

    fn matvec_batched(
        &self,
        npu: u32,
        matrix: &[bf16],
        vectors: &[bf16],
        out: &mut [bf16],
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        let batch = vectors.len().checked_div(cols).unwrap_or(0);
        if self.on_cpu((rows * cols * batch) as u64) {
            self.cpu.matvec_batched(0, matrix, vectors, out, rows, cols)
        } else {
            self.npu
                .matvec_batched(npu, matrix, vectors, out, rows, cols)
        }
    }

    fn allocate(&self, npu: u32, rows: usize, cols: usize, role: DataRole) -> Result<RawHandle> {
        self.npu.allocate(npu, rows, cols, role)
    }

    fn free(&self, handle: RawHandle) -> Result<()> {
        self.npu.free(handle)
    }

    fn upload(&self, handle: RawHandle, data: &[bf16]) -> Result<()> {
        self.npu.upload(handle, data)
    }

    fn download(&self, handle: RawHandle, out: &mut [bf16]) -> Result<()> {
        self.npu.download(handle, out)
    }

    fn matvec_resident(
        &self,
        matrix: RawHandle,
        vectors: RawHandle,
        result: RawHandle,
    ) -> Result<()> {
        self.npu.matvec_resident(matrix, vectors, result)
    }

    fn driver_version(&self, npu: u32) -> Result<String> {
        self.npu.driver_version(npu)
    }

    fn device_info(&self, npu: u32) -> Result<DeviceInfo> {
        self.npu.device_info(npu)
    }

    fn perf_counters(&self, npu: u32) -> Result<PerfCounters> {
        self.npu.perf_counters(npu)
    }

    fn reset_perf_counters(&self, npu: u32) -> Result<()> {
        self.npu.reset_perf_counters(npu)
    }
}
