//! Profiling: per-operation counts, time, data moved and work done.
//!
//! [`Npu::profiled`](crate::Npu::profiled) returns a handle that records every primitive call it
//! makes, and a [`Profiler`] to read the numbers:
//!
//! ```
//! # fn main() -> qanten::Result<()> {
//! use qanten::{Npu, Tensor, ops};
//! let (npu, profiler) = Npu::simulator().profiled();
//! let w = Tensor::rand_uniform([64, 128], -1., 1., 1);
//! let x = Tensor::rand_uniform([32, 128], -1., 1., 2);
//! ops::linear(&npu, &x, &w, None)?;
//! let report = profiler.report();
//! println!("{report}");
//! assert_eq!(report.total().macs, 64 * 128 * 32);
//! # Ok(()) }
//! ```
//!
//! Every call is also emitted as a `tracing` event at TRACE level (target
//! `qanten::profile`), so any `tracing` subscriber can log or export them.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use half::bf16;

use crate::backend::{Backend, BackendKind, DataRole, NpuEntry, RawHandle};
use crate::error::Result;
use crate::info::{DeviceInfo, PerfCounters};

/// The primitive operations that are recorded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum OpKind {
    /// Elementwise multiply.
    Mul,
    /// Photonic periodic nonlinearity.
    PeriodicNl,
    /// Matrix-vector product with operands streamed from the host.
    MatVec,
    /// Matrix-vector product with all operands in device memory.
    MatVecResident,
    /// Host → device copy into a buffer.
    Upload,
    /// Device → host copy from a buffer.
    Download,
    /// Device memory allocation.
    Allocate,
    /// Device memory release.
    Free,
}

impl OpKind {
    /// Short name used in reports.
    pub fn name(self) -> &'static str {
        match self {
            OpKind::Mul => "mul",
            OpKind::PeriodicNl => "periodic_nl",
            OpKind::MatVec => "matvec",
            OpKind::MatVecResident => "matvec_resident",
            OpKind::Upload => "upload",
            OpKind::Download => "download",
            OpKind::Allocate => "allocate",
            OpKind::Free => "free",
        }
    }
}

/// Accumulated numbers for one [`OpKind`] (or a total).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct OpStats {
    /// Number of calls.
    pub calls: u64,
    /// Calls that returned an error.
    pub errors: u64,
    /// Time spent inside the backend.
    pub time: Duration,
    /// Bytes sent from the host to the NPU.
    pub bytes_to_device: u64,
    /// Bytes copied from the NPU back to the host.
    pub bytes_from_device: u64,
    /// Multiply-accumulates (or elementwise operations) performed on the NPU.
    pub macs: u64,
}

impl OpStats {
    fn add(&mut self, other: &OpStats) {
        self.calls += other.calls;
        self.errors += other.errors;
        self.time += other.time;
        self.bytes_to_device += other.bytes_to_device;
        self.bytes_from_device += other.bytes_from_device;
        self.macs += other.macs;
    }

    /// Multiply-accumulates per second, in billions.
    pub fn gmacs_per_sec(&self) -> f64 {
        let t = self.time.as_secs_f64();
        if t > 0.0 {
            self.macs as f64 / t / 1e9
        } else {
            0.0
        }
    }
}

/// Snapshot of everything a [`Profiler`] recorded.
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct Report {
    /// Backend name.
    pub backend: String,
    /// Wall time since the profiler was created or last reset.
    pub wall: Duration,
    /// Statistics per operation kind.
    pub ops: BTreeMap<OpKind, OpStats>,
    /// Device memory currently allocated through this handle, in bytes.
    pub live_device_bytes: u64,
    /// Highest device memory use seen, in bytes.
    pub peak_device_bytes: u64,
}

impl Report {
    /// Sum over all operation kinds.
    pub fn total(&self) -> OpStats {
        let mut t = OpStats::default();
        for s in self.ops.values() {
            t.add(s);
        }
        t
    }

    /// Statistics of one kind (zeros if it never ran).
    pub fn op(&self, kind: OpKind) -> OpStats {
        self.ops.get(&kind).copied().unwrap_or_default()
    }

    /// Fraction of wall time spent inside the backend, in `[0, 1]`.
    pub fn backend_share(&self) -> f64 {
        let w = self.wall.as_secs_f64();
        if w > 0.0 {
            (self.total().time.as_secs_f64() / w).min(1.0)
        } else {
            0.0
        }
    }
}

fn human_bytes(b: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = b as f64;
    let mut u = 0;
    while v >= 1024.0 && u < UNITS.len() - 1 {
        v /= 1024.0;
        u += 1;
    }
    if u == 0 {
        format!("{b} B")
    } else {
        format!("{v:.1} {}", UNITS[u])
    }
}

impl fmt::Display for Report {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "qanten profile: {} ({:.1} ms wall)",
            self.backend,
            self.wall.as_secs_f64() * 1e3
        )?;
        writeln!(
            f,
            "{:<16} {:>8} {:>11} {:>11} {:>11} {:>12} {:>9}",
            "op", "calls", "time ms", "to NPU", "from NPU", "MACs", "GMAC/s"
        )?;
        let row = |f: &mut fmt::Formatter<'_>, name: &str, s: &OpStats| {
            writeln!(
                f,
                "{:<16} {:>8} {:>11.3} {:>11} {:>11} {:>12} {:>9.2}",
                name,
                s.calls,
                s.time.as_secs_f64() * 1e3,
                human_bytes(s.bytes_to_device),
                human_bytes(s.bytes_from_device),
                s.macs,
                s.gmacs_per_sec()
            )
        };
        for (k, s) in &self.ops {
            row(f, k.name(), s)?;
        }
        row(f, "total", &self.total())?;
        write!(
            f,
            "backend share of wall time {:.1} %, device memory live {} / peak {}",
            100.0 * self.backend_share(),
            human_bytes(self.live_device_bytes),
            human_bytes(self.peak_device_bytes)
        )
    }
}

#[derive(Debug)]
struct State {
    since: Instant,
    ops: BTreeMap<OpKind, OpStats>,
    /// `(rows, cols)` of every live allocation.
    allocs: HashMap<RawHandle, (u64, u64)>,
    live: u64,
    peak: u64,
}

impl State {
    fn new() -> Self {
        Self {
            since: Instant::now(),
            ops: BTreeMap::new(),
            allocs: HashMap::new(),
            live: 0,
            peak: 0,
        }
    }
}

/// Reads the numbers recorded by a profiled [`Npu`](crate::Npu). Cheap to clone.
#[derive(Debug, Clone)]
pub struct Profiler {
    state: Arc<Mutex<State>>,
    backend: String,
}

impl Profiler {
    /// A snapshot of everything recorded so far.
    pub fn report(&self) -> Report {
        let s = lock(&self.state);
        Report {
            backend: self.backend.clone(),
            wall: s.since.elapsed(),
            ops: s.ops.clone(),
            live_device_bytes: s.live,
            peak_device_bytes: s.peak,
        }
    }

    /// Clears the counters (live device memory is kept; the peak restarts from it).
    pub fn reset(&self) {
        let mut s = lock(&self.state);
        s.since = Instant::now();
        s.ops.clear();
        s.peak = s.live;
    }
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

/// Backend wrapper that records every call before forwarding it.
#[derive(Debug)]
pub(crate) struct ProfiledBackend {
    inner: Arc<dyn Backend>,
    state: Arc<Mutex<State>>,
}

pub(crate) fn wrap(inner: Arc<dyn Backend>) -> (Arc<dyn Backend>, Profiler) {
    let state = Arc::new(Mutex::new(State::new()));
    let backend = inner.name();
    let wrapped: Arc<dyn Backend> = Arc::new(ProfiledBackend {
        inner,
        state: Arc::clone(&state),
    });
    (wrapped, Profiler { state, backend })
}

const BF16: u64 = 2;

impl ProfiledBackend {
    fn record<T>(
        &self,
        kind: OpKind,
        to: u64,
        from: u64,
        macs: u64,
        f: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        let start = Instant::now();
        let r = f();
        let time = start.elapsed();
        tracing::trace!(
            target: "qanten::profile",
            op = kind.name(),
            us = time.as_micros() as u64,
            bytes_to_device = to,
            bytes_from_device = from,
            macs,
            ok = r.is_ok()
        );
        let mut s = lock(&self.state);
        let e = s.ops.entry(kind).or_default();
        e.calls += 1;
        e.time += time;
        if r.is_ok() {
            e.bytes_to_device += to;
            e.bytes_from_device += from;
            e.macs += macs;
        } else {
            e.errors += 1;
        }
        r
    }

    fn shape(&self, h: RawHandle) -> (u64, u64) {
        lock(&self.state).allocs.get(&h).copied().unwrap_or((0, 0))
    }
}

impl Backend for ProfiledBackend {
    fn kind(&self) -> BackendKind {
        self.inner.kind()
    }

    fn name(&self) -> String {
        format!("{} [profiled]", self.inner.name())
    }

    fn init_npu(&self, npu: u32) -> Result<()> {
        self.inner.init_npu(npu)
    }

    fn release_npu(&self, npu: u32) -> Result<()> {
        self.inner.release_npu(npu)
    }

    fn available_npus(&self) -> Result<Vec<NpuEntry>> {
        self.inner.available_npus()
    }

    fn mul(&self, npu: u32, a: &[bf16], b: &[bf16], out: &mut [bf16]) -> Result<()> {
        let n = a.len() as u64;
        self.record(OpKind::Mul, 2 * n * BF16, n * BF16, n, || {
            self.inner.mul(npu, a, b, out)
        })
    }

    fn periodic_nl(&self, npu: u32, u: &[bf16], v: &[bf16], out: &mut [bf16]) -> Result<()> {
        let n = u.len() as u64;
        self.record(OpKind::PeriodicNl, 2 * n * BF16, n * BF16, n, || {
            self.inner.periodic_nl(npu, u, v, out)
        })
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
        let n_vecs = vectors.len().checked_div(cols).unwrap_or(0) as u64;
        let to = (matrix.len() + vectors.len()) as u64 * BF16;
        let macs = rows as u64 * cols as u64 * n_vecs;
        self.record(OpKind::MatVec, to, out.len() as u64 * BF16, macs, || {
            self.inner
                .matvec_batched(npu, matrix, vectors, out, rows, cols)
        })
    }

    fn allocate(&self, npu: u32, rows: usize, cols: usize, role: DataRole) -> Result<RawHandle> {
        let h = self.record(OpKind::Allocate, 0, 0, 0, || {
            self.inner.allocate(npu, rows, cols, role)
        })?;
        let mut s = lock(&self.state);
        s.allocs.insert(h, (rows as u64, cols as u64));
        s.live += (rows * cols) as u64 * BF16;
        s.peak = s.peak.max(s.live);
        Ok(h)
    }

    fn free(&self, handle: RawHandle) -> Result<()> {
        let r = self.record(OpKind::Free, 0, 0, 0, || self.inner.free(handle));
        let mut s = lock(&self.state);
        if let Some((rows, cols)) = s.allocs.remove(&handle) {
            s.live = s.live.saturating_sub(rows * cols * BF16);
        }
        r
    }

    fn upload(&self, handle: RawHandle, data: &[bf16]) -> Result<()> {
        self.record(OpKind::Upload, data.len() as u64 * BF16, 0, 0, || {
            self.inner.upload(handle, data)
        })
    }

    fn download(&self, handle: RawHandle, out: &mut [bf16]) -> Result<()> {
        self.record(OpKind::Download, 0, out.len() as u64 * BF16, 0, || {
            self.inner.download(handle, out)
        })
    }

    fn matvec_resident(
        &self,
        matrix: RawHandle,
        vectors: RawHandle,
        result: RawHandle,
    ) -> Result<()> {
        // MACs = matrix elements × number of vectors (rows of the vector block).
        let (m_rows, m_cols) = self.shape(matrix);
        let (n_vecs, _) = self.shape(vectors);
        self.record(
            OpKind::MatVecResident,
            0,
            0,
            m_rows * m_cols * n_vecs,
            || self.inner.matvec_resident(matrix, vectors, result),
        )
    }

    fn driver_version(&self, npu: u32) -> Result<String> {
        self.inner.driver_version(npu)
    }

    fn device_info(&self, npu: u32) -> Result<DeviceInfo> {
        self.inner.device_info(npu)
    }

    fn perf_counters(&self, npu: u32) -> Result<PerfCounters> {
        self.inner.perf_counters(npu)
    }

    fn reset_perf_counters(&self, npu: u32) -> Result<()> {
        self.inner.reset_perf_counters(npu)
    }
}
