//! Pure-Rust simulator of the NPU primitives.

use std::collections::{BTreeMap, HashMap};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use half::bf16;
use rayon::prelude::*;

use super::{Backend, BackendKind, DataRole, NpuEntry, RawHandle, check_len};
use crate::error::{Error, Result};
use crate::info::{DeviceInfo, PerfCounters, SensorReadings, VersionInfo};
use crate::rng::SplitMix64;

/// Simulator settings.
#[derive(Debug, Clone)]
pub struct SimConfig {
    /// Number of simulated NPUs (ids `0..num_npus`).
    pub num_npus: u32,
    /// Standard deviation of Gaussian noise added to every NPU output value.
    /// `0.0` (the default) gives exact results. Use it to test how robust a
    /// model is against analog imprecision; it is not a calibrated model of
    /// real hardware.
    pub noise_std: f32,
    /// Seed of the noise generator.
    pub seed: u64,
    /// Simulated device memory per NPU, in bytes. `None` means unlimited.
    pub memory_limit_bytes: Option<usize>,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            num_npus: 1,
            noise_std: 0.0,
            seed: 0x5EED,
            memory_limit_bytes: None,
        }
    }
}

#[derive(Debug)]
struct Block {
    npu: u32,
    data: Vec<bf16>,
    rows: usize,
    cols: usize,
    role: DataRole,
}

#[derive(Debug)]
struct Perf {
    since: Instant,
    busy: Duration,
}

/// CPU simulator. Thread safe; cheap to share behind an `Arc`.
#[derive(Debug)]
pub struct Simulator {
    config: SimConfig,
    store: Mutex<BTreeMap<u64, Block>>,
    next_key: AtomicU64,
    rng: Mutex<SplitMix64>,
    perf: Mutex<HashMap<u32, Perf>>,
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn sim_err(npu: u32, message: impl Into<String>) -> Error {
    Error::Driver {
        npu,
        code: -1,
        message: message.into(),
    }
}

impl Default for Simulator {
    fn default() -> Self {
        Self::new(SimConfig::default())
    }
}

impl Simulator {
    pub fn new(config: SimConfig) -> Self {
        let rng = SplitMix64::new(config.seed);
        Self {
            config,
            store: Mutex::new(BTreeMap::new()),
            next_key: AtomicU64::new(1),
            rng: Mutex::new(rng),
            perf: Mutex::new(HashMap::new()),
        }
    }

    pub fn config(&self) -> &SimConfig {
        &self.config
    }

    /// Number of live device memory blocks (useful for leak tests).
    pub fn live_allocations(&self) -> usize {
        lock(&self.store).len()
    }

    fn check_npu(&self, npu: u32) -> Result<()> {
        if npu >= self.config.num_npus {
            return Err(sim_err(
                npu,
                format!(
                    "no such NPU (simulator has {} NPU(s))",
                    self.config.num_npus
                ),
            ));
        }
        Ok(())
    }

    /// Runs `f`, records its runtime as busy time, then applies noise to `out`.
    fn timed<T>(&self, npu: u32, f: impl FnOnce() -> T) -> T {
        let start = Instant::now();
        let r = f();
        let elapsed = start.elapsed();
        lock(&self.perf)
            .entry(npu)
            .or_insert_with(|| Perf {
                since: start,
                busy: Duration::ZERO,
            })
            .busy += elapsed;
        r
    }

    fn add_noise(&self, out: &mut [bf16]) {
        if self.config.noise_std <= 0.0 {
            return;
        }
        let mut rng = lock(&self.rng);
        for x in out {
            *x = bf16::from_f32(x.to_f32() + self.config.noise_std * rng.next_normal());
        }
    }

    fn used_bytes(store: &BTreeMap<u64, Block>, npu: u32) -> usize {
        store
            .values()
            .filter(|b| b.npu == npu)
            .map(|b| b.data.len() * 2)
            .sum()
    }
}

/// `out[v, r] = sum_c m[r, c] * x[v, c]`, accumulated in f32.
fn matvec(m: &[bf16], x: &[bf16], out: &mut [bf16], rows: usize, cols: usize) {
    if rows == 0 || cols == 0 {
        out.fill(bf16::ZERO);
        return;
    }
    out.par_chunks_mut(rows)
        .zip(x.par_chunks(cols))
        .for_each(|(out_row, vec)| {
            for (r, o) in out_row.iter_mut().enumerate() {
                let w = &m[r * cols..(r + 1) * cols];
                let s: f32 = w
                    .iter()
                    .zip(vec)
                    .map(|(a, b)| a.to_f32() * b.to_f32())
                    .sum();
                *o = bf16::from_f32(s);
            }
        });
}

impl Backend for Simulator {
    fn kind(&self) -> BackendKind {
        BackendKind::Simulator
    }

    fn name(&self) -> String {
        if self.config.noise_std > 0.0 {
            format!("qanten simulator (noise σ={})", self.config.noise_std)
        } else {
            "qanten simulator".to_owned()
        }
    }

    fn init_npu(&self, npu: u32) -> Result<()> {
        self.check_npu(npu)?;
        lock(&self.perf).entry(npu).or_insert_with(|| Perf {
            since: Instant::now(),
            busy: Duration::ZERO,
        });
        Ok(())
    }

    fn release_npu(&self, npu: u32) -> Result<()> {
        self.check_npu(npu)
    }

    fn available_npus(&self) -> Result<Vec<NpuEntry>> {
        Ok((0..self.config.num_npus)
            .map(|id| NpuEntry {
                id,
                serial: format!("SIM-{id:04}"),
            })
            .collect())
    }

    fn mul(&self, npu: u32, a: &[bf16], b: &[bf16], out: &mut [bf16]) -> Result<()> {
        self.check_npu(npu)?;
        check_len(npu, "mul rhs", b.len(), a.len())?;
        check_len(npu, "mul output", out.len(), a.len())?;
        self.timed(npu, || {
            out.par_iter_mut()
                .zip(a.par_iter().zip(b))
                .for_each(|(o, (x, y))| *o = bf16::from_f32(x.to_f32() * y.to_f32()));
        });
        self.add_noise(out);
        Ok(())
    }

    fn periodic_nl(&self, npu: u32, u: &[bf16], v: &[bf16], out: &mut [bf16]) -> Result<()> {
        self.check_npu(npu)?;
        check_len(npu, "periodic_nl weights", v.len(), u.len())?;
        check_len(npu, "periodic_nl output", out.len(), u.len())?;
        self.timed(npu, || {
            out.par_iter_mut()
                .zip(u.par_iter().zip(v))
                .for_each(|(o, (x, y))| *o = bf16::from_f32(x.to_f32().cos() * y.to_f32()));
        });
        self.add_noise(out);
        Ok(())
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
        self.check_npu(npu)?;
        check_len(npu, "matrix", matrix.len(), rows * cols)?;
        if cols == 0 || !vectors.len().is_multiple_of(cols) {
            return Err(Error::Shape(format!(
                "vector batch of {} values is not a multiple of {cols} columns",
                vectors.len()
            )));
        }
        let n_vecs = vectors.len() / cols;
        check_len(npu, "matvec output", out.len(), n_vecs * rows)?;
        self.timed(npu, || matvec(matrix, vectors, out, rows, cols));
        self.add_noise(out);
        Ok(())
    }

    fn allocate(&self, npu: u32, rows: usize, cols: usize, role: DataRole) -> Result<RawHandle> {
        self.check_npu(npu)?;
        let len = rows
            .checked_mul(cols)
            .ok_or_else(|| sim_err(npu, "allocation size overflows"))?;
        let mut store = lock(&self.store);
        if let Some(limit) = self.config.memory_limit_bytes {
            let used = Self::used_bytes(&store, npu);
            if used + len * 2 > limit {
                return Err(sim_err(
                    npu,
                    format!(
                        "out of device memory: {used} of {limit} bytes used, {} requested",
                        len * 2
                    ),
                ));
            }
        }
        let key = self.next_key.fetch_add(1, Ordering::Relaxed);
        store.insert(
            key,
            Block {
                npu,
                data: vec![bf16::ZERO; len],
                rows,
                cols,
                role,
            },
        );
        Ok(RawHandle { npu, key })
    }

    fn free(&self, handle: RawHandle) -> Result<()> {
        match lock(&self.store).remove(&handle.key) {
            Some(_) => Ok(()),
            None => Err(sim_err(handle.npu, "invalid memory handle")),
        }
    }

    fn upload(&self, handle: RawHandle, data: &[bf16]) -> Result<()> {
        let mut store = lock(&self.store);
        let block = store
            .get_mut(&handle.key)
            .ok_or_else(|| sim_err(handle.npu, "invalid memory handle"))?;
        check_len(handle.npu, "upload", data.len(), block.data.len())?;
        block.data.copy_from_slice(data);
        Ok(())
    }

    fn download(&self, handle: RawHandle, out: &mut [bf16]) -> Result<()> {
        let store = lock(&self.store);
        let block = store
            .get(&handle.key)
            .ok_or_else(|| sim_err(handle.npu, "invalid memory handle"))?;
        if block.role != DataRole::Features {
            return Err(sim_err(handle.npu, "weights memory cannot be downloaded"));
        }
        check_len(handle.npu, "download", out.len(), block.data.len())?;
        out.copy_from_slice(&block.data);
        Ok(())
    }

    fn matvec_resident(
        &self,
        matrix: RawHandle,
        vectors: RawHandle,
        result: RawHandle,
    ) -> Result<()> {
        let npu = matrix.npu;
        let mut store = lock(&self.store);
        let (Some(m), Some(v), Some(r)) = (
            store.get(&matrix.key),
            store.get(&vectors.key),
            store.get(&result.key),
        ) else {
            return Err(sim_err(npu, "invalid memory handle"));
        };
        if m.npu != v.npu || m.npu != r.npu {
            return Err(sim_err(npu, "operands live on different NPUs"));
        }
        if m.role != DataRole::Weights
            || v.role != DataRole::Features
            || r.role != DataRole::Features
        {
            return Err(sim_err(npu, "expected roles (Weights, Features, Features)"));
        }
        if m.cols != v.cols || r.rows != v.rows || r.cols != m.rows {
            return Err(Error::Shape(format!(
                "matrix {}x{}, vectors {}x{}, result {}x{} do not fit",
                m.rows, m.cols, v.rows, v.cols, r.rows, r.cols
            )));
        }
        let (rows, cols) = (m.rows, m.cols);
        let mut out = vec![bf16::ZERO; r.data.len()];
        self.timed(npu, || matvec(&m.data, &v.data, &mut out, rows, cols));
        self.add_noise(&mut out);
        if let Some(r) = store.get_mut(&result.key) {
            r.data = out;
        }
        Ok(())
    }

    fn driver_version(&self, npu: u32) -> Result<String> {
        self.check_npu(npu)?;
        Ok(format!(
            "qanten-simulator;version:{};commit:none;",
            env!("CARGO_PKG_VERSION")
        ))
    }

    fn device_info(&self, npu: u32) -> Result<DeviceInfo> {
        self.check_npu(npu)?;
        Ok(DeviceInfo {
            sensors: SensorReadings {
                temp_pd1: 35.0,
                temp_pd2: 35.0,
                temp_pd3: 35.0,
                temp_dac_1: 35.0,
                temp_dac_2: 35.0,
                voltage_1v8: 1.8,
                voltage_3v3: 3.3,
                voltage_6v0_1: 6.0,
                voltage_6v0_2: 6.0,
                voltage_6v2: 6.2,
                voltage_6v5: 6.5,
                voltage_8v0_1: 8.0,
                voltage_8v0_2: 8.0,
                voltage_12v: 12.0,
                voltage_12v5: 12.5,
                voltage_n12v5: -12.5,
                current_12v: 8.0,
            },
            version: VersionInfo {
                firmware: "simulated".into(),
                zephyr: "simulated".into(),
                board_serial: format!("SIM-{npu:04}"),
            },
        })
    }

    fn perf_counters(&self, npu: u32) -> Result<PerfCounters> {
        self.check_npu(npu)?;
        let perf = lock(&self.perf);
        Ok(match perf.get(&npu) {
            Some(p) => {
                let total = p.since.elapsed();
                PerfCounters {
                    elapsed: total,
                    stalled: total.saturating_sub(p.busy),
                }
            }
            None => PerfCounters::default(),
        })
    }

    fn reset_perf_counters(&self, npu: u32) -> Result<()> {
        self.check_npu(npu)?;
        lock(&self.perf).insert(
            npu,
            Perf {
                since: Instant::now(),
                busy: Duration::ZERO,
            },
        );
        Ok(())
    }
}
