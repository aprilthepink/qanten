//! The [`Npu`] handle.

#[cfg(feature = "driver")]
use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};

use crate::backend::{Backend, BackendKind, NpuEntry, SimConfig, Simulator};
use crate::error::{Error, Result};
use crate::info::{DeviceInfo, PerfCounters};

/// Environment variable read by [`Npu::from_env`]: `driver`, `sim` or `auto`.
pub const BACKEND_ENV: &str = "QANTEN_BACKEND";

struct Inner {
    id: u32,
    backend: Arc<dyn Backend>,
    /// Set for wrapper handles (e.g. profiled): the parent owns the device and
    /// releases it; the wrapper just keeps it alive.
    parent: Option<Npu>,
}

impl Drop for Inner {
    fn drop(&mut self) {
        if self.parent.is_some() {
            return;
        }
        if let Err(err) = self.backend.release_npu(self.id) {
            tracing::warn!(npu = self.id, %err, "failed to release NPU");
        }
    }
}

/// A handle to one initialised NPU.
///
/// Opening initialises the device; dropping the last clone releases it.
/// `Npu` is cheap to clone and can be shared across threads.
///
/// ```
/// # fn main() -> qanten::Result<()> {
/// // Real hardware if the driver is installed, the simulator otherwise.
/// let npu = qanten::Npu::auto(0)?;
/// let a = qanten::Tensor::vector(&[1.0, 2.0, 3.0]);
/// let b = qanten::Tensor::vector(&[4.0, 5.0, 6.0]);
/// let c = qanten::ops::mul(&npu, &a, &b)?;
/// assert_eq!(c.to_f32_vec(), vec![4.0, 10.0, 18.0]);
/// # Ok(()) }
/// ```
#[derive(Clone)]
pub struct Npu {
    inner: Arc<Inner>,
}

impl std::fmt::Debug for Npu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Npu")
            .field("id", &self.inner.id)
            .field("backend", &self.inner.backend.kind())
            .finish()
    }
}

// Opened hardware NPUs, so that opening the same id twice shares one handle
// (and one init/release pair) instead of initialising the device twice.
#[cfg(feature = "driver")]
static OPEN_DRIVER_NPUS: Mutex<Option<HashMap<u32, Weak<Inner>>>> = Mutex::new(None);

static DEFAULT_SIM: Mutex<Option<Weak<Simulator>>> = Mutex::new(None);

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

impl Npu {
    /// Opens NPU `id` on real hardware through the Q.ANT driver.
    pub fn open(id: u32) -> Result<Self> {
        #[cfg(feature = "driver")]
        {
            let mut guard = lock(&OPEN_DRIVER_NPUS);
            let map = guard.get_or_insert_with(HashMap::new);
            if let Some(inner) = map.get(&id).and_then(Weak::upgrade) {
                return Ok(Self { inner });
            }
            let backend: Arc<dyn Backend> = crate::backend::DriverBackend::global()?;
            let npu = Self::with_backend(backend, id)?;
            map.insert(id, Arc::downgrade(&npu.inner));
            Ok(npu)
        }
        #[cfg(not(feature = "driver"))]
        {
            let _ = id;
            Err(Error::DriverFeatureDisabled)
        }
    }

    /// NPU 0 of the shared default simulator (exact, noise-free).
    pub fn simulator() -> Self {
        let mut slot = lock(&DEFAULT_SIM);
        let sim = slot.as_ref().and_then(Weak::upgrade).unwrap_or_else(|| {
            let s = Arc::new(Simulator::default());
            *slot = Some(Arc::downgrade(&s));
            s
        });
        drop(slot);
        // The simulator only rejects ids it does not have; 0 always exists.
        let _ = sim.init_npu(0);
        Self {
            inner: Arc::new(Inner {
                id: 0,
                backend: sim,
                parent: None,
            }),
        }
    }

    /// NPU 0 of a fresh simulator with custom settings (noise, memory limit, ...).
    pub fn simulator_with(config: SimConfig) -> Self {
        let sim = Arc::new(Simulator::new(config));
        let _ = sim.init_npu(0);
        Self {
            inner: Arc::new(Inner {
                id: 0,
                backend: sim,
                parent: None,
            }),
        }
    }

    /// NPU `id` of a shared simulator instance. Use this to simulate several
    /// NPUs that share one backend (see [`SimConfig::num_npus`]).
    pub fn simulated(sim: &Arc<Simulator>, id: u32) -> Result<Self> {
        Self::with_backend(Arc::clone(sim) as Arc<dyn Backend>, id)
    }

    /// Opens hardware NPU `id` if the driver is available, otherwise falls
    /// back to the simulator and logs a warning. Check [`Npu::is_simulated`]
    /// if it matters.
    pub fn auto(id: u32) -> Result<Self> {
        match Self::open(id) {
            Ok(npu) => Ok(npu),
            Err(Error::DriverUnavailable(_) | Error::DriverFeatureDisabled) => {
                tracing::warn!("Q.ANT driver not available, using the simulator");
                Ok(Self::simulator())
            }
            Err(e) => Err(e),
        }
    }

    /// Picks the backend from the `QANTEN_BACKEND` environment variable:
    /// `driver`/`hw` → [`Npu::open`], `sim`/`simulator` → [`Npu::simulator`],
    /// unset or `auto` → [`Npu::auto`].
    pub fn from_env(id: u32) -> Result<Self> {
        let choice = std::env::var(BACKEND_ENV).unwrap_or_default();
        match choice.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Self::auto(id),
            "driver" | "hw" | "hardware" => Self::open(id),
            "sim" | "simulator" | "cpu" => Ok(Self::simulator()),
            other => Err(Error::InvalidArgument(format!(
                "{BACKEND_ENV}={other:?}: expected driver, sim or auto"
            ))),
        }
    }

    /// Opens NPU `id` on a custom backend.
    pub fn with_backend(backend: Arc<dyn Backend>, id: u32) -> Result<Self> {
        backend.init_npu(id)?;
        Ok(Self {
            inner: Arc::new(Inner {
                id,
                backend,
                parent: None,
            }),
        })
    }

    /// The NPU index.
    pub fn id(&self) -> u32 {
        self.inner.id
    }

    /// The backend this NPU runs on.
    pub fn backend(&self) -> &Arc<dyn Backend> {
        &self.inner.backend
    }

    /// Which kind of backend this NPU runs on.
    pub fn backend_kind(&self) -> BackendKind {
        self.inner.backend.kind()
    }

    /// `true` if this NPU is the built-in simulator.
    pub fn is_simulated(&self) -> bool {
        self.backend_kind() == BackendKind::Simulator
    }

    /// `true` if both handles refer to the same device on the same backend.
    pub fn same_device(&self, other: &Npu) -> bool {
        self.id() == other.id()
            && std::ptr::addr_eq(
                Arc::as_ptr(&self.inner.backend),
                Arc::as_ptr(&other.inner.backend),
            )
    }

    /// Current sensor readings and firmware info.
    /// A handle to the same device that records every call it makes, plus the
    /// [`Profiler`](crate::profile::Profiler) to read the numbers. Use the
    /// returned handle for the work you want measured; buffers and layers
    /// created through it belong to it (mixing them with buffers from the
    /// original handle gives [`Error::DeviceMismatch`]).
    pub fn profiled(&self) -> (Npu, crate::profile::Profiler) {
        let (backend, profiler) = crate::profile::wrap(Arc::clone(&self.inner.backend));
        let npu = Npu {
            inner: Arc::new(Inner {
                id: self.id(),
                backend,
                parent: Some(self.clone()),
            }),
        };
        (npu, profiler)
    }

    /// Current sensor readings and firmware info.
    pub fn device_info(&self) -> Result<DeviceInfo> {
        self.inner.backend.device_info(self.id())
    }

    /// Driver version string.
    pub fn driver_version(&self) -> Result<String> {
        self.inner.backend.driver_version(self.id())
    }

    /// Performance counters since the last reset.
    pub fn perf_counters(&self) -> Result<PerfCounters> {
        self.inner.backend.perf_counters(self.id())
    }

    /// Restarts the performance counters.
    pub fn reset_perf_counters(&self) -> Result<()> {
        self.inner.backend.reset_perf_counters(self.id())
    }
}

/// Lists the NPUs the Q.ANT driver can see.
pub fn devices() -> Result<Vec<NpuEntry>> {
    #[cfg(feature = "driver")]
    {
        crate::backend::DriverBackend::global()?.available_npus()
    }
    #[cfg(not(feature = "driver"))]
    {
        Err(Error::DriverFeatureDisabled)
    }
}

/// `true` if the Q.ANT driver library can be loaded on this machine.
pub fn driver_available() -> bool {
    #[cfg(feature = "driver")]
    {
        crate::backend::DriverBackend::global().is_ok()
    }
    #[cfg(not(feature = "driver"))]
    {
        false
    }
}

/// Makes the Q.ANT driver write its own logs into `folder`.
#[cfg(feature = "driver")]
pub fn setup_driver_logging(
    folder: impl AsRef<std::path::Path>,
    level: tracing::Level,
) -> Result<()> {
    crate::backend::DriverBackend::global()?.setup_logging(folder.as_ref(), level)
}
