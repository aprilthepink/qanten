//! Async and background execution.
//!
//! NPU calls block the calling thread. An [`NpuExecutor`] runs them on
//! dedicated worker threads instead (one per NPU) and hands back a
//! [`JobHandle`], which is a `Future` usable from any async runtime (tokio,
//! async-std, smol, …) and can also be waited on synchronously:
//!
//! ```
//! # fn main() -> qanten::Result<()> {
//! use qanten::executor::NpuExecutor;
//! use qanten::{Npu, Tensor, ops};
//!
//! let exec = NpuExecutor::new(Npu::simulator());
//! let job = exec.run(|npu| ops::mul(npu, &Tensor::ones([4]), &Tensor::ones([4])));
//! // In async code: `let y = job.await?;`
//! let y = job.wait()?;
//! assert_eq!(y.to_f32_vec(), vec![1.0; 4]);
//! # Ok(()) }
//! ```
//!
//! With a pool, every NPU gets a worker and each job goes to the worker with
//! the fewest jobs in flight, so independent requests spread across all cards. For models, see
//! [`NpuExecutor::forward`].

use std::future::Future;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::task::{Context, Poll, Waker};
use std::thread::JoinHandle as ThreadHandle;

use crate::device::Npu;
use crate::error::{Error, Result};
use crate::nn::Module;
use crate::pool::NpuPool;
use crate::tensor::Tensor;

/// A queued job; gets the worker's NPU and its in-flight counter, which it
/// decrements before publishing the result (so `queue_depths` is never stale
/// once a caller has the result).
type Job = Box<dyn FnOnce(&Npu, &AtomicUsize) + Send + 'static>;

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

/// One worker thread with its own queue.
#[derive(Debug)]
struct Worker {
    sender: Option<Sender<Job>>,
    /// Jobs queued or running on this worker.
    in_flight: Arc<AtomicUsize>,
    thread: Option<ThreadHandle<()>>,
}

/// Worker threads that run NPU work in the background.
///
/// Each job goes to the worker with the fewest jobs queued or running
/// (round-robin among equals), so concurrent jobs spread over all NPUs of a
/// pool. Dropping the executor finishes the queued jobs, then stops the workers.
#[derive(Debug)]
pub struct NpuExecutor {
    workers: Vec<Worker>,
    next: AtomicUsize,
}

impl NpuExecutor {
    /// One worker thread for one NPU.
    pub fn new(npu: Npu) -> Self {
        Self::with_npus(vec![npu])
    }

    /// One worker per NPU of `pool`; each job goes to the least busy one.
    pub fn for_pool(pool: &NpuPool) -> Self {
        Self::with_npus(pool.npus().to_vec())
    }

    /// `workers` threads sharing one NPU (useful when jobs also do host-side
    /// work that can overlap with NPU calls).
    pub fn with_workers(npu: Npu, workers: usize) -> Self {
        Self::with_npus(vec![npu; workers.max(1)])
    }

    fn with_npus(npus: Vec<Npu>) -> Self {
        let workers = npus
            .into_iter()
            .enumerate()
            .filter_map(|(i, npu)| {
                let (sender, receiver) = mpsc::channel::<Job>();
                let in_flight = Arc::new(AtomicUsize::new(0));
                let counter = Arc::clone(&in_flight);
                let thread = std::thread::Builder::new()
                    .name(format!("qanten-npu-{i}"))
                    .spawn(move || {
                        // Ends when the executor drops its sender (after draining the queue).
                        for job in receiver {
                            job(&npu, &counter);
                        }
                    });
                match thread {
                    Ok(t) => Some(Worker {
                        sender: Some(sender),
                        in_flight,
                        thread: Some(t),
                    }),
                    Err(err) => {
                        tracing::error!(%err, "failed to start NPU worker thread");
                        None
                    }
                }
            })
            .collect();
        Self {
            workers,
            next: AtomicUsize::new(0),
        }
    }

    /// Number of worker threads.
    pub fn workers(&self) -> usize {
        self.workers.len()
    }

    /// Jobs queued or running, per worker (in pool order).
    pub fn queue_depths(&self) -> Vec<usize> {
        self.workers
            .iter()
            .map(|w| w.in_flight.load(Ordering::Acquire))
            .collect()
    }

    /// Runs `f` on the least busy worker with that worker's NPU. A panic
    /// inside `f` is caught and returned as an error instead of killing the worker.
    pub fn run<T, F>(&self, f: F) -> JobHandle<T>
    where
        T: Send + 'static,
        F: FnOnce(&Npu) -> Result<T> + Send + 'static,
    {
        let slot = Arc::new(Slot::<T>::default());
        let done = Arc::clone(&slot);
        let job: Job = Box::new(move |npu: &Npu, in_flight: &AtomicUsize| {
            let result = catch_unwind(AssertUnwindSafe(|| f(npu))).unwrap_or_else(|_| {
                Err(Error::Unsupported("job panicked on the NPU worker".into()))
            });
            in_flight.fetch_sub(1, Ordering::AcqRel);
            done.complete(result);
        });
        let n = self.workers.len();
        if n == 0 {
            slot.complete(Err(Error::Unsupported(
                "executor has no running workers".into(),
            )));
            return JobHandle { slot };
        }
        // Least busy worker, starting the scan at a rotating offset to break ties.
        let start = self.next.fetch_add(1, Ordering::Relaxed);
        let pick = (0..n)
            .map(|k| (start + k) % n)
            .min_by_key(|&i| self.workers[i].in_flight.load(Ordering::Acquire))
            .unwrap_or(0);
        let w = &self.workers[pick];
        w.in_flight.fetch_add(1, Ordering::AcqRel);
        let sent = w.sender.as_ref().map(|s| s.send(job));
        if !matches!(sent, Some(Ok(()))) {
            w.in_flight.fetch_sub(1, Ordering::AcqRel);
            slot.complete(Err(Error::Unsupported(
                "executor worker has stopped".into(),
            )));
        }
        JobHandle { slot }
    }

    /// Runs `model.forward(x)` in the background. The model must be shared
    /// (`Arc`) because it outlives this call; build it on the same NPU(s).
    pub fn forward<M>(&self, model: &Arc<M>, x: Tensor) -> JobHandle<Tensor>
    where
        M: Module + ?Sized + 'static,
    {
        let model = Arc::clone(model);
        self.run(move |_| model.forward(&x))
    }
}

impl Drop for NpuExecutor {
    fn drop(&mut self) {
        // Closing every queue lets the workers drain their jobs and exit.
        for w in &mut self.workers {
            w.sender.take();
        }
        for w in &mut self.workers {
            if let Some(t) = w.thread.take() {
                let _ = t.join();
            }
        }
    }
}

struct Slot<T> {
    state: Mutex<(Option<Result<T>>, Option<Waker>)>,
    ready: Condvar,
}

impl<T> Default for Slot<T> {
    fn default() -> Self {
        Self {
            state: Mutex::new((None, None)),
            ready: Condvar::new(),
        }
    }
}

impl<T> Slot<T> {
    fn complete(&self, value: Result<T>) {
        let waker = {
            let mut s = lock(&self.state);
            s.0 = Some(value);
            s.1.take()
        };
        self.ready.notify_all();
        if let Some(w) = waker {
            w.wake();
        }
    }
}

/// The result of a background job: `.await` it, or call [`wait`](JobHandle::wait).
#[must_use = "a JobHandle does nothing unless awaited or waited on (the job still runs)"]
pub struct JobHandle<T> {
    slot: Arc<Slot<T>>,
}

impl<T> std::fmt::Debug for JobHandle<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JobHandle")
            .field("done", &self.is_done())
            .finish()
    }
}

impl<T> JobHandle<T> {
    /// Blocks the current thread until the job finishes.
    pub fn wait(self) -> Result<T> {
        let mut s = lock(&self.slot.state);
        loop {
            if let Some(v) = s.0.take() {
                return v;
            }
            s = self.slot.ready.wait(s).unwrap_or_else(|p| p.into_inner());
        }
    }

    /// `true` once the result is available.
    pub fn is_done(&self) -> bool {
        lock(&self.slot.state).0.is_some()
    }
}

impl<T> Future for JobHandle<T> {
    type Output = Result<T>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut s = lock(&self.slot.state);
        match s.0.take() {
            Some(v) => Poll::Ready(v),
            None => {
                s.1 = Some(cx.waker().clone());
                Poll::Pending
            }
        }
    }
}
