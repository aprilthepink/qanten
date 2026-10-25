//! Vibration-based predictive maintenance on Q.ANT NPUs.
//!
//! An example project for the `qanten` crates. Accelerometers on pumps, fans
//! and compressors send a 0.4 s window every hour. The NPU runs the
//! signal-processing front end (filter bank, envelope demodulation,
//! spectra) and a small classifier that names the developing fault. See
//! `README.md` for the walkthrough.
//!
//! - [`sim`]: synthetic machines and their fault signatures
//! - [`features`]: the NPU feature pipeline
//! - [`model`]: dataset, training on the NPU, the deployable [`model::Diagnoser`]
//! - [`novelty`]: PCA novelty detector for faults the classifier never saw
//!   (built on `qanten::compute`)
//! - [`fleet`]: a simulated plant and the alarm logic

pub mod features;
pub mod fleet;
pub mod model;
pub mod novelty;
pub mod sim;

pub use model::{Assessment, Diagnoser, INPUT_WIDTH, OUTPUT_WIDTH};
pub use sim::{Condition, Machine, SAMPLE_RATE, WINDOW};
