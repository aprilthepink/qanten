//! The fault classifier: dataset generation, training on the NPU, and the
//! deployable [`Diagnoser`].

use std::fmt;
use std::path::Path;

use qanten::compute::Matrix;
use qanten::nn::{self, Module, StateDict};
use qanten::rng::SplitMix64;
use qanten::train::{self, Adam, Dense, Model, Relu};
use qanten::{Error, Npu, Result, Tensor};

use crate::features::{FeatureExtractor, NUM_FEATURES};
use crate::novelty::{self, FitReport, NoveltyDetector};
use crate::sim::{Condition, Machine, Measurement, WINDOW};

const HIDDEN: [usize; 2] = [64, 32];
const CLASSES: usize = Condition::ALL.len();

/// Feature vectors with their labels.
#[derive(Debug, Clone)]
pub struct Dataset {
    /// One feature vector per row (`len × NUM_FEATURES`).
    pub features: Matrix,
    /// One condition index per row.
    pub labels: Vec<usize>,
}

impl Dataset {
    /// Number of examples.
    pub fn len(&self) -> usize {
        self.labels.len()
    }

    /// `true` when there are no examples.
    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    /// Simulates `per_class` measurements of every model condition on random
    /// machines (speeds 15–45 Hz, severities in `severity`) and extracts
    /// their features on the NPU.
    pub fn generate(
        extractor: &FeatureExtractor,
        per_class: usize,
        severity: (f32, f32),
        seed: u64,
    ) -> Result<Self> {
        let conditions: Vec<Condition> = (0..per_class * CLASSES)
            .map(|i| Condition::ALL[i % CLASSES])
            .collect();
        Self::simulate(extractor, &conditions, severity, seed)
    }

    /// Like [`generate`](Self::generate), for `count` measurements of one
    /// condition (which may be one the model never saw, e.g. cavitation).
    pub fn generate_condition(
        extractor: &FeatureExtractor,
        condition: Condition,
        count: usize,
        severity: (f32, f32),
        seed: u64,
    ) -> Result<Self> {
        Self::simulate(extractor, &vec![condition; count], severity, seed)
    }

    fn simulate(
        extractor: &FeatureExtractor,
        conditions: &[Condition],
        severity: (f32, f32),
        seed: u64,
    ) -> Result<Self> {
        let mut rng = SplitMix64::new(seed);
        let mut features = Matrix::zeros(0, NUM_FEATURES);
        for chunk in conditions.chunks(64) {
            let batch: Vec<Measurement> = chunk
                .iter()
                .map(|&c| {
                    let machine = Machine::random(rng.next_u64());
                    let s = rng.uniform(severity.0, severity.1);
                    machine.measure(c, s, rng.next_u64())
                })
                .collect();
            for row in extractor.extract_batch(&batch)?.iter_rows() {
                features.push_row(row)?;
            }
        }
        Ok(Self {
            features,
            // Unseen conditions have no class; label them "healthy" (unused by callers).
            labels: conditions.iter().map(|c| c.class().unwrap_or(0)).collect(),
        })
    }
}

/// Per-feature mean and standard deviation (the model's input normalisation).
#[derive(Debug, Clone)]
pub struct Normalizer {
    mean: Vec<f32>,
    std: Vec<f32>,
}

impl Normalizer {
    /// Fitted on a feature matrix.
    pub fn fit(features: &Matrix) -> Self {
        let mean: Vec<f64> = features.column_means();
        let mut var = vec![0.0f64; features.cols()];
        for row in features.iter_rows() {
            for ((v, x), m) in var.iter_mut().zip(row).zip(&mean) {
                *v += (*x as f64 - m).powi(2);
            }
        }
        let n = features.rows().max(1) as f64;
        Self {
            mean: mean.iter().map(|m| *m as f32).collect(),
            std: var
                .iter()
                .map(|v| ((v / n).sqrt() as f32).max(1e-3))
                .collect(),
        }
    }

    /// A normalised copy.
    pub fn apply(&self, features: &Matrix) -> Matrix {
        let mut out = features.clone();
        for row in out.iter_rows_mut() {
            for ((v, m), s) in row.iter_mut().zip(&self.mean).zip(&self.std) {
                *v = (*v - m) / s;
            }
        }
        out
    }

    fn save(&self, sd: &mut StateDict) -> Result<()> {
        sd.insert("norm.mean", Tensor::vector(&self.mean));
        sd.insert("norm.std", Tensor::vector(&self.std));
        Ok(())
    }

    fn load(sd: &StateDict) -> Result<Self> {
        Ok(Self {
            mean: sd.get("norm.mean")?.to_f32_vec(),
            std: sd.get("norm.std")?.to_f32_vec(),
        })
    }
}

/// Training settings.
#[derive(Debug, Clone)]
pub struct TrainConfig {
    /// Epochs over the training set.
    pub epochs: usize,
    /// Mini-batch size.
    pub batch: usize,
    /// Adam learning rate.
    pub lr: f32,
    /// Seed for weight init and shuffling.
    pub seed: u64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            epochs: 40,
            batch: 64,
            lr: 3e-3,
            seed: 1,
        }
    }
}

/// Trains the classifier on `data` with the NPU in the loop. `log(epoch,
/// loss, accuracy)` is called after every epoch. Returns the weights plus the
/// normalisation, ready for [`Diagnoser::load`].
pub fn train(
    npu: &Npu,
    data: &Dataset,
    config: &TrainConfig,
    mut log: impl FnMut(usize, f32, f32),
) -> Result<StateDict> {
    let norm = Normalizer::fit(&data.features);
    let features = norm.apply(&data.features);
    let s = config.seed;
    let mut model = Model::new()
        .add(Dense::new(npu, NUM_FEATURES, HIDDEN[0], s)?)
        .add(Relu::default())
        .add(Dense::new(npu, HIDDEN[0], HIDDEN[1], s + 1)?)
        .add(Relu::default())
        .add(Dense::new(npu, HIDDEN[1], CLASSES, s + 2)?);
    let mut opt = Adam::new(config.lr);
    let mut rng = SplitMix64::new(s);
    let mut order: Vec<usize> = (0..data.len()).collect();
    for epoch in 1..=config.epochs {
        rng.shuffle(&mut order);
        let (mut loss_sum, mut hits) = (0.0, 0.0);
        for idx in order.chunks(config.batch.max(1)) {
            let x = features.select_rows(idx).to_tensor()?;
            let y: Vec<usize> = idx.iter().map(|&i| data.labels[i]).collect();
            let logits = model.forward(&x)?;
            let (loss, grad) = train::cross_entropy(&logits, &y)?;
            loss_sum += loss * idx.len() as f32;
            hits += train::accuracy(&logits, &y) * idx.len() as f32;
            model.backward(&grad)?;
            model.clip_grad_norm(5.0);
            model.step(&mut opt)?;
        }
        let n = data.len().max(1) as f32;
        log(epoch, loss_sum / n, hits / n);
    }
    let mut sd = model.state_dict()?;
    norm.save(&mut sd)?;
    Ok(sd)
}

/// Fits the novelty detector (see [`novelty`]) on the training features and
/// calibrates it on `validation`, adding it to the weights from [`train()`].
pub fn add_novelty(
    npu: &Npu,
    sd: &mut StateDict,
    train: &Dataset,
    validation: &Dataset,
) -> Result<FitReport> {
    let norm = Normalizer::load(sd)?;
    novelty::fit(
        npu,
        &norm.apply(&train.features),
        &norm.apply(&validation.features),
        sd,
    )
}

/// Saves trained weights as `.safetensors`.
pub fn save(sd: &StateDict, path: &Path) -> anyhow::Result<()> {
    std::fs::write(path, sd.to_safetensors_bytes()?)?;
    Ok(())
}

/// Width of one [`Diagnoser`] input row: the shaft speed in Hz followed by the window.
pub const INPUT_WIDTH: usize = 1 + WINDOW;
/// Width of one [`Diagnoser`] output row: six condition probabilities, then
/// the novelty score (above 1 = unlike any known condition; 0 if the model
/// has no novelty detector).
pub const OUTPUT_WIDTH: usize = CLASSES + 1;

/// Classifier output plus novelty score for one measurement.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Assessment {
    /// Probabilities in [`Condition::ALL`] order.
    pub probs: [f32; CLASSES],
    /// Novelty score (max of SPE and T² over their thresholds); above 1 = unusual.
    pub novelty: f32,
}

impl Assessment {
    /// From one [`OUTPUT_WIDTH`] row.
    pub fn from_row(row: &[f32]) -> Self {
        Self {
            probs: std::array::from_fn(|i| row[i]),
            novelty: row[CLASSES],
        }
    }

    /// The row form: probabilities, then the novelty score.
    pub fn to_row(&self) -> [f32; OUTPUT_WIDTH] {
        std::array::from_fn(|i| {
            if i < CLASSES {
                self.probs[i]
            } else {
                self.novelty
            }
        })
    }

    /// The most likely condition and its probability.
    pub fn most_likely(&self) -> (Condition, f32) {
        Condition::ALL
            .into_iter()
            .zip(self.probs)
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap_or((Condition::Healthy, 0.0))
    }
}

/// The deployed pipeline on one NPU: measurements in, assessments out.
#[derive(Debug)]
pub struct Diagnoser {
    extractor: FeatureExtractor,
    norm: Normalizer,
    net: nn::Sequential,
    novelty: Option<NoveltyDetector>,
}

impl Diagnoser {
    /// Builds the pipeline on `npu` from trained weights.
    pub fn load(npu: &Npu, sd: &StateDict) -> Result<Self> {
        let net = nn::Sequential::new()
            .add(nn::Linear::load(npu, sd, "0")?)
            .add(nn::ReLU)
            .add(nn::Linear::load(npu, sd, "2")?)
            .add(nn::ReLU)
            .add(nn::Linear::load(npu, sd, "4")?)
            .add(nn::Softmax);
        Ok(Self {
            extractor: FeatureExtractor::new(npu)?,
            norm: Normalizer::load(sd)?,
            net,
            novelty: NoveltyDetector::load(npu, sd)?,
        })
    }

    /// Reads weights from a `.safetensors` file.
    pub fn from_file(npu: &Npu, path: &Path) -> anyhow::Result<Self> {
        let sd = StateDict::load_safetensors(path).map_err(|e| {
            anyhow::anyhow!(
                "{}: {e} (run `machine-monitor train` first)",
                path.display()
            )
        })?;
        Ok(Self::load(npu, &sd)?)
    }

    /// The feature extractor this pipeline uses.
    pub fn extractor(&self) -> &FeatureExtractor {
        &self.extractor
    }

    /// Condition probabilities, one row per feature vector (`rows × 6`).
    pub fn classify_features(&self, features: &Matrix) -> Result<Tensor> {
        self.net.forward(&self.norm.apply(features).to_tensor()?)
    }

    /// Novelty scores, one per feature vector (zeros without a detector).
    pub fn novelty_scores(&self, features: &Matrix) -> Result<Vec<f32>> {
        match &self.novelty {
            Some(d) => d.score(&self.norm.apply(features)),
            None => Ok(vec![0.0; features.rows()]),
        }
    }

    /// `true` if the model includes a novelty detector.
    pub fn has_novelty(&self) -> bool {
        self.novelty.is_some()
    }

    /// NPU self-test through the resident novelty matrix (relative error,
    /// see [`NoveltyDetector::self_check`]); `None` without a detector.
    pub fn self_check(&self, seed: u64) -> Result<Option<f64>> {
        self.novelty
            .as_ref()
            .map(|d| d.self_check(32, seed))
            .transpose()
    }

    /// Full assessment of every measurement.
    pub fn assess(&self, batch: &[Measurement]) -> Result<Vec<Assessment>> {
        let features = self.extractor.extract_batch(batch)?;
        let probs = Matrix::from_tensor(&self.classify_features(&features)?)?;
        let novelty = self.novelty_scores(&features)?;
        Ok(probs
            .iter_rows()
            .zip(novelty)
            .map(|(p, n)| Assessment {
                probs: std::array::from_fn(|i| p[i]),
                novelty: n,
            })
            .collect())
    }
}

/// Input `[rows, 1 + WINDOW]` (shaft speed in Hz, then the samples), output
/// `[rows, 7]`: probabilities in [`Condition::ALL`] order, then the novelty
/// score. This is what the HTTP server and the multi-NPU pool run.
impl Module for Diagnoser {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let rows = Matrix::from_tensor(x)?;
        if rows.cols() != INPUT_WIDTH {
            return Err(Error::Shape(format!(
                "each row is the shaft speed in Hz followed by {WINDOW} samples \
                 ({INPUT_WIDTH} values), got {}",
                rows.cols()
            )));
        }
        let batch: Vec<Measurement> = rows
            .iter_rows()
            .map(|r| Measurement {
                shaft_hz: r[0],
                samples: r[1..].to_vec(),
            })
            .collect();
        let out: Vec<[f32; OUTPUT_WIDTH]> = self
            .assess(&batch)?
            .iter()
            .map(Assessment::to_row)
            .collect();
        Matrix::from_rows(&out)?.to_tensor()
    }

    fn name(&self) -> &'static str {
        "Diagnoser"
    }
}

/// `counts[true][predicted]` for a test set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Confusion {
    counts: [[usize; CLASSES]; CLASSES],
}

impl Confusion {
    /// Tallies predicted classes (argmax of `probs` rows) against `labels`.
    pub fn new(probs: &Tensor, labels: &[usize]) -> Self {
        let mut counts = [[0; CLASSES]; CLASSES];
        for (p, &y) in probs.argmax_last().iter().zip(labels) {
            counts[y][*p] += 1;
        }
        Self { counts }
    }

    /// Share of correctly classified examples.
    pub fn accuracy(&self) -> f32 {
        let total: usize = self.counts.iter().flatten().sum();
        let right: usize = (0..CLASSES).map(|i| self.counts[i][i]).sum();
        right as f32 / total.max(1) as f32
    }

    /// Share of class `c`'s examples recognised as `c`.
    pub fn recall(&self, c: Condition) -> f32 {
        let row = &self.counts[c.index()];
        row[c.index()] as f32 / row.iter().sum::<usize>().max(1) as f32
    }
}

impl fmt::Display for Confusion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:>14}", "true \\ pred")?;
        for c in Condition::ALL {
            write!(f, "{:>13}", c.label())?;
        }
        writeln!(f, "{:>9}", "recall")?;
        for (c, row) in Condition::ALL.iter().zip(&self.counts) {
            write!(f, "{:>14}", c.label())?;
            for v in row {
                write!(f, "{v:>13}")?;
            }
            writeln!(f, "{:>8.1}%", 100.0 * self.recall(*c))?;
        }
        Ok(())
    }
}
