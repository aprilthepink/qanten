//! The fault classifier: dataset generation, training on the NPU, and the
//! deployable [`Diagnoser`].

use std::path::Path;

use qanten::nn::{self, Module, StateDict};
use qanten::train::{self, Adam, Dense, Model, Relu};
use qanten::{Error, Npu, Result, Tensor};

use crate::features::{FeatureExtractor, NUM_FEATURES};
use crate::novelty::{self, FitReport, NoveltyDetector};
use crate::sim::{Condition, Machine, Rng, WINDOW};

const HIDDEN: [usize; 2] = [64, 32];

/// Feature vectors with their labels.
#[derive(Debug, Clone, Default)]
pub struct Dataset {
    /// `len × NUM_FEATURES`, row-major.
    pub features: Vec<f32>,
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

    /// Simulates `per_class` windows of every condition on random machines
    /// (speeds 15–45 Hz, severities `severity.0..severity.1`) and extracts
    /// their features on the NPU.
    pub fn generate(
        extractor: &FeatureExtractor,
        per_class: usize,
        severity: (f32, f32),
        seed: u64,
    ) -> Result<Self> {
        let mut rng = Rng::new(seed);
        let mut data = Dataset::default();
        let chunk = 64;
        let mut pending: Vec<(f32, Vec<f32>, usize)> = Vec::with_capacity(chunk);
        let total = per_class * Condition::ALL.len();
        for i in 0..total {
            let condition = Condition::ALL[i % Condition::ALL.len()];
            let machine = Machine::random(rng.next_u64());
            let s = rng.uniform(severity.0, severity.1);
            pending.push((
                machine.shaft_hz,
                machine.measure(condition, s, rng.next_u64()),
                condition.index(),
            ));
            if pending.len() == chunk || i + 1 == total {
                let rows: Vec<(f32, &[f32])> =
                    pending.iter().map(|(f, x, _)| (*f, x.as_slice())).collect();
                data.features.extend(extractor.extract_batch(&rows)?);
                data.labels.extend(pending.drain(..).map(|(_, _, y)| y));
            }
        }
        Ok(data)
    }

    fn rows(&self, idx: &[usize]) -> Result<(Tensor, Vec<usize>)> {
        let mut x = Vec::with_capacity(idx.len() * NUM_FEATURES);
        for &i in idx {
            x.extend_from_slice(&self.features[i * NUM_FEATURES..(i + 1) * NUM_FEATURES]);
        }
        let y = idx.iter().map(|&i| self.labels[i]).collect();
        Ok((Tensor::from_f32(&x, [idx.len(), NUM_FEATURES])?, y))
    }
}

/// Per-feature mean and standard deviation (the model's input normalisation).
#[derive(Debug, Clone)]
pub struct Normalizer {
    mean: Vec<f32>,
    std: Vec<f32>,
}

impl Normalizer {
    /// Fitted on a dataset.
    pub fn fit(data: &Dataset) -> Self {
        let n = data.len().max(1) as f32;
        let mut mean = vec![0.0f32; NUM_FEATURES];
        let mut sq = vec![0.0f32; NUM_FEATURES];
        for row in data.features.chunks(NUM_FEATURES) {
            for (k, v) in row.iter().enumerate() {
                mean[k] += v / n;
                sq[k] += v * v / n;
            }
        }
        let std = mean
            .iter()
            .zip(&sq)
            .map(|(m, s)| (s - m * m).max(0.0).sqrt().max(1e-3))
            .collect();
        Self { mean, std }
    }

    /// Normalises `rows × NUM_FEATURES` values in place.
    pub fn apply(&self, features: &mut [f32]) {
        for row in features.chunks_mut(NUM_FEATURES) {
            for ((v, m), s) in row.iter_mut().zip(&self.mean).zip(&self.std) {
                *v = (*v - m) / s;
            }
        }
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
    let norm = Normalizer::fit(data);
    let mut normalized = data.clone();
    norm.apply(&mut normalized.features);

    let s = config.seed;
    let mut model = Model::new()
        .add(Dense::new(npu, NUM_FEATURES, HIDDEN[0], s)?)
        .add(Relu::default())
        .add(Dense::new(npu, HIDDEN[0], HIDDEN[1], s + 1)?)
        .add(Relu::default())
        .add(Dense::new(npu, HIDDEN[1], Condition::ALL.len(), s + 2)?);
    let mut opt = Adam::new(config.lr);
    let mut rng = Rng::new(s);
    let mut order: Vec<usize> = (0..data.len()).collect();
    for epoch in 1..=config.epochs {
        // Fisher–Yates shuffle.
        for i in (1..order.len()).rev() {
            order.swap(i, rng.below(i + 1));
        }
        let (mut loss_sum, mut hits, mut seen) = (0.0, 0.0, 0);
        for idx in order.chunks(config.batch.max(1)) {
            let (x, y) = normalized.rows(idx)?;
            let logits = model.forward(&x)?;
            let (loss, grad) = train::cross_entropy(&logits, &y)?;
            loss_sum += loss * idx.len() as f32;
            hits += train::accuracy(&logits, &y) * idx.len() as f32;
            seen += idx.len();
            model.backward(&grad)?;
            model.clip_grad_norm(5.0);
            model.step(&mut opt)?;
        }
        log(epoch, loss_sum / seen as f32, hits / seen as f32);
    }
    let mut sd = model.state_dict()?;
    sd.insert("norm.mean", Tensor::from_f32(&norm.mean, [NUM_FEATURES])?);
    sd.insert("norm.std", Tensor::from_f32(&norm.std, [NUM_FEATURES])?);
    Ok(sd)
}

impl Normalizer {
    fn from_state_dict(sd: &StateDict) -> Result<Self> {
        Ok(Self {
            mean: sd.get("norm.mean")?.to_f32_vec(),
            std: sd.get("norm.std")?.to_f32_vec(),
        })
    }
}

/// Fits the novelty detector (see [`novelty`]) on the training features and
/// calibrates it on `validation`, adding it to the weights from [`train()`].
pub fn add_novelty(
    npu: &Npu,
    sd: &mut StateDict,
    train: &Dataset,
    validation: &Dataset,
) -> Result<FitReport> {
    let norm = Normalizer::from_state_dict(sd)?;
    let (mut t, mut v) = (train.features.clone(), validation.features.clone());
    norm.apply(&mut t);
    norm.apply(&mut v);
    novelty::fit(npu, &t, &v, sd)
}

/// Saves trained weights as `.safetensors`.
pub fn save(sd: &StateDict, path: &Path) -> anyhow::Result<()> {
    std::fs::write(path, sd.to_safetensors_bytes()?)?;
    Ok(())
}

/// The deployed pipeline on one NPU: raw window in, condition probabilities out.
#[derive(Debug)]
pub struct Diagnoser {
    extractor: FeatureExtractor,
    norm: Normalizer,
    net: nn::Sequential,
    novelty: Option<NoveltyDetector>,
}

/// Width of one [`Diagnoser`] input row: the shaft speed in Hz followed by the window.
pub const INPUT_WIDTH: usize = 1 + WINDOW;
/// Width of one [`Diagnoser`] output row: six condition probabilities, then
/// the novelty score (above 1 = unlike any known condition; 0 if the model
/// has no novelty detector).
pub const OUTPUT_WIDTH: usize = 7;

/// Classifier output plus novelty score for one window.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Assessment {
    /// Probabilities in [`Condition::ALL`] order.
    pub probs: [f32; 6],
    /// Novelty score (max of SPE and T² over their thresholds); above 1 = unusual.
    pub novelty: f32,
}

impl Assessment {
    /// From one [`OUTPUT_WIDTH`] row.
    pub fn from_row(row: &[f32]) -> Self {
        let mut probs = [0.0; 6];
        probs.copy_from_slice(&row[..6]);
        Self {
            probs,
            novelty: row[6],
        }
    }
}

impl Diagnoser {
    /// Builds the pipeline on `npu` from trained weights.
    pub fn load(npu: &Npu, sd: &StateDict) -> Result<Self> {
        let norm = Normalizer::from_state_dict(sd)?;
        let net = nn::Sequential::new()
            .add(nn::Linear::load(npu, sd, "0")?)
            .add(nn::ReLU)
            .add(nn::Linear::load(npu, sd, "2")?)
            .add(nn::ReLU)
            .add(nn::Linear::load(npu, sd, "4")?)
            .add(nn::Softmax);
        Ok(Self {
            extractor: FeatureExtractor::new(npu)?,
            norm,
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

    /// Condition probabilities `[rows, 6]` from already extracted features.
    pub fn classify_features(&self, features: &[f32]) -> Result<Tensor> {
        let mut f = features.to_vec();
        self.norm.apply(&mut f);
        let rows = f.len() / NUM_FEATURES;
        self.net
            .forward(&Tensor::from_f32(&f, [rows, NUM_FEATURES])?)
    }

    /// Novelty scores from already extracted features (zeros without a detector).
    pub fn novelty_scores(&self, features: &[f32]) -> Result<Vec<f32>> {
        let rows = features.len() / NUM_FEATURES;
        match &self.novelty {
            Some(d) => {
                let mut f = features.to_vec();
                self.norm.apply(&mut f);
                d.score(&f)
            }
            None => Ok(vec![0.0; rows]),
        }
    }

    /// `true` if the model includes a novelty detector.
    pub fn has_novelty(&self) -> bool {
        self.novelty.is_some()
    }

    /// NPU self-test through the resident novelty projector (relative error,
    /// see [`NoveltyDetector::self_check`]); `None` without a detector.
    pub fn self_check(&self, seed: u64) -> Result<Option<f64>> {
        self.novelty
            .as_ref()
            .map(|d| d.self_check(32, seed))
            .transpose()
    }

    /// Full assessment of raw windows `(shaft_hz, samples)`.
    pub fn assess(&self, windows: &[(f32, &[f32])]) -> Result<Vec<Assessment>> {
        let features = self.extractor.extract_batch(windows)?;
        let probs = self.classify_features(&features)?.to_f32_vec();
        let novelty = self.novelty_scores(&features)?;
        Ok(probs
            .chunks(6)
            .zip(novelty)
            .map(|(p, n)| {
                let mut row = p.to_vec();
                row.push(n);
                Assessment::from_row(&row)
            })
            .collect())
    }
}

/// Input `[rows, 1 + WINDOW]` (shaft speed in Hz, then the samples), output
/// `[rows, 7]`: probabilities in [`Condition::ALL`] order, then the novelty
/// score. This is what the HTTP server and the multi-NPU pool run.
impl Module for Diagnoser {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let [_, width] = x.dims::<2>()?;
        if width != INPUT_WIDTH {
            return Err(Error::Shape(format!(
                "each row is the shaft speed in Hz followed by {WINDOW} samples \
                 ({INPUT_WIDTH} values), got {width}"
            )));
        }
        let v = x.to_f32_vec();
        let rows: Vec<(f32, &[f32])> = v.chunks(INPUT_WIDTH).map(|r| (r[0], &r[1..])).collect();
        let out: Vec<f32> = self
            .assess(&rows)?
            .iter()
            .flat_map(|a| a.probs.into_iter().chain([a.novelty]))
            .collect();
        Tensor::from_f32(&out, [rows.len(), OUTPUT_WIDTH])
    }

    fn name(&self) -> &'static str {
        "Diagnoser"
    }
}

/// `confusion[true][predicted]` counts.
pub fn confusion(probs: &Tensor, labels: &[usize]) -> Vec<Vec<usize>> {
    let n = Condition::ALL.len();
    let mut m = vec![vec![0; n]; n];
    for (p, &y) in probs.argmax_last().iter().zip(labels) {
        m[y][*p] += 1;
    }
    m
}
