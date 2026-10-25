use machine_monitor::features::{FeatureExtractor, NUM_FEATURES, feature_names};
use machine_monitor::fleet::{AlarmTracker, Diagnosis, Event, Monitor, Outcome, Plant};
use machine_monitor::model::{self, Confusion, Dataset, TrainConfig};
use machine_monitor::sim::BPFO;
use machine_monitor::{
    Assessment, Condition, Diagnoser, INPUT_WIDTH, Machine, OUTPUT_WIDTH, WINDOW,
};
use qanten::compute::Matrix;
use qanten::nn::Module;
use qanten::{Npu, Tensor};

fn feature(name: &str) -> usize {
    feature_names().iter().position(|n| n == name).unwrap()
}

/// Features of one simulated measurement.
fn features_of(
    fx: &FeatureExtractor,
    m: &Machine,
    c: Condition,
    severity: f32,
    seed: u64,
) -> Vec<f32> {
    fx.extract(&m.measure(c, severity, seed)).unwrap()
}

#[test]
fn signatures_show_up_in_the_right_features() {
    let npu = Npu::simulator();
    let fx = FeatureExtractor::new(&npu).unwrap();
    let m = Machine::random_at(25.0, 3);
    let get = |c, s| features_of(&fx, &m, c, s, 11);
    let healthy = get(Condition::Healthy, 0.0);
    assert_eq!(healthy.len(), NUM_FEATURES);

    let imbalance = get(Condition::Imbalance, 1.0);
    let x1 = feature("spectrum 1.00x");
    assert!(
        imbalance[x1] > healthy[x1] + 0.3,
        "1x: {} vs {}",
        imbalance[x1],
        healthy[x1]
    );

    let misalignment = get(Condition::Misalignment, 1.0);
    let x2 = feature("spectrum 2.00x");
    assert!(misalignment[x2] > healthy[x2] + 0.3);

    // Outer race: the envelope spectrum peaks at BPFO (≈3.5x bin), and impacts raise kurtosis.
    let outer = get(Condition::OuterRace, 0.8);
    let bpfo = feature("envelope 3.50x");
    const { assert!(BPFO > 3.375 && BPFO < 3.625) };
    assert!(
        outer[bpfo] > healthy[bpfo] + 0.5,
        "{} vs {}",
        outer[bpfo],
        healthy[bpfo]
    );
    let k = feature("log kurtosis");
    assert!(outer[k] > healthy[k] + 0.2);

    // Cavitation raises the broadband features.
    let cav = get(Condition::Cavitation, 0.8);
    for name in ["log band rms", "log envelope floor"] {
        let i = feature(name);
        assert!(
            cav[i] > healthy[i] + 0.3,
            "{name}: {} vs {}",
            cav[i],
            healthy[i]
        );
    }
}

#[test]
fn order_features_do_not_depend_on_speed() {
    let npu = Npu::simulator();
    let fx = FeatureExtractor::new(&npu).unwrap();
    let bpfo = feature("envelope 3.50x");
    for hz in [16.0, 30.0, 44.0] {
        let f = features_of(
            &fx,
            &Machine::random_at(hz, 5),
            Condition::OuterRace,
            0.8,
            1,
        );
        for n in [feature("envelope 2.50x"), feature("envelope 4.50x")] {
            assert!(f[bpfo] > f[n] + 0.3, "at {hz} Hz: {} vs {}", f[bpfo], f[n]);
        }
    }
}

#[test]
fn trains_and_deploys() {
    let npu = Npu::simulator();
    let fx = FeatureExtractor::new(&npu).unwrap();
    let data = Dataset::generate(&fx, 20, (0.4, 1.0), 1).unwrap();
    assert_eq!(
        data.features.shape(),
        (20 * Condition::ALL.len(), NUM_FEATURES)
    );
    let config = TrainConfig {
        epochs: 30,
        batch: 32,
        ..Default::default()
    };
    let mut sd = model::train(&npu, &data, &config, |_, _, _| {}).unwrap();
    let validation = Dataset::generate(&fx, 10, (0.4, 1.0), 3).unwrap();
    let fit = model::add_novelty(&npu, &mut sd, &data, &validation).unwrap();
    assert!(fit.components > 0 && fit.explained >= 0.9);
    assert!(fit.covariance_error < 1e-2, "{}", fit.covariance_error);

    // Round-trip through safetensors like a deployment would.
    let path = std::env::temp_dir().join(format!(
        "machine-monitor-{}.safetensors",
        std::process::id()
    ));
    model::save(&sd, &path).unwrap();
    let diag = Diagnoser::from_file(&npu, &path).unwrap();
    let _ = std::fs::remove_file(&path);

    let test = Dataset::generate(&fx, 6, (0.5, 1.0), 77).unwrap();
    let confusion = Confusion::new(
        &diag.classify_features(&test.features).unwrap(),
        &test.labels,
    );
    assert!(
        confusion.accuracy() > 0.8,
        "test accuracy {}\n{confusion}",
        confusion.accuracy()
    );

    // The Module interface (used by the server and the pool) gives the same verdict.
    let m = Machine::random_at(31.0, 9).measure(Condition::InnerRace, 0.9, 4);
    let row: Vec<f32> = std::iter::once(m.shaft_hz)
        .chain(m.samples.iter().copied())
        .collect();
    let via_module = diag
        .forward(&Tensor::from_f32(&row, [1, INPUT_WIDTH]).unwrap())
        .unwrap();
    assert_eq!(via_module.shape(), [1, OUTPUT_WIDTH]);
    let a = Assessment::from_row(&via_module.to_f32_vec());
    assert_eq!(a.most_likely().0, Condition::InnerRace);
    // The direct path skips the bf16 rounding of the Module input; same verdict.
    assert_eq!(
        diag.assess(std::slice::from_ref(&m)).unwrap()[0]
            .most_likely()
            .0,
        Condition::InnerRace
    );
    assert!(diag.forward(&Tensor::zeros([1, WINDOW])).is_err());

    // Novelty: severe cavitation (never trained on) scores far above known conditions.
    assert!(diag.has_novelty());
    let mut known = diag.novelty_scores(&test.features).unwrap();
    known.sort_by(f32::total_cmp);
    let known_median = known[known.len() / 2];
    let cavitation =
        Dataset::generate_condition(&fx, Condition::Cavitation, 6, (1.0, 1.0), 100).unwrap();
    let scores = diag.novelty_scores(&cavitation.features).unwrap();
    assert!(
        scores.iter().filter(|s| **s > 1.0).count() >= 4,
        "cavitation scores {scores:?}, known median {known_median}"
    );
    assert!(known_median < 0.8, "{known_median}");
    let err = diag.self_check(1).unwrap().unwrap();
    assert!(err < 0.02, "self-check {err}");
}

fn assessment(probs: [f32; 6], novelty: f32) -> Assessment {
    Assessment { probs, novelty }
}

#[test]
fn alarms_need_a_few_confirming_hours() {
    let mut t = AlarmTracker::default();
    let healthy = assessment([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.1);
    let outer = assessment([0.0, 0.0, 0.0, 0.0, 1.0, 0.0], 0.2);
    let inner = assessment([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], 0.2);
    // One bad reading between good ones: no alarm.
    for p in [&healthy, &outer, &healthy, &healthy] {
        assert_eq!(t.update(p), Event::None);
    }
    let events: Vec<Event> = (0..8).map(|_| t.update(&outer)).collect();
    assert_eq!(
        events
            .iter()
            .filter(|e| matches!(e, Event::Alarm { .. }))
            .count(),
        1
    );
    assert_eq!(t.active, Some(Diagnosis::Fault(Condition::OuterRace)));
    let later: Vec<Event> = (0..6).map(|_| t.update(&inner)).collect();
    assert!(later.iter().any(|e| matches!(
        e,
        Event::Reclassified {
            diagnosis: Diagnosis::Fault(Condition::InnerRace),
            ..
        }
    )));
}

#[test]
fn sustained_novelty_overrides_the_classifier() {
    // The classifier is sure it's healthy, but the pattern is unlike anything known.
    let odd = assessment([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 3.0);
    let mut t = AlarmTracker::default();
    let events: Vec<Event> = (0..6).map(|_| t.update(&odd)).collect();
    assert!(events.iter().any(|e| matches!(
        e,
        Event::Alarm {
            diagnosis: Diagnosis::Unrecognised,
            ..
        }
    )));
    // A known-fault alarm is upgraded to "unrecognised" once novelty persists.
    let mut t = AlarmTracker::default();
    for _ in 0..5 {
        t.update(&assessment([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], 0.3));
    }
    assert_eq!(t.active, Some(Diagnosis::Fault(Condition::InnerRace)));
    let events: Vec<Event> = (0..8)
        .map(|_| t.update(&assessment([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], 3.0)))
        .collect();
    assert!(events.iter().any(|e| matches!(
        e,
        Event::Reclassified {
            diagnosis: Diagnosis::Unrecognised,
            ..
        }
    )));
    // A single odd hour does nothing.
    let mut t = AlarmTracker::default();
    t.update(&odd);
    for _ in 0..5 {
        assert_eq!(
            t.update(&assessment([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.1)),
            Event::None
        );
    }
}

#[test]
fn plant_ground_truth_and_scoring() {
    let plant = Plant::simulate(12, 100, 1);
    assert_eq!(plant.assets.len(), 12);
    assert!(
        plant
            .assets
            .iter()
            .any(|a| a.degradation.map(|d| d.condition) == Some(Condition::Cavitation)),
        "the plant includes a fault the classifier never saw"
    );
    let faulty: Vec<_> = plant
        .assets
        .iter()
        .filter_map(|a| a.degradation.map(|d| (a, d)))
        .collect();
    assert_eq!(faulty.len(), 4);
    for (a, d) in &faulty {
        assert_ne!(d.condition, Condition::Healthy);
        assert_eq!(
            a.state_at(d.onset_hour.saturating_sub(1)).0,
            Condition::Healthy
        );
        assert_eq!(a.state_at(d.onset_hour + d.ramp_hours).1, 1.0);
    }
    assert!(plant.measure(5).iter().all(|m| m.samples.len() == WINDOW));

    // A perfect classifier fed the ground truth: every developed fault is detected.
    let mut monitor = Monitor::new(plant.assets.len());
    for hour in 0..plant.hours {
        let truth: Vec<Assessment> = plant
            .assets
            .iter()
            .map(|a| {
                let (c, s) = a.state_at(hour);
                let mut probs = [0.0; 6];
                match c.class() {
                    Some(k) if s > 0.1 => probs[k] = 1.0,
                    None if s > 0.1 => probs[0] = 1.0,
                    _ => probs[0] = 1.0,
                }
                assessment(
                    probs,
                    if c.class().is_none() && s > 0.1 {
                        3.0
                    } else {
                        0.1
                    },
                )
            })
            .collect();
        monitor.step(hour, &truth);
    }
    for (i, a) in plant.assets.iter().enumerate() {
        let outcome = monitor.outcome(&plant, i);
        match a.degradation {
            None => assert_eq!(outcome, Outcome::Quiet, "{}", a.name),
            Some(d) => assert!(
                matches!(outcome, Outcome::Detected { new_fault_type, .. } if new_fault_type == d.condition.class().is_none()),
                "{}: {outcome:?}",
                a.name
            ),
        }
    }
    // Matrix input: the same plant measurement as Module rows.
    let rows = Matrix::from_rows(
        &plant
            .measure(0)
            .into_iter()
            .map(|m| {
                std::iter::once(m.shaft_hz)
                    .chain(m.samples)
                    .collect::<Vec<f32>>()
            })
            .collect::<Vec<_>>(),
    )
    .unwrap();
    assert_eq!(rows.shape(), (12, INPUT_WIDTH));
}

#[test]
fn condition_names_round_trip() {
    for c in Condition::ALL.into_iter().chain(Condition::UNSEEN) {
        assert_eq!(c.label().parse::<Condition>().unwrap(), c);
    }
    assert_eq!(Condition::Cavitation.class(), None);
    assert!("broken".parse::<Condition>().is_err());
}
