use machine_monitor::features::{FeatureExtractor, NUM_FEATURES, feature_names};
use machine_monitor::fleet::{self, AlarmTracker, Diagnosis, Event};
use machine_monitor::model::{self, Dataset, TrainConfig};
use machine_monitor::sim::BPFO;
use machine_monitor::{
    Assessment, Condition, Diagnoser, INPUT_WIDTH, Machine, OUTPUT_WIDTH, WINDOW,
};
use qanten::nn::Module;
use qanten::{Npu, Tensor};

fn feature(name: &str) -> usize {
    feature_names().iter().position(|n| n == name).unwrap()
}

#[test]
fn signatures_show_up_in_the_right_features() {
    let npu = Npu::simulator();
    let fx = FeatureExtractor::new(&npu).unwrap();
    let m = Machine::random_at(25.0, 3);
    let get = |c, s| fx.extract(m.shaft_hz, &m.measure(c, s, 11)).unwrap();
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
}

#[test]
fn cavitation_raises_the_broadband_features() {
    let npu = Npu::simulator();
    let fx = FeatureExtractor::new(&npu).unwrap();
    let m = Machine::random_at(25.0, 3);
    let healthy = fx
        .extract(m.shaft_hz, &m.measure(Condition::Healthy, 0.0, 11))
        .unwrap();
    let cav = fx
        .extract(m.shaft_hz, &m.measure(Condition::Cavitation, 0.8, 11))
        .unwrap();
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
        let m = Machine::random_at(hz, 5);
        let f = fx
            .extract(hz, &m.measure(Condition::OuterRace, 0.8, 1))
            .unwrap();
        let neighbours = [feature("envelope 2.50x"), feature("envelope 4.50x")];
        for n in neighbours {
            assert!(f[bpfo] > f[n] + 0.3, "at {hz} Hz: {} vs {}", f[bpfo], f[n]);
        }
    }
}

#[test]
fn trains_and_deploys() {
    let npu = Npu::simulator();
    let fx = FeatureExtractor::new(&npu).unwrap();
    let data = Dataset::generate(&fx, 20, (0.4, 1.0), 1).unwrap();
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
    let probs = diag.classify_features(&test.features).unwrap();
    let acc = qanten::train::accuracy(&probs, &test.labels);
    assert!(acc > 0.8, "test accuracy {acc}");

    // The Module interface (used by the server and the pool) gives the same answer.
    let m = Machine::random_at(31.0, 9);
    let w = m.measure(Condition::InnerRace, 0.9, 4);
    let mut row = vec![m.shaft_hz];
    row.extend(&w);
    let via_module = diag
        .forward(&Tensor::from_f32(&row, [1, INPUT_WIDTH]).unwrap())
        .unwrap();
    assert_eq!(via_module.shape(), [1, OUTPUT_WIDTH]);
    let a = Assessment::from_row(&via_module.to_f32_vec());
    let best = (0..6)
        .max_by(|&i, &j| a.probs[i].total_cmp(&a.probs[j]))
        .unwrap();
    assert_eq!(best, Condition::InnerRace.index());
    // The direct path skips the bf16 rounding of the HTTP/Module input; same verdict.
    let direct = diag.assess(&[(m.shaft_hz, &w)]).unwrap()[0];
    assert!(direct.probs[best] > 0.5);

    // Novelty: severe cavitation (never trained on) scores far above known conditions.
    assert!(diag.has_novelty());
    let known: Vec<f32> = diag.novelty_scores(&test.features).unwrap();
    let mut sorted = known.clone();
    sorted.sort_by(f32::total_cmp);
    let known_median = sorted[sorted.len() / 2];
    let machines: Vec<Machine> = (0..6).map(|i| Machine::random(100 + i)).collect();
    let cav: Vec<Vec<f32>> = machines
        .iter()
        .zip(0..)
        .map(|(m, i)| m.measure(Condition::Cavitation, 1.0, i))
        .collect();
    let rows: Vec<(f32, &[f32])> = machines
        .iter()
        .zip(&cav)
        .map(|(m, x)| (m.shaft_hz, x.as_slice()))
        .collect();
    let cav_scores: Vec<f32> = diag
        .assess(&rows)
        .unwrap()
        .iter()
        .map(|a| a.novelty)
        .collect();
    let flagged = cav_scores.iter().filter(|s| **s > 1.0).count();
    assert!(
        flagged >= 4,
        "cavitation scores {cav_scores:?}, known median {known_median}"
    );
    assert!(known_median < 0.8, "{known_median}");
    let err = diag.self_check(1).unwrap().unwrap();
    assert!(err < 0.02, "self-check {err}");
    assert!(diag.forward(&Tensor::zeros([1, WINDOW])).is_err());
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
    let alarms = events
        .iter()
        .filter(|e| matches!(e, Event::Alarm { .. }))
        .count();
    assert_eq!(alarms, 1);
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
    let mut t = AlarmTracker::default();
    // The classifier is sure it's healthy, but the pattern is unlike anything known.
    let odd = assessment([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 3.0);
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
fn plant_ground_truth() {
    let assets = fleet::plant(12, 100, 1);
    assert_eq!(assets.len(), 12);
    assert!(
        assets
            .iter()
            .any(|a| a.degradation.map(|d| d.condition) == Some(Condition::Cavitation)),
        "the plant includes a fault the classifier never saw"
    );
    let faulty: Vec<_> = assets
        .iter()
        .filter_map(|a| a.degradation.map(|d| (a, d)))
        .collect();
    assert_eq!(faulty.len(), 4);
    for (a, d) in faulty {
        assert_ne!(d.condition, Condition::Healthy);
        assert_eq!(
            a.state_at(d.onset_hour.saturating_sub(1)).0,
            Condition::Healthy
        );
        assert_eq!(a.state_at(d.onset_hour + d.ramp_hours).1, 1.0);
        assert_eq!(a.measure(d.onset_hour, 1).len(), WINDOW);
    }
}

#[test]
fn condition_names_round_trip() {
    for c in Condition::ALL.into_iter().chain(Condition::UNSEEN) {
        assert_eq!(c.label().parse::<Condition>().unwrap(), c);
    }
    assert_eq!(Condition::Cavitation.class(), None);
    assert!("broken".parse::<Condition>().is_err());
}
