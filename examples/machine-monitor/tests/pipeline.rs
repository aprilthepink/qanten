use machine_monitor::features::{FeatureExtractor, NUM_FEATURES, feature_names};
use machine_monitor::fleet::{self, AlarmTracker, Event};
use machine_monitor::model::{self, Dataset, TrainConfig};
use machine_monitor::sim::BPFO;
use machine_monitor::{Condition, Diagnoser, INPUT_WIDTH, Machine, WINDOW};
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
    let sd = model::train(&npu, &data, &config, |_, _, _| {}).unwrap();

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
    assert_eq!(via_module.shape(), [1, Condition::ALL.len()]);
    assert_eq!(via_module.argmax_last(), [Condition::InnerRace.index()]);
    assert!(diag.forward(&Tensor::zeros([1, WINDOW])).is_err());
}

#[test]
fn alarms_need_a_few_confirming_hours() {
    let mut t = AlarmTracker::default();
    let healthy = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let outer = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let inner = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    // One bad reading between good ones: no alarm.
    for p in [healthy, outer, healthy, healthy] {
        assert_eq!(t.update(&p), Event::None);
    }
    let events: Vec<Event> = (0..8).map(|_| t.update(&outer)).collect();
    let alarms = events
        .iter()
        .filter(|e| matches!(e, Event::Alarm { .. }))
        .count();
    assert_eq!(alarms, 1);
    assert_eq!(t.active, Some(Condition::OuterRace));
    let later: Vec<Event> = (0..6).map(|_| t.update(&inner)).collect();
    assert!(later.iter().any(|e| matches!(
        e,
        Event::Reclassified {
            condition: Condition::InnerRace,
            ..
        }
    )));
}

#[test]
fn plant_ground_truth() {
    let assets = fleet::plant(12, 100, 1);
    assert_eq!(assets.len(), 12);
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
    for c in Condition::ALL {
        assert_eq!(c.label().parse::<Condition>().unwrap(), c);
    }
    assert!("broken".parse::<Condition>().is_err());
}
