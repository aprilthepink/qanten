//! Neural-field layers: Fourier features, SIREN's sine on the NPU, SIREN
//! initialisation and training, and loading a trained SIREN for inference.

use qanten::compute::Matrix;
use qanten::nn::{self, Module};
use qanten::train::{self, Adam, Dense, Layer};
use qanten::{Npu, Tensor};

#[test]
fn fourier_features_match_the_formula() {
    let npu = Npu::simulator();
    let ff = nn::FourierFeatures::gaussian(&npu, 16, 2, 10.0, 3);
    assert_eq!((ff.in_features(), ff.out_features()), (2, 32));
    let coords = Matrix::from_fn(50, 2, |r, c| (r as f32 * 0.037 + c as f32 * 0.5).sin());
    let got = ff.encode(&coords).unwrap();
    let b = ff.matrix();
    for r in 0..coords.rows() {
        for k in 0..16 {
            let phase: f64 = (0..2)
                .map(|d| b[(k, d)] as f64 * coords[(r, d)] as f64)
                .sum::<f64>()
                * std::f64::consts::TAU;
            // bf16 output of the NPU: ~0.4 % of 1, plus the phase rounding.
            assert!((got[(r, k)] as f64 - phase.cos()).abs() < 0.01);
            assert!((got[(r, k + 16)] as f64 - phase.sin()).abs() < 0.01);
        }
    }
    assert!(ff.encode(&Matrix::zeros(3, 5)).is_err());
}

#[test]
fn sine_runs_on_the_npu_and_its_gradient_is_right() {
    let npu = Npu::simulator();
    let x = Tensor::rand_uniform([4, 32], -1.0, 1.0, 1);
    let omega = 30.0;
    let want: Vec<f32> = x.to_f32_vec().iter().map(|v| (omega * v).sin()).collect();
    let inference = nn::Sine::new(&npu, omega).forward(&x).unwrap().to_f32_vec();
    let mut layer = train::Sine::on_npu(&npu, omega);
    let trained = layer.forward(&x).unwrap().to_f32_vec();
    for ((a, b), w) in inference.iter().zip(&trained).zip(&want) {
        assert!((a - w).abs() < 0.01 && (b - w).abs() < 0.01);
    }
    // Backward: ω₀·cos(ω₀·x) · g.
    let g = Tensor::ones([4, 32]);
    let dx = layer.backward(&g).unwrap().to_f32_vec();
    for (d, v) in dx.iter().zip(x.to_f32_vec()) {
        let exact = omega * (omega * v).cos();
        assert!((d - exact).abs() < 0.01 * omega, "{d} vs {exact}");
    }
}

#[test]
fn fourier_feature_gradient_matches_finite_differences() {
    let npu = Npu::simulator();
    let b = Matrix::from_rows(&[[0.5f32, -0.25], [0.125, 0.75]]).unwrap();
    let mut layer = train::FourierFeatures::new(nn::FourierFeatures::new(&npu, b.clone()));
    let x = [0.3f32, -0.2];
    layer
        .forward(&Tensor::from_f32(&x, [1, 2]).unwrap())
        .unwrap();
    // Loss = Σ outputs, so g = 1 everywhere.
    let dx = layer.backward(&Tensor::ones([1, 4])).unwrap().to_f32_vec();
    let loss = |x: [f64; 2]| -> f64 {
        (0..2)
            .map(|k| {
                let p = std::f64::consts::TAU * (b[(k, 0)] as f64 * x[0] + b[(k, 1)] as f64 * x[1]);
                p.cos() + p.sin()
            })
            .sum()
    };
    let h = 1e-5;
    for d in 0..2 {
        let mut hi = [x[0] as f64, x[1] as f64];
        let mut lo = hi;
        hi[d] += h;
        lo[d] -= h;
        let numeric = (loss(hi) - loss(lo)) / (2.0 * h);
        assert!(
            (dx[d] as f64 - numeric).abs() < 0.05,
            "{} vs {numeric}",
            dx[d]
        );
    }
    // B is exported as a buffer, so inference can load it.
    assert_eq!(layer.buffers()[0].0, "b");
}

#[test]
fn siren_initialisation_bounds() {
    let npu = Npu::simulator();
    let mut first = Dense::siren(&npu, 2, 64, 30.0, true, 1).unwrap();
    let mut hidden = Dense::siren(&npu, 64, 64, 30.0, false, 2).unwrap();
    let max = |l: &mut Dense| {
        l.params()[0]
            .value
            .iter()
            .fold(0.0f32, |m, v| m.max(v.abs()))
    };
    assert!(max(&mut first) <= 0.5);
    let bound = (6.0f32 / 64.0).sqrt() / 30.0;
    assert!(max(&mut hidden) <= bound && max(&mut hidden) > 0.8 * bound);
}

#[test]
fn a_siren_learns_a_high_frequency_signal_and_reloads_for_inference() {
    let npu = Npu::simulator();
    // y = sin(12x) + 0.5·sin(31x): too wiggly for a small ReLU net to learn quickly.
    let n = 256;
    let xs: Vec<f32> = (0..n)
        .map(|i| -1.0 + 2.0 * i as f32 / (n - 1) as f32)
        .collect();
    let ys: Vec<f32> = xs
        .iter()
        .map(|x| (12.0 * x).sin() + 0.5 * (31.0 * x).sin())
        .collect();
    let x = Tensor::from_f32(&xs, [n, 1]).unwrap();
    let y = Tensor::from_f32(&ys, [n, 1]).unwrap();

    let mut model = train::siren(&npu, &[1, 48, 48, 1], 30.0, 7).unwrap();
    let mut opt = Adam::new(2e-3);
    let (first, _) = train::mse(&model.forward(&x).unwrap(), &y).unwrap();
    for _ in 0..300 {
        let pred = model.forward(&x).unwrap();
        let (_, grad) = train::mse(&pred, &y).unwrap();
        model.backward(&grad).unwrap();
        model.step(&mut opt).unwrap();
    }
    let (last, _) = train::mse(&model.forward(&x).unwrap(), &y).unwrap();
    assert!(last < first / 20.0, "loss {first} → {last}");
    assert!(last < 0.02, "final loss {last}");

    // The state dict loads into an inference model that fits as well. Its
    // `Linear` layers round differently from training's `Dense` (bf16
    // tensors vs scaled f32), which ω₀ = 30 magnifies, so compare the fit
    // rather than the raw outputs.
    let sd = model.state_dict().unwrap();
    let inference = nn::Sequential::load_siren(&npu, &sd, 30.0).unwrap();
    assert_eq!(inference.len(), 5);
    let (reloaded, _) = train::mse(&inference.forward(&x).unwrap(), &y).unwrap();
    assert!(
        reloaded < 2.0 * last + 0.01,
        "inference loss {reloaded} vs {last}"
    );
}
