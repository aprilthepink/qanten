#![allow(clippy::needless_range_loop)] // reference implementations index on purpose

use std::f32::consts::PI;

use qanten::nn::{self, Module};
use qanten::train::{self, Adam, Dense, Kan, Layer, Model, Relu, Sgd, Tanh};
use qanten::{Npu, SimConfig, Tensor};

fn close(a: &[f32], b: &[f32], tol: f32) -> f32 {
    let err = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max);
    assert!(err < tol, "max error {err} >= {tol}");
    err
}

#[test]
fn dense_gradients_match_reference() {
    let npu = Npu::simulator();
    let (n_in, n_out, b) = (5, 3, 4);
    let w: Vec<f32> = (0..15)
        .map(|i| ((i * 7) % 11) as f32 / 11.0 - 0.5)
        .collect();
    let bias = vec![0.1, -0.2, 0.3];
    let weights = qanten::compute::Matrix::new(n_out, n_in, w.clone()).unwrap();
    let mut d = Dense::from_weights(&npu, weights, bias.clone()).unwrap();
    let x: Vec<f32> = (0..20).map(|i| ((i * 3) % 7) as f32 / 7.0 - 0.4).collect();
    let g: Vec<f32> = (0..12).map(|i| ((i * 5) % 9) as f32 / 9.0 - 0.5).collect();
    let y = d
        .forward(&Tensor::from_f32(&x, [b, n_in]).unwrap())
        .unwrap();
    let dx = d
        .backward(&Tensor::from_f32(&g, [b, n_out]).unwrap())
        .unwrap();
    // Reference in f32.
    let want_y: Vec<f32> = (0..b * n_out)
        .map(|k| {
            let (r, o) = (k / n_out, k % n_out);
            bias[o]
                + (0..n_in)
                    .map(|i| x[r * n_in + i] * w[o * n_in + i])
                    .sum::<f32>()
        })
        .collect();
    let want_dx: Vec<f32> = (0..b * n_in)
        .map(|k| {
            let (r, i) = (k / n_in, k % n_in);
            (0..n_out).map(|o| g[r * n_out + o] * w[o * n_in + i]).sum()
        })
        .collect();
    let want_dw: Vec<f32> = (0..n_out * n_in)
        .map(|k| {
            let (o, i) = (k / n_in, k % n_in);
            (0..b).map(|r| g[r * n_out + o] * x[r * n_in + i]).sum()
        })
        .collect();
    let want_db: Vec<f32> = (0..n_out)
        .map(|o| (0..b).map(|r| g[r * n_out + o]).sum())
        .collect();
    close(&y.to_f32_vec(), &want_y, 0.02);
    close(&dx.to_f32_vec(), &want_dx, 0.02);
    let p = d.params();
    close(p[0].grad, &want_dw, 0.02);
    close(p[1].grad, &want_db, 0.002); // the incoming gradient is stored in bf16
}

#[test]
fn kan_gradients_match_finite_differences() {
    let npu = Npu::simulator();
    let ks = [1.0, 2.0];
    let mut kan = Kan::new(&npu, 2, 2, &ks, 5).unwrap();
    let x = [0.3f32, -0.7, 0.1, 0.9];
    let g = [1.0f32, -0.5, 0.25, 2.0];
    kan.forward(&Tensor::from_f32(&x, [2, 2]).unwrap()).unwrap();
    let dx = kan
        .backward(&Tensor::from_f32(&g, [2, 2]).unwrap())
        .unwrap()
        .to_f32_vec();
    let (phis, ampls): (Vec<f64>, Vec<f64>) = {
        let p = kan.params();
        (
            p[0].value.iter().map(|&v| v as f64).collect(),
            p[1].value.iter().map(|&v| v as f64).collect(),
        )
    };
    // L = Σ_b,j g[b,j] · y[b,j] with y from the exact f64 formula.
    let loss = |x: &[f64], phis: &[f64], ampls: &[f64]| -> f64 {
        let mut l = 0.0;
        for b in 0..2 {
            for j in 0..2 {
                for i in 0..2 {
                    for k in 0..2 {
                        let p = (j * 2 + i) * 2 + k;
                        l += g[b * 2 + j] as f64
                            * (ks[k] as f64 * x[b * 2 + i] + phis[p]).cos()
                            * ampls[p];
                    }
                }
            }
        }
        l
    };
    let xd: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    let h = 1e-6;
    let num =
        |f: &dyn Fn(usize) -> f64, n: usize| -> Vec<f32> { (0..n).map(|i| f(i) as f32).collect() };
    let num_dx = num(
        &|i| {
            let mut a = xd.clone();
            a[i] += h;
            let mut b = xd.clone();
            b[i] -= h;
            (loss(&a, &phis, &ampls) - loss(&b, &phis, &ampls)) / (2.0 * h)
        },
        4,
    );
    let num_dphi = num(
        &|i| {
            let mut a = phis.clone();
            a[i] += h;
            let mut b = phis.clone();
            b[i] -= h;
            (loss(&xd, &a, &ampls) - loss(&xd, &b, &ampls)) / (2.0 * h)
        },
        8,
    );
    let num_damp = num(
        &|i| {
            let mut a = ampls.clone();
            a[i] += h;
            let mut b = ampls.clone();
            b[i] -= h;
            (loss(&xd, &phis, &a) - loss(&xd, &phis, &b)) / (2.0 * h)
        },
        8,
    );
    close(&dx, &num_dx, 0.05);
    let p = kan.params();
    close(p[0].grad, &num_dphi, 0.05);
    close(p[1].grad, &num_damp, 0.05);
}

#[test]
fn cross_entropy_gradient() {
    let logits = Tensor::from_f32(&[1.0, 2.0, 0.5, -1.0, 0.0, 3.0], [2, 3]).unwrap();
    let (_, grad) = train::cross_entropy(&logits, &[1, 2]).unwrap();
    // Central differences with a step bf16 represents exactly at these values.
    let h = 0.0625;
    let base = logits.to_f32_vec();
    for k in 0..6 {
        let (mut up, mut down) = (base.clone(), base.clone());
        up[k] += h;
        down[k] -= h;
        let (lu, _) =
            train::cross_entropy(&Tensor::from_f32(&up, [2, 3]).unwrap(), &[1, 2]).unwrap();
        let (ld, _) =
            train::cross_entropy(&Tensor::from_f32(&down, [2, 3]).unwrap(), &[1, 2]).unwrap();
        let numeric = (lu - ld) / (2.0 * h);
        assert!(
            (numeric - grad.to_f32_vec()[k]).abs() < 0.01,
            "logit {k}: {numeric} vs {}",
            grad.to_f32_vec()[k]
        );
    }
    assert!(train::cross_entropy(&logits, &[3, 0]).is_err());
}

/// Two interleaved spirals, the classic non-linear toy problem.
fn spirals(n: usize) -> (Tensor, Vec<usize>) {
    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..n {
        let t = i as f32 / n as f32 * 3.0 * PI;
        let r = 0.1 + t / (3.0 * PI);
        for (class, phase) in [(0, 0.0), (1, PI)] {
            x.extend([r * (t + phase).cos(), r * (t + phase).sin()]);
            y.push(class);
        }
    }
    (Tensor::from_f32(&x, [2 * n, 2]).unwrap(), y)
}

#[test]
fn mlp_learns_spirals_and_exports() {
    let npu = Npu::simulator();
    let (x, y) = spirals(100);
    let mut model = Model::new()
        .add(Dense::new(&npu, 2, 32, 1).unwrap())
        .add(Tanh::default())
        .add(Dense::new(&npu, 32, 32, 2).unwrap())
        .add(Tanh::default())
        .add(Dense::new(&npu, 32, 2, 3).unwrap());
    let mut opt = Adam::new(0.02);
    for _ in 0..400 {
        let logits = model.forward(&x).unwrap();
        let (_, grad) = train::cross_entropy(&logits, &y).unwrap();
        model.backward(&grad).unwrap();
        model.step(&mut opt).unwrap();
    }
    let acc = train::accuracy(&model.forward(&x).unwrap(), &y);
    assert!(acc > 0.9, "spiral accuracy {acc}");

    // Export → inference layers give the same predictions.
    let sd = model.state_dict().unwrap();
    let inference = nn::Sequential::new()
        .add(nn::Linear::load(&npu, &sd, "0").unwrap())
        .add(nn::Tanh)
        .add(nn::Linear::load(&npu, &sd, "2").unwrap())
        .add(nn::Tanh)
        .add(nn::Linear::load(&npu, &sd, "4").unwrap());
    let a = inference.forward(&x).unwrap().argmax_last();
    let b = model.forward(&x).unwrap().argmax_last();
    assert_eq!(a, b);
    assert_eq!(model.num_params(), 2 * 32 + 32 + 32 * 32 + 32 + 32 * 2 + 2);
}

#[test]
fn kan_fits_a_curve_and_exports() {
    let npu = Npu::simulator();
    let xs: Vec<f32> = (0..64).map(|i| -1.0 + 2.0 * i as f32 / 63.0).collect();
    let ys: Vec<f32> = xs.iter().map(|x| (3.0 * x).sin() + 0.3 * x).collect();
    let (x, y) = (
        Tensor::from_f32(&xs, [64, 1]).unwrap(),
        Tensor::from_f32(&ys, [64, 1]).unwrap(),
    );
    let mut model = Model::new().add(Kan::new(&npu, 1, 1, &[0.5, 1.0, 2.0, 3.0], 7).unwrap());
    let mut opt = Adam::new(0.05);
    let (start, _) = train::mse(&model.forward(&x).unwrap(), &y).unwrap();
    for _ in 0..300 {
        let (_, g) = train::mse(&model.forward(&x).unwrap(), &y).unwrap();
        model.backward(&g).unwrap();
        model.step(&mut opt).unwrap();
    }
    let (end, _) = train::mse(&model.forward(&x).unwrap(), &y).unwrap();
    assert!(
        end < start / 20.0 && end < 0.02,
        "KAN loss {start} -> {end}"
    );
    let sd = model.state_dict().unwrap();
    let layer = nn::Kan::load(&npu, &sd, "0").unwrap();
    let err = layer
        .forward(&x)
        .unwrap()
        .max_abs_diff(&model.forward(&x).unwrap())
        .unwrap();
    assert!(err < 0.05, "exported KAN differs by {err}");
}

#[test]
fn training_against_a_noisy_npu_still_converges() {
    let npu = Npu::simulator_with(SimConfig {
        noise_std: 0.02,
        ..SimConfig::default()
    });
    let x = Tensor::rand_uniform([128, 4], -1., 1., 1);
    let y = Tensor::from_f32(
        &x.to_f32_vec()
            .chunks(4)
            .map(|r| r[0] * 0.5 - r[2])
            .collect::<Vec<_>>(),
        [128, 1],
    )
    .unwrap();
    let mut model = Model::new()
        .add(Dense::new(&npu, 4, 16, 1).unwrap())
        .add(Relu::default())
        .add(Dense::new(&npu, 16, 1, 2).unwrap());
    // Aggressive SGD (lr 0.1, momentum 0.9) diverges on this noisy NPU after
    // ~130 steps without clipping; gradient clipping keeps it stable.
    let mut opt = Sgd::new(0.1, 0.9);
    let (start, _) = train::mse(&model.forward(&x).unwrap(), &y).unwrap();
    let mut worst = 0.0f32;
    for _ in 0..200 {
        let (loss, g) = train::mse(&model.forward(&x).unwrap(), &y).unwrap();
        worst = worst.max(loss);
        model.backward(&g).unwrap();
        model.clip_grad_norm(0.5);
        model.step(&mut opt).unwrap();
    }
    let (end, _) = train::mse(&model.forward(&x).unwrap(), &y).unwrap();
    assert!(end < start / 10.0, "noisy training {start} -> {end}");
    assert!(worst <= start * 1.01, "loss never blew up (worst {worst})");
}

#[test]
fn backward_without_forward_is_an_error() {
    let mut d = Dense::new(&Npu::simulator(), 2, 2, 0).unwrap();
    assert!(d.backward(&Tensor::zeros([1, 2])).is_err());
}

#[test]
fn small_gradients_survive_a_noisy_npu() {
    // Cross-entropy gradients are ~1/batch; without per-tensor gradient scaling
    // absolute NPU noise of 0.05 drowns them and this stays at chance level.
    let npu = Npu::simulator_with(SimConfig {
        noise_std: 0.05,
        ..SimConfig::default()
    });
    let (x, y) = spirals(100);
    let mut model = Model::new()
        .add(Dense::new(&npu, 2, 32, 1).unwrap())
        .add(Tanh::default())
        .add(Dense::new(&npu, 32, 32, 2).unwrap())
        .add(Tanh::default())
        .add(Dense::new(&npu, 32, 2, 9).unwrap());
    let mut opt = Adam::new(0.02);
    for _ in 0..300 {
        let (_, g) = train::cross_entropy(&model.forward(&x).unwrap(), &y).unwrap();
        model.backward(&g).unwrap();
        model.clip_grad_norm(1.0);
        model.step(&mut opt).unwrap();
    }
    let acc = train::accuracy(&model.forward(&x).unwrap(), &y);
    assert!(acc > 0.9, "noisy accuracy {acc}");
}
