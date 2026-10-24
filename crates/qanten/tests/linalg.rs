//! The mixed-precision solvers must reach f64-level accuracy despite bf16 NPU products.

use qanten::Npu;
use qanten::linalg::{self, LinearOperator, SolveOptions};

fn rng(seed: u64) -> impl FnMut() -> f64 {
    let mut s = seed;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
}

/// A = MᵀM + n·I: symmetric positive definite, moderately conditioned.
fn spd(n: usize, seed: u64) -> Vec<f64> {
    let mut r = rng(seed);
    let m: Vec<f64> = (0..n * n).map(|_| r()).collect();
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            a[i * n + j] = (0..n).map(|k| m[k * n + i] * m[k * n + j]).sum::<f64>()
                + if i == j { n as f64 * 0.5 } else { 0.0 };
        }
    }
    a
}

#[test]
fn cg_refinement_reaches_f64_accuracy() {
    let npu = Npu::simulator();
    let n = 64;
    let a = spd(n, 1);
    let op = LinearOperator::new(&npu, &a, n, n).unwrap();
    let mut r = rng(2);
    let x_true: Vec<f64> = (0..n).map(|_| r()).collect();
    let b = op.apply_exact(&x_true);

    // Without refinement, bf16 limits the accuracy...
    let one_shot = SolveOptions {
        max_refinements: 1,
        ..SolveOptions::default()
    };
    let (_, rep1) = linalg::conjugate_gradient(&op, &b, &one_shot).unwrap();
    assert!(
        rep1.relative_residual > 1e-6,
        "single pass is bf16-limited: {}",
        rep1.relative_residual
    );

    // ...with refinement it reaches the f64 target.
    let (x, rep) = linalg::conjugate_gradient(&op, &b, &SolveOptions::default()).unwrap();
    assert!(rep.converged, "{rep:?}");
    let err = x
        .iter()
        .zip(&x_true)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(err < 1e-8, "solution error {err}");
    println!("CG: {rep:?}, max error {err:e}");
}

#[test]
fn least_squares_matches_normal_equations() {
    let npu = Npu::simulator();
    let (rows, cols) = (80, 12);
    let mut r = rng(3);
    let a: Vec<f64> = (0..rows * cols).map(|_| r()).collect();
    let b: Vec<f64> = (0..rows).map(|_| r()).collect();
    let op = LinearOperator::with_transpose(&npu, &a, rows, cols).unwrap();
    let (x, rep) = linalg::least_squares(&op, &b, &SolveOptions::default()).unwrap();
    assert!(rep.converged, "{rep:?}");
    // Reference: solve AᵀA x = Aᵀb with f64 Cholesky.
    let mut ata = vec![0.0; cols * cols];
    for i in 0..cols {
        for j in 0..cols {
            ata[i * cols + j] = (0..rows).map(|k| a[k * cols + i] * a[k * cols + j]).sum();
        }
    }
    let atb: Vec<f64> = (0..cols)
        .map(|i| (0..rows).map(|k| a[k * cols + i] * b[k]).sum())
        .collect();
    let mut l = vec![0.0; cols * cols];
    for i in 0..cols {
        for j in 0..=i {
            let s: f64 = (0..j).map(|k| l[i * cols + k] * l[j * cols + k]).sum();
            l[i * cols + j] = if i == j {
                (ata[i * cols + i] - s).sqrt()
            } else {
                (ata[i * cols + j] - s) / l[j * cols + j]
            };
        }
    }
    let mut y = vec![0.0; cols];
    for i in 0..cols {
        y[i] = (atb[i] - (0..i).map(|k| l[i * cols + k] * y[k]).sum::<f64>()) / l[i * cols + i];
    }
    let mut want = vec![0.0; cols];
    for i in (0..cols).rev() {
        want[i] = (y[i]
            - (i + 1..cols)
                .map(|k| l[k * cols + i] * want[k])
                .sum::<f64>())
            / l[i * cols + i];
    }
    let err = x
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(err < 1e-8, "least squares error {err}");
}

#[test]
fn eigenpairs_of_known_spectrum() {
    let npu = Npu::simulator();
    let n = 40;
    // A = Q·diag(λ)·Qᵀ with a random orthogonal Q.
    let mut r = rng(4);
    let mut q: Vec<Vec<f64>> = (0..n).map(|_| (0..n).map(|_| r()).collect()).collect();
    for i in 0..n {
        for j in 0..i {
            let d: f64 = q[i].iter().zip(&q[j]).map(|(a, b)| a * b).sum();
            let qj = q[j].clone();
            q[i].iter_mut().zip(&qj).for_each(|(a, b)| *a -= d * b);
        }
        let nrm = q[i].iter().map(|v| v * v).sum::<f64>().sqrt();
        q[i].iter_mut().for_each(|v| *v /= nrm);
    }
    let lambdas: Vec<f64> = (0..n)
        .map(|i| {
            if i < 3 {
                [10.0, -7.0, 5.0][i]
            } else {
                1.0 / (i as f64)
            }
        })
        .collect();
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            a[i * n + j] = (0..n).map(|k| q[k][i] * lambdas[k] * q[k][j]).sum();
        }
    }
    let op = LinearOperator::new(&npu, &a, n, n).unwrap();
    let eig = linalg::top_eigenpairs(&op, 3, 300, 1).unwrap();
    for (got, want) in eig.values.iter().zip([10.0, -7.0, 5.0]) {
        assert!((got - want).abs() < 1e-9, "eigenvalue {got} vs {want}");
    }
    println!(
        "eigen: {} NPU iterations, {} exact refinements",
        eig.iterations, eig.refinements
    );
    // Eigenvectors match ±q_k.
    for (v, k) in eig.vectors.iter().zip(0..3) {
        let d: f64 = v.iter().zip(&q[k]).map(|(a, b)| a * b).sum();
        assert!(d.abs() > 1.0 - 1e-9, "eigenvector {k} alignment {d}");
    }
}

#[test]
fn pca_finds_the_main_direction() {
    let npu = Npu::simulator();
    let (n, d) = (500, 5);
    let mut r = rng(5);
    let dir = [0.6, 0.0, 0.8, 0.0, 0.0];
    let data: Vec<f64> = (0..n)
        .flat_map(|_| {
            let t = r() * 5.0;
            let noise: Vec<f64> = (0..d).map(|_| r() * 0.1).collect();
            (0..d)
                .map(move |j| 3.0 + t * dir[j] + noise[j])
                .collect::<Vec<_>>()
        })
        .collect();
    let p = linalg::pca(&npu, &data, n, d, 2).unwrap();
    let align: f64 = p.components[0].iter().zip(dir).map(|(a, b)| a * b).sum();
    assert!(align.abs() > 0.999, "first component alignment {align}");
    assert!(p.explained_ratio[0] > 0.95);
    assert!((p.mean[0] - 3.0).abs() < 0.2);
    assert_eq!(p.transform(&data[..d]).len(), 2);
}

#[test]
fn shape_errors() {
    let npu = Npu::simulator();
    assert!(LinearOperator::new(&npu, &[1.0, 2.0], 2, 2).is_err());
    let op = LinearOperator::new(&npu, &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0], 3, 2).unwrap();
    assert!(linalg::conjugate_gradient(&op, &[1.0, 1.0, 1.0], &SolveOptions::default()).is_err());
    assert!(
        op.apply_t_npu(&[1.0, 1.0, 1.0]).is_err(),
        "no transpose uploaded"
    );
}
