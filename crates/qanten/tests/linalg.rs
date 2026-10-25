//! The mixed-precision solvers must reach f64-level accuracy despite bf16 NPU products.

use qanten::Npu;
use qanten::compute::{Matrix, Matrix64};
use qanten::linalg::{LinearOperator, Pca, SolveOptions};
use qanten::rng::SplitMix64;

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

/// A = MᵀM + (n/2)·I: symmetric positive definite, moderately conditioned.
fn spd(n: usize, seed: u64) -> Matrix64 {
    let m = Matrix64::random_normal(n, n, seed);
    let mut a = m.transpose().matmul(&m).unwrap();
    (0..n).for_each(|i| a[(i, i)] += n as f64 * 0.5);
    a
}

/// `Qᵀ·diag(λ)·Q` for a random orthogonal `Q` (rows are the eigenvectors).
fn with_spectrum(lambdas: &[f64], seed: u64) -> (Matrix64, Matrix64) {
    let n = lambdas.len();
    let mut q = Matrix64::random_normal(n, n, seed);
    q.orthonormalize_rows();
    let scaled = Matrix64::from_fn(n, n, |i, j| q[(i, j)] * lambdas[i]);
    (q.transpose().matmul(&scaled).unwrap(), q)
}

#[test]
fn cg_refinement_reaches_f64_accuracy() {
    let npu = Npu::simulator();
    let op = LinearOperator::new(&npu, &spd(64, 1)).unwrap();
    let mut rng = SplitMix64::new(2);
    let x_true: Vec<f64> = (0..64).map(|_| rng.uniform(-1.0, 1.0) as f64).collect();
    let b = op.apply_exact(&x_true);

    // Without refinement, bf16 limits the accuracy...
    let one_shot = SolveOptions {
        max_refinements: 1,
        ..SolveOptions::default()
    };
    let (_, rep1) = op.conjugate_gradient(&b, &one_shot).unwrap();
    assert!(
        rep1.relative_residual > 1e-6,
        "single pass is bf16-limited: {}",
        rep1.relative_residual
    );

    // ...with refinement it reaches the f64 target.
    let (x, rep) = op.conjugate_gradient(&b, &SolveOptions::default()).unwrap();
    assert!(rep.converged, "{rep:?}");
    let err = max_abs_diff(&x, &x_true);
    assert!(err < 1e-8, "solution error {err}");
}

#[test]
fn least_squares_solves_the_normal_equations() {
    let npu = Npu::simulator();
    let a = Matrix64::random_normal(80, 12, 3);
    let b: Vec<f64> = Matrix64::random_normal(1, 80, 4).into_vec();
    let op = LinearOperator::with_transpose(&npu, &a).unwrap();
    let (x, rep) = op.least_squares(&b, &SolveOptions::default()).unwrap();
    assert!(rep.converged, "{rep:?}");
    // At the minimum the gradient Aᵀ(A·x − b) vanishes.
    let residual: Vec<f64> = op
        .apply_exact(&x)
        .iter()
        .zip(&b)
        .map(|(ax, b)| ax - b)
        .collect();
    let gradient = op.apply_t_exact(&residual);
    let scale = op
        .apply_t_exact(&b)
        .iter()
        .map(|v| v.abs())
        .fold(0.0, f64::max);
    assert!(
        gradient.iter().all(|g| g.abs() < 1e-9 * scale),
        "{gradient:?}"
    );
}

#[test]
fn eigenpairs_of_known_spectrum() {
    let npu = Npu::simulator();
    let lambdas: Vec<f64> = (0..40)
        .map(|i| [10.0, -7.0, 5.0].get(i).copied().unwrap_or(1.0 / i as f64))
        .collect();
    let (a, q) = with_spectrum(&lambdas, 4);
    let eig = LinearOperator::new(&npu, &a)
        .unwrap()
        .top_eigenpairs(3, 300, 1)
        .unwrap();
    for (got, want) in eig.values.iter().zip([10.0, -7.0, 5.0]) {
        assert!((got - want).abs() < 1e-9, "eigenvalue {got} vs {want}");
    }
    // Eigenvectors match ±q_k.
    for k in 0..3 {
        let d: f64 = eig
            .vectors
            .row(k)
            .iter()
            .zip(q.row(k))
            .map(|(a, b)| a * b)
            .sum();
        assert!(d.abs() > 1.0 - 1e-9, "eigenvector {k} alignment {d}");
    }
}

#[test]
fn exact_symmetric_eigen_on_the_host() {
    let (a, _) = with_spectrum(&[3.0, 2.0, -1.0, 0.5], 9);
    let (values, vectors) = a.symmetric_eigen().unwrap();
    assert!(
        max_abs_diff(&values, &[3.0, 2.0, 0.5, -1.0]) < 1e-12,
        "{values:?}"
    );
    for (k, &l) in values.iter().enumerate() {
        let v = vectors.row(k);
        assert!(
            max_abs_diff(
                &a.matvec(v).unwrap(),
                &v.iter().map(|x| x * l).collect::<Vec<_>>()
            ) < 1e-12
        );
    }
}

#[test]
fn pca_finds_the_main_direction() {
    let npu = Npu::simulator();
    let dir = [0.6, 0.0, 0.8, 0.0, 0.0];
    let mut rng = SplitMix64::new(5);
    let samples: Vec<[f64; 5]> = (0..500)
        .map(|_| {
            let t = rng.uniform(-5.0, 5.0) as f64;
            dir.map(|d| 3.0 + t * d + rng.uniform(-0.1, 0.1) as f64)
        })
        .collect();
    let data = Matrix64::from_rows(&samples).unwrap();
    let p = Pca::fit(&npu, &data, 2).unwrap();
    let align: f64 = p
        .components
        .row(0)
        .iter()
        .zip(dir)
        .map(|(a, b)| a * b)
        .sum();
    assert!(align.abs() > 0.999, "first component alignment {align}");
    assert!(p.explained_ratio[0] > 0.95);
    assert!((p.mean[0] - 3.0).abs() < 0.2);
    assert_eq!(p.transform(&data.slice_rows(0..3)).unwrap().shape(), (3, 2));
}

#[test]
fn shape_errors() {
    let npu = Npu::simulator();
    assert!(Matrix64::new(2, 2, vec![1.0, 2.0]).is_err());
    let op = LinearOperator::new(
        &npu,
        &Matrix64::from_rows(&[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]).unwrap(),
    )
    .unwrap();
    assert!(
        op.conjugate_gradient(&[1.0, 1.0, 1.0], &SolveOptions::default())
            .is_err()
    );
    assert!(
        op.apply_t_npu(&Matrix::new(1, 3, vec![1.0; 3]).unwrap())
            .is_err(),
        "no transpose uploaded"
    );
    assert!(Pca::fit(&npu, &Matrix64::zeros(1, 3), 1).is_err());
}
