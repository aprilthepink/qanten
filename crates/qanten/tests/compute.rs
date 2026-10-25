use qanten::compute::{AutoPolicy, Matrix, NpuMatrix, Placement, Tiling};
use qanten::pool::NpuPool;
use qanten::{Npu, SimConfig};

#[test]
fn matrix_basics() {
    let a = Matrix::from_fn(2, 3, |i, j| (i * 3 + j) as f32);
    assert_eq!(a.row(1), &[3.0, 4.0, 5.0]);
    assert_eq!(a[(1, 2)], 5.0);
    let t = a.transpose();
    assert_eq!((t.rows(), t.cols()), (3, 2));
    assert_eq!(t[(2, 1)], 5.0);
    let p = a.matmul(&t).unwrap();
    assert_eq!(p.as_slice(), &[5.0, 14.0, 14.0, 50.0]);
    assert_eq!(Matrix::identity(3).matmul(&t).unwrap(), t);
    assert_eq!(a.columns(1..3).row(0), &[1.0, 2.0]);
    assert!(Matrix::new(2, 2, vec![1.0]).is_err());
    let mut grown = Matrix::zeros(0, 0);
    grown.push_row(&[1.0, 2.0]).unwrap();
    grown.push_row(&[3.0, 4.0]).unwrap();
    assert!(grown.push_row(&[1.0]).is_err());
    assert_eq!(grown.select_rows(&[1, 0]).row(0), &[3.0, 4.0]);
    assert_eq!(grown.cast::<f64>().cast::<f32>(), grown);
    assert_eq!(grown.column_means(), vec![2.0, 3.0]);
    assert!(a.matmul(&a).is_err());
    let r = Matrix::from_rows(&[[1.0, 2.0], [3.0, 4.0]]).unwrap();
    assert_eq!(Matrix::from_tensor(&r.to_tensor().unwrap()).unwrap(), r);
}

#[test]
fn resident_matrix_matches_cpu() {
    let npu = Npu::simulator();
    let a = Matrix::random_normal(96, 80, 1);
    let x = Matrix::random_normal(50, 80, 2);
    let exact = x.matmul(&a.transpose()).unwrap();
    let m = NpuMatrix::upload(&npu, &a).unwrap();
    let y = m.apply_batch(&x).unwrap();
    assert!(
        y.relative_error(&exact) < 1e-2,
        "{}",
        y.relative_error(&exact)
    );
    let one = m.apply(x.row(3)).unwrap();
    assert_eq!(one, y.row(3));
    assert!(m.apply(&[1.0; 3]).is_err());
}

#[test]
fn scaling_keeps_relative_accuracy_for_tiny_and_huge_values() {
    let npu = Npu::simulator();
    // Rows that differ by 1e12 in magnitude, inputs at 1e-20.
    let a = Matrix::from_fn(4, 64, |i, j| {
        (j as f32 + 1.0) * 10f32.powi(4 * i as i32 - 6)
    });
    let x = Matrix::from_fn(3, 64, |i, j| ((i + j) % 7) as f32 * 1e-20);
    let exact = x.matmul(&a.transpose()).unwrap();
    let y = NpuMatrix::upload(&npu, &a)
        .unwrap()
        .apply_batch(&x)
        .unwrap();
    for (got, want) in y.as_slice().iter().zip(exact.as_slice()) {
        assert!(((got - want) / want).abs() < 1e-2, "{got} vs {want}");
    }
}

#[test]
fn tiles_and_pools_give_the_same_answer() {
    let npu = Npu::simulator();
    let a = Matrix::random_normal(70, 90, 3);
    let x = Matrix::random_normal(20, 90, 4);
    let exact = x.matmul(&a.transpose()).unwrap();
    let tiled = NpuMatrix::upload_tiled(
        std::slice::from_ref(&npu),
        &a,
        Tiling {
            max_rows: Some(32),
            max_cols: Some(40),
        },
    )
    .unwrap();
    assert_eq!(tiled.tiles(), 9);
    let y = tiled.apply_batch(&x).unwrap();
    assert!(y.relative_error(&exact) < 1e-2);

    let pool = NpuPool::simulated(3).unwrap();
    let (npus, profilers): (Vec<_>, Vec<_>) = pool.npus().iter().map(Npu::profiled).unzip();
    let pool = NpuPool::new(npus).unwrap();
    let sharded = NpuMatrix::upload_sharded(&pool, &a).unwrap();
    assert_eq!(sharded.tiles(), 3);
    let y = sharded.apply_batch(&x).unwrap();
    assert!(y.relative_error(&exact) < 1e-2);
    for p in &profilers {
        assert!(
            p.report().total().macs > 0,
            "every NPU holds a shard and works"
        );
    }

    // Replicated: a full copy per NPU, the batch split between them.
    let before: Vec<u64> = profilers
        .iter()
        .map(|p| p.report().total().bytes_to_device)
        .collect();
    let replicated = NpuMatrix::upload_replicated(&pool, &a).unwrap();
    assert_eq!((replicated.tiles(), replicated.npus()), (3, 3));
    let y = replicated.apply_batch(&x).unwrap();
    assert!(y.relative_error(&exact) < 1e-2);
    assert_eq!(
        y.row(7),
        NpuMatrix::upload(&npu, &a)
            .unwrap()
            .apply(x.row(7))
            .unwrap()
    );
    let sent: u64 = profilers
        .iter()
        .zip(&before)
        .map(|(p, b)| p.report().total().bytes_to_device - b)
        .sum();
    // 3 matrix copies + each vector once (bf16), not three times.
    assert_eq!(sent, (3 * 70 * 90 + 20 * 90) as u64 * 2);
    let small = Matrix::random_normal(2, 90, 9); // fewer vectors than NPUs
    assert_eq!(replicated.apply_batch(&small).unwrap().rows(), 2);
}

#[test]
fn matmul_on_picks_the_resident_side_and_checks_out() {
    let npu = Npu::simulator();
    for (m, k, n) in [(120, 64, 30), (30, 64, 120)] {
        let a = Matrix::random_normal(m, k, 5);
        let b = Matrix::random_normal(k, n, 6);
        let c = a.matmul_on(&b, Placement::Npu(&npu)).unwrap();
        assert_eq!((c.rows(), c.cols()), (m, n));
        let exact = a.matmul(&b).unwrap();
        assert!(c.relative_error(&exact) < 1e-2);
        let est = c.product_error(&a, &b, 4);
        assert!(est < 1e-2, "{est}");
        // A wrong product is caught.
        let mut bad = c.clone();
        bad[(m / 2, n / 2)] += 50.0;
        assert!(bad.product_error(&a, &b, 4) > 0.05);
    }
    let z = Matrix::zeros(2, 3);
    assert!(z.matmul_on(&z, Placement::auto(&npu)).is_err());
    assert_eq!(z.product_error(&z, &z, 1), f64::INFINITY);
}

#[test]
fn auto_placement() {
    let p = AutoPolicy::default();
    assert!(!p.use_npu(8, 8, 8), "tiny: CPU");
    assert!(!p.use_npu(4096, 4096, 1), "single vector: no reuse");
    assert!(p.use_npu(512, 512, 512));
    // Auto on a tiny product is exact (CPU).
    let npu = Npu::simulator();
    let a = Matrix::from_fn(3, 3, |i, j| (i + 2 * j) as f32 + 0.123);
    let b = Matrix::identity(3);
    assert_eq!(a.matmul_on(&b, Placement::auto(&npu)).unwrap(), a);
    let x = [1.0, 2.0, 3.0];
    assert_eq!(
        a.matvec_on(&x, Placement::auto(&npu)).unwrap(),
        a.matvec(&x).unwrap()
    );
}

#[test]
fn repeated_application_markov_chain() {
    let npu = Npu::simulator();
    // Column-stochastic 3-state chain; its stationary distribution is known.
    let p = Matrix::from_rows(&[
        vec![0.9, 0.2, 0.1],
        vec![0.05, 0.7, 0.3],
        vec![0.05, 0.1, 0.6],
    ])
    .unwrap();
    let m = NpuMatrix::upload(&npu, &p).unwrap();
    let v = m.apply_repeated(&[1.0, 0.0, 0.0], 200).unwrap();
    let total: f32 = v.iter().sum();
    let v: Vec<f32> = v.iter().map(|x| x / total).collect();
    // Exact stationary distribution by CPU power iteration.
    let mut e = vec![1.0f32, 0.0, 0.0];
    for _ in 0..500 {
        e = p.matvec(&e).unwrap();
    }
    for (a, b) in v.iter().zip(&e) {
        assert!((a - b).abs() < 0.02, "{v:?} vs {e:?}");
    }
}

#[test]
fn elementwise_cos_sin_mul() {
    let npu = Npu::simulator();
    let x: Vec<f32> = (0..200).map(|i| i as f32 * 0.37 - 30.0).collect();
    let c = npu.cos(&x).unwrap();
    let s = npu.sin(&x).unwrap();
    for ((v, c), s) in x.iter().zip(&c).zip(&s) {
        assert!((c - v.cos()).abs() < 0.01, "cos {v}: {c}");
        assert!((s - v.sin()).abs() < 0.01, "sin {v}: {s}");
    }
    // Large arguments are range-reduced on the host.
    let big = npu.cos(&[1000.0, 12345.0]).unwrap();
    assert!((big[0] - 1000f32.cos()).abs() < 0.01);
    assert!((big[1] - 12345f32.cos()).abs() < 0.02);
    let m = npu.mul(&[1.5, -2.0, 3.0], &[2.0, 4.0, 0.5]).unwrap();
    assert_eq!(m, vec![3.0, -8.0, 1.5]);
    assert!(npu.mul(&[1.0], &[1.0, 2.0]).is_err());
    assert!(npu.cos(&[]).unwrap().is_empty());
}

#[test]
fn noisy_npu_error_is_measurable() {
    let npu = Npu::simulator_with(SimConfig {
        noise_std: 0.02,
        ..SimConfig::default()
    });
    let a = Matrix::random_normal(64, 64, 7);
    let b = Matrix::random_normal(64, 64, 8);
    let c = a.matmul_on(&b, Placement::Npu(&npu)).unwrap();
    let est = c.product_error(&a, &b, 8);
    let exact = c.relative_error(&a.matmul(&b).unwrap());
    assert!(est > 1e-3 && est < 0.2, "{est}");
    assert!(
        (est / exact) > 0.2 && (est / exact) < 5.0,
        "estimate {est} vs exact {exact}"
    );
}
