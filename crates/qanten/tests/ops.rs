//! Every op is checked against a naive f32 reference implementation on the simulator.

use qanten::ops::{self, Conv2dParams, ConvTranspose2dParams, Hw, Pool2dParams};
use qanten::{Npu, Tensor};

fn npu() -> Npu {
    Npu::simulator()
}

#[track_caller]
fn assert_close(got: &Tensor, want: &[f32], shape: &[usize], tol: f32) {
    assert_eq!(got.shape(), shape, "shape");
    let g = got.to_f32_vec();
    for (i, (a, b)) in g.iter().zip(want).enumerate() {
        let scale = 1.0f32.max(b.abs());
        assert!(
            (a - b).abs() <= tol * scale,
            "element {i}: got {a}, want {b} (tol {tol})"
        );
    }
}

fn naive_conv(x: &Tensor, w: &Tensor, p: &Conv2dParams) -> (Vec<f32>, Vec<usize>) {
    let [n, c, h, wd] = x.dims::<4>().unwrap();
    let [oc, icg, kh, kw] = w.dims::<4>().unwrap();
    let g = p.groups;
    let ocg = oc / g;
    let out = p.output_size(Hw::new(h, wd), Hw::new(kh, kw)).unwrap();
    let (xv, wv) = (x.to_f32_vec(), w.to_f32_vec());
    let mut y = vec![0.0; n * oc * out.h * out.w];
    for b in 0..n {
        for o in 0..oc {
            let grp = o / ocg;
            for oy in 0..out.h {
                for ox in 0..out.w {
                    let mut s = 0.0;
                    for ci in 0..icg {
                        let cin = grp * icg + ci;
                        for ky in 0..kh {
                            for kx in 0..kw {
                                let iy = (oy * p.stride.h + ky * p.dilation.h) as isize
                                    - p.padding.h as isize;
                                let ix = (ox * p.stride.w + kx * p.dilation.w) as isize
                                    - p.padding.w as isize;
                                if iy >= 0 && ix >= 0 && (iy as usize) < h && (ix as usize) < wd {
                                    s += xv[((b * c + cin) * h + iy as usize) * wd + ix as usize]
                                        * wv[((o * icg + ci) * kh + ky) * kw + kx];
                                }
                            }
                        }
                    }
                    y[((b * oc + o) * out.h + oy) * out.w + ox] = s;
                }
            }
        }
    }
    (y, vec![n, oc, out.h, out.w])
}

fn naive_conv_transpose(
    x: &Tensor,
    w: &Tensor,
    p: &ConvTranspose2dParams,
) -> (Vec<f32>, Vec<usize>) {
    let [n, c, h, wd] = x.dims::<4>().unwrap();
    let [_, ocg, kh, kw] = w.dims::<4>().unwrap();
    let g = p.groups;
    let icg = c / g;
    let oc = ocg * g;
    let out = p.output_size(Hw::new(h, wd), Hw::new(kh, kw)).unwrap();
    let (xv, wv) = (x.to_f32_vec(), w.to_f32_vec());
    let mut y = vec![0.0f32; n * oc * out.h * out.w];
    // Scatter formulation: every input pixel adds a kernel-shaped stamp.
    for b in 0..n {
        for ci in 0..c {
            let grp = ci / icg;
            for iy in 0..h {
                for ix in 0..wd {
                    let v = xv[((b * c + ci) * h + iy) * wd + ix];
                    for o in 0..ocg {
                        for ky in 0..kh {
                            for kx in 0..kw {
                                let oy = (iy * p.stride.h + ky * p.dilation.h) as isize
                                    - p.padding.h as isize;
                                let ox = (ix * p.stride.w + kx * p.dilation.w) as isize
                                    - p.padding.w as isize;
                                if oy >= 0
                                    && ox >= 0
                                    && (oy as usize) < out.h
                                    && (ox as usize) < out.w
                                {
                                    let oc_i = grp * ocg + o;
                                    y[((b * oc + oc_i) * out.h + oy as usize) * out.w
                                        + ox as usize] +=
                                        v * wv[((ci * ocg + o) * kh + ky) * kw + kx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    (y, vec![n, oc, out.h, out.w])
}

#[test]
fn mul_and_periodic_nl() {
    let npu = npu();
    let a = Tensor::rand_uniform([3, 7], -2., 2., 1);
    let b = Tensor::rand_uniform([3, 7], -2., 2., 2);
    let want: Vec<f32> = a
        .to_f32_vec()
        .iter()
        .zip(b.to_f32_vec())
        .map(|(x, y)| x * y)
        .collect();
    assert_close(&ops::mul(&npu, &a, &b).unwrap(), &want, &[3, 7], 0.01);
    let want: Vec<f32> = a
        .to_f32_vec()
        .iter()
        .zip(b.to_f32_vec())
        .map(|(x, y)| x.cos() * y)
        .collect();
    assert_close(
        &ops::periodic_nl(&npu, &a, &b).unwrap(),
        &want,
        &[3, 7],
        0.01,
    );
    assert!(ops::mul(&npu, &a, &Tensor::zeros([7, 3])).is_err());
}

#[test]
fn matmul_matches_reference() {
    let npu = npu();
    let a = Tensor::rand_uniform([5, 9], -1., 1., 3);
    let b = Tensor::rand_uniform([9, 4], -1., 1., 4);
    let (av, bv) = (a.to_f32_vec(), b.to_f32_vec());
    let mut want = vec![0.0; 20];
    for i in 0..5 {
        for j in 0..4 {
            want[i * 4 + j] = (0..9).map(|k| av[i * 9 + k] * bv[k * 4 + j]).sum();
        }
    }
    assert_close(&ops::matmul(&npu, &a, &b).unwrap(), &want, &[5, 4], 0.02);
    assert!(ops::matmul(&npu, &a, &a).is_err());
}

#[test]
fn linear_1d_and_batched_with_bias() {
    let npu = npu();
    // Values from Q.ANT's upstream test suite.
    let w = Tensor::from_f32(
        &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1., 1.],
        [4, 3],
    )
    .unwrap();
    let x = Tensor::from_f32(&[0., 1., 0.5, 0.5, 0.5, 0.5], [2, 3]).unwrap();
    let y = ops::linear(&npu, &x, &w, None).unwrap();
    assert_close(
        &y,
        &[0.35, 0.8, 1.25, 1.5, 0.3, 0.75, 1.2, 1.5],
        &[2, 4],
        0.02,
    );
    let y1 = ops::linear(
        &npu,
        &Tensor::vector(&[0., 1., 0.5]),
        &w,
        Some(&Tensor::ones([4])),
    )
    .unwrap();
    assert_close(&y1, &[1.35, 1.8, 2.25, 2.5], &[4], 0.02);
    // Bias added to every row of the batch (upstream's in-place variant only touched row 0).
    let yb = ops::linear(&npu, &x, &w, Some(&Tensor::ones([4]))).unwrap();
    assert_close(
        &yb,
        &[1.35, 1.8, 2.25, 2.5, 1.3, 1.75, 2.2, 2.5],
        &[2, 4],
        0.02,
    );
}

#[test]
fn conv2d_matches_reference_for_many_configs() {
    let npu = npu();
    let configs = [
        (Conv2dParams::new(), [2, 3, 6, 5], [4, 3, 3, 3]),
        (Conv2dParams::new().padding(1), [1, 2, 4, 4], [3, 2, 3, 3]),
        (
            Conv2dParams::new().stride(2).padding(1),
            [2, 3, 7, 7],
            [5, 3, 3, 3],
        ),
        (
            Conv2dParams::new().stride((2, 1)).padding((0, 2)),
            [1, 1, 6, 5],
            [2, 1, 2, 3],
        ),
        (
            Conv2dParams::new().dilation(2).padding(2),
            [1, 2, 8, 8],
            [2, 2, 3, 3],
        ),
        (Conv2dParams::new().groups(2), [2, 4, 5, 5], [6, 2, 3, 3]),
        (
            Conv2dParams::new().groups(4).padding(1),
            [1, 4, 5, 5],
            [4, 1, 3, 3],
        ), // depthwise
        (Conv2dParams::new(), [1, 3, 4, 4], [8, 3, 1, 1]), // pointwise
    ];
    for (i, (params, xs, ws)) in configs.into_iter().enumerate() {
        let x = Tensor::rand_uniform(xs, -1., 1., 10 + i as u64);
        let w = Tensor::rand_uniform(ws, -1., 1., 20 + i as u64);
        let (want, shape) = naive_conv(&x, &w, &params);
        let got = ops::conv2d(&npu, &x, &w, None, params).unwrap();
        assert_close(&got, &want, &shape, 0.03);
    }
}

#[test]
fn conv2d_upstream_values_and_bias() {
    let npu = npu();
    let x = Tensor::from_f32(
        &[
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 0.11, 0.21, 0.31, 0.41, 0.51, 0.61,
        ],
        [1, 1, 4, 4],
    )
    .unwrap();
    let w = Tensor::from_f32(&[1., 0., 0., 1.], [1, 1, 2, 2]).unwrap();
    let y = ops::conv2d(
        &npu,
        &x,
        &w,
        Some(&Tensor::vector(&[1.0])),
        Conv2dParams::new(),
    )
    .unwrap();
    let want: Vec<f32> = [0.7, 0.9, 1.1, 1.5, 0.71, 0.91, 1.31, 1.51, 0.72]
        .iter()
        .map(|v| v + 1.0)
        .collect();
    assert_close(&y, &want, &[1, 1, 3, 3], 0.02);
}

#[test]
fn conv2d_rejects_bad_input() {
    let npu = npu();
    let x = Tensor::zeros([1, 3, 2, 2]);
    assert!(
        ops::conv2d(
            &npu,
            &x,
            &Tensor::zeros([1, 3, 3, 3]),
            None,
            Conv2dParams::new()
        )
        .is_err()
    );
    assert!(
        ops::conv2d(
            &npu,
            &x,
            &Tensor::zeros([1, 2, 1, 1]),
            None,
            Conv2dParams::new()
        )
        .is_err()
    );
    assert!(
        ops::conv2d(
            &npu,
            &x,
            &Tensor::zeros([1, 3, 1, 1]),
            None,
            Conv2dParams::new().stride(0)
        )
        .is_err()
    );
}

#[test]
fn conv_transpose2d_matches_reference() {
    let npu = npu();
    let configs = [
        (ConvTranspose2dParams::new(), [1, 2, 3, 3], [2, 3, 2, 2]),
        (
            ConvTranspose2dParams::new().stride(2),
            [2, 2, 3, 4],
            [2, 3, 3, 3],
        ),
        (
            ConvTranspose2dParams::new().stride(2).padding(1),
            [1, 3, 4, 4],
            [3, 2, 4, 4],
        ),
        (
            ConvTranspose2dParams::new()
                .stride(2)
                .padding(1)
                .output_padding(1),
            [1, 2, 3, 3],
            [2, 2, 3, 3],
        ),
        (
            ConvTranspose2dParams::new().dilation(2),
            [1, 1, 3, 3],
            [1, 2, 2, 2],
        ),
        (
            ConvTranspose2dParams::new().groups(2).stride(2),
            [1, 4, 3, 3],
            [4, 3, 2, 2],
        ),
    ];
    for (i, (params, xs, ws)) in configs.into_iter().enumerate() {
        let x = Tensor::rand_uniform(xs, -1., 1., 30 + i as u64);
        let w = Tensor::rand_uniform(ws, -1., 1., 40 + i as u64);
        let (want, shape) = naive_conv_transpose(&x, &w, &params);
        let got = ops::conv_transpose2d(&npu, &x, &w, None, params).unwrap();
        assert_close(&got, &want, &shape, 0.03);
    }
    let bad = ConvTranspose2dParams::new().output_padding(1);
    assert!(
        ops::conv_transpose2d(
            &npu,
            &Tensor::zeros([1, 1, 2, 2]),
            &Tensor::zeros([1, 1, 2, 2]),
            None,
            bad
        )
        .is_err()
    );
}

#[test]
fn pooling() {
    let x = Tensor::from_f32(
        &[
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 0.11, 0.21, 0.31, 0.41, 0.51, 0.61,
        ],
        [1, 1, 4, 4],
    )
    .unwrap();
    assert_close(
        &ops::max_pool2d(&x, Pool2dParams::new(2)).unwrap(),
        &[0.6, 0.8, 1.0, 0.61],
        &[1, 1, 2, 2],
        0.01,
    );
    assert_close(
        &ops::avg_pool2d(&x, Pool2dParams::new(2)).unwrap(),
        &[0.35, 0.55, 0.655, 0.36],
        &[1, 1, 2, 2],
        0.01,
    );
    // Padding: PyTorch gives [[0.1,0.3,0.4],[0.9,1.0,0.8],[0.31,0.51,0.61]] for max and
    // count_include_pad=False averages for avg.
    let p = Pool2dParams::new(2).padding(1);
    assert_close(
        &ops::max_pool2d(&x, p).unwrap(),
        &[0.1, 0.3, 0.4, 0.9, 1.0, 0.8, 0.31, 0.51, 0.61],
        &[1, 1, 3, 3],
        0.01,
    );
    let avg_ex = ops::avg_pool2d(&x, p.count_include_pad(false)).unwrap();
    assert_close(
        &avg_ex,
        &[0.1, 0.25, 0.4, 0.7, 0.6025, 0.505, 0.31, 0.46, 0.61],
        &[1, 1, 3, 3],
        0.01,
    );
    let avg_in = ops::avg_pool2d(&x, p).unwrap();
    assert_close(
        &avg_in,
        &[
            0.025, 0.125, 0.1, 0.35, 0.6025, 0.2525, 0.0775, 0.23, 0.1525,
        ],
        &[1, 1, 3, 3],
        0.01,
    );
    // Batched + multi-channel.
    let xb = Tensor::rand_uniform([3, 2, 6, 6], 0., 1., 9);
    assert_eq!(
        ops::max_pool2d(&xb, Pool2dParams::new(3).stride(2))
            .unwrap()
            .shape(),
        &[3, 2, 2, 2]
    );
}

#[test]
fn adaptive_pooling_non_divisible() {
    // 5 -> 3 uses windows [0,2), [1,4), [3,5) like PyTorch.
    let x = Tensor::from_f32(&(0..5).map(|v| v as f32).collect::<Vec<_>>(), [1, 1, 1, 5]).unwrap();
    let y = ops::adaptive_avg_pool2d(&x, (1, 3)).unwrap();
    assert_close(&y, &[0.5, 2.0, 3.5], &[1, 1, 1, 3], 0.01);
    let y = ops::adaptive_max_pool2d(&x, (1, 3)).unwrap();
    assert_close(&y, &[1.0, 3.0, 4.0], &[1, 1, 1, 3], 0.01);
    let g = ops::global_avg_pool2d(&Tensor::ones([2, 3, 4, 4])).unwrap();
    assert_close(&g, &[1.0; 6], &[2, 3], 0.001);
}

#[test]
fn batch_norm_and_layer_norm() {
    let npu = npu();
    let x = Tensor::from_f32(&[1., 2., 3., 4., 10., 20., 30., 40.], [2, 2, 1, 2]).unwrap();
    let mean = Tensor::vector(&[2.0, 25.0]);
    let var = Tensor::vector(&[4.0, 100.0]);
    let w = Tensor::vector(&[1.0, 2.0]);
    let b = Tensor::vector(&[0.0, 1.0]);
    let y = ops::batch_norm2d(&npu, &x, &mean, &var, Some(&w), Some(&b), 0.0).unwrap();
    assert_close(
        &y,
        &[-0.5, 0.0, -3.4, -3.2, 4.0, 9.0, 2.0, 4.0],
        &[2, 2, 1, 2],
        0.02,
    );

    let x = Tensor::from_f32(&[1., 2., 3., 4.], [1, 4]).unwrap();
    let y = ops::layer_norm(&x, None, None, 0.0).unwrap();
    assert_close(&y, &[-1.3416, -0.4472, 0.4472, 1.3416], &[1, 4], 0.01);
}

#[test]
fn activations_and_softmax() {
    let x = Tensor::from_f32(&[-1., 0., 2., 1., 1., 1.], [2, 3]).unwrap();
    assert_close(&ops::relu(&x), &[0., 0., 2., 1., 1., 1.], &[2, 3], 0.0);
    assert_close(
        &ops::sigmoid(&x),
        &[0.2689, 0.5, 0.8808, 0.7311, 0.7311, 0.7311],
        &[2, 3],
        0.01,
    );
    let s = ops::softmax(&x).unwrap();
    assert_close(
        &s,
        &[0.0420, 0.1142, 0.8438, 1. / 3., 1. / 3., 1. / 3.],
        &[2, 3],
        0.01,
    );
    // Large values do not overflow.
    let big = ops::softmax(&Tensor::vector(&[1000., 1000.])).unwrap();
    assert_close(&big, &[0.5, 0.5], &[2], 0.01);
}

#[test]
#[allow(clippy::needless_range_loop)]
fn kan_layer_matches_formula() {
    let npu = npu();
    let (nb, ni, no, nk) = (3, 4, 2, 3);
    let x = Tensor::rand_uniform([nb, ni], -1., 1., 50);
    let phis = Tensor::rand_uniform([no, ni, nk], -3., 3., 51);
    let ampls = Tensor::rand_uniform([no, ni, nk], -1., 1., 52);
    let ks = Tensor::vector(&[1.0, 2.0, 3.0]);
    let (xv, pv, av, kv) = (
        x.to_f32_vec(),
        phis.to_f32_vec(),
        ampls.to_f32_vec(),
        ks.to_f32_vec(),
    );
    let mut want = vec![0.0; nb * no];
    for b in 0..nb {
        for j in 0..no {
            for i in 0..ni {
                for k in 0..nk {
                    let idx = (j * ni + i) * nk + k;
                    want[b * no + j] += (kv[k] * xv[b * ni + i] + pv[idx]).cos() * av[idx];
                }
            }
        }
    }
    let got = ops::kan_layer(&npu, &x, &phis, &ampls, &ks).unwrap();
    assert_close(&got, &want, &[nb, no], 0.1);
}
