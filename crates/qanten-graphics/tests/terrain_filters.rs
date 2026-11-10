//! Terrain tiles and image filters: NPU path vs exact CPU path, and the
//! properties each one must have.

use glam::{Vec2, Vec3};
use qanten::Npu;
use qanten::compute::Matrix;
use qanten_graphics::filters::{Bloom, Separable, Stencil3x3, downsample_matrix, gaussian_matrix};
use qanten_graphics::terrain::{PlaneWave, Terrain, fbm_spectrum};

fn hills() -> Terrain {
    Terrain::new(
        fbm_spectrum(48, 400.0, 6.0, 40.0, 1.0, 7),
        Vec2::splat(128.0),
        65,
        65,
    )
    .unwrap()
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
fn terrain_tiles_on_the_npu_match_the_cpu() {
    let cpu = hills();
    let scale = cpu.max_height();
    let npu = hills().with_npu(&Npu::simulator()).unwrap();
    // Including a tile far from the origin: phases are reduced on the host.
    let origins = [Vec2::ZERO, Vec2::new(128.0, 0.0), Vec2::new(1.0e5, -3.0e4)];
    let want = cpu.tiles_cpu(&origins);
    let got = npu.tiles(&origins).unwrap();
    for (g, w) in got.iter().zip(&want) {
        let err = max_diff(&g.heights, &w.heights);
        assert!(
            err < 0.01 * scale,
            "height error {err} of {scale} at {}",
            g.origin
        );
        let nerr = g
            .normals
            .iter()
            .zip(&w.normals)
            .map(|(a, b)| (*a - *b).length())
            .fold(0.0, f32::max);
        assert!(nerr < 0.05, "normal error {nerr}");
    }
    assert!(npu.self_check().unwrap().unwrap() < 0.02);
}

#[test]
fn tiles_are_seamless_and_match_point_queries() {
    let t = hills();
    let tiles = t.tiles_cpu(&[Vec2::ZERO, Vec2::new(128.0, 0.0)]);
    let (a, b) = (&tiles[0], &tiles[1]);
    for j in 0..a.nz {
        let (right, left) = (a.index(a.nx - 1, j), b.index(0, j));
        assert!((a.heights[right] - b.heights[left]).abs() < 1e-4);
    }
    for (i, j) in [(0, 0), (10, 20), (64, 64)] {
        let p = a.position(i, j);
        assert!((t.height_at(p.x, p.z) - p.y).abs() < 1e-4);
        assert!((t.normal_at(p.x, p.z) - a.normals[a.index(i, j)]).length() < 1e-4);
    }
}

#[test]
fn slopes_and_masks() {
    let flat = Terrain::new(
        vec![PlaneWave {
            direction: Vec2::X,
            wavelength: 10.0,
            amplitude: 0.0,
            phase: 0.0,
        }],
        Vec2::splat(10.0),
        8,
        8,
    )
    .unwrap();
    let tile = flat.tile(Vec2::ZERO).unwrap();
    assert!(tile.heights.iter().all(|&h| h == 0.0));
    assert!(tile.slopes().iter().all(|&s| s < 1e-6));
    assert!(tile.slope_mask(0.1).iter().all(|&m| !m));
    // A steep sine: slope = atan(A·k) at its zero crossings.
    let steep = Terrain::new(
        vec![PlaneWave {
            direction: Vec2::X,
            wavelength: std::f32::consts::TAU,
            amplitude: 1.0,
            phase: 0.0,
        }],
        Vec2::splat(10.0),
        8,
        8,
    )
    .unwrap();
    let n = steep.normal_at(0.0, 0.0);
    assert!((n - Vec3::new(-1.0, 1.0, 0.0).normalize()).length() < 1e-5);
    assert!(Terrain::new(vec![], Vec2::ONE, 8, 8).is_err());
}

// ---- filters ---------------------------------------------------------------

fn image(h: usize, w: usize) -> Matrix {
    Matrix::from_fn(h, w, |y, x| {
        ((x as f32 * 0.31).sin() + (y as f32 * 0.17).cos()) * 0.5 + 0.5
    })
}

#[test]
fn gaussian_blur_matches_a_direct_convolution() {
    let (h, w, sigma) = (24, 32, 2.0f32);
    let img = image(h, w);
    let g = gaussian_matrix(w, sigma);
    for r in 0..w {
        assert!((g.row(r).iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }
    // Direct 2-D convolution with clamped edges.
    let radius = (3.0 * sigma).ceil() as isize;
    let taps: Vec<f32> = (-radius..=radius)
        .map(|d| (-(d * d) as f32 / (2.0 * sigma * sigma)).exp())
        .collect();
    let norm: f32 = taps.iter().sum::<f32>().powi(2);
    let direct = Matrix::from_fn(h, w, |y, x| {
        let mut acc = 0.0;
        for (a, ty) in taps.iter().enumerate() {
            for (b, tx) in taps.iter().enumerate() {
                let yy = (y as isize + a as isize - radius).clamp(0, h as isize - 1) as usize;
                let xx = (x as isize + b as isize - radius).clamp(0, w as isize - 1) as usize;
                acc += ty * tx * img[(yy, xx)];
            }
        }
        acc / norm
    });
    let blur = Separable::gaussian(w, h, sigma);
    let cpu = blur.apply_cpu(std::slice::from_ref(&img)).unwrap();
    assert!(max_diff(cpu[0].as_slice(), direct.as_slice()) < 1e-5);

    let npu = blur.with_npu(&Npu::simulator()).unwrap();
    let got = npu.apply(&[img.clone(), img.map(|v| 1.0 - v)]).unwrap();
    assert_eq!(got.len(), 2);
    assert!(max_diff(got[0].as_slice(), direct.as_slice()) < 0.02);
    assert!(
        Separable::gaussian(w, h, 1.0)
            .apply(&[img, Matrix::zeros(3, 3)])
            .is_err()
    );
}

#[test]
fn downsampling_halves_and_keeps_constants() {
    assert_eq!(downsample_matrix(9).shape(), (5, 9));
    let flat = Matrix::from_fn(16, 10, |_, _| 0.7);
    let out = Separable::downsample(10, 16).apply(&[flat]).unwrap();
    assert_eq!(out[0].shape(), (8, 5));
    assert!(out[0].as_slice().iter().all(|v| (v - 0.7).abs() < 1e-6));
}

#[test]
fn bloom_adds_glow_only_around_bright_pixels() {
    let (h, w) = (32, 32);
    let mut img = Matrix::from_fn(h, w, |_, _| 0.2);
    img[(16, 16)] = 10.0;
    let bloom = Bloom::new(w, h, &[1.0, 3.0, 6.0], 1.0, 0.5).unwrap();
    let cpu = bloom.apply_cpu(std::slice::from_ref(&img)).unwrap();
    // Glow spreads from the bright pixel and fades with distance.
    assert!(cpu[0][(16, 20)] > cpu[0][(16, 28)]);
    assert!(cpu[0][(16, 28)] >= 0.2);
    // Below the threshold nothing changes.
    let dim = Matrix::from_fn(h, w, |_, _| 0.5);
    let same = bloom.apply_cpu(std::slice::from_ref(&dim)).unwrap();
    assert!(max_diff(same[0].as_slice(), dim.as_slice()) < 1e-6);
    // NPU path: two calls for three scales.
    let npu = bloom.with_npu(&Npu::simulator()).unwrap();
    let got = npu.apply(std::slice::from_ref(&img)).unwrap();
    let peak = cpu[0][(16, 16)];
    assert!(max_diff(got[0].as_slice(), cpu[0].as_slice()) < 0.01 * peak);
}

#[test]
fn sobel_sharpen_and_normal_maps() {
    let ramp = Matrix::from_fn(12, 12, |_, x| x as f32);
    let sobel = Stencil3x3::sobel();
    let g = sobel.apply_cpu(std::slice::from_ref(&ramp)).unwrap();
    for y in 0..12 {
        for x in 1..11 {
            assert!((g[0][0][(y, x)] - 1.0).abs() < 1e-6, "gx");
            assert!(g[0][1][(y, x)].abs() < 1e-6, "gy");
        }
    }
    // The /8 normalisation makes a ramp of slope 1 give exactly 1.
    let edges = sobel.edges(&ramp).unwrap();
    assert!((edges[(5, 5)] - 1.0).abs() < 1e-6);

    let flat = Matrix::from_fn(8, 8, |_, _| 3.0);
    let normals = sobel.normal_map(&flat, 4.0).unwrap();
    assert!(normals.iter().all(|n| (*n - Vec3::Z).length() < 1e-6));
    let sharp = Stencil3x3::sharpen(0.8).apply_cpu(&[flat]).unwrap();
    assert!(
        sharp[0][0]
            .as_slice()
            .iter()
            .all(|v| (v - 3.0).abs() < 1e-5)
    );
    assert!(Stencil3x3::sharpen(1.0).edges(&ramp).is_err());

    let npu = Stencil3x3::sobel().with_npu(&Npu::simulator()).unwrap();
    let img = image(16, 16);
    let a = npu.apply(std::slice::from_ref(&img)).unwrap();
    let b = sobel.apply_cpu(std::slice::from_ref(&img)).unwrap();
    assert!(max_diff(a[0][0].as_slice(), b[0][0].as_slice()) < 0.01);
}
