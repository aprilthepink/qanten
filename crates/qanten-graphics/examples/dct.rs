//! JPEG-style compression with the 8×8 DCT on the NPU: original, quality 50
//! and quality 10, side by side.
//!
//! ```sh
//! cargo run --release -p qanten-graphics --example dct -- [output.png]
//! ```

mod common;

use common::Canvas;
use glam::Vec3;
use qanten::Npu;
use qanten_graphics::dct::Dct8x8;

/// The standard JPEG luminance quantisation table.
const Q50: [f32; 64] = [
    16., 11., 10., 16., 24., 40., 51., 61., 12., 12., 14., 19., 26., 58., 60., 55., 14., 13., 16.,
    24., 40., 57., 69., 56., 14., 17., 22., 29., 51., 87., 80., 62., 18., 22., 37., 56., 68., 109.,
    103., 77., 24., 35., 55., 64., 81., 104., 113., 92., 49., 64., 78., 87., 103., 121., 120.,
    101., 72., 92., 95., 98., 112., 100., 103., 99.,
];

/// A colourful procedural test image (rings, gradient, hard edges).
fn test_image(size: usize) -> [Vec<f32>; 3] {
    let mut ch = [
        vec![0.0; size * size],
        vec![0.0; size * size],
        vec![0.0; size * size],
    ];
    for y in 0..size {
        for x in 0..size {
            let (u, v) = (x as f32 / size as f32, y as f32 / size as f32);
            let r = ((u - 0.5).powi(2) + (v - 0.5).powi(2)).sqrt();
            let rings = 0.5 + 0.5 * (r * 60.0).cos();
            let square = if (0.62..0.9).contains(&u) && (0.1..0.38).contains(&v) {
                1.0
            } else {
                0.0
            };
            let c = Vec3::new(u, 0.3 + 0.4 * v, 1.0 - u) * (0.35 + 0.65 * rings)
                + Vec3::splat(square * 0.8);
            for k in 0..3 {
                ch[k][y * size + x] = c[k].min(1.0) * 255.0;
            }
        }
    }
    ch
}

fn compress(
    dct: &Dct8x8,
    plane: &[f32],
    size: usize,
    quality: f32,
) -> anyhow::Result<(Vec<f32>, usize)> {
    let scale = if quality < 50.0 {
        50.0 / quality
    } else {
        2.0 - quality / 50.0
    };
    let centered: Vec<f32> = plane.iter().map(|v| v - 128.0).collect();
    let mut coeffs = dct.forward(&centered, size, size)?;
    let mut zeros = 0;
    for (k, c) in coeffs.iter_mut().enumerate() {
        let (y, x) = ((k / size) % 8, (k % size) % 8);
        let step = (Q50[y * 8 + x] * scale).max(1.0);
        *c = (*c / step).round() * step;
        zeros += usize::from(*c == 0.0);
    }
    let back = dct.inverse(&coeffs, size, size)?;
    Ok((
        back.iter().map(|v| (v + 128.0).clamp(0.0, 255.0)).collect(),
        zeros,
    ))
}

fn main() -> anyhow::Result<()> {
    let out = std::env::args().nth(1).unwrap_or_else(|| "dct.png".into());
    let npu = Npu::auto(0)?;
    let dct = Dct8x8::new().with_npu(&npu)?;
    let size = 256;
    let img = test_image(size);

    let mut canvas = Canvas::new(size * 3 + 16, size, |_, _| Vec3::ZERO);
    let mut put = |panel: usize, planes: &[Vec<f32>; 3]| {
        for y in 0..size {
            for x in 0..size {
                // Undo the PNG writer's tonemap+gamma so pixels come out as given (approximately).
                let c = Vec3::new(
                    planes[0][y * size + x],
                    planes[1][y * size + x],
                    planes[2][y * size + x],
                ) / 255.0;
                canvas.color[y * canvas.w + panel * (size + 8) + x] = inverse_display(c);
            }
        }
    };
    put(0, &img);
    for (panel, quality) in [(1, 50.0), (2, 10.0)] {
        let mut planes: [Vec<f32>; 3] = Default::default();
        let mut zeros = 0;
        for k in 0..3 {
            let (p, z) = compress(&dct, &img[k], size, quality)?;
            planes[k] = p;
            zeros += z;
        }
        let mse: f32 = (0..3)
            .flat_map(|k| img[k].iter().zip(&planes[k]).map(|(a, b)| (a - b).powi(2)))
            .sum::<f32>()
            / (3 * size * size) as f32;
        println!(
            "quality {quality:>3}: PSNR {:.1} dB, {:.0}% of coefficients zero",
            10.0 * (255.0f32 * 255.0 / mse).log10(),
            100.0 * zeros as f32 / (3 * size * size) as f32
        );
        put(panel, &planes);
    }
    canvas.save(&out)?;
    println!("wrote {out}");
    Ok(())
}

/// Inverse of the example renderer's ACES + gamma, so a displayed value `c` is stored.
fn inverse_display(c: Vec3) -> Vec3 {
    let lin = c.clamp(Vec3::ZERO, Vec3::splat(0.999)).powf(2.2);
    // Solve ACES(x) = lin for x (quadratic).
    let f = |y: f32| {
        let (a, b, cc) = (2.51 - 2.43 * y, 0.03 - 0.59 * y, -0.14 * y);
        if a.abs() < 1e-6 {
            -cc / b
        } else {
            (-b + (b * b - 4.0 * a * cc).max(0.0).sqrt()) / (2.0 * a)
        }
    };
    Vec3::new(f(lin.x), f(lin.y), f(lin.z))
}
