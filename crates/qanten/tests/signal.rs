use std::f32::consts::TAU;

use qanten::Npu;
use qanten::compute::Matrix;
use qanten::rng::SplitMix64;
use qanten::signal::{Dft, FirBank, Stft, TemplateMatcher, design};

fn tone(freq: f32, rate: f32, n: usize, amp: f32) -> Vec<f32> {
    (0..n)
        .map(|t| amp * (TAU * freq * t as f32 / rate).sin())
        .collect()
}

#[test]
fn dft_matches_naive_reference() {
    let npu = Npu::simulator();
    let n = 128;
    let x = Matrix::from_fn(2, n, |f, t| (((f * n + t) * 37) % 17) as f32 / 17.0 - 0.5);
    let s = Dft::new(&npu, n).unwrap().forward(&x).unwrap();
    assert_eq!((s.frames(), s.bins()), (2, n / 2 + 1));
    let mut max_err = 0.0f64;
    let mut max_mag = 0.0f64;
    for f in 0..2 {
        for k in 0..=n / 2 {
            let (mut re, mut im) = (0.0f64, 0.0f64);
            for (t, &v) in x.row(f).iter().enumerate() {
                let a = std::f64::consts::TAU * (k * t) as f64 / n as f64;
                re += v as f64 * a.cos();
                im -= v as f64 * a.sin();
            }
            max_err = max_err
                .max((s.re[(f, k)] as f64 - re).abs())
                .max((s.im[(f, k)] as f64 - im).abs());
            max_mag = max_mag.max(re.hypot(im));
        }
    }
    assert!(
        max_err < 0.02 * max_mag,
        "DFT error {max_err} vs peak {max_mag}"
    );
}

#[test]
fn stft_tracks_a_chirp() {
    let npu = Npu::simulator();
    let rate = 8000.0;
    // Frequency rises linearly from 200 Hz to 3000 Hz over one second.
    let sig: Vec<f32> = (0..8000)
        .map(|t| {
            let s = t as f32 / rate;
            (TAU * (200.0 * s + 0.5 * 2800.0 * s * s)).sin()
        })
        .collect();
    let stft = Stft::new(&npu, 256, 128).unwrap();
    let spec = stft.spectrogram(&sig).unwrap();
    assert_eq!(spec.frames(), (8000 - 256) / 128 + 1);
    let first = stft.bin_frequency(spec.peak_bin(0), rate);
    let last = stft.bin_frequency(spec.peak_bin(spec.frames() - 1), rate);
    assert!((first - 290.0).abs() < 100.0, "start {first} Hz");
    assert!((last - 2910.0).abs() < 100.0, "end {last} Hz");
}

#[test]
fn fir_bank_separates_frequencies() {
    let npu = Npu::simulator();
    let rate = 1000.0;
    let low = tone(20.0, rate, 2000, 1.0);
    let high = tone(300.0, rate, 2000, 1.0);
    let mix: Vec<f32> = low.iter().zip(&high).map(|(a, b)| a + b).collect();
    let bank = FirBank::new(
        &npu,
        &[design::lowpass(0.05, 101), design::highpass(0.15, 101)],
    )
    .unwrap();
    let out = bank.apply(&mix).unwrap();
    assert_eq!(out.shape(), (2, 2000));
    // After the filter settles, each output should look like one component (delay = 50 taps).
    let rms = |v: &[f32]| (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt();
    let err_low: Vec<f32> = (200..2000).map(|t| out[(0, t)] - low[t - 50]).collect();
    let err_high: Vec<f32> = (200..2000).map(|t| out[(1, t)] - high[t - 50]).collect();
    assert!(rms(&err_low) < 0.05, "low-pass residual {}", rms(&err_low));
    assert!(
        rms(&err_high) < 0.05,
        "high-pass residual {}",
        rms(&err_high)
    );
}

#[test]
fn correlation_finds_templates_in_noise() {
    let npu = Npu::simulator();
    let mut rng = SplitMix64::new(7);
    let chirp: Vec<f32> = (0..64)
        .map(|t| (TAU * (0.02 * t as f32 + 0.003 * (t * t) as f32)).sin())
        .collect();
    let click: Vec<f32> = (0..64)
        .map(|t| if t % 16 < 2 { 1.0 } else { -0.1 })
        .collect();
    let mut sig: Vec<f32> = (0..2000).map(|_| rng.uniform(-0.2, 0.2)).collect();
    for (pos, tpl) in [(300usize, &chirp), (1200, &chirp), (700, &click)] {
        for (i, v) in tpl.iter().enumerate() {
            sig[pos + i] += v;
        }
    }
    let matcher = TemplateMatcher::new(&npu, &[chirp.clone(), click.clone()]).unwrap();
    let mut found = matcher.scores(&sig).unwrap().matches(0.7);
    found.sort_by_key(|m| m.offset);
    let got: Vec<(usize, usize)> = found.iter().map(|m| (m.template, m.offset)).collect();
    assert_eq!(got, vec![(0, 300), (1, 700), (0, 1200)], "{found:?}");
}

#[test]
fn bad_inputs() {
    let npu = Npu::simulator();
    assert!(Dft::new(&npu, 1).is_err());
    assert!(
        Dft::new(&npu, 8)
            .unwrap()
            .forward(&Matrix::zeros(1, 9))
            .is_err()
    );
    assert!(
        Stft::new(&npu, 64, 32)
            .unwrap()
            .spectrogram(&[0.0; 10])
            .is_err()
    );
    assert!(FirBank::new(&npu, &[vec![1.0], vec![1.0, 2.0]]).is_err());
    let matcher = TemplateMatcher::new(&npu, &[vec![1.0, -1.0, 1.0]]).unwrap();
    assert!(matcher.scores(&[0.0, 1.0]).is_err());
    assert!(TemplateMatcher::new(&npu, &[vec![1.0]]).is_err());
}
