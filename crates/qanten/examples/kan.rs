//! Q.ANT's photonic Kolmogorov-Arnold layer, which maps directly onto the
//! NPU's periodic nonlinearity.
//!
//! Here a single KAN layer with hand-set parameters represents
//! `f(x) = sin(x) + 0.5 cos(2x)` exactly, since `sin(x) = cos(x - π/2)`.

use std::f32::consts::FRAC_PI_2;

use qanten::prelude::*;

fn main() -> qanten::Result<()> {
    let npu = Npu::auto(0)?;
    // 1 input, 1 output, 2 frequency components: ks = [1, 2].
    let ks = Tensor::vector(&[1.0, 2.0]);
    let phis = Tensor::from_f32(&[-FRAC_PI_2, 0.0], [1, 1, 2])?;
    let ampls = Tensor::from_f32(&[1.0, 0.5], [1, 1, 2])?;
    let layer = nn::Kan::new(&npu, phis, ampls, ks);

    let xs: Vec<f32> = (0..9).map(|i| -2.0 + 0.5 * i as f32).collect();
    let y = layer.forward(&Tensor::from_f32(&xs, [xs.len(), 1])?)?;
    println!("{:>6} {:>10} {:>10}", "x", "npu", "exact");
    for (x, got) in xs.iter().zip(y.to_f32_vec()) {
        println!(
            "{x:>6.2} {got:>10.4} {:>10.4}",
            x.sin() + 0.5 * (2.0 * x).cos()
        );
    }
    Ok(())
}
