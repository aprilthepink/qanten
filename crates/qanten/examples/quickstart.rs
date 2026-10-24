//! The basics: open an NPU, run ops, keep weights on the device.
//!
//! `cargo run --example quickstart` (uses the simulator if no driver is installed).

use qanten::prelude::*;

fn main() -> qanten::Result<()> {
    // Hardware if the Q.ANT driver is installed, otherwise the simulator.
    let npu = Npu::auto(0)?;
    println!("NPU {} on {}", npu.id(), npu.backend().name());

    // Elementwise product and the photonic periodic nonlinearity.
    let a = Tensor::vector(&[0.5, 1.0, 1.5, 2.0]);
    let b = Tensor::vector(&[2.0, 2.0, 2.0, 2.0]);
    println!("a * b        = {:?}", ops::mul(&npu, &a, &b)?.to_f32_vec());
    println!(
        "tcos(a) * b  = {:?}",
        ops::periodic_nl(&npu, &a, &b)?.to_f32_vec()
    );

    // Matrix products.
    let x = Tensor::from_f32(&[1., 2., 3., 4., 5., 6.], [2, 3])?;
    let w = Tensor::from_f32(&[1., 0., 0., 0., 1., 0.], [2, 3])?; // [out, in]
    println!(
        "linear       = {:?}",
        ops::linear(&npu, &x, &w, None)?.to_f32_vec()
    );
    println!(
        "matmul       = {:?}",
        ops::matmul(&npu, &x, &Tensor::ones([3, 2]))?.to_f32_vec()
    );

    // Upload weights once, reuse them for many inputs.
    let layer = nn::Linear::new(&npu, &w, Some(Tensor::vector(&[10., 20.])))?;
    for i in 0..3 {
        let input = Tensor::full([1, 3], i as f32);
        println!("resident #{i}  = {:?}", layer.forward(&input)?.to_f32_vec());
    }

    // Device telemetry.
    let info = npu.device_info()?;
    println!(
        "board {} at {:.1} °C",
        info.version.board_serial, info.sensors.temp_pd1
    );
    Ok(())
}
