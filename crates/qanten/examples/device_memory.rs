//! Typed device memory: upload weights once, stream features, download results.

use qanten::{DeviceBuffer, Features, Npu, Tensor, Weights, matvec_resident};

fn main() -> qanten::Result<()> {
    let npu = Npu::auto(0)?;
    let (out_f, in_f, batch) = (256, 512, 32);

    let w = DeviceBuffer::<Weights>::from_tensor(
        &npu,
        &Tensor::rand_uniform([out_f, in_f], -1., 1., 1),
    )?;
    let x = DeviceBuffer::<Features>::alloc(&npu, batch, in_f)?;
    let mut y = DeviceBuffer::<Features>::alloc(&npu, batch, out_f)?;
    println!("weights on device: {} KiB", w.size_bytes() / 1024);

    npu.reset_perf_counters()?;
    for step in 0..5 {
        x.upload(&Tensor::rand_uniform([batch, in_f], -1., 1., 100 + step))?;
        matvec_resident(&w, &x, &mut y)?;
        let out = y.download()?;
        println!("step {step}: y[0][0..4] = {:?}", &out.to_f32_vec()[..4]);
    }
    let p = npu.perf_counters()?;
    println!("NPU utilisation: {:.1}%", 100.0 * p.utilization());
    // `w.download()` would not compile: weights cannot be read back.
    Ok(())
}
