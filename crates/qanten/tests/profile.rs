use qanten::nn::{self, Module};
use qanten::profile::OpKind;
use qanten::{DeviceBuffer, Error, Features, Npu, Tensor, ops};

#[test]
fn counts_streamed_ops() {
    let (npu, prof) = Npu::simulator().profiled();
    let w = Tensor::rand_uniform([64, 128], -1., 1., 1);
    let x = Tensor::rand_uniform([32, 128], -1., 1., 2);
    ops::linear(&npu, &x, &w, None).unwrap();
    ops::mul(&npu, &x, &x).unwrap();
    let r = prof.report();
    let mv = r.op(OpKind::MatVec);
    assert_eq!((mv.calls, mv.macs), (1, 64 * 128 * 32));
    assert_eq!(mv.bytes_to_device, (64 * 128 + 32 * 128) * 2);
    assert_eq!(mv.bytes_from_device, 32 * 64 * 2);
    assert_eq!(r.op(OpKind::Mul).macs, 32 * 128);
    assert_eq!(r.total().calls, 2);
    assert!(r.to_string().contains("matvec"));
    assert!(npu.backend().name().contains("[profiled]"));
}

#[test]
fn counts_resident_ops_and_memory() {
    let (npu, prof) = Npu::simulator().profiled();
    {
        let layer = nn::Linear::new(&npu, &Tensor::ones([16, 8]), None).unwrap();
        prof.reset();
        layer.forward(&Tensor::ones([4, 8])).unwrap();
        let r = prof.report();
        assert_eq!(r.op(OpKind::MatVecResident).macs, 16 * 8 * 4);
        assert_eq!(r.op(OpKind::Allocate).calls, 2); // input + output buffers
        assert_eq!(r.op(OpKind::Free).calls, 2);
        assert_eq!(r.op(OpKind::Upload).bytes_to_device, 4 * 8 * 2);
        assert_eq!(r.op(OpKind::Download).bytes_from_device, 4 * 16 * 2);
        assert_eq!(r.live_device_bytes, 16 * 8 * 2); // the layer's weights
        assert!(r.peak_device_bytes >= (16 * 8 + 4 * 8 + 4 * 16) * 2);
    }
    assert_eq!(
        prof.report().live_device_bytes,
        0,
        "weights freed with the layer"
    );
}

#[test]
fn errors_are_counted_and_parent_survives() {
    let parent = Npu::simulator();
    let (npu, prof) = parent.profiled();
    let buf = DeviceBuffer::<Features>::alloc(&npu, 2, 2).unwrap();
    assert!(buf.upload_raw(&[qanten::bf16::ONE; 3]).is_err()); // rejected before the backend
    drop(buf);
    // A backend-level error: matvec with an inconsistent batch.
    assert!(
        npu.backend()
            .matvec_batched(
                0,
                &[qanten::bf16::ONE; 4],
                &[qanten::bf16::ONE; 3],
                &mut [],
                2,
                2
            )
            .is_err()
    );
    assert_eq!(prof.report().op(OpKind::MatVec).errors, 1);
    drop(npu);
    // The parent handle still works after the profiled one is gone.
    assert!(ops::mul(&parent, &Tensor::ones([2]), &Tensor::ones([2])).is_ok());
    // Buffers from different handles don't mix.
    let (p2, _) = parent.profiled();
    let w = DeviceBuffer::<qanten::Weights>::alloc(&parent, 1, 1).unwrap();
    let x = DeviceBuffer::<Features>::alloc(&p2, 1, 1).unwrap();
    let mut y = DeviceBuffer::<Features>::alloc(&p2, 1, 1).unwrap();
    assert!(matches!(
        qanten::matvec_resident(&w, &x, &mut y),
        Err(Error::DeviceMismatch(_))
    ));
}
