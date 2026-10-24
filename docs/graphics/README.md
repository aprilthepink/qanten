# qanten-graphics

Moves CPU-side 3D work onto the NPU.

Engines keep some geometry work on the CPU because the result is needed there, not on
the GPU: wave heights for buoyancy, deformed meshes for collision and picking, baked
lighting in asset pipelines. Each module rewrites one of these jobs as a large matrix
product. The big, constant matrix stays on the NPU, so each frame only moves small inputs.

| | | |
|---|---|---|
| ![](../images/ocean.png) [Ocean waves](ocean.md) | ![](../images/crowd.png) [Skinning](skinning.md) | ![](../images/blendshapes.png) [Blendshapes](blendshapes.md) |
| ![](../images/probes.png) [Light probes](sh.md) | ![](../images/dct.png) [Block DCT](dct.md) | |

## The pattern every module follows

```rust
use qanten::Npu;
use qanten_graphics::dct::Dct8x8;

fn main() -> qanten::Result<()> {
    let npu = Npu::auto(0)?;
    let cpu_only = Dct8x8::new();               // works without an NPU
    let accelerated = Dct8x8::new().with_npu(&npu)?; // uploads its constant matrix
    let img = vec![0.5f32; 64 * 64];
    let a = cpu_only.forward(&img, 64, 64)?;
    let b = accelerated.forward(&img, 64, 64)?;  // same API, runs on the NPU
    assert_eq!(a.len(), b.len());
    Ok(())
}
```

- `new(...)` builds a CPU-only object. `with_npu(&npu)` uploads the constant part and
  returns it accelerated. The same methods then run on the NPU.
- Each module also has an explicit `*_cpu` method. It is the accuracy reference in the
  tests and the fallback for batches too small to be worth a transfer.
- `qanten_graphics::glam` re-exports the [glam](https://docs.rs/glam) math types used
  in the API (`Vec2`, `Vec3`, `Mat4`).

## Measured on the simulator

Every example prints NPU-path vs CPU-reference timing and the largest error
(4 CPU threads, release build):

| Example | CPU reference | NPU path (simulated) | Max error |
|---|---|---|---|
| `ocean`: 400², 64 waves | 103 ms | 63 ms | 2 cm of 11 m wave height |
| `crowd`: skinning 100 × 1280 vertices | 2 ms | 19 ms | 8.5 mm on a 3 m mesh |
| `crowd`: blendshapes 200 × 4000 vertices × 52 | 22 ms | 65 ms | 0.14 mm |
| `probes`: 1024 probes × 2048 samples | 5 ms | 46 ms | 0.005 (DC ≈ 1.1) |
| `dct`: 256² RGB, JPEG q50 | – | – | 42.7 dB PSNR |

The simulator does the same multiplications as the NPU plus format conversions, so for
most rows it shows correctness, not hardware speed. The ocean is the exception: its
matrix formulation beats the usual per-point loop even on a plain CPU. Real speedups for
the other modules depend on the card's throughput and transfer cost, which still needs
measuring on hardware.

## Precision

The NPU computes in bfloat16 (about 3 significant digits), so each output carries an
error of roughly 0.5 % of its magnitude. The modules are arranged so the magnitudes stay
small:

- **ocean**: wave phases are reduced modulo 2π on the host in f64; the NPU only sees
  values in `[-1, 1] × amplitude`.
- **skinning, blendshapes**: the NPU computes offsets from the rest pose; positions are
  added back in f32 on the host. The error scales with how far the mesh moved, not with
  its size or its distance from the origin.
- **sh, dct**: inputs are bounded by nature (radiance, pixel values).

Good for physics proxies, culling, picking, baking and previews. Not a replacement for
exact final-frame geometry.

## Running the examples

```sh
cargo run --release -p qanten-graphics --example ocean        # ocean.png
cargo run --release -p qanten-graphics --example crowd        # crowd.png
cargo run --release -p qanten-graphics --example blendshapes  # blendshapes.png
cargo run --release -p qanten-graphics --example probes       # probes.png
cargo run --release -p qanten-graphics --example dct          # dct.png
```

They use the hardware when the driver is installed and the simulator otherwise. The
images are drawn by a small software rasterizer in `examples/common`.
