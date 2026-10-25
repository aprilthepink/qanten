# Block DCT (`qanten_graphics::dct`)

![Original, quality 50, quality 10](../images/dct.png)

*Left to right: the original, JPEG-style quality 50 (42.7 dB PSNR, 91 % of
coefficients zero) and quality 10 (32.5 dB, 96 % zero). Forward and inverse DCT ran
on the NPU.*

The 8×8 DCT-II is the transform behind JPEG and many texture and video codecs. Texture
pipelines run it on the CPU over millions of blocks.

## Usage

```rust
use qanten::Npu;
use qanten::compute::Matrix;
use qanten_graphics::dct::Dct8x8;

fn main() -> qanten::Result<()> {
    let npu = Npu::auto(0)?;
    let dct = Dct8x8::new().with_npu(&npu)?;
    // A 32 × 64 plane (height × width, both multiples of 8).
    let plane = Matrix::from_fn(32, 64, |y, x| ((y * 64 + x) % 255) as f32 - 128.0);

    let mut coeffs = dct.forward(&plane)?;         // same layout; DC at each block's top-left
    for c in coeffs.as_mut_slice() {
        *c = (*c / 16.0).round() * 16.0;           // crude uniform quantisation
    }
    let back = dct.inverse(&coeffs)?;
    assert_eq!(back.shape(), plane.shape());
    Ok(())
}
```

## API

| Item | Purpose |
|---|---|
| `Dct8x8::new()` | builds the orthonormal forward and inverse 64×64 matrices |
| `.with_npu(&npu)` / `.with_pool(&pool)` | uploads both |
| `forward(&plane)` | `Matrix` (height × width) with each 8×8 block replaced by its coefficients |
| `inverse(&coeffs)` | back to pixels |

Width and height must be multiples of 8 (`Error::Shape` otherwise); pad images first.

## How it maps to the NPU

A 2-D 8×8 DCT is linear in the 64 pixels of a block, so it's one 64×64 matrix: the
Kronecker product of two 1-D DCT matrices, `K[(u,v),(y,x)] = C[u][y] · C[v][x]`. That
matrix stays on the NPU and every block of the image is one row: a 4096 × 2048
texture is 131,072 rows in a single call. The inverse is the transpose (the transform is
orthonormal).

## Accuracy

Coefficients come back within about 0.05 for pixel values in 0..1, well below any
quantisation step a codec would use. The `dct` example shows the full JPEG-style round
trip with the standard luminance table.
