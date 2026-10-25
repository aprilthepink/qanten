# Blendshapes (`qanten_graphics::blendshapes`)

![Six blobs morphing between four targets](../images/blendshapes.png)

*One sphere, four morph targets (squash, bulge, spikes, ripple). Six characters with
different weights are evaluated in one NPU call; the last one mixes all four.*

Morph targets (blendshapes) drive faces, muscles and corrective shapes. Evaluating
them for a crowd is one big matrix product with a constant matrix, a perfect NPU fit.

## Usage

```rust
use qanten::Npu;
use qanten::compute::Matrix;
use qanten_graphics::blendshapes::Blendshapes;
use qanten_graphics::glam::Vec3;

fn main() -> qanten::Result<()> {
    let npu = Npu::auto(0)?;
    let base = vec![Vec3::ZERO, Vec3::X, Vec3::Y];
    let smile = vec![Vec3::new(0.0, 0.1, 0.0); 3];   // offsets per vertex
    let blink = vec![Vec3::new(0.0, -0.05, 0.0); 3];
    let shapes = Blendshapes::new(base, &[smile, blink])?.with_npu(&npu)?;

    // One row of shape weights per character.
    let weights = Matrix::from_rows(&[[1.0, 0.0], [0.5, 1.0]])?;
    let meshes = shapes.evaluate_batch(&weights)?;   // one mesh per character
    assert_eq!((meshes.len(), meshes[0].len()), (2, 3));
    println!("{:?}", meshes[0]);
    Ok(())
}
```

Use `Blendshapes::from_target_positions(base, targets)` if your targets are absolute
positions (as in glTF files exported from some tools) rather than offsets.

## API

| Item | Purpose |
|---|---|
| `Blendshapes::new(base, &deltas)` | `deltas[s][v]` = offset of target `s` at vertex `v` |
| `from_target_positions(base, &targets)` | same, from absolute target positions |
| `.with_npu(&npu)` / `.with_pool(&pool)` | uploads the `[3·vertices, shapes]` delta matrix |
| `evaluate(&weights)` | one character (`&[f32]`, one weight per shape) |
| `evaluate_batch(&weights)` | `Matrix` with one row per character → one `Vec<Vec3>` mesh per character |
| `evaluate_batch_cpu(&weights)` | exact CPU reference |
| `self_check()`, `is_accelerated()` | NPU accuracy check, whether it's uploaded |
| `vertex_count()`, `shape_count()` | sizes |

## How it maps to the NPU

`vertex = base + Σ_s weight_s · delta_s`. The deltas form a constant
`[3·vertices, shapes]` matrix, uploaded once. Each frame uploads only the weights,
`characters × shapes` values (52 per character for an ARKit-style face rig), and
downloads the offsets. The base mesh is added on the host in f32.

## Accuracy

Only the offsets go through bf16, so the error scales with the size of the
deformation: 0.14 mm on 200 faces × 4000 vertices × 52 shapes in the `crowd` example,
0.004 on the unit-sized blobs above.
