# Skinning (`qanten_graphics::skinning`)

![100 skinned tentacles](../images/crowd.png)

*100 characters × 1280 vertices × 12 bones, skinned in one NPU call per frame.*

GPUs skin meshes for rendering, but engines often need the deformed mesh on the CPU
as well: hit detection and ray picking, collision proxies, cloth attachment points,
audio occlusion. This module does linear blend skinning (LBS) for whole crowds as one
NPU matrix product.

## Usage

```rust
use qanten::Npu;
use qanten_graphics::glam::{Mat4, Vec3};
use qanten_graphics::skinning::SkinnedMesh;

fn main() -> qanten::Result<()> {
    let npu = Npu::auto(0)?;
    // A 2-bone strip: vertices blend from bone 0 to bone 1 along y.
    let rest: Vec<Vec3> = (0..100).map(|i| Vec3::new(0.0, i as f32 * 0.02, 0.0)).collect();
    let normals = Some(vec![Vec3::X; 100]);
    let influences = (0..100)
        .map(|i| {
            let t = i as f32 / 99.0;
            [(0, 1.0 - t), (1, t), (0, 0.0), (0, 0.0)] // (bone, weight) × 4
        })
        .collect();
    let mesh = SkinnedMesh::new(rest, normals, influences, 2)?.with_npu(&npu)?;

    // Skinning matrices (animated world × inverse bind) for 3 characters × 2 bones.
    let poses: Vec<Mat4> = (0..3)
        .flat_map(|c| [Mat4::IDENTITY, Mat4::from_rotation_z(0.3 * c as f32)])
        .collect();
    let out = mesh.skin_batch(&poses)?;
    assert_eq!(out.positions.len(), 3 * 100);   // characters × vertices
    assert_eq!(out.normals.len(), 3 * 100);
    Ok(())
}
```

## API

| Item | Purpose |
|---|---|
| `Influences = [(u16, f32); 4]` | up to 4 `(bone, weight)` pairs per vertex; weights are renormalised |
| `SkinnedMesh::new(rest, normals, influences, bones)` | validates bone indices; normals are optional |
| `.with_npu(&npu)` | uploads the constant skinning matrices |
| `skin(&bones)` / `skin_batch(&poses)` | one character / `characters × bones` matrices → `Skinned { positions, normals }` |
| `skin_batch_cpu(&poses)` | standard LBS on the CPU (reference) |

Bone matrices are the usual skinning matrices: the bone's animated world transform
times its inverse bind matrix.

## How it maps to the NPU

LBS is `v' = Σ_b w_vb · M_b · (v, 1)`. Rewritten as an offset from the rest pose,

```text
v' = v + Σ_b w_vb · (M_b − I) · (v, 1)
```

the sum is a matrix product between:

- a **constant** matrix `A[v, (b, j)] = w_vb · ṽ_j` with `vertices × 4·bones` entries,
  uploaded once (`ṽ = (x, y, z, 1)`);
- per character, the three rows of every `(M_b − I)`: `3 × 4·bones` numbers.

All characters go into one call: `[3·characters, 4·bones] × [4·bones, vertices]`.
Normals use a second resident matrix with the rotation part (`vertices × 3·bones`).

Working with offsets keeps the bf16 error proportional to how far the mesh moved,
not to its size: a character standing still at x = 10 km has no error.

## Accuracy and memory

- The crowd example measures 8.5 mm of error on a 3 m mesh in strong poses, and 0.005
  on normals.
- The resident matrices are dense: `vertices × 4·bones` (+ `vertices × 3·bones` for
  normals) bf16 values. 20k vertices × 64 bones ≈ 10 MB. Most entries are zero
  (≤ 4 bones per vertex); the dense form is what the NPU's matrix unit wants.
- Normals are transformed with the bone matrices directly (correct for rotations and
  uniform scale), then renormalised.
