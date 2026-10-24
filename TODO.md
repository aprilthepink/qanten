# TODO

Planned work that is designed but not built yet. Each item has enough notes to
start from. Ordered roughly by value.

## glTF loading (`qanten-graphics`, feature `gltf`)

Load `.gltf` / `.glb` straight into the graphics types.

- Meshes: positions, normals, indices (all primitives of a mesh, merged).
- Skins: `JOINTS_0` / `WEIGHTS_0` → `skinning::Influences`, joint list, inverse
  bind matrices → `SkinnedMesh`.
- Morph targets: `targets[*].POSITION` (already offsets in glTF) → `Blendshapes`;
  default weights from `mesh.weights`.
- Skeleton + animation: node hierarchy, `animations[*].channels` (translation,
  rotation, scale; LINEAR and STEP; CUBICSPLINE later). API sketch:
  `GltfScene::load(path)`, `scene.skinned_mesh(i)`, `scene.blendshapes(i)`,
  `scene.animation("Walk").skinning_matrices(skin, t) -> Vec<Mat4>`.
- Crate: `gltf` (check MSRV ≤ 1.88). Tests need small assets: generate them with
  a script (JSON + base64 buffer) rather than downloading Khronos samples.
- Docs: extend `docs/graphics/skinning.md` and `blendshapes.md`; picture of a
  loaded, animated model rendered by the example rasterizer.

## More graphics workloads (`qanten-graphics`)

- **Procedural terrain** (`terrain` module): heightfields from a sum of plane
  waves with a power-law spectrum (fBm-like), using the same separable
  angle-sum trick as `ocean` → one matmul per tile. Outputs heights, normals,
  slope mask; exact `height_at(x, z)` for gameplay. Static, so tiles can be
  generated in background batches.
- **Image filters** (`filters` module): separable Gaussian blur as two banded
  matrix products (rows, then columns), bloom (threshold + multi-scale blur),
  Sobel/edge detection and sharpen via `ops::conv2d`. Useful in texture
  pipelines (mip generation, normal-map from height).
- Examples + pictures for both, guides in `docs/graphics/`.
- Not a good fit (sparse, branchy): cloth/soft-body springs, BVH building.

## Neural graphics (SIREN / Fourier features)

- `nn::FourierFeatures { b: [m, d] }`: `[cos(2π B x), sin(2π B x)]`. The matvec
  runs on the NPU; phases are reduced modulo 2π on the host before the
  nonlinearity (bf16 can't hold large phases), then `periodic_nl`.
- `nn::Sine { omega0 }` layer (SIREN): `sin(ω₀ (Wx + b))` =
  `cos(ω₀ (Wx + b) − π/2)`, same host reduction.
- SIREN initialisation (uniform ±√(6/n)/ω₀, first layer ±1/n).
- Needs training (core TODO list) for backward passes: `d/dz sin(ωz) = ω cos(ωz)`,
  also computable with `periodic_nl`.
- Examples: fit an image (picture: target vs NPU-trained SIREN), fit a 2-D/3-D
  SDF and mesh it with marching squares/cubes.

## Bevy plugin (`qanten-bevy` crate)

- `QantenPlugin`: inserts an `Npu` resource (`Npu::from_env`), logs the backend.
- `OceanPlugin`: `Ocean` component owning a `GerstnerOcean`; a system updates the
  mesh asset each frame from `evaluate(time)`; `Buoyancy { points }` component
  moves/tilts entities from `sample()` (kinematic; Rapier/Avian forces optional).
- `NpuSkinnedCollider`: CPU-side skinned copy of a `SkinnedMesh` for picking and
  collision, updated from Bevy's joint transforms.
- `ProbeBaker`: bake light-probe volumes into an `Image` / storage buffer.
- Headless test with `MinimalPlugins`; windowed example. Bevy is a large build
  (several minutes, ~2 GB target dir): keep it out of the default CI job.

## crates.io release workflow

- `.github/workflows/release.yml` on tags `v*`: run the full CI matrix, check that
  the tag matches `workspace.package.version`, then `cargo publish` in dependency
  order: `qanten-sys` → `qanten` → `qanten-graphics` → `qanten-cli`
  (`qanten-mock-driver` stays unpublished). Needs a `CARGO_REGISTRY_TOKEN` secret.
- Create a GitHub release with the CHANGELOG section as notes.
- `RELEASING.md`: bump version, update CHANGELOG, tag, push.
- Before the first publish: fill in `authors`, decide on the trademark question
  (see README), and run `qanten selftest` on real hardware if possible.
