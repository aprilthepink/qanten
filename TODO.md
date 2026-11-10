# TODO

Planned work that is designed but not built yet. Each item has enough notes to
start from. Ordered roughly by value.

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
