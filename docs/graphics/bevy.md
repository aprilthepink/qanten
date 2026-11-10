# Bevy plugins (`qanten-bevy`)

`qanten-bevy` connects `qanten-graphics` to [Bevy](https://bevy.org) 0.17 (the newest
release that builds with Rust 1.88, the workspace MSRV).

| Item | What it does |
|---|---|
| `QantenPlugin` | opens an NPU (`Npu::from_env(0)`: hardware if the driver is installed, the simulator otherwise; `QANTEN_BACKEND` overrides) as the `NpuResource` and logs the backend. `QantenPlugin::with_npu(&npu)` uses a given one. |
| `OceanPlugin`, `Ocean` | an `Ocean` component owns a `GerstnerOcean`; every frame its mesh asset is rewritten from `evaluate(time)`, one NPU matrix product |
| `Buoyancy { points, draft, responsiveness }` | floats an entity on the ocean: height from the mean exact surface height under `points`, tilt from the mean normal, yaw kept. Kinematic; add physics forces on top if needed |
| `SkinnedColliderPlugin`, `NpuSkinnedCollider` | a CPU-side copy of an animated mesh, skinned on the NPU from Bevy's joint `GlobalTransform`s, for picking and collision (`ray_cast`). Characters sharing one `Arc<SkinnedMesh>` are skinned in one NPU call per frame. `from_gltf` (feature `gltf`) builds one from a `GltfScene` model |
| `ProbeBaker`, `ProbeVolume` | bakes a grid of light probes to SH9 (all probes in one NPU call) and packs them into a 3-D `Rgba32Float` `Image` |
| `Convert`, `to_bevy`, `from_bevy` | `qanten-graphics` and Bevy may use different `glam` versions; these convert `Vec2`, `Vec3` and `Mat4` |

## Usage

<!-- doccheck: ignore -->
```rust
use bevy::prelude::*;
use qanten_bevy::{Buoyancy, NpuResource, Ocean, OceanPlugin, QantenPlugin};
use qanten_graphics::glam::Vec2 as GVec2;
use qanten_graphics::ocean::{GerstnerOcean, Grid, wind_sea};

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, QantenPlugin::default(), OceanPlugin))
        .add_systems(Startup, setup)
        .run();
}

fn setup(mut commands: Commands, npu: Res<NpuResource>, mut meshes: ResMut<Assets<Mesh>>) {
    let waves = wind_sea(64, GVec2::new(1.0, 0.4), 30.0, 0.75, 7);
    let grid = Grid::new(GVec2::splat(-60.0), GVec2::splat(120.0), 192, 192);
    let ocean = GerstnerOcean::new(waves, grid).unwrap().with_npu(&npu).unwrap();
    let ocean = Ocean::new(ocean, &mut meshes);
    commands.spawn((ocean.mesh3d(), ocean)); // + a MeshMaterial3d
    commands.spawn((
        Transform::from_xyz(0.0, 0.0, 10.0),
        Buoyancy::new(vec![Vec3::new(-1.5, 0.0, -3.0), Vec3::new(1.5, 0.0, 3.0)]),
    ));
}
```

The full windowed example (ocean, floating boxes, light, camera):

```sh
cargo run --release --manifest-path crates/qanten-bevy/Cargo.toml --features windowed --example ocean
```

On Linux, the windowed build needs `libasound2-dev libudev-dev libwayland-dev
libxkbcommon-dev`.

## Skinned colliders

```text
joints' GlobalTransform ─┐
inverse bind matrices ───┼─► bones in the collider's local space ─► SkinnedMesh::skin_batch (NPU)
collider's GlobalTransform┘                                         └─► NpuSkinnedCollider.positions
```

The system runs in `PostUpdate` after transform propagation, so the positions match
the frame's animation. Bevy itself skins on the GPU for rendering; this copy is for
gameplay code on the CPU.

## Building and testing

`qanten-bevy` is a separate Cargo workspace (`crates/qanten-bevy`), excluded from the
main one, because Bevy takes several minutes and about 2 GB of disk to build.
`scripts/ci-bevy.sh` runs its checks: rustfmt, clippy, headless tests
(`MinimalPlugins` with a manual clock, no window or GPU) and a type check of the
windowed example. CI runs the same steps in a separate `qanten-bevy` job.

The crate is not published to crates.io yet (`publish = false`).
