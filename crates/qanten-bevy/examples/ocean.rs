//! A windowed ocean whose mesh is evaluated on the NPU every frame, with a
//! few boxes floating on it.
//!
//! ```sh
//! cargo run --release --manifest-path crates/qanten-bevy/Cargo.toml --features windowed --example ocean
//! ```

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

fn setup(
    mut commands: Commands,
    npu: Res<NpuResource>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let waves = wind_sea(64, GVec2::new(1.0, 0.4), 30.0, 0.75, 7);
    let grid = Grid::new(GVec2::splat(-60.0), GVec2::splat(120.0), 192, 192);
    let ocean = GerstnerOcean::new(waves, grid)
        .and_then(|o| o.with_npu(&npu))
        .expect("ocean");
    let ocean = Ocean::new(ocean, &mut meshes);
    commands.spawn((
        ocean.mesh3d(),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.05, 0.2, 0.35),
            perceptual_roughness: 0.15,
            ..default()
        })),
        ocean,
    ));

    let hull = meshes.add(Cuboid::new(3.0, 1.0, 6.0));
    let paint = materials.add(Color::srgb(0.85, 0.35, 0.2));
    for (k, x) in [-15.0f32, 0.0, 15.0].into_iter().enumerate() {
        let corners = [(-1.5, -3.0), (1.5, -3.0), (-1.5, 3.0), (1.5, 3.0)]
            .map(|(x, z)| Vec3::new(x, 0.0, z))
            .to_vec();
        commands.spawn((
            Mesh3d(hull.clone()),
            MeshMaterial3d(paint.clone()),
            Transform::from_xyz(x, 0.0, k as f32 * 8.0 - 8.0)
                .with_rotation(Quat::from_rotation_y(k as f32 * 0.7)),
            Buoyancy {
                draft: 0.2,
                responsiveness: 0.2,
                ..Buoyancy::new(corners)
            },
        ));
    }

    commands.spawn((
        DirectionalLight {
            illuminance: 12_000.0,
            ..default()
        },
        Transform::from_xyz(-20.0, 30.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(-35.0, 18.0, -40.0).looking_at(Vec3::new(0.0, 0.0, 5.0), Vec3::Y),
    ));
}
