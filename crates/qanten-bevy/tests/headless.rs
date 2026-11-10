//! The plugins in a headless app (`MinimalPlugins`, no window or GPU).

use std::sync::Arc;
use std::time::Duration;

use bevy::prelude::*;
use bevy::time::TimeUpdateStrategy;
use qanten::Npu;
use qanten_bevy::{
    Buoyancy, Convert, NpuResource, NpuSkinnedCollider, Ocean, OceanPlugin, ProbeBaker,
    ProbeVolume, QantenPlugin, SkinnedColliderPlugin,
};
use qanten_graphics::glam::Vec2 as GVec2;
use qanten_graphics::ocean::{GerstnerOcean, Grid, wind_sea};
use qanten_graphics::skinning::SkinnedMesh;

fn app() -> App {
    let mut app = App::new();
    app.add_plugins((
        MinimalPlugins,
        AssetPlugin::default(),
        bevy::transform::TransformPlugin,
        QantenPlugin::with_npu(&Npu::simulator()),
        OceanPlugin,
        SkinnedColliderPlugin,
    ))
    .init_asset::<Mesh>()
    .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_millis(
        250,
    )));
    app
}

fn elapsed(app: &App) -> f64 {
    app.world().resource::<Time>().elapsed_secs_f64()
}

#[test]
fn the_plugin_provides_an_npu() {
    let app = app();
    assert!(app.world().resource::<NpuResource>().is_simulated());
}

#[test]
fn ocean_meshes_and_buoys_follow_the_waves() {
    let mut app = app();
    let npu = app.world().resource::<NpuResource>().0.clone();
    let waves = wind_sea(24, GVec2::new(1.0, 0.2), 30.0, 0.7, 3);
    let grid = Grid::new(GVec2::splat(-20.0), GVec2::splat(40.0), 48, 48);
    let reference = GerstnerOcean::new(waves.clone(), grid).unwrap();
    let ocean = GerstnerOcean::new(waves, grid)
        .unwrap()
        .with_npu(&npu)
        .unwrap();
    let ocean = app
        .world_mut()
        .resource_scope(|_, mut meshes: Mut<Assets<Mesh>>| Ocean::new(ocean, &mut meshes));
    let handle = ocean.mesh.clone();
    app.world_mut().spawn((ocean.mesh3d(), ocean));
    let buoy = app
        .world_mut()
        .spawn((
            Transform::from_xyz(3.0, 0.0, 4.0),
            Buoyancy {
                draft: 0.1,
                ..Buoyancy::new(vec![Vec3::ZERO])
            },
        ))
        .id();

    for _ in 0..4 {
        app.update();
    }
    let t = elapsed(&app);
    assert!(t > 0.5, "time advanced: {t}");

    // The mesh holds the NPU frame at time t.
    let want = reference.evaluate_cpu(t);
    let meshes = app.world().resource::<Assets<Mesh>>();
    let positions = meshes
        .get(&handle)
        .unwrap()
        .attribute(Mesh::ATTRIBUTE_POSITION)
        .unwrap()
        .as_float3()
        .unwrap();
    let worst = positions
        .iter()
        .zip(&want.positions)
        .map(|(a, b)| {
            (Vec3::from_array(*a) - Vec3::from_array(b.to_array()))
                .abs()
                .max_element()
        })
        .fold(0.0, f32::max);
    assert!(
        worst < 0.02 * reference.max_height().max(1.0),
        "mesh error {worst}"
    );

    // The buoy sits at the exact surface height minus its draft.
    let y = app.world().get::<Transform>(buoy).unwrap().translation.y;
    let surface = reference.sample(3.0, 4.0, t).height;
    assert!((y - (surface - 0.1)).abs() < 1e-4, "{y} vs {surface}");
}

#[test]
fn skinned_colliders_follow_their_joints() {
    let mut app = app();
    let npu = app.world().resource::<NpuResource>().0.clone();
    // A vertical strip of 11 vertices, blended from joint 0 (bottom) to joint 1.
    let rest: Vec<qanten_graphics::glam::Vec3> = (0..11)
        .map(|i| qanten_graphics::glam::Vec3::new(0.0, i as f32 * 0.2, 0.0))
        .collect();
    let influences = (0..11)
        .map(|i| {
            let t = i as f32 / 10.0;
            [(0, 1.0 - t), (1, t), (0, 0.0), (0, 0.0)]
        })
        .collect();
    let mesh = Arc::new(
        SkinnedMesh::new(rest.clone(), None, influences, 2)
            .unwrap()
            .with_npu(&npu)
            .unwrap(),
    );
    // Joint 1 sits at y = 1 in the bind pose and bends 90° about z.
    let root = app.world_mut().spawn(Transform::IDENTITY).id();
    let tip = app
        .world_mut()
        .spawn(
            Transform::from_xyz(0.0, 1.0, 0.0)
                .with_rotation(Quat::from_rotation_z(std::f32::consts::FRAC_PI_2)),
        )
        .id();
    let inverse_bind = vec![Mat4::IDENTITY, Mat4::from_translation(-Vec3::Y)];
    // Two characters share the mesh, so they are skinned in one call.
    let triangles = vec![[0, 5, 10]];
    let mut characters = Vec::new();
    for x in [0.0, 5.0] {
        let c = app
            .world_mut()
            .spawn((
                Transform::from_xyz(x, 0.0, 0.0),
                NpuSkinnedCollider::new(
                    Arc::clone(&mesh),
                    vec![root, tip],
                    inverse_bind.clone(),
                    triangles.clone(),
                ),
            ))
            .id();
        characters.push(c);
    }
    app.update();

    // Hand-computed LBS for the first character (at the origin).
    let bones = [
        Mat4::IDENTITY,
        Mat4::from_translation(Vec3::Y)
            * Mat4::from_rotation_z(std::f32::consts::FRAC_PI_2)
            * Mat4::from_translation(-Vec3::Y),
    ];
    let collider = app
        .world()
        .get::<NpuSkinnedCollider>(characters[0])
        .unwrap();
    for (i, p) in collider.positions.iter().enumerate() {
        let t = i as f32 / 10.0;
        let v: Vec3 = rest[i].convert();
        let want = bones[0].transform_point3(v) * (1.0 - t) + bones[1].transform_point3(v) * t;
        assert!((*p - want).length() < 0.02, "vertex {i}: {p} vs {want}");
    }
    // The second character's joints are the same entities, but it stands at
    // x = 5: in its local space the joints are 5 m to the left.
    let other = app
        .world()
        .get::<NpuSkinnedCollider>(characters[1])
        .unwrap();
    assert!((other.positions[0] - Vec3::new(-5.0, 0.0, 0.0)).length() < 0.02);

    // Ray picking against the skinned triangle (0, 5, 10).
    let a = collider.positions[0];
    let b = collider.positions[5];
    let c = collider.positions[10];
    let centre = (a + b + c) / 3.0;
    let normal = (b - a).cross(c - a).normalize();
    let hit = collider.ray_cast(centre + normal * 2.0, -normal);
    assert!(
        hit.is_some_and(|(t, k)| (t - 2.0).abs() < 1e-3 && k == 0),
        "{hit:?}"
    );
    assert!(collider.ray_cast(centre + normal * 2.0, normal).is_none());
}

#[test]
fn probe_volumes_bake_into_sh_and_an_image() {
    let npu = Npu::simulator();
    let baker = ProbeBaker::new(256, Some(&npu)).unwrap();
    let volume = ProbeVolume {
        min: Vec3::ZERO,
        max: Vec3::splat(10.0),
        resolution: UVec3::new(3, 2, 2),
    };
    assert_eq!(volume.len(), 12);
    assert_eq!(
        volume.position(UVec3::new(2, 1, 0)),
        Vec3::new(10.0, 10.0, 0.0)
    );
    // Light from above, brighter higher up.
    let probes = baker
        .bake(&volume, |pos, dir| {
            Vec3::splat(dir.y.max(0.0) * (1.0 + pos.y / 10.0))
        })
        .unwrap();
    assert_eq!(probes.len(), 12);
    let up = qanten_graphics::glam::Vec3::Y;
    let low = probes[0].irradiance(up).x;
    let high = probes[3].irradiance(up).x; // (0, 1, 0): x varies fastest, then y
    assert!(high > 1.5 * low, "{high} vs {low}");
    assert!(probes[0].irradiance(-up).x < 0.2 * low);

    let image = ProbeBaker::to_image(&volume, &probes).unwrap();
    let size = image.texture_descriptor.size;
    assert_eq!(
        (size.width, size.height, size.depth_or_array_layers),
        (3, 2, 14)
    );
    assert_eq!(image.data.as_ref().map(Vec::len), Some(3 * 2 * 14 * 16));
    assert!(ProbeBaker::to_image(&volume, &probes[..3]).is_none());
}
