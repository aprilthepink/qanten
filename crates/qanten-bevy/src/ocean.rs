//! Gerstner oceans as Bevy meshes, updated on the NPU every frame, and
//! kinematic buoyancy for floating entities.

use std::sync::Arc;

use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, Mesh3d, PrimitiveTopology};
use bevy::prelude::*;
use qanten_graphics::ocean::GerstnerOcean;

use crate::Convert;

/// Updates every [`Ocean`] mesh each frame, then moves [`Buoyancy`]
/// entities. Needs `Assets<Mesh>` (an `AssetPlugin` and
/// `init_asset::<Mesh>()`, or Bevy's default plugins).
#[derive(Debug, Clone, Copy, Default)]
pub struct OceanPlugin;

impl Plugin for OceanPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, (update_oceans, float_buoyant).chain());
    }
}

/// An ocean surface: a [`GerstnerOcean`] (use `with_npu` to evaluate it on
/// the NPU) and the mesh it writes into every frame.
#[derive(Component, Debug, Clone)]
pub struct Ocean {
    /// The wave model; shared, so the component stays cheap to clone.
    pub ocean: Arc<GerstnerOcean>,
    /// The mesh updated every frame (positions and normals).
    pub mesh: Handle<Mesh>,
    /// Simulation seconds per real second.
    pub time_scale: f64,
}

impl Ocean {
    /// Builds the grid mesh (at `t = 0`) and adds it to `meshes`. Spawn the
    /// result together with [`mesh3d`](Self::mesh3d) and a material.
    pub fn new(ocean: GerstnerOcean, meshes: &mut Assets<Mesh>) -> Self {
        let (nx, nz) = ocean.grid().resolution;
        let frame = ocean.evaluate_cpu(0.0);
        let mut mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        );
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, to_arrays(&frame.positions));
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, to_arrays(&frame.normals));
        let uv: Vec<[f32; 2]> = (0..nx * nz)
            .map(|k| {
                [
                    (k % nx) as f32 / (nx - 1).max(1) as f32,
                    (k / nx) as f32 / (nz - 1).max(1) as f32,
                ]
            })
            .collect();
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uv);
        let mut indices = Vec::with_capacity((nx - 1) * (nz - 1) * 6);
        for j in 0..nz.saturating_sub(1) {
            for i in 0..nx.saturating_sub(1) {
                let a = (j * nx + i) as u32;
                let (b, c, d) = (a + 1, a + nx as u32, a + nx as u32 + 1);
                indices.extend_from_slice(&[a, c, b, b, c, d]);
            }
        }
        mesh.insert_indices(Indices::U32(indices));
        Self {
            ocean: Arc::new(ocean),
            mesh: meshes.add(mesh),
            time_scale: 1.0,
        }
    }

    /// The `Mesh3d` component for this ocean's mesh.
    pub fn mesh3d(&self) -> Mesh3d {
        Mesh3d(self.mesh.clone())
    }

    /// Exact surface height at world `(x, z)` and time `t` (host, f64).
    pub fn height_at(&self, x: f32, z: f32, t: f64) -> f32 {
        self.ocean.sample(x, z, t).height
    }
}

fn to_arrays(v: &[qanten_graphics::glam::Vec3]) -> Vec<[f32; 3]> {
    v.iter().map(|p| p.to_array()).collect()
}

fn update_oceans(time: Res<Time>, oceans: Query<&Ocean>, mut meshes: ResMut<Assets<Mesh>>) {
    for ocean in &oceans {
        let t = time.elapsed_secs_f64() * ocean.time_scale;
        let frame = match ocean.ocean.evaluate(t) {
            Ok(frame) => frame,
            Err(e) => {
                warn_once!("qanten ocean: NPU evaluation failed ({e}); using the CPU");
                ocean.ocean.evaluate_cpu(t)
            }
        };
        if let Some(mesh) = meshes.get_mut(&ocean.mesh) {
            mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, to_arrays(&frame.positions));
            mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, to_arrays(&frame.normals));
        }
    }
}

/// Floats an entity on the first [`Ocean`]: every frame its height follows
/// the mean surface height under `points` and it tilts with the mean
/// surface normal, keeping its yaw. Kinematic (no forces); add a physics
/// engine's forces on top if you need them.
#[derive(Component, Debug, Clone)]
pub struct Buoyancy {
    /// Sample points in the entity's local frame (for example the corners of a hull).
    pub points: Vec<Vec3>,
    /// How deep the entity sits: its origin goes this far below the mean surface.
    pub draft: f32,
    /// How quickly it follows the waves, `0..=1` per frame (1 = instantly).
    pub responsiveness: f32,
}

impl Buoyancy {
    /// Samples at `points` with no draft, following the waves instantly.
    pub fn new(points: Vec<Vec3>) -> Self {
        Self {
            points,
            draft: 0.0,
            responsiveness: 1.0,
        }
    }
}

fn float_buoyant(
    time: Res<Time>,
    oceans: Query<&Ocean>,
    mut floats: Query<(&mut Transform, &Buoyancy)>,
) {
    let Some(ocean) = oceans.iter().next() else {
        return;
    };
    let t = time.elapsed_secs_f64() * ocean.time_scale;
    for (mut transform, b) in &mut floats {
        if b.points.is_empty() {
            continue;
        }
        let (yaw, _, _) = transform.rotation.to_euler(EulerRot::YXZ);
        let heading = Quat::from_rotation_y(yaw);
        let (mut height, mut normal) = (0.0, Vec3::ZERO);
        for p in &b.points {
            let world = transform.translation + heading * *p;
            let s = ocean.ocean.sample(world.x, world.z, t);
            height += s.height;
            let n: Vec3 = s.normal.convert();
            normal += n;
        }
        height /= b.points.len() as f32;
        let normal = normal.normalize_or(Vec3::Y);
        let k = b.responsiveness.clamp(0.0, 1.0);
        let target_y = height - b.draft;
        transform.translation.y += (target_y - transform.translation.y) * k;
        let target = Quat::from_rotation_arc(Vec3::Y, normal) * heading;
        transform.rotation = transform.rotation.slerp(target, k);
    }
}
