//! CPU-side skinned copies of animated meshes for picking and collision,
//! skinned on the NPU from Bevy's joint transforms.

use std::collections::HashMap;
use std::sync::Arc;

use bevy::prelude::*;
use qanten_graphics::skinning::SkinnedMesh;

use crate::Convert;

/// Skins every [`NpuSkinnedCollider`] after transforms are propagated.
/// Colliders that share one [`SkinnedMesh`] (the same `Arc`) are skinned in
/// a single NPU call per frame.
#[derive(Debug, Clone, Copy, Default)]
pub struct SkinnedColliderPlugin;

impl Plugin for SkinnedColliderPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            PostUpdate,
            update_colliders.after(bevy::transform::TransformSystems::Propagate),
        );
    }
}

/// The deformed mesh of an animated entity, in the entity's local space,
/// for ray picking, collision proxies or attachment points.
///
/// Bevy skins meshes on the GPU for rendering; this keeps a CPU copy that
/// follows the same joints, skinned on the NPU.
#[derive(Component, Debug, Clone)]
pub struct NpuSkinnedCollider {
    /// The bind-pose mesh and weights (use `with_npu` to skin on the NPU).
    /// Share one `Arc` between characters to skin them together.
    pub mesh: Arc<SkinnedMesh>,
    /// The joint entities, in the order of the mesh's bones.
    pub joints: Vec<Entity>,
    /// Inverse bind matrix of each joint.
    pub inverse_bind: Vec<Mat4>,
    /// Triangles (indices into `positions`), for ray casts.
    pub triangles: Vec<[u32; 3]>,
    /// Skinned positions after the last update, in the entity's local space.
    pub positions: Vec<Vec3>,
}

impl NpuSkinnedCollider {
    /// A collider for `mesh` driven by `joints`. Positions start at the rest pose.
    pub fn new(
        mesh: Arc<SkinnedMesh>,
        joints: Vec<Entity>,
        inverse_bind: Vec<Mat4>,
        triangles: Vec<[u32; 3]>,
    ) -> Self {
        let rest = vec![Vec3::ZERO; mesh.vertex_count()];
        Self {
            mesh,
            joints,
            inverse_bind,
            triangles,
            positions: rest,
        }
    }

    /// From a skinned glTF model; `joints[k]` is the entity of the skin's `k`-th joint.
    #[cfg(feature = "gltf")]
    pub fn from_gltf(
        model: qanten_graphics::gltf::SkinnedModel,
        npu: Option<&qanten::Npu>,
        joints: Vec<Entity>,
    ) -> qanten::Result<Self> {
        let mesh = match npu {
            Some(npu) => model.mesh.with_npu(npu)?,
            None => model.mesh,
        };
        Ok(Self::new(
            Arc::new(mesh),
            joints,
            model
                .skin
                .inverse_bind
                .iter()
                .map(|m| m.convert())
                .collect(),
            model.triangles,
        ))
    }

    /// Nearest hit of the ray `origin + t·dir` (local space) with the
    /// skinned triangles: `(t, triangle index)`.
    pub fn ray_cast(&self, origin: Vec3, dir: Vec3) -> Option<(f32, usize)> {
        let mut best: Option<(f32, usize)> = None;
        for (k, tri) in self.triangles.iter().enumerate() {
            let [a, b, c] = tri.map(|i| self.positions.get(i as usize).copied());
            let (Some(a), Some(b), Some(c)) = (a, b, c) else {
                continue;
            };
            if let Some(t) = ray_triangle(origin, dir, a, b, c)
                && best.is_none_or(|(bt, _)| t < bt)
            {
                best = Some((t, k));
            }
        }
        best
    }
}

/// Möller–Trumbore ray/triangle intersection: the ray parameter of the hit.
fn ray_triangle(o: Vec3, d: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Option<f32> {
    let (e1, e2) = (b - a, c - a);
    let p = d.cross(e2);
    let det = e1.dot(p);
    if det.abs() < 1e-9 {
        return None;
    }
    let inv = 1.0 / det;
    let s = o - a;
    let u = s.dot(p) * inv;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = s.cross(e1);
    let v = d.dot(q) * inv;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = e2.dot(q) * inv;
    (t >= 0.0).then_some(t)
}

/// One collider in a group: its entity and bone matrices.
type Member = (Entity, Vec<qanten_graphics::glam::Mat4>);

fn update_colliders(
    mut colliders: Query<(Entity, &GlobalTransform, &mut NpuSkinnedCollider)>,
    joints: Query<&GlobalTransform>,
) {
    // Bone matrices of every collider, grouped by shared mesh.
    let mut groups: HashMap<usize, (Arc<SkinnedMesh>, Vec<Member>)> = HashMap::new();
    for (entity, global, collider) in &colliders {
        let to_local = global.affine().inverse();
        let bones: Option<Vec<qanten_graphics::glam::Mat4>> = collider
            .joints
            .iter()
            .zip(&collider.inverse_bind)
            .map(|(&j, ibm)| {
                let joint = joints.get(j).ok()?;
                Some((Mat4::from(to_local) * joint.to_matrix() * *ibm).convert())
            })
            .collect();
        let Some(bones) = bones else {
            warn_once!("qanten: a skinned collider's joint entity is missing");
            continue;
        };
        let key = Arc::as_ptr(&collider.mesh) as usize;
        groups
            .entry(key)
            .or_insert_with(|| (Arc::clone(&collider.mesh), Vec::new()))
            .1
            .push((entity, bones));
    }
    for (mesh, members) in groups.into_values() {
        let poses: Vec<&[qanten_graphics::glam::Mat4]> =
            members.iter().map(|(_, b)| b.as_slice()).collect();
        let skinned = match mesh.skin_batch(&poses) {
            Ok(s) => s,
            Err(e) => {
                warn_once!("qanten: NPU skinning failed ({e}); using the CPU");
                mesh.skin_batch_cpu(&poses)
            }
        };
        for ((entity, _), result) in members.iter().zip(skinned) {
            if let Ok((_, _, mut collider)) = colliders.get_mut(*entity) {
                collider.positions = result.positions.iter().map(|p| p.convert()).collect();
            }
        }
    }
}
