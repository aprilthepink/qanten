//! Load glTF 2.0 assets (`.gltf` with embedded or side-by-side buffers, or
//! `.glb`) straight into the graphics types: meshes, [`SkinnedMesh`]es with
//! their skeleton, [`Blendshapes`] from morph targets, and animation clips
//! sampled into skinning matrices.
//!
//! Requires the `gltf` feature.
//!
//! ```no_run
//! # fn main() -> qanten_graphics::Result<()> {
//! use qanten_graphics::gltf::GltfScene;
//!
//! let scene = GltfScene::load("character.glb")?;
//! let model = scene.skinned_mesh(0)?;
//! let walk = scene.animation("Walk").expect("clip");
//! let bones = walk.skinning_matrices(&model.skin, 0.5);
//! let posed = model.mesh.skin(&bones)?;
//! # Ok(()) }
//! ```
//!
//! # What is read
//!
//! - **Meshes**: positions, normals and indices of every triangle primitive of
//!   a mesh, merged into one vertex list (strips and fans become triangles).
//! - **Skins**: `JOINTS_0` / `WEIGHTS_0` become [`Influences`] (bone `k` is
//!   the skin's `k`-th joint), plus the joint nodes and inverse bind matrices.
//! - **Morph targets**: `targets[*].POSITION`, which glTF already stores as
//!   offsets, become [`Blendshapes`]; the default weights come from the mesh.
//! - **Animations**: translation, rotation, scale and morph-weight channels
//!   with `STEP`, `LINEAR` and `CUBICSPLINE` interpolation.
//!
//! Textures, materials and cameras are ignored.
//!
//! # Untrusted files
//!
//! Every accessor is bounds-checked before it is read, node hierarchies with
//! cycles are rejected, and external buffers must be plain relative paths
//! next to the file (no `..`, absolute paths or URL schemes).

use std::path::{Component, Path};

use glam::{Mat4, Quat, Vec3};
use gltf::accessor::{DataType, Dimensions};
use gltf::animation::util::ReadOutputs;
use gltf::animation::{Interpolation, Property};
use gltf::buffer::Source;
use gltf::mesh::Mode;

use crate::blendshapes::Blendshapes;
use crate::skinning::{Influences, SkinnedMesh};
use crate::{Error, Result};

fn err(msg: impl std::fmt::Display) -> Error {
    Error::Format(format!("glTF: {msg}"))
}

/// Largest buffer accepted (declared or read from disk): 1 GiB.
const MAX_BUFFER_BYTES: usize = 1 << 30;

/// Triangle geometry of one mesh, all primitives merged.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct MeshData {
    /// Vertex positions.
    pub positions: Vec<Vec3>,
    /// Vertex normals, if every primitive has them.
    pub normals: Option<Vec<Vec3>>,
    /// Triangles as indices into `positions`.
    pub triangles: Vec<[u32; 3]>,
}

/// A skin: which nodes drive the bones, and their inverse bind matrices.
#[derive(Debug, Clone, PartialEq)]
pub struct Skin {
    /// The skin's name, if it has one.
    pub name: Option<String>,
    /// Node index of each joint; bone `k` of the mesh is `joints[k]`.
    pub joints: Vec<usize>,
    /// Inverse bind matrix of each joint (identity if the file has none).
    pub inverse_bind: Vec<Mat4>,
}

/// A skinned mesh with its skeleton, ready for [`SkinnedMesh::skin`].
#[derive(Debug)]
pub struct SkinnedModel {
    /// The mesh bound to the skin; enable the NPU with `with_npu`.
    pub mesh: SkinnedMesh,
    /// The skeleton driving it.
    pub skin: Skin,
    /// The node that instantiates the mesh.
    pub node: usize,
    /// Triangles for rendering.
    pub triangles: Vec<[u32; 3]>,
}

/// A mesh with morph targets.
#[derive(Debug)]
pub struct MorphModel {
    /// Base positions plus one offset set per target.
    pub shapes: Blendshapes,
    /// The mesh's default weights (`mesh.weights`, zeros if absent).
    pub default_weights: Vec<f32>,
    /// The node that instantiates the mesh, if any (animations target it).
    pub node: Option<usize>,
    /// Triangles for rendering.
    pub triangles: Vec<[u32; 3]>,
}

/// Translation, rotation and scale of a node.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Trs {
    /// Translation.
    pub translation: Vec3,
    /// Rotation (unit quaternion).
    pub rotation: Quat,
    /// Scale.
    pub scale: Vec3,
}

impl Trs {
    /// The transform as a matrix (`T · R · S`).
    pub fn matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }
}

/// What an animation channel drives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Target {
    Translation,
    Rotation,
    Scale,
    Weights,
}

#[derive(Debug, Clone)]
struct Channel {
    node: usize,
    target: Target,
    interpolation: Interpolation,
    times: Vec<f32>,
    /// `times.len() × (3 if cubic) × width` values.
    values: Vec<f32>,
    /// Components per keyframe value (3, 4, or the number of morph targets).
    width: usize,
}

impl Channel {
    /// Keyframe `k`'s value (for cubic splines, part 0/1/2 = in-tangent,
    /// value, out-tangent).
    fn key(&self, k: usize, part: usize) -> &[f32] {
        let parts = if self.interpolation == Interpolation::CubicSpline {
            3
        } else {
            1
        };
        let at = (k * parts + if parts == 3 { part } else { 0 }) * self.width;
        &self.values[at..at + self.width]
    }

    /// The channel's value at time `t` (clamped to its keyframes).
    fn sample(&self, t: f32) -> Vec<f32> {
        let n = self.times.len();
        let k = self.times.partition_point(|&x| x <= t);
        if k == 0 {
            return self.key(0, 1).to_vec();
        }
        if k == n {
            return self.key(n - 1, 1).to_vec();
        }
        let (k0, k1) = (k - 1, k);
        let dt = self.times[k1] - self.times[k0];
        let s = if dt > 0.0 {
            (t - self.times[k0]) / dt
        } else {
            0.0
        };
        match self.interpolation {
            Interpolation::Step => self.key(k0, 1).to_vec(),
            Interpolation::Linear if self.target == Target::Rotation => {
                let q = |v: &[f32]| Quat::from_xyzw(v[0], v[1], v[2], v[3]).normalize();
                q(self.key(k0, 1))
                    .slerp(q(self.key(k1, 1)), s)
                    .to_array()
                    .to_vec()
            }
            Interpolation::Linear => self
                .key(k0, 1)
                .iter()
                .zip(self.key(k1, 1))
                .map(|(a, b)| a + (b - a) * s)
                .collect(),
            Interpolation::CubicSpline => {
                // Hermite spline (glTF 2.0 spec, appendix C).
                let (s2, s3) = (s * s, s * s * s);
                let (h00, h10, h01, h11) = (
                    2.0 * s3 - 3.0 * s2 + 1.0,
                    s3 - 2.0 * s2 + s,
                    -2.0 * s3 + 3.0 * s2,
                    s3 - s2,
                );
                let (v0, b0, v1, a1) = (
                    self.key(k0, 1),
                    self.key(k0, 2),
                    self.key(k1, 1),
                    self.key(k1, 0),
                );
                let mut out: Vec<f32> = (0..self.width)
                    .map(|c| h00 * v0[c] + h10 * dt * b0[c] + h01 * v1[c] + h11 * dt * a1[c])
                    .collect();
                if self.target == Target::Rotation {
                    let q = Quat::from_slice(&out).normalize();
                    out = q.to_array().to_vec();
                }
                out
            }
        }
    }
}

#[derive(Debug, Clone)]
struct AnimationData {
    name: Option<String>,
    channels: Vec<Channel>,
    start: f32,
    end: f32,
}

/// Node transforms (and animated morph weights) at one moment.
#[derive(Debug, Clone, PartialEq)]
pub struct Pose {
    local: Vec<Trs>,
    global: Vec<Mat4>,
    weights: Vec<Option<Vec<f32>>>,
}

impl Pose {
    /// World transform of node `node`.
    pub fn node_transform(&self, node: usize) -> Option<Mat4> {
        self.global.get(node).copied()
    }

    /// Local translation, rotation and scale of node `node`.
    pub fn local(&self, node: usize) -> Option<Trs> {
        self.local.get(node).copied()
    }

    /// Morph weights animated on node `node`, if a channel drives them.
    pub fn morph_weights(&self, node: usize) -> Option<&[f32]> {
        self.weights.get(node)?.as_deref()
    }

    /// One matrix per joint, `global(joint) · inverse_bind(joint)`, for
    /// [`SkinnedMesh::skin`]. As the glTF spec says, the transform of the
    /// node holding the mesh is not applied: positions come out in the
    /// skeleton's space.
    pub fn skinning_matrices(&self, skin: &Skin) -> Vec<Mat4> {
        skin.joints
            .iter()
            .zip(&skin.inverse_bind)
            .map(|(&j, ibm)| self.global[j] * *ibm)
            .collect()
    }
}

/// One animation of a [`GltfScene`].
#[derive(Debug, Clone, Copy)]
pub struct Clip<'a> {
    scene: &'a GltfScene,
    data: &'a AnimationData,
}

impl Clip<'_> {
    /// The animation's name, if it has one.
    pub fn name(&self) -> Option<&str> {
        self.data.name.as_deref()
    }

    /// Time of the first keyframe, in seconds.
    pub fn start(&self) -> f32 {
        self.data.start
    }

    /// Time of the last keyframe, in seconds.
    pub fn end(&self) -> f32 {
        self.data.end
    }

    /// `end − start`.
    pub fn duration(&self) -> f32 {
        self.data.end - self.data.start
    }

    /// The scene posed at time `t` (clamped to the clip). Nodes the clip
    /// does not animate keep their rest transform.
    pub fn sample(&self, t: f32) -> Pose {
        let mut local = self.scene.rest.clone();
        let mut weights = vec![None; local.len()];
        for c in &self.data.channels {
            let v = c.sample(t);
            let l = &mut local[c.node];
            match c.target {
                Target::Translation => l.translation = Vec3::from_slice(&v),
                Target::Rotation => l.rotation = Quat::from_slice(&v).normalize(),
                Target::Scale => l.scale = Vec3::from_slice(&v),
                Target::Weights => weights[c.node] = Some(v),
            }
        }
        self.scene.pose_from(local, weights)
    }

    /// Skinning matrices for `skin` at time `t`: `sample(t).skinning_matrices(skin)`.
    pub fn skinning_matrices(&self, skin: &Skin, t: f32) -> Vec<Mat4> {
        self.sample(t).skinning_matrices(skin)
    }
}

/// A loaded glTF document with its buffers and node hierarchy.
#[derive(Debug)]
pub struct GltfScene {
    doc: gltf::Document,
    buffers: Vec<Vec<u8>>,
    /// Nodes ordered so parents come before their children.
    order: Vec<usize>,
    parents: Vec<Option<usize>>,
    rest: Vec<Trs>,
    animations: Vec<AnimationData>,
}

impl GltfScene {
    /// Loads a `.gltf` or `.glb` file. External buffers are resolved
    /// relative to the file's directory.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path)?;
        Self::from_slice(&bytes, path.parent())
    }

    /// Parses glTF JSON or GLB bytes. `base_dir` is where external buffer
    /// files are looked up; without it only embedded buffers work.
    pub fn from_slice(bytes: &[u8], base_dir: Option<&Path>) -> Result<Self> {
        let gltf::Gltf { document, mut blob } = parse(bytes)?;
        let buffers = document
            .buffers()
            .map(|b| {
                let mut data = match b.source() {
                    Source::Bin => blob.take().ok_or_else(|| err("GLB has no binary chunk"))?,
                    Source::Uri(uri) => read_uri(uri, base_dir)?,
                };
                if b.length() > MAX_BUFFER_BYTES || data.len() < b.length() {
                    return Err(err(format!(
                        "buffer {} declares {} bytes, has {}",
                        b.index(),
                        b.length(),
                        data.len()
                    )));
                }
                data.truncate(b.length());
                Ok(data)
            })
            .collect::<Result<Vec<_>>>()?;
        for a in document.accessors() {
            check_accessor(&a, &buffers)?;
        }
        check_animation_targets(&document)?;
        let (parents, order) = hierarchy(&document)?;
        let rest = document
            .nodes()
            .map(|n| {
                let (t, r, s) = n.transform().decomposed();
                Trs {
                    translation: Vec3::from(t),
                    rotation: Quat::from_array(r).normalize(),
                    scale: Vec3::from(s),
                }
            })
            .collect();
        let mut scene = Self {
            doc: document,
            buffers,
            order,
            parents,
            rest,
            animations: Vec::new(),
        };
        scene.animations = scene
            .doc
            .animations()
            .map(|a| scene.read_animation(&a))
            .collect::<Result<_>>()?;
        Ok(scene)
    }

    /// Number of meshes.
    pub fn mesh_count(&self) -> usize {
        self.doc.meshes().len()
    }

    /// Mesh names (`None` for unnamed meshes), by index.
    pub fn mesh_names(&self) -> Vec<Option<&str>> {
        self.doc.meshes().map(|m| m.name()).collect()
    }

    /// Number of nodes.
    pub fn node_count(&self) -> usize {
        self.rest.len()
    }

    /// Parent of node `node`, if it has one.
    pub fn parent(&self, node: usize) -> Option<usize> {
        self.parents.get(node).copied().flatten()
    }

    /// Node names (`None` for unnamed nodes), by index.
    pub fn node_names(&self) -> Vec<Option<&str>> {
        self.doc.nodes().map(|n| n.name()).collect()
    }

    /// Names of the animations, in file order.
    pub fn animation_names(&self) -> Vec<Option<&str>> {
        self.animations.iter().map(|a| a.name.as_deref()).collect()
    }

    /// The animation called `name`.
    pub fn animation(&self, name: &str) -> Option<Clip<'_>> {
        self.animations
            .iter()
            .find(|a| a.name.as_deref() == Some(name))
            .map(|data| Clip { scene: self, data })
    }

    /// Animation `index` in file order.
    pub fn animation_at(&self, index: usize) -> Option<Clip<'_>> {
        self.animations
            .get(index)
            .map(|data| Clip { scene: self, data })
    }

    /// The scene in its rest pose (every node at its file transform).
    pub fn rest_pose(&self) -> Pose {
        self.pose_from(self.rest.clone(), vec![None; self.rest.len()])
    }

    /// Mesh `index` as plain triangle geometry.
    pub fn mesh(&self, index: usize) -> Result<MeshData> {
        Ok(self.read_mesh(index)?.data)
    }

    /// Mesh `index` bound to the skin of the first node that uses it.
    pub fn skinned_mesh(&self, index: usize) -> Result<SkinnedModel> {
        let node = self
            .doc
            .nodes()
            .find(|n| n.mesh().is_some_and(|m| m.index() == index) && n.skin().is_some())
            .ok_or_else(|| err(format!("no node uses mesh {index} with a skin")))?;
        let gskin = node.skin().ok_or_else(|| err("node has no skin"))?;
        let joints: Vec<usize> = gskin.joints().map(|j| j.index()).collect();
        let inverse_bind = match gskin.inverse_bind_matrices() {
            Some(acc) => {
                expect(
                    &acc,
                    Dimensions::Mat4,
                    &[DataType::F32],
                    "inverseBindMatrices",
                )?;
                let ibm: Vec<Mat4> = gskin
                    .reader(|b| self.buffer(b))
                    .read_inverse_bind_matrices()
                    .ok_or_else(|| err("unreadable inverse bind matrices"))?
                    .map(|m| Mat4::from_cols_array_2d(&m))
                    .collect();
                if ibm.len() != joints.len() {
                    return Err(err(format!(
                        "{} inverse bind matrices for {} joints",
                        ibm.len(),
                        joints.len()
                    )));
                }
                ibm
            }
            None => vec![Mat4::IDENTITY; joints.len()],
        };
        let merged = self.read_mesh(index)?;
        let influences = merged
            .influences
            .ok_or_else(|| err(format!("mesh {index} has no JOINTS_0 / WEIGHTS_0")))?;
        let mesh = SkinnedMesh::new(
            merged.data.positions,
            merged.data.normals,
            influences,
            joints.len(),
        )?;
        Ok(SkinnedModel {
            mesh,
            skin: Skin {
                name: gskin.name().map(str::to_owned),
                joints,
                inverse_bind,
            },
            node: node.index(),
            triangles: merged.data.triangles,
        })
    }

    /// Mesh `index`'s morph targets as [`Blendshapes`].
    pub fn blendshapes(&self, index: usize) -> Result<MorphModel> {
        let merged = self.read_mesh(index)?;
        if merged.targets.is_empty() {
            return Err(err(format!("mesh {index} has no morph targets")));
        }
        let mesh = self
            .doc
            .meshes()
            .nth(index)
            .ok_or_else(|| err(format!("no mesh {index}")))?;
        let mut default_weights = mesh.weights().map(<[f32]>::to_vec).unwrap_or_default();
        default_weights.resize(merged.targets.len(), 0.0);
        let node = self
            .doc
            .nodes()
            .find(|n| n.mesh().is_some_and(|m| m.index() == index))
            .map(|n| n.index());
        Ok(MorphModel {
            shapes: Blendshapes::new(merged.data.positions, &merged.targets)?,
            default_weights,
            node,
            triangles: merged.data.triangles,
        })
    }

    // ---- internals

    fn buffer(&self, b: gltf::Buffer<'_>) -> Option<&[u8]> {
        self.buffers.get(b.index()).map(Vec::as_slice)
    }

    fn pose_from(&self, local: Vec<Trs>, weights: Vec<Option<Vec<f32>>>) -> Pose {
        let mut global = vec![Mat4::IDENTITY; local.len()];
        for &n in &self.order {
            let m = local[n].matrix();
            global[n] = match self.parents[n] {
                Some(p) => global[p] * m,
                None => m,
            };
        }
        Pose {
            local,
            global,
            weights,
        }
    }

    fn read_mesh(&self, index: usize) -> Result<Merged> {
        let mesh = self
            .doc
            .meshes()
            .nth(index)
            .ok_or_else(|| err(format!("no mesh {index}")))?;
        let mut out = Merged::default();
        let mut normals = Some(Vec::new());
        let mut joints_seen = false;
        let mut influences = Vec::new();
        for (pi, prim) in mesh.primitives().enumerate() {
            let what = |s: &str| format!("mesh {index} primitive {pi} {s}");
            let reader = prim.reader(|b| self.buffer(b));
            let pos_acc = prim
                .get(&gltf::Semantic::Positions)
                .ok_or_else(|| err(what("has no POSITION")))?;
            expect(&pos_acc, Dimensions::Vec3, &[DataType::F32], "POSITION")?;
            let positions: Vec<Vec3> = reader
                .read_positions()
                .ok_or_else(|| err(what("POSITION is unreadable")))?
                .map(Vec3::from)
                .collect();
            let n = positions.len();
            let base = out.data.positions.len();
            let base32 = u32::try_from(base).map_err(|_| err("too many vertices"))?;

            // Normals: kept only if every primitive has them.
            match (&mut normals, prim.get(&gltf::Semantic::Normals)) {
                (Some(ns), Some(acc)) => {
                    expect(&acc, Dimensions::Vec3, &[DataType::F32], "NORMAL")?;
                    let read: Vec<Vec3> = reader
                        .read_normals()
                        .ok_or_else(|| err(what("NORMAL is unreadable")))?
                        .map(Vec3::from)
                        .collect();
                    if read.len() != n {
                        return Err(err(what("has a different number of normals")));
                    }
                    ns.extend(read);
                }
                _ => normals = None,
            }

            // Triangles.
            let local: Vec<u32> = match prim.indices() {
                Some(acc) => {
                    expect(
                        &acc,
                        Dimensions::Scalar,
                        &[DataType::U8, DataType::U16, DataType::U32],
                        "indices",
                    )?;
                    reader
                        .read_indices()
                        .ok_or_else(|| err(what("indices are unreadable")))?
                        .into_u32()
                        .collect()
                }
                None => (0..u32::try_from(n).map_err(|_| err("too many vertices"))?).collect(),
            };
            if local.iter().any(|&i| i as usize >= n) {
                return Err(err(what("has an index past its vertices")));
            }
            let tris = triangles(prim.mode(), &local).ok_or_else(|| {
                err(what(&format!(
                    "uses mode {:?}; only triangles are supported",
                    prim.mode()
                )))
            })?;
            for t in tris {
                out.data.triangles.push([
                    t[0].checked_add(base32)
                        .ok_or_else(|| err("too many vertices"))?,
                    t[1].checked_add(base32)
                        .ok_or_else(|| err("too many vertices"))?,
                    t[2].checked_add(base32)
                        .ok_or_else(|| err("too many vertices"))?,
                ]);
            }

            // Skin influences.
            let joints = prim.get(&gltf::Semantic::Joints(0));
            let weights = prim.get(&gltf::Semantic::Weights(0));
            match (joints, weights) {
                (Some(ja), Some(wa)) => {
                    expect(
                        &ja,
                        Dimensions::Vec4,
                        &[DataType::U8, DataType::U16],
                        "JOINTS_0",
                    )?;
                    expect(
                        &wa,
                        Dimensions::Vec4,
                        &[DataType::F32, DataType::U8, DataType::U16],
                        "WEIGHTS_0",
                    )?;
                    let j: Vec<[u16; 4]> = reader
                        .read_joints(0)
                        .ok_or_else(|| err(what("JOINTS_0 is unreadable")))?
                        .into_u16()
                        .collect();
                    let w: Vec<[f32; 4]> = reader
                        .read_weights(0)
                        .ok_or_else(|| err(what("WEIGHTS_0 is unreadable")))?
                        .into_f32()
                        .collect();
                    if j.len() != n || w.len() != n {
                        return Err(err(what(
                            "has joints or weights for a different vertex count",
                        )));
                    }
                    joints_seen = true;
                    influences.extend(
                        j.iter()
                            .zip(&w)
                            .map(|(j, w)| -> Influences { std::array::from_fn(|k| (j[k], w[k])) }),
                    );
                }
                _ => influences.extend(std::iter::repeat_n([(0u16, 0.0f32); 4], n)),
            }

            // Morph targets: one offset set per target, merged across primitives.
            // The reader reads all three attributes, so check them all first.
            for t in prim.morph_targets() {
                for acc in [t.positions(), t.normals(), t.tangents()]
                    .into_iter()
                    .flatten()
                {
                    expect(&acc, Dimensions::Vec3, &[DataType::F32], "morph target")?;
                }
            }
            let prim_targets: Vec<Vec<Vec3>> = reader
                .read_morph_targets()
                .map(|(p, _, _)| match p {
                    Some(it) => it.map(Vec3::from).collect(),
                    None => vec![Vec3::ZERO; n],
                })
                .collect();
            if pi == 0 {
                out.targets = vec![Vec::new(); prim_targets.len()];
            } else if prim_targets.len() != out.targets.len() {
                return Err(err(what("has a different number of morph targets")));
            }
            for (dst, t) in out.targets.iter_mut().zip(prim_targets) {
                if t.len() != n {
                    return Err(err(what("has a morph target of the wrong size")));
                }
                dst.extend(t);
            }
            out.data.positions.extend(positions);
        }
        out.data.normals = normals;
        out.influences = joints_seen.then_some(influences);
        Ok(out)
    }

    fn read_animation(&self, a: &gltf::Animation<'_>) -> Result<AnimationData> {
        let mut channels = Vec::new();
        for (ci, c) in a.channels().enumerate() {
            let what = |s: &str| format!("animation {} channel {ci} {s}", a.index());
            let sampler = c.sampler();
            expect(
                &sampler.input(),
                Dimensions::Scalar,
                &[DataType::F32],
                "sampler input",
            )?;
            let ints = [
                DataType::F32,
                DataType::I8,
                DataType::U8,
                DataType::I16,
                DataType::U16,
            ];
            let (dims, types): (Dimensions, &[DataType]) = match c.target().property() {
                Property::Translation | Property::Scale => (Dimensions::Vec3, &[DataType::F32]),
                Property::Rotation => (Dimensions::Vec4, &ints),
                Property::MorphTargetWeights => (Dimensions::Scalar, &ints),
            };
            expect(&sampler.output(), dims, types, "sampler output")?;
            let reader = c.reader(|b| self.buffer(b));
            let times: Vec<f32> = reader
                .read_inputs()
                .ok_or_else(|| err(what("has unreadable times")))?
                .collect();
            if times.is_empty()
                || times.iter().any(|t| !t.is_finite())
                || times.windows(2).any(|w| w[1] < w[0])
            {
                return Err(err(what("needs finite, increasing keyframe times")));
            }
            let (target, values, width) = match reader
                .read_outputs()
                .ok_or_else(|| err(what("has unreadable values")))?
            {
                ReadOutputs::Translations(it) => (Target::Translation, flat(it), 3),
                ReadOutputs::Scales(it) => (Target::Scale, flat(it), 3),
                ReadOutputs::Rotations(r) => (Target::Rotation, flat(r.into_f32()), 4),
                ReadOutputs::MorphTargetWeights(w) => {
                    let v: Vec<f32> = w.into_f32().collect();
                    let parts = if sampler.interpolation() == Interpolation::CubicSpline {
                        3
                    } else {
                        1
                    };
                    let width = v.len() / (times.len() * parts).max(1);
                    (Target::Weights, v, width.max(1))
                }
            };
            debug_assert_eq!(
                target == Target::Weights,
                c.target().property() == Property::MorphTargetWeights
            );
            let parts = if sampler.interpolation() == Interpolation::CubicSpline {
                3
            } else {
                1
            };
            if values.len() != times.len() * parts * width {
                return Err(err(what(&format!(
                    "has {} values for {} keyframes",
                    values.len(),
                    times.len()
                ))));
            }
            if values.iter().any(|v| !v.is_finite()) {
                return Err(err(what("has non-finite values")));
            }
            channels.push(Channel {
                node: c.target().node().index(),
                target,
                interpolation: sampler.interpolation(),
                times,
                values,
                width,
            });
        }
        let start = channels
            .iter()
            .map(|c| c.times[0])
            .fold(f32::INFINITY, f32::min);
        let end = channels
            .iter()
            .filter_map(|c| c.times.last().copied())
            .fold(f32::NEG_INFINITY, f32::max);
        Ok(AnimationData {
            name: a.name().map(str::to_owned),
            channels,
            start: if start.is_finite() { start } else { 0.0 },
            end: if end.is_finite() { end } else { 0.0 },
        })
    }
}

#[derive(Debug, Default)]
struct Merged {
    data: MeshData,
    influences: Option<Vec<Influences>>,
    targets: Vec<Vec<Vec3>>,
}

fn flat<const N: usize>(it: impl Iterator<Item = [f32; N]>) -> Vec<f32> {
    it.flatten().collect()
}

/// Triangles of a primitive in `mode`, or `None` for points and lines.
fn triangles(mode: Mode, idx: &[u32]) -> Option<Vec<[u32; 3]>> {
    Some(match mode {
        Mode::Triangles => idx.chunks_exact(3).map(|t| [t[0], t[1], t[2]]).collect(),
        Mode::TriangleStrip => (2..idx.len())
            .map(|k| {
                // Alternate the winding so every triangle faces the same way.
                if k % 2 == 0 {
                    [idx[k - 2], idx[k - 1], idx[k]]
                } else {
                    [idx[k - 1], idx[k - 2], idx[k]]
                }
            })
            .collect(),
        Mode::TriangleFan => (2..idx.len())
            .map(|k| [idx[0], idx[k - 1], idx[k]])
            .collect(),
        _ => return None,
    })
}

/// Parents of every node, and an order with parents before children.
/// Rejects nodes with two parents and cycles.
fn hierarchy(doc: &gltf::Document) -> Result<(Vec<Option<usize>>, Vec<usize>)> {
    let n = doc.nodes().len();
    let mut parents = vec![None; n];
    for node in doc.nodes() {
        for child in node.children() {
            let c = child.index();
            if parents[c].replace(node.index()).is_some() || c == node.index() {
                return Err(err(format!("node {c} has more than one parent")));
            }
        }
    }
    let mut order = Vec::with_capacity(n);
    let mut stack: Vec<usize> = (0..n).filter(|&i| parents[i].is_none()).collect();
    while let Some(i) = stack.pop() {
        order.push(i);
        if let Some(node) = doc.nodes().nth(i) {
            stack.extend(node.children().map(|c| c.index()));
        }
    }
    if order.len() != n {
        return Err(err("the node hierarchy has a cycle"));
    }
    Ok((parents, order))
}

/// Parses and validates a document. The `gltf` crate's validation itself
/// indexes POSITION accessors before checking them, so those indices are
/// checked on an unvalidated parse first; any other panic inside the crate
/// is turned into an error as a last line of defence.
fn parse(bytes: &[u8]) -> Result<gltf::Gltf> {
    let raw = gltf::Gltf::from_slice_without_validation(bytes).map_err(err)?;
    let root = raw.document.as_json();
    let positions = gltf::json::validation::Checked::Valid(gltf::Semantic::Positions);
    for m in &root.meshes {
        for p in &m.primitives {
            if p.attributes
                .get(&positions)
                .is_some_and(|i| i.value() >= root.accessors.len())
            {
                return Err(err("a POSITION accessor index is out of range"));
            }
        }
    }
    std::panic::catch_unwind(|| gltf::Gltf::from_slice(bytes))
        .map_err(|_| err("the gltf parser failed on this file"))?
        .map_err(err)
}

/// The `gltf` crate's validation skips animation channel targets, whose
/// node and property it later unwraps: check them here.
fn check_animation_targets(doc: &gltf::Document) -> Result<()> {
    let root = doc.as_json();
    for (ai, a) in root.animations.iter().enumerate() {
        for (ci, c) in a.channels.iter().enumerate() {
            let node_ok = c.target.node.value() < root.nodes.len();
            let path_ok = matches!(c.target.path, gltf::json::validation::Checked::Valid(_));
            if !node_ok || !path_ok {
                return Err(err(format!(
                    "animation {ai} channel {ci} has an invalid target"
                )));
            }
        }
    }
    Ok(())
}

/// Checks that an accessor's dimensions and component type are what a
/// semantic needs (the reader would otherwise misread it).
fn expect(
    acc: &gltf::Accessor<'_>,
    dims: Dimensions,
    types: &[DataType],
    what: &str,
) -> Result<()> {
    if acc.dimensions() != dims || !types.contains(&acc.data_type()) {
        return Err(err(format!(
            "{what} accessor {} is {:?} {:?}, expected {dims:?} of {types:?}",
            acc.index(),
            acc.dimensions(),
            acc.data_type()
        )));
    }
    Ok(())
}

/// Checks that every element an accessor (and its sparse parts) reads lies
/// inside its buffer view and buffer, with overflow-checked arithmetic, so
/// the `gltf` reader never computes a wrapped-around range.
fn check_accessor(acc: &gltf::Accessor<'_>, buffers: &[Vec<u8>]) -> Result<()> {
    let what = || format!("accessor {}", acc.index());
    let count = acc.count();
    if count == 0 || count > qanten::limits::MAX_ELEMENTS {
        return Err(err(format!("{}: count {count} is out of range", what())));
    }
    if let Some(view) = acc.view() {
        span(&view, acc.offset(), acc.size(), count, buffers)
            .map_err(|e| err(format!("{}: {e}", what())))?;
    }
    if let Some(sparse) = acc.sparse() {
        let n = sparse.count();
        if n == 0 || n > count {
            return Err(err(format!("{}: sparse count {n} is out of range", what())));
        }
        let idx = sparse.indices();
        span(
            &idx.view(),
            idx.offset(),
            idx.index_type().size(),
            n,
            buffers,
        )
        .map_err(|e| err(format!("{} sparse indices: {e}", what())))?;
        let vals = sparse.values();
        span(&vals.view(), vals.offset(), acc.size(), n, buffers)
            .map_err(|e| err(format!("{} sparse values: {e}", what())))?;
    }
    Ok(())
}

/// `count` elements of `size` bytes from `offset` in `view` fit the view,
/// and the view fits its buffer.
fn span(
    view: &gltf::buffer::View<'_>,
    offset: usize,
    size: usize,
    count: usize,
    buffers: &[Vec<u8>],
) -> std::result::Result<(), String> {
    let stride = view.stride().unwrap_or(size);
    if size == 0 || stride < size {
        return Err(format!(
            "stride {stride} is smaller than the element size {size}"
        ));
    }
    let end = stride
        .checked_mul(count - 1)
        .and_then(|x| x.checked_add(size))
        .and_then(|x| x.checked_add(offset))
        .ok_or("size overflows")?;
    if end > view.length() {
        return Err(format!(
            "reads {end} bytes of a {}-byte view",
            view.length()
        ));
    }
    let buf = buffers.get(view.buffer().index()).ok_or("missing buffer")?;
    let view_end = view
        .offset()
        .checked_add(view.length())
        .ok_or("view size overflows")?;
    if view_end > buf.len() {
        return Err(format!(
            "view ends at byte {view_end} of a {}-byte buffer",
            buf.len()
        ));
    }
    Ok(())
}

/// Buffer contents from a `data:` URI or a relative file next to the asset.
fn read_uri(uri: &str, base_dir: Option<&Path>) -> Result<Vec<u8>> {
    if let Some(rest) = uri.strip_prefix("data:") {
        let (_, payload) = rest
            .split_once(";base64,")
            .ok_or_else(|| err("only base64 data URIs are supported"))?;
        return base64_decode(payload).ok_or_else(|| err("invalid base64 in a data URI"));
    }
    let rel = Path::new(uri);
    let plain = !uri.contains(':')
        && !uri.contains('\\')
        && !uri.contains('%')
        && rel
            .components()
            .all(|c| matches!(c, Component::Normal(_) | Component::CurDir));
    if !plain {
        return Err(err(format!(
            "buffer URI {uri:?} must be a data URI or a relative path inside the asset's folder"
        )));
    }
    let dir =
        base_dir.ok_or_else(|| err(format!("external buffer {uri:?} needs a base folder")))?;
    let path = dir.join(rel);
    let len = std::fs::metadata(&path)?.len();
    if len > MAX_BUFFER_BYTES as u64 {
        return Err(err(format!("buffer file {uri:?} is larger than 1 GiB")));
    }
    Ok(std::fs::read(path)?)
}

/// Standard base64 (RFC 4648) with optional padding; whitespace is not allowed.
fn base64_decode(s: &str) -> Option<Vec<u8>> {
    let s = s.trim_end_matches('=');
    let value = |c: u8| -> Option<u32> {
        Some(match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => return None,
        } as u32)
    };
    let mut out = Vec::with_capacity(s.len() / 4 * 3 + 3);
    for chunk in s.as_bytes().chunks(4) {
        let mut acc = 0u32;
        for (k, &c) in chunk.iter().enumerate() {
            acc |= value(c)? << (18 - 6 * k);
        }
        let bytes = acc.to_be_bytes();
        match chunk.len() {
            4 => out.extend_from_slice(&bytes[1..4]),
            3 => out.extend_from_slice(&bytes[1..3]),
            2 => out.push(bytes[1]),
            _ => return None,
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_round_trips() {
        assert_eq!(base64_decode("aGVsbG8=").unwrap(), b"hello");
        assert_eq!(base64_decode("aGVsbG8").unwrap(), b"hello");
        assert_eq!(base64_decode("AAEC").unwrap(), [0, 1, 2]);
        assert!(base64_decode("a").is_none());
        assert!(base64_decode("a b=").is_none());
    }

    #[test]
    fn strips_and_fans_become_triangles() {
        assert_eq!(
            triangles(Mode::TriangleStrip, &[0, 1, 2, 3]).unwrap(),
            [[0, 1, 2], [2, 1, 3]]
        );
        assert_eq!(
            triangles(Mode::TriangleFan, &[0, 1, 2, 3]).unwrap(),
            [[0, 1, 2], [0, 2, 3]]
        );
        assert!(triangles(Mode::Lines, &[0, 1]).is_none());
    }

    #[test]
    fn unsafe_buffer_uris_are_refused() {
        let dir = Path::new(".");
        for uri in [
            "../secret.bin",
            "/etc/passwd",
            "http://x/y.bin",
            "a\\b.bin",
            "x%2e.bin",
        ] {
            assert!(read_uri(uri, Some(dir)).is_err(), "{uri}");
        }
    }
}
