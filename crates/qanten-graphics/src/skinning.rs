//! Linear blend skinning on the NPU, for the CPU-side copies of animated
//! meshes engines need for collision, ray picking, cloth attachment, or
//! audio occlusion.
//!
//! # Formulation
//!
//! Skinning is `v' = Σ_b w_vb · M_b · (v, 1)`. Written as an offset from the
//! rest pose, `v' = v + Σ_b w_vb · (M_b − I) · (v, 1)`, it becomes one matrix
//! product per frame:
//!
//! - a constant matrix `A[v, (b, j)] = w_vb · ṽ_j` (`vertices × 4·bones`),
//!   uploaded once and resident on the NPU;
//! - per character, the three rows of every `(M_b − I)` (`3 × 4·bones`).
//!
//! Crowds sharing one mesh go through in a single call. Computing offsets
//! instead of positions keeps the bf16 error proportional to how far the
//! mesh moved, not how large it is.

use glam::{Mat4, Vec3};
use rayon::prelude::*;

use crate::accel::ResidentMatrix;
use crate::{Error, Npu, Result};

/// Up to four bone influences per vertex (unused slots have weight 0).
pub type Influences = [(u16, f32); 4];

/// Skinned positions (and normals, if the mesh has them) of one or more characters.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Skinned {
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
}

#[derive(Debug)]
struct NpuSkin {
    positions: ResidentMatrix,
    normals: Option<ResidentMatrix>,
}

/// A mesh bound to a skeleton.
#[derive(Debug)]
pub struct SkinnedMesh {
    rest: Vec<Vec3>,
    rest_normals: Option<Vec<Vec3>>,
    influences: Vec<Influences>,
    bones: usize,
    npu: Option<NpuSkin>,
}

impl SkinnedMesh {
    /// Weights are renormalised to sum to 1 per vertex.
    pub fn new(
        rest: Vec<Vec3>,
        rest_normals: Option<Vec<Vec3>>,
        mut influences: Vec<Influences>,
        bones: usize,
    ) -> Result<Self> {
        if influences.len() != rest.len() {
            return Err(Error::Shape(format!(
                "{} influence sets for {} vertices",
                influences.len(),
                rest.len()
            )));
        }
        if rest_normals.as_ref().is_some_and(|n| n.len() != rest.len()) {
            return Err(Error::Shape("one normal per vertex required".into()));
        }
        for inf in &mut influences {
            if inf.iter().any(|&(b, w)| w != 0.0 && b as usize >= bones) {
                return Err(Error::InvalidArgument(format!(
                    "bone index out of range (have {bones} bones)"
                )));
            }
            let sum: f32 = inf.iter().map(|&(_, w)| w).sum();
            if sum > 0.0 {
                inf.iter_mut().for_each(|(_, w)| *w /= sum);
            }
        }
        Ok(Self {
            rest,
            rest_normals,
            influences,
            bones,
            npu: None,
        })
    }

    /// Uploads the constant skinning matrices to `npu`.
    ///
    /// Memory: `vertices × 4·bones` (plus `vertices × 3·bones` with normals)
    /// bf16 values, e.g. 20k vertices × 64 bones ≈ 10 MB.
    pub fn with_npu(mut self, npu: &Npu) -> Result<Self> {
        let (v, b) = (self.rest.len(), self.bones);
        let mut a = vec![0.0f32; v * 4 * b];
        for (vi, (p, inf)) in self.rest.iter().zip(&self.influences).enumerate() {
            let h = [p.x, p.y, p.z, 1.0];
            for &(bone, w) in inf {
                for j in 0..4 {
                    a[vi * 4 * b + bone as usize * 4 + j] += w * h[j];
                }
            }
        }
        let positions = ResidentMatrix::upload(npu, &a, v, 4 * b)?;
        let normals = match &self.rest_normals {
            Some(ns) => {
                let mut m = vec![0.0f32; v * 3 * b];
                for (vi, (n, inf)) in ns.iter().zip(&self.influences).enumerate() {
                    for &(bone, w) in inf {
                        for j in 0..3 {
                            m[vi * 3 * b + bone as usize * 3 + j] += w * n[j];
                        }
                    }
                }
                Some(ResidentMatrix::upload(npu, &m, v, 3 * b)?)
            }
            None => None,
        };
        self.npu = Some(NpuSkin { positions, normals });
        Ok(self)
    }

    pub fn vertex_count(&self) -> usize {
        self.rest.len()
    }

    pub fn bone_count(&self) -> usize {
        self.bones
    }

    /// Skins one character. `bones[b]` is the skinning matrix of bone `b`
    /// (animated world transform × inverse bind matrix).
    pub fn skin(&self, bones: &[Mat4]) -> Result<Skinned> {
        self.skin_batch(bones)
    }

    /// Skins `characters` poses in one call. `poses` is `characters × bones`
    /// matrices; the result holds `characters × vertices` entries.
    pub fn skin_batch(&self, poses: &[Mat4]) -> Result<Skinned> {
        let chars = self.check(poses)?;
        let Some(npu) = &self.npu else {
            return Ok(self.skin_batch_cpu(poses));
        };
        let b = self.bones;
        let v = self.rest.len();
        // Rows (char, i): entries (bone, j) = (M_b - I)[i][j].
        let mut delta_rows = vec![0.0f32; chars * 3 * 4 * b];
        for (c, pose) in poses.chunks(b).enumerate() {
            for (bi, m) in pose.iter().enumerate() {
                for i in 0..3 {
                    for j in 0..4 {
                        let ident = if i == j { 1.0 } else { 0.0 };
                        delta_rows[(c * 3 + i) * 4 * b + bi * 4 + j] = m.col(j)[i] - ident;
                    }
                }
            }
        }
        let offsets = npu.positions.apply(&delta_rows, chars * 3)?; // [chars*3, V]
        let positions: Vec<Vec3> = (0..chars * v)
            .into_par_iter()
            .map(|k| {
                let (c, vi) = (k / v, k % v);
                let o = |i: usize| offsets[(c * 3 + i) * v + vi];
                self.rest[vi] + Vec3::new(o(0), o(1), o(2))
            })
            .collect();

        let normals = match &npu.normals {
            Some(nm) => {
                let mut rot_rows = vec![0.0f32; chars * 3 * 3 * b];
                for (c, pose) in poses.chunks(b).enumerate() {
                    for (bi, m) in pose.iter().enumerate() {
                        for i in 0..3 {
                            for j in 0..3 {
                                rot_rows[(c * 3 + i) * 3 * b + bi * 3 + j] = m.col(j)[i];
                            }
                        }
                    }
                }
                let out = nm.apply(&rot_rows, chars * 3)?;
                (0..chars * v)
                    .into_par_iter()
                    .map(|k| {
                        let (c, vi) = (k / v, k % v);
                        let o = |i: usize| out[(c * 3 + i) * v + vi];
                        Vec3::new(o(0), o(1), o(2)).normalize_or(Vec3::Y)
                    })
                    .collect()
            }
            None => Vec::new(),
        };
        Ok(Skinned { positions, normals })
    }

    /// Standard CPU linear blend skinning (reference / fallback).
    pub fn skin_batch_cpu(&self, poses: &[Mat4]) -> Skinned {
        let b = self.bones.max(1);
        let v = self.rest.len();
        let chars = poses.len() / b;
        let positions = (0..chars * v)
            .into_par_iter()
            .map(|k| {
                let (c, vi) = (k / v, k % v);
                let pose = &poses[c * b..(c + 1) * b];
                self.influences[vi]
                    .iter()
                    .filter(|(_, w)| *w != 0.0)
                    .map(|&(bone, w)| w * pose[bone as usize].transform_point3(self.rest[vi]))
                    .sum()
            })
            .collect();
        let normals = match &self.rest_normals {
            Some(ns) => (0..chars * v)
                .into_par_iter()
                .map(|k| {
                    let (c, vi) = (k / v, k % v);
                    let pose = &poses[c * b..(c + 1) * b];
                    self.influences[vi]
                        .iter()
                        .filter(|(_, w)| *w != 0.0)
                        .map(|&(bone, w)| w * pose[bone as usize].transform_vector3(ns[vi]))
                        .sum::<Vec3>()
                        .normalize_or(Vec3::Y)
                })
                .collect(),
            None => Vec::new(),
        };
        Skinned { positions, normals }
    }

    fn check(&self, poses: &[Mat4]) -> Result<usize> {
        if self.bones == 0 || !poses.len().is_multiple_of(self.bones) {
            return Err(Error::Shape(format!(
                "{} matrices is not a multiple of {} bones",
                poses.len(),
                self.bones
            )));
        }
        Ok(poses.len() / self.bones)
    }
}
