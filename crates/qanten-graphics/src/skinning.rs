//! Linear blend skinning on the NPU, for the CPU-side copies of animated
//! meshes engines need for collision, ray picking, cloth attachment, or
//! audio occlusion.
//!
//! ![100 tentacles skinned in one NPU call (example `crowd`)](https://raw.githubusercontent.com/aprilthepink/qanten/main/docs/images/crowd.png)
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
//! Memory on the NPU: `vertices × 4·bones` (plus `vertices × 3·bones` with
//! normals) bf16 values, e.g. 20k vertices × 64 bones ≈ 10 MB. The matrices
//! are built only when the mesh is accelerated.
//!
//! Crowds sharing one mesh go through in a single call. Computing offsets
//! instead of positions keeps the bf16 error proportional to how far the
//! mesh moved, not how large it is.

use glam::{Mat4, Vec3};
use qanten::compute::Matrix;
use rayon::prelude::*;

use crate::accel::{Accelerated, MatrixOp, accelerated_methods};
use crate::{Error, Result};

/// Up to four bone influences per vertex (unused slots have weight 0).
pub type Influences = [(u16, f32); 4];

/// Skinned positions (and normals, if the mesh has them) of one character.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Skinned {
    /// Skinned positions, one per vertex.
    pub positions: Vec<Vec3>,
    /// Skinned unit normals (empty if the mesh has none).
    pub normals: Vec<Vec3>,
}

/// The NPU formulation: `A` for positions and, with normals, `N[v, (b, j)] = w_vb · n_j`.
#[derive(Debug)]
struct SkinMatrices {
    positions: MatrixOp,
    normals: Option<MatrixOp>,
}

/// A mesh bound to a skeleton.
#[derive(Debug)]
pub struct SkinnedMesh {
    rest: Vec<Vec3>,
    rest_normals: Option<Vec<Vec3>>,
    influences: Vec<Influences>,
    bones: usize,
    matrices: Option<SkinMatrices>,
}

impl Accelerated for SkinnedMesh {
    fn ops(&self) -> Vec<&MatrixOp> {
        self.matrices
            .iter()
            .flat_map(|m| std::iter::once(&m.positions).chain(&m.normals))
            .collect()
    }

    fn ops_mut(&mut self) -> Vec<&mut MatrixOp> {
        self.matrices
            .iter_mut()
            .flat_map(|m| std::iter::once(&mut m.positions).chain(&mut m.normals))
            .collect()
    }

    fn prepare(&mut self) {
        let positions = self.weighted(4, |v, j| self.rest[v].extend(1.0)[j]);
        let normals = self
            .rest_normals
            .as_ref()
            .map(|ns| self.weighted(3, |v, j| ns[v][j]));
        self.matrices = Some(SkinMatrices {
            positions: MatrixOp::new(positions),
            normals: normals.map(MatrixOp::new),
        });
    }
}

accelerated_methods!(SkinnedMesh);

impl SkinnedMesh {
    /// `influences[v]` lists `(bone, weight)` pairs for vertex `v`.
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
            matrices: None,
        })
    }

    /// `M[v, (b, j)] = Σ w_vb · value(v, j)` for `j < width` (`vertices × width·bones`).
    fn weighted(&self, width: usize, value: impl Fn(usize, usize) -> f32) -> Matrix {
        let mut m = Matrix::zeros(self.rest.len(), width * self.bones);
        for (v, inf) in self.influences.iter().enumerate() {
            for &(bone, w) in inf {
                for j in 0..width {
                    m[(v, bone as usize * width + j)] += w * value(v, j);
                }
            }
        }
        m
    }

    /// Vertices in the mesh.
    pub fn vertex_count(&self) -> usize {
        self.rest.len()
    }

    /// Bones in the skeleton.
    pub fn bone_count(&self) -> usize {
        self.bones
    }

    /// Skins one character. `bones[b]` is the skinning matrix of bone `b`
    /// (animated world transform × inverse bind matrix).
    pub fn skin(&self, bones: &[Mat4]) -> Result<Skinned> {
        Ok(self.skin_batch(&[bones])?.remove(0))
    }

    /// Skins many characters in one call: `poses[c]` holds character `c`'s
    /// bone matrices.
    pub fn skin_batch<P: AsRef<[Mat4]> + Sync>(&self, poses: &[P]) -> Result<Vec<Skinned>> {
        self.check(poses)?;
        let Some(m) = &self.matrices else {
            return Ok(self.skin_batch_cpu(poses));
        };
        // Per character, rows i = 0..3 with entries (bone, j) = (M_b − I)[i][j].
        let delta = self.pose_rows(poses, 4, |pose, i, j| {
            pose.col(j)[i] - if i == j { 1.0 } else { 0.0 }
        });
        let offsets = m.positions.apply(&delta)?; // [3·characters, V]
        let normals = match &m.normals {
            Some(n) => Some(n.apply(&self.pose_rows(poses, 3, |pose, i, j| pose.col(j)[i]))?),
            None => None,
        };
        Ok((0..poses.len())
            .map(|c| Skinned {
                positions: Self::columns_as_vec3(&offsets, c)
                    .zip(&self.rest)
                    .map(|(o, r)| *r + o)
                    .collect(),
                normals: normals.as_ref().map_or_else(Vec::new, |n| {
                    Self::columns_as_vec3(n, c)
                        .map(|v| v.normalize_or(Vec3::Y))
                        .collect()
                }),
            })
            .collect())
    }

    /// `3·characters × width·bones` rows: `entry(M_b, i, j)` for each character.
    fn pose_rows<P: AsRef<[Mat4]>>(
        &self,
        poses: &[P],
        width: usize,
        entry: impl Fn(&Mat4, usize, usize) -> f32,
    ) -> Matrix {
        Matrix::from_fn(3 * poses.len(), width * self.bones, |r, col| {
            entry(&poses[r / 3].as_ref()[col / width], r % 3, col % width)
        })
    }

    /// Character `c`'s three rows of a `[3·characters, V]` result, as one Vec3 per vertex.
    fn columns_as_vec3(m: &Matrix, c: usize) -> impl Iterator<Item = Vec3> + '_ {
        let (x, y, z) = (m.row(3 * c), m.row(3 * c + 1), m.row(3 * c + 2));
        x.iter()
            .zip(y)
            .zip(z)
            .map(|((x, y), z)| Vec3::new(*x, *y, *z))
    }

    /// Standard CPU linear blend skinning (reference / fallback).
    pub fn skin_batch_cpu<P: AsRef<[Mat4]> + Sync>(&self, poses: &[P]) -> Vec<Skinned> {
        poses
            .par_iter()
            .map(|pose| {
                let pose = pose.as_ref();
                let blend = |v: usize, f: &dyn Fn(&Mat4) -> Vec3| -> Vec3 {
                    self.influences[v]
                        .iter()
                        .filter(|(_, w)| *w != 0.0)
                        .map(|&(bone, w)| w * f(&pose[bone as usize]))
                        .sum()
                };
                Skinned {
                    positions: (0..self.rest.len())
                        .map(|v| blend(v, &|m| m.transform_point3(self.rest[v])))
                        .collect(),
                    normals: self.rest_normals.as_ref().map_or_else(Vec::new, |ns| {
                        (0..ns.len())
                            .map(|v| {
                                blend(v, &|m| m.transform_vector3(ns[v])).normalize_or(Vec3::Y)
                            })
                            .collect()
                    }),
                }
            })
            .collect()
    }

    fn check<P: AsRef<[Mat4]>>(&self, poses: &[P]) -> Result<()> {
        if poses.is_empty() || poses.iter().any(|p| p.as_ref().len() != self.bones) {
            return Err(Error::Shape(format!(
                "every pose needs one matrix per bone ({})",
                self.bones
            )));
        }
        Ok(())
    }
}
