//! Light-probe volumes baked to spherical harmonics on the NPU.

use bevy::asset::RenderAssetUsages;
use bevy::image::Image;
use bevy::prelude::*;
use qanten::Npu;
use qanten_graphics::sh::{Sh9, ShProjector};
use wgpu_types::{Extent3d, TextureDimension, TextureFormat};

use crate::Convert;

/// A regular grid of probes in a box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProbeVolume {
    /// Corner with the smallest coordinates.
    pub min: Vec3,
    /// Corner with the largest coordinates.
    pub max: Vec3,
    /// Probes along x, y and z (at least 1 each).
    pub resolution: UVec3,
}

impl ProbeVolume {
    /// Number of probes.
    pub fn len(&self) -> usize {
        (self.resolution.x * self.resolution.y * self.resolution.z) as usize
    }

    /// `true` if the volume has no probes.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// World position of probe `(x, y, z)`; probes sit on the box corners and edges.
    pub fn position(&self, p: UVec3) -> Vec3 {
        let step = |i: u32, n: u32| {
            if n > 1 {
                i as f32 / (n - 1) as f32
            } else {
                0.5
            }
        };
        self.min
            + (self.max - self.min)
                * Vec3::new(
                    step(p.x, self.resolution.x),
                    step(p.y, self.resolution.y),
                    step(p.z, self.resolution.z),
                )
    }

    /// Probe coordinates in storage order (x fastest, then y, then z).
    pub fn probes(&self) -> impl Iterator<Item = UVec3> + '_ {
        let r = self.resolution;
        (0..r.z).flat_map(move |z| {
            (0..r.y).flat_map(move |y| (0..r.x).map(move |x| UVec3::new(x, y, z)))
        })
    }
}

/// Bakes probe volumes: samples radiance in many directions at every probe
/// and projects all probes onto SH9 in one NPU call.
#[derive(Resource, Debug)]
pub struct ProbeBaker {
    projector: ShProjector,
}

impl ProbeBaker {
    /// `directions` Fibonacci-sphere samples per probe (a few hundred is
    /// plenty for diffuse lighting), projected on `npu` (or the CPU with `None`).
    pub fn new(directions: usize, npu: Option<&Npu>) -> qanten::Result<Self> {
        let projector = ShProjector::fibonacci(directions);
        let projector = match npu {
            Some(npu) => projector.with_npu(npu)?,
            None => projector,
        };
        Ok(Self { projector })
    }

    /// SH9 of every probe (storage order of [`ProbeVolume::probes`]).
    /// `radiance(position, direction)` is the light arriving at `position`
    /// from `direction`, typically from your own ray tracer or a cube map.
    pub fn bake(
        &self,
        volume: &ProbeVolume,
        radiance: impl Fn(Vec3, Vec3) -> Vec3 + Sync,
    ) -> qanten::Result<Vec<Sh9>> {
        let dirs: Vec<Vec3> = self
            .projector
            .directions()
            .iter()
            .map(|d| d.convert())
            .collect();
        let samples: Vec<Vec<qanten_graphics::glam::Vec3>> = volume
            .probes()
            .map(|p| {
                let pos = volume.position(p);
                dirs.iter().map(|&d| radiance(pos, d).convert()).collect()
            })
            .collect();
        self.projector.project_batch(&samples)
    }

    /// Packs baked probes into a 3-D `Rgba32Float` image for shaders:
    /// `resolution.x × resolution.y × 7·resolution.z` texels, where texels
    /// `(x, y, 7z .. 7z + 7)` hold probe `(x, y, z)`'s 27 coefficients
    /// (`c0.rgb, c1.rgb, …, c8.rgb`, then one zero), four per texel.
    pub fn to_image(volume: &ProbeVolume, probes: &[Sh9]) -> Option<Image> {
        if probes.len() != volume.len() || volume.is_empty() {
            return None;
        }
        let r = volume.resolution;
        let mut texels = vec![[0.0f32; 4]; (r.x * r.y * r.z * 7) as usize];
        for (probe, sh) in volume.probes().zip(probes) {
            let flat: Vec<f32> = sh
                .coeffs
                .iter()
                .flat_map(|c| c.to_array())
                .chain(std::iter::once(0.0))
                .collect();
            for (k, chunk) in flat.chunks(4).enumerate() {
                let z = probe.z * 7 + k as u32;
                let i = ((z * r.y + probe.y) * r.x + probe.x) as usize;
                texels[i][..chunk.len()].copy_from_slice(chunk);
            }
        }
        let bytes: Vec<u8> = texels
            .iter()
            .flatten()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        Some(Image::new(
            Extent3d {
                width: r.x,
                height: r.y,
                depth_or_array_layers: 7 * r.z,
            },
            TextureDimension::D3,
            bytes,
            TextureFormat::Rgba32Float,
            RenderAssetUsages::default(),
        ))
    }
}
