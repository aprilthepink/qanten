//! Tiny software renderer shared by the examples (z-buffered triangles → PNG).
#![allow(dead_code)]

use std::path::Path;

use glam::{Mat4, Vec3, Vec4Swizzles};

pub struct Canvas {
    pub w: usize,
    pub h: usize,
    pub color: Vec<Vec3>,
    depth: Vec<f32>,
}

impl Canvas {
    pub fn new(w: usize, h: usize, bg: impl Fn(f32, f32) -> Vec3) -> Self {
        let color = (0..w * h)
            .map(|k| bg((k % w) as f32 / w as f32, (k / w) as f32 / h as f32))
            .collect();
        Self {
            w,
            h,
            color,
            depth: vec![f32::INFINITY; w * h],
        }
    }

    /// Gouraud-shaded triangle in world space.
    pub fn triangle(&mut self, view_proj: &Mat4, p: [Vec3; 3], c: [Vec3; 3]) {
        self.triangle_with(view_proj, p, |b| c[0] * b.x + c[1] * b.y + c[2] * b.z);
    }

    /// Triangle with a per-pixel shader that receives perspective-correct barycentrics.
    pub fn triangle_with(&mut self, view_proj: &Mat4, p: [Vec3; 3], shade: impl Fn(Vec3) -> Vec3) {
        let mut inv_w = [0.0f32; 3];
        let mut s = [Vec3::ZERO; 3];
        for k in 0..3 {
            let clip = *view_proj * p[k].extend(1.0);
            if clip.w < 0.05 {
                return;
            }
            inv_w[k] = 1.0 / clip.w;
            let ndc = clip.xyz() / clip.w;
            s[k] = Vec3::new(
                (ndc.x * 0.5 + 0.5) * self.w as f32,
                (0.5 - ndc.y * 0.5) * self.h as f32,
                ndc.z,
            );
        }
        let edge =
            |a: Vec3, b: Vec3, x: f32, y: f32| (b.x - a.x) * (y - a.y) - (b.y - a.y) * (x - a.x);
        let area = edge(s[0], s[1], s[2].x, s[2].y);
        if area.abs() < 1e-8 {
            return;
        }
        let x0 = s
            .iter()
            .map(|v| v.x)
            .fold(f32::MAX, f32::min)
            .floor()
            .max(0.0) as usize;
        let x1 = s
            .iter()
            .map(|v| v.x)
            .fold(f32::MIN, f32::max)
            .ceil()
            .min(self.w as f32 - 1.0);
        let y0 = s
            .iter()
            .map(|v| v.y)
            .fold(f32::MAX, f32::min)
            .floor()
            .max(0.0) as usize;
        let y1 = s
            .iter()
            .map(|v| v.y)
            .fold(f32::MIN, f32::max)
            .ceil()
            .min(self.h as f32 - 1.0);
        if x1 < 0.0 || y1 < 0.0 {
            return;
        }
        for y in y0..=y1 as usize {
            for x in x0..=x1 as usize {
                let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
                let w0 = edge(s[1], s[2], px, py) / area;
                let w1 = edge(s[2], s[0], px, py) / area;
                let w2 = 1.0 - w0 - w1;
                if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                    continue;
                }
                let z = w0 * s[0].z + w1 * s[1].z + w2 * s[2].z;
                let i = y * self.w + x;
                if z < self.depth[i] {
                    self.depth[i] = z;
                    let b = Vec3::new(w0 * inv_w[0], w1 * inv_w[1], w2 * inv_w[2]);
                    self.color[i] = shade(b / (b.x + b.y + b.z));
                }
            }
        }
    }

    /// ACES-fitted tonemap + sRGB gamma, written as 8-bit PNG.
    pub fn save(&self, path: impl AsRef<Path>) -> anyhow::Result<()> {
        let aces =
            |x: f32| ((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)).clamp(0.0, 1.0);
        let bytes: Vec<u8> = self
            .color
            .iter()
            .flat_map(|c| [c.x, c.y, c.z])
            .map(|v| (aces(v.max(0.0)).powf(1.0 / 2.2) * 255.0 + 0.5) as u8)
            .collect();
        let file = std::io::BufWriter::new(std::fs::File::create(path)?);
        let mut enc = png::Encoder::new(file, self.w as u32, self.h as u32);
        enc.set_color(png::ColorType::Rgb);
        enc.set_depth(png::BitDepth::Eight);
        enc.write_header()?.write_image_data(&bytes)?;
        Ok(())
    }
}

/// Ray direction through pixel (u, v) in [0,1]² for a camera.
pub fn camera_ray(eye: Vec3, target: Vec3, fov_y: f32, aspect: f32, u: f32, v: f32) -> Vec3 {
    let f = (target - eye).normalize();
    let r = f.cross(Vec3::Y).normalize();
    let up = r.cross(f);
    let t = (fov_y * 0.5).tan();
    (f + r * ((u * 2.0 - 1.0) * t * aspect) + up * ((1.0 - 2.0 * v) * t)).normalize()
}

pub fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

pub fn ms(d: std::time::Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

pub fn max_err(a: &[Vec3], b: &[Vec3]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (*x - *y).abs().max_element())
        .fold(0.0, f32::max)
}

/// Right-handed look-at × OpenGL-style perspective (NDC z in [-1, 1]).
pub fn view_proj(eye: Vec3, target: Vec3, fov_y: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    let f = (target - eye).normalize();
    let s = f.cross(Vec3::Y).normalize();
    let u = s.cross(f);
    let view = Mat4::from_cols(
        glam::Vec4::new(s.x, u.x, -f.x, 0.0),
        glam::Vec4::new(s.y, u.y, -f.y, 0.0),
        glam::Vec4::new(s.z, u.z, -f.z, 0.0),
        glam::Vec4::new(-s.dot(eye), -u.dot(eye), f.dot(eye), 1.0),
    );
    let t = 1.0 / (fov_y * 0.5).tan();
    let proj = Mat4::from_cols(
        glam::Vec4::new(t / aspect, 0.0, 0.0, 0.0),
        glam::Vec4::new(0.0, t, 0.0, 0.0),
        glam::Vec4::new(0.0, 0.0, (far + near) / (near - far), -1.0),
        glam::Vec4::new(0.0, 0.0, 2.0 * far * near / (near - far), 0.0),
    );
    proj * view
}
