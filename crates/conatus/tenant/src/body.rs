// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! Bodies: rigid palette meshes and glTF.
//!
//! A palette mesh's colours reach the GPU as a palette texture addressed by
//! UV, the shape of glTF's base-colour texture, so a part tree exported as
//! glTF and a mesher's greedy quads draw through one material.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use kiss3d::glamx::{Quat, Vec2, Vec3};
use kiss3d::resource::{GpuMesh3d, Texture};

/// Palette entries per texture row.
const ROW: u32 = 256;

/// A body the tenant draws.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BodyId(pub(crate) u64);

/// Colours as sRGB-encoded RGBA bytes, as a glTF base-colour texture holds them.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Palette {
    pub colors: Vec<[u8; 4]>,
}

/// Non-indexed triangles, each vertex naming its palette entry.
/// Normals are per face, from the winding (counter-clockwise is front).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PaletteMesh {
    pub positions: Vec<[f32; 3]>,
    pub palette_indices: Vec<u32>,
}

/// A body's placement: translation, rotation as a unit quaternion
/// `[x, y, z, w]`, and scale.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pose {
    pub translation: [f32; 3],
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
}

impl Default for Pose {
    fn default() -> Self {
        Self {
            translation: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0; 3],
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum BodyError {
    /// The vertex count is not a multiple of three.
    NotTriangles(usize),
    /// Positions and palette indices differ in length.
    IndexCount { positions: usize, indices: usize },
    /// A palette index past the palette's end.
    PaletteIndex { index: u32, len: usize },
    /// A non-finite position or pose value.
    NotFinite,
    /// The glTF bytes did not load.
    Gltf(String),
    /// No body has this id.
    Unknown(BodyId),
}

impl std::fmt::Display for BodyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for BodyError {}

impl Palette {
    /// Linear RGBA to sRGB bytes.
    pub fn from_linear(colors: &[[f32; 4]]) -> Self {
        Self {
            colors: colors.iter().map(|c| srgb_bytes(*c)).collect(),
        }
    }

    /// The texture coordinate that samples entry `index`'s texel centre.
    pub fn uv(&self, index: u32) -> [f32; 2] {
        let rows = self.rows();
        [
            ((index % ROW) as f32 + 0.5) / ROW.min(self.len_u32()).max(1) as f32,
            ((index / ROW) as f32 + 0.5) / rows as f32,
        ]
    }

    fn len_u32(&self) -> u32 {
        self.colors.len() as u32
    }

    fn rows(&self) -> u32 {
        self.len_u32().div_ceil(ROW).max(1)
    }

    /// The palette as a texture: `ROW` entries a row, rows padded with zeros.
    pub(crate) fn texture(&self) -> Arc<Texture> {
        let width = ROW.min(self.len_u32()).max(1);
        let rows = self.rows();
        let mut bytes = vec![0u8; (width * rows * 4) as usize];
        for (i, color) in self.colors.iter().enumerate() {
            let (x, y) = (i as u32 % ROW, i as u32 / ROW);
            let at = ((y * width + x) * 4) as usize;
            bytes[at..at + 4].copy_from_slice(color);
        }
        Texture::new(
            width,
            rows,
            &bytes,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            wgpu::AddressMode::ClampToEdge,
            wgpu::FilterMode::Nearest,
            false,
            1,
        )
    }
}

impl PaletteMesh {
    /// Triangles coloured per vertex, as a mesher emits them, folded into a
    /// mesh over a palette of their distinct colours (linear RGBA in).
    pub fn from_colored(positions: Vec<[f32; 3]>, colors: &[[f32; 4]]) -> (Self, Palette) {
        let mut palette = Palette::default();
        let mut seen = std::collections::HashMap::new();
        let palette_indices = colors
            .iter()
            .map(|c| {
                let bytes = srgb_bytes(*c);
                *seen.entry(bytes).or_insert_with(|| {
                    palette.colors.push(bytes);
                    palette.colors.len() as u32 - 1
                })
            })
            .collect();
        let mesh = Self {
            positions,
            palette_indices,
        };
        (mesh, palette)
    }

    pub fn validate(&self, palette: &Palette) -> Result<(), BodyError> {
        let (n, m) = (self.positions.len(), self.palette_indices.len());
        if n % 3 != 0 {
            return Err(BodyError::NotTriangles(n));
        }
        if n != m {
            return Err(BodyError::IndexCount {
                positions: n,
                indices: m,
            });
        }
        if let Some(&index) = self
            .palette_indices
            .iter()
            .find(|&&i| i as usize >= palette.colors.len())
        {
            return Err(BodyError::PaletteIndex {
                index,
                len: palette.colors.len(),
            });
        }
        if !self.positions.iter().flatten().all(|x| x.is_finite()) {
            return Err(BodyError::NotFinite);
        }
        Ok(())
    }

    /// The mesh on the GPU: shared-nothing vertices, face normals, palette UVs.
    pub(crate) fn gpu(&self, palette: &Palette) -> Rc<RefCell<GpuMesh3d>> {
        let coords: Vec<Vec3> = self.positions.iter().map(|p| Vec3::from(*p)).collect();
        let faces = (0..coords.len() as u32 / 3)
            .map(|t| [3 * t, 3 * t + 1, 3 * t + 2])
            .collect();
        let normals = coords
            .chunks_exact(3)
            .flat_map(|t| [(t[1] - t[0]).cross(t[2] - t[0]).normalize_or_zero(); 3])
            .collect();
        let uvs = self
            .palette_indices
            .iter()
            .map(|i| Vec2::from(palette.uv(*i)))
            .collect();
        Rc::new(RefCell::new(GpuMesh3d::new(
            coords,
            faces,
            Some(normals),
            Some(uvs),
            false,
        )))
    }
}

impl Pose {
    pub(crate) fn is_finite(&self) -> bool {
        self.translation
            .iter()
            .chain(&self.rotation)
            .chain(&self.scale)
            .all(|x| x.is_finite())
    }

    pub(crate) fn parts(&self) -> (kiss3d::glamx::Pose3, Vec3) {
        let rotation = Quat::from_array(self.rotation).normalize();
        let pose = kiss3d::glamx::Pose3::from_parts(Vec3::from(self.translation), rotation);
        (pose, Vec3::from(self.scale))
    }
}

/// Linear to sRGB-encoded bytes; alpha stays linear.
fn srgb_bytes(c: [f32; 4]) -> [u8; 4] {
    let encode = |x: f32| {
        let x = x.clamp(0.0, 1.0);
        let s = if x <= 0.003_130_8 {
            12.92 * x
        } else {
            1.055 * x.powf(1.0 / 2.4) - 0.055
        };
        (s * 255.0).round() as u8
    };
    let alpha = (c[3].clamp(0.0, 1.0) * 255.0).round() as u8;
    [encode(c[0]), encode(c[1]), encode(c[2]), alpha]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn colored_triangles_fold_into_a_palette() {
        let red = [1.0, 0.0, 0.0, 1.0];
        let grey = [0.214, 0.214, 0.214, 1.0];
        let (mesh, palette) =
            PaletteMesh::from_colored(vec![[0.0; 3]; 6], &[red, red, red, grey, grey, red]);
        assert_eq!(palette.colors, vec![[255, 0, 0, 255], [127, 127, 127, 255]]);
        assert_eq!(mesh.palette_indices, vec![0, 0, 0, 1, 1, 0]);
        assert_eq!(mesh.validate(&palette), Ok(()));
    }

    #[test]
    fn uvs_hit_texel_centres() {
        let palette = Palette {
            colors: vec![[0; 4]; 300],
        };
        assert_eq!(palette.uv(0), [0.5 / 256.0, 0.25]);
        assert_eq!(palette.uv(255), [255.5 / 256.0, 0.25]);
        assert_eq!(palette.uv(256), [0.5 / 256.0, 0.75]);
        let small = Palette {
            colors: vec![[0; 4]; 4],
        };
        assert_eq!(small.uv(3), [3.5 / 4.0, 0.5]);
    }

    #[test]
    fn malformed_meshes_are_refused() {
        let palette = Palette {
            colors: vec![[0; 4]; 2],
        };
        let mesh = |n: usize, idx: Vec<u32>| PaletteMesh {
            positions: vec![[0.0; 3]; n],
            palette_indices: idx,
        };
        assert_eq!(
            mesh(4, vec![0; 4]).validate(&palette),
            Err(BodyError::NotTriangles(4))
        );
        assert_eq!(
            mesh(3, vec![0; 2]).validate(&palette),
            Err(BodyError::IndexCount {
                positions: 3,
                indices: 2
            })
        );
        assert_eq!(
            mesh(3, vec![0, 1, 2]).validate(&palette),
            Err(BodyError::PaletteIndex { index: 2, len: 2 })
        );
    }
}
