// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! The stack's light and environment block (wing ruling 472).
//!
//! The block is plain data the sim's fields produce and both the tracer and
//! this rasteriser read, so terrain and bodies are lit by one sun and one
//! set of point lights. Colours are linear RGB; intensities are linear
//! multipliers, 1.0 a unit-strength light.

use kiss3d::color::Color;
use kiss3d::glamx::{Pose3, Vec3};
use kiss3d::light::Light;
use kiss3d::scene::SceneNode3d;

/// One frame's lighting.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LightBlock {
    /// The sun, or `None` at night or underground.
    pub sun: Option<Sun>,
    pub ambient: Ambient,
    pub points: Vec<PointLight>,
}

/// A directional light from the sim's day and night fields.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sun {
    /// World-space direction the light travels.
    pub direction: [f32; 3],
    pub color: [f32; 3],
    pub intensity: f32,
    pub casts_shadows: bool,
}

/// Light from everywhere, unshadowed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ambient {
    pub color: [f32; 3],
    pub intensity: f32,
}

impl Default for Ambient {
    fn default() -> Self {
        Self {
            color: [1.0; 3],
            intensity: 0.2,
        }
    }
}

/// A light at a point: a torch, a lamp, a glowing body.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PointLight {
    pub position: [f32; 3],
    pub color: [f32; 3],
    pub intensity: f32,
    /// Distance beyond which the light contributes nothing.
    pub radius: f32,
    pub casts_shadows: bool,
}

impl LightBlock {
    /// Whether every value is finite, every intensity and radius
    /// non-negative, and the sun's direction non-zero.
    pub fn is_valid(&self) -> bool {
        let finite = |v: &[f32]| v.iter().all(|x| x.is_finite());
        let sun = self.sun.is_none_or(|s| {
            finite(&s.direction)
                && finite(&s.color)
                && s.intensity >= 0.0
                && s.direction.iter().any(|x| *x != 0.0)
        });
        let ambient = finite(&self.ambient.color) && self.ambient.intensity >= 0.0;
        let points = self.points.iter().all(|p| {
            finite(&p.position)
                && finite(&p.color)
                && p.intensity >= 0.0
                && p.radius > 0.0
                && p.radius.is_finite()
        });
        sun && ambient && points
    }
}

fn color(rgb: [f32; 3]) -> Color {
    Color::new(rgb[0], rgb[1], rgb[2], 1.0)
}

/// Rebuilds `group`'s children as the block's lights.
pub(crate) fn place(block: &LightBlock, group: &mut SceneNode3d) {
    let children = group.data().children().to_vec();
    for mut child in children {
        child.remove();
    }
    if let Some(sun) = block.sun {
        let light = Light::directional(Vec3::from(sun.direction).normalize())
            .with_color(color(sun.color))
            .with_intensity(sun.intensity)
            .with_casts_shadows(sun.casts_shadows);
        group.add_light(light);
    }
    for point in &block.points {
        let light = Light::point(point.radius)
            .with_color(color(point.color))
            .with_intensity(point.intensity)
            .with_casts_shadows(point.casts_shadows);
        group
            .add_light(light)
            .set_pose(Pose3::from_translation(Vec3::from(point.position)));
    }
}

/// The ambient term as kiss3d takes it.
pub(crate) fn ambient(block: &LightBlock) -> (f32, Color) {
    (block.ambient.intensity, color(block.ambient.color))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn torch() -> PointLight {
        PointLight {
            position: [1.0, 2.0, 3.0],
            color: [1.0, 0.8, 0.6],
            intensity: 4.0,
            radius: 6.0,
            casts_shadows: true,
        }
    }

    #[test]
    fn validity_rejects_what_the_shaders_cannot_use() {
        let mut block = LightBlock {
            points: vec![torch()],
            ..Default::default()
        };
        assert!(block.is_valid());
        block.points[0].radius = 0.0;
        assert!(!block.is_valid(), "zero radius");
        block.points[0] = torch();
        block.sun = Some(Sun {
            direction: [0.0; 3],
            color: [1.0; 3],
            intensity: 1.0,
            casts_shadows: true,
        });
        assert!(!block.is_valid(), "zero sun direction");
        block.sun = None;
        block.ambient.intensity = f32::NAN;
        assert!(!block.is_valid(), "NaN ambient");
    }
}
