// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! The shared shader's walk on the CPU.
//!
//! [`BrickMap::trace`] is `brick_dda` from [`BRICK_DDA_WGSL`] in f32,
//! operation for operation and in the shader's order, reading the map's
//! pointer volume and atlas as `brick_material_at` reads their textures, so
//! a pick lands on the voxel the pixel shows. Rust never fuses a multiply
//! into an add; where a GPU fuses one, the two can round differently, and
//! then only at an exact tie.
//!
//! [`BRICK_DDA_WGSL`]: crate::BRICK_DDA_WGSL

use crate::{BRICK_EDGE, BrickMap, BrickTraceSpace};

/// The voxel a walk stopped in: the shader's `BrickHit`, and which voxel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BrickHit {
    /// The world voxel.
    pub voxel: [i32; 3],
    /// Its material, never zero.
    pub material: u8,
    /// How far along the ray the walk entered it, in units of the ray's
    /// direction.
    pub t: f32,
    /// The face the walk entered it through, pointing back along the ray;
    /// `[0, 1, 0]` for the voxel the walk started in.
    pub normal: [f32; 3],
}

/// Where a [`BrickMap::trace`] walk ended.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BrickTrace {
    /// The first voxel with a material.
    Hit(BrickHit),
    /// Air from where the ray enters the pointer volume until it leaves it
    /// or passes `far`, or a ray that misses the volume.
    Clear,
    /// The shader's 1,024-cell budget ran out first. The shader draws this
    /// as a miss; a pick should not take the unwalked rest for clear air.
    Exhausted,
}

impl BrickMap {
    /// The first voxel with a material along `eye + direction * t`, for `t`
    /// up to `far`, walked as `brick_dda` walks it over this map.
    ///
    /// The ray is taken as the shader takes it: `direction` need not be of
    /// unit length, `t` and `far` are in its units, and nothing is checked.
    /// Each voxel crossing is measured afresh from `eye`, so the rounding
    /// does not grow along the walk and a voxel's next step does not depend
    /// on where the walk entered the pointer volume.
    pub fn trace(&self, eye: [f32; 3], direction: [f32; 3], far: f32) -> BrickTrace {
        let space = BrickTraceSpace::from_map(self);
        let [enter, exit] = brick_ray_box(&space, far, eye, direction);
        if enter > exit || exit < 0.0 {
            return BrickTrace::Clear;
        }
        let start_t = enter.max(0.0) + 0.0001;
        let mut voxel = [0, 1, 2].map(|i| (eye[i] + direction[i] * start_t).floor() as i32);
        let step = direction.map(|v| if v >= 0.0 { 1 } else { -1 });
        let mut crossing = [0, 1, 2].map(|i| brick_crossing(eye[i], direction[i], voxel[i]));
        let mut t = start_t;
        let mut normal = [0.0, 1.0, 0.0];
        for _ in 0..1024 {
            let material = brick_material_at(self, voxel);
            if material != 0 {
                return BrickTrace::Hit(BrickHit {
                    voxel,
                    material,
                    t,
                    normal,
                });
            }
            let axis = if crossing[0] <= crossing[1] && crossing[0] <= crossing[2] {
                0
            } else if crossing[1] <= crossing[2] {
                1
            } else {
                2
            };
            t = crossing[axis];
            voxel[axis] += step[axis];
            crossing[axis] = brick_crossing(eye[axis], direction[axis], voxel[axis]);
            normal = [0.0; 3];
            normal[axis] = -step[axis] as f32;
            if t > exit || t > far {
                return BrickTrace::Clear;
            }
        }
        BrickTrace::Exhausted
    }
}

/// `brick_ray_box`: the ray's `[enter, exit]` through the pointer volume,
/// cut at `far`; `enter > exit` is a miss.
pub(crate) fn brick_ray_box(
    space: &BrickTraceSpace,
    far: f32,
    eye: [f32; 3],
    direction: [f32; 3],
) -> [f32; 2] {
    let low = [0, 1, 2].map(|i| space.world_min[i]);
    let high = [0, 1, 2].map(|i| low[i] + space.pointer_extent[i] as f32 * 8.0);
    let mut enter = 0.0f32;
    let mut exit = far;
    for axis in 0..3 {
        let d = direction[axis];
        if d.abs() < 1e-6 {
            if eye[axis] < low[axis] || eye[axis] >= high[axis] {
                return [1.0, -1.0];
            }
            continue;
        }
        let a = (low[axis] - eye[axis]) / d;
        let b = (high[axis] - eye[axis]) / d;
        enter = enter.max(a.min(b));
        exit = exit.min(a.max(b));
    }
    [enter, exit]
}

/// `brick_crossing`: when the ray leaves `voxel` along one axis, measured
/// from the eye.
fn brick_crossing(origin: f32, direction: f32, voxel: i32) -> f32 {
    if direction > 1e-6 {
        return ((voxel + 1) as f32 - origin) / direction;
    }
    if direction < -1e-6 {
        return (voxel as f32 - origin) / direction;
    }
    1e30
}

/// `brick_material_at`: the pointer volume, then that slot's atlas box,
/// with none of the map's key bookkeeping. Air outside the pointer volume.
pub(crate) fn brick_material_at(map: &BrickMap, at: [i32; 3]) -> u8 {
    let edge = BRICK_EDGE as i32;
    let local = [0, 1, 2].map(|axis| at[axis] - i32::from(map.origin()[axis]) * edge);
    if local.iter().any(|axis| *axis < 0) {
        return 0;
    }
    let slot = map.pointer_at(local.map(|axis| (axis / edge) as u32));
    let Some(index) = slot.and_then(|slot| slot.checked_sub(1)) else {
        return 0;
    };
    let [sx, _, sz] = map.slots();
    let spot = [index % sx, index / (sx * sz), (index / sx) % sz];
    let texel = [0, 1, 2].map(|axis| spot[axis] * BRICK_EDGE + (local[axis] % edge) as u32);
    let [width, height, _] = map.atlas_extent();
    map.atlas()[((texel[2] * height + texel[1]) * width + texel[0]) as usize]
}
