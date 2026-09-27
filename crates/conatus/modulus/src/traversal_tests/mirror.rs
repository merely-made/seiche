// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! `brick_dda.wgsl` on the CPU in f32, operation for operation and in the
//! shader's order, so a test can walk the shader's rays without a GPU. Rust
//! never fuses a multiply into an add, so where a GPU fuses one this mirror
//! and the shader can round differently at a near-tie.

use super::*;

/// What `brick_dda` returns, plus the voxel it stopped in.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Hit {
    pub(super) voxel: [i32; 3],
    pub(super) material: u8,
    pub(super) t: f32,
    pub(super) normal: [f32; 3],
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

/// `brick_ray_box`: the ray's `[enter, exit]` through the pointer volume,
/// cut at `far`; `enter > exit` is a miss.
pub(super) fn brick_ray_box(
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

/// `brick_initial_crossing`: the first boundary crossing on one axis,
/// measured from the start point and offset by its distance.
fn brick_initial_crossing(position: f32, direction: f32, voxel: i32, start_t: f32) -> f32 {
    if direction > 1e-6 {
        return start_t + ((voxel + 1) as f32 - position) / direction;
    }
    if direction < -1e-6 {
        return start_t + (voxel as f32 - position) / direction;
    }
    1e30
}

/// `brick_dda`, over whatever `material` reads for a voxel.
pub(super) fn brick_dda(
    space: &BrickTraceSpace,
    far: f32,
    eye: [f32; 3],
    direction: [f32; 3],
    material: impl Fn([i32; 3]) -> u8,
) -> Option<Hit> {
    let [enter, exit] = brick_ray_box(space, far, eye, direction);
    if enter > exit || exit < 0.0 {
        return None;
    }
    let start_t = enter.max(0.0) + 0.0001;
    let start = [0, 1, 2].map(|i| eye[i] + direction[i] * start_t);
    let mut voxel = start.map(|v| v.floor() as i32);
    let step = direction.map(|v| if v >= 0.0 { 1 } else { -1 });
    let mut crossing =
        [0, 1, 2].map(|i| brick_initial_crossing(start[i], direction[i], voxel[i], start_t));
    let delta = direction.map(|v| if v.abs() > 1e-6 { 1.0 / v.abs() } else { 1e30 });
    let mut t = start_t;
    let mut normal = [0.0, 1.0, 0.0];
    for _ in 0..1024 {
        let found = material(voxel);
        if found != 0 {
            return Some(Hit {
                voxel,
                material: found,
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
        crossing[axis] += delta[axis];
        voxel[axis] += step[axis];
        normal = [0.0; 3];
        normal[axis] = -step[axis] as f32;
        if t > exit || t > far {
            break;
        }
    }
    None
}
