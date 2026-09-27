// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! The shader's walk until 2026-09-27, frozen as the instrument's positive
//! control: `brick_dda` as it stood before `a404cd48`, in f32 and in its
//! order. The first crossing on each axis is measured from the start point
//! and each later one is the last plus `1 / |direction|`, so every
//! addition's rounding stays in the walk, and the start voxel is a bare
//! floor of the start point. It shares only the box clip and the voxel read
//! with [`BrickMap::trace`], which never changed.

use super::*;
use crate::trace::brick_ray_box;

fn initial_crossing(position: f32, direction: f32, voxel: i32, start_t: f32) -> f32 {
    if direction > 1e-6 {
        return start_t + ((voxel + 1) as f32 - position) / direction;
    }
    if direction < -1e-6 {
        return start_t + (voxel as f32 - position) / direction;
    }
    1e30
}

/// The old walk of one ray over `map`; `None` for a miss or a spent
/// budget, both of which the old shader drew as a miss.
pub(super) fn accumulated(
    map: &BrickMap,
    far: f32,
    eye: [f32; 3],
    direction: [f32; 3],
) -> Option<BrickHit> {
    let space = BrickTraceSpace::from_map(map);
    let [enter, exit] = brick_ray_box(&space, far, eye, direction);
    if enter > exit || exit < 0.0 {
        return None;
    }
    let start_t = enter.max(0.0) + 0.0001;
    let start = [0, 1, 2].map(|i| eye[i] + direction[i] * start_t);
    let mut voxel = start.map(|v| v.floor() as i32);
    let step = direction.map(|v| if v >= 0.0 { 1 } else { -1 });
    let mut crossing =
        [0, 1, 2].map(|i| initial_crossing(start[i], direction[i], voxel[i], start_t));
    let delta = direction.map(|v| if v.abs() > 1e-6 { 1.0 / v.abs() } else { 1e30 });
    let mut t = start_t;
    let mut normal = [0.0, 1.0, 0.0];
    for _ in 0..1024 {
        let material = brick_material_at(map, voxel);
        if material != 0 {
            return Some(BrickHit {
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
