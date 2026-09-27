// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

// Product-neutral fragment-compatible voxel traversal. Pointer zero is air;
// nonzero values identify one dense 8-cubed material slot in the atlas.

struct BrickTraceSpace {
    world_min: vec4<f32>,
    pointer_extent: vec4<u32>,
    atlas_slots: vec4<u32>,
};

@group(0) @binding(0) var brick_pointers: texture_3d<u32>;
@group(0) @binding(1) var brick_atlas: texture_3d<u32>;

struct BrickHit {
    material: u32,
    t: f32,
    normal: vec3<f32>,
    found: bool,
};

fn brick_material_at(space: BrickTraceSpace, voxel: vec3<i32>) -> u32 {
    let local = voxel - vec3<i32>(space.world_min.xyz);
    if (any(local < vec3<i32>(0))) {
        return 0u;
    }
    let cell = local / 8;
    let limits = vec3<i32>(space.pointer_extent.xyz);
    if (any(cell < vec3<i32>(0)) || any(cell >= limits)) {
        return 0u;
    }
    let slot = textureLoad(brick_pointers, cell, 0).x;
    if (slot == 0u) {
        return 0u;
    }
    let index = slot - 1u;
    let sx = space.atlas_slots.x;
    let sz = space.atlas_slots.z;
    let slot_x = index % sx;
    let slot_z = (index / sx) % sz;
    let slot_y = index / (sx * sz);
    let within = local - cell * 8;
    let atlas_at = vec3<i32>(
        i32(slot_x * 8u) + within.x,
        i32(slot_y * 8u) + within.y,
        i32(slot_z * 8u) + within.z,
    );
    return textureLoad(brick_atlas, atlas_at, 0).x;
}

fn brick_ray_box(
    space: BrickTraceSpace,
    far: f32,
    eye: vec3<f32>,
    direction: vec3<f32>,
) -> vec2<f32> {
    let low = space.world_min.xyz;
    let high = low + vec3<f32>(space.pointer_extent.xyz) * 8.0;
    var enter = 0.0;
    var exit = far;
    for (var axis = 0; axis < 3; axis = axis + 1) {
        let d = direction[axis];
        if (abs(d) < 1e-6) {
            if (eye[axis] < low[axis] || eye[axis] >= high[axis]) {
                return vec2(1.0, -1.0);
            }
            continue;
        }
        let a = (low[axis] - eye[axis]) / d;
        let b = (high[axis] - eye[axis]) / d;
        enter = max(enter, min(a, b));
        exit = min(exit, max(a, b));
    }
    return vec2(enter, exit);
}

// When the ray leaves `voxel` along one axis, measured from the eye. Each
// crossing is taken afresh rather than added to the one before, so its error
// is one subtraction's and one division's however long the walk, and the
// walk steps the same way from a voxel wherever it began.
fn brick_crossing(origin: f32, direction: f32, voxel: i32) -> f32 {
    if (direction > 1e-6) {
        return (f32(voxel + 1) - origin) / direction;
    }
    if (direction < -1e-6) {
        return (f32(voxel) - origin) / direction;
    }
    return 1e30;
}

fn brick_dda(
    space: BrickTraceSpace,
    far: f32,
    eye: vec3<f32>,
    direction: vec3<f32>,
) -> BrickHit {
    let interval = brick_ray_box(space, far, eye, direction);
    if (interval.x > interval.y || interval.y < 0.0) {
        return BrickHit(0u, far, vec3(0.0, 1.0, 0.0), false);
    }
    let start_t = max(interval.x, 0.0) + 0.0001;
    var voxel = vec3<i32>(floor(eye + direction * start_t));
    let step = vec3<i32>(
        select(-1, 1, direction.x >= 0.0),
        select(-1, 1, direction.y >= 0.0),
        select(-1, 1, direction.z >= 0.0),
    );
    var crossing = vec3(
        brick_crossing(eye.x, direction.x, voxel.x),
        brick_crossing(eye.y, direction.y, voxel.y),
        brick_crossing(eye.z, direction.z, voxel.z),
    );
    var t = start_t;
    var normal = vec3(0.0, 1.0, 0.0);
    for (var count = 0; count < 1024; count = count + 1) {
        let material = brick_material_at(space, voxel);
        if (material != 0u) {
            return BrickHit(material, t, normal, true);
        }
        if (crossing.x <= crossing.y && crossing.x <= crossing.z) {
            t = crossing.x;
            voxel.x = voxel.x + step.x;
            crossing.x = brick_crossing(eye.x, direction.x, voxel.x);
            normal = vec3(-f32(step.x), 0.0, 0.0);
        } else if (crossing.y <= crossing.z) {
            t = crossing.y;
            voxel.y = voxel.y + step.y;
            crossing.y = brick_crossing(eye.y, direction.y, voxel.y);
            normal = vec3(0.0, -f32(step.y), 0.0);
        } else {
            t = crossing.z;
            voxel.z = voxel.z + step.z;
            crossing.z = brick_crossing(eye.z, direction.z, voxel.z);
            normal = vec3(0.0, 0.0, -f32(step.z));
        }
        if (t > interval.y || t > far) {
            break;
        }
    }
    return BrickHit(0u, far, vec3(0.0, 1.0, 0.0), false);
}
