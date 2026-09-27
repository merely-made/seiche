// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! The exact walk the f32 traversal is measured against: the same f32 eye,
//! direction, box and far cut taken as exact values, every crossing computed
//! afresh from the eye in f64, and the shader's own rules kept (its 1e-4
//! start offset, its 1e-6 parallel threshold, its x-then-y-then-z tie order
//! and its 1,024-cell budget), so that only rounding separates the two.
//!
//! Each walk also records its closest call: of every choice an f32 walk of
//! this ray has to make, how near it came to going the other way, in f32
//! ulps of the numbers that choice rounds. A disagreement on a ray whose
//! closest call is a few ulps is a tie at f32's resolution; one on a ray
//! whose closest call is wide is a fault.

use super::*;

/// Where the exact walk stopped, and how close it came to stopping elsewhere.
pub(super) struct Exact {
    pub(super) hit: Option<Hit>,
    /// The closest call, in f32 ulps; infinite for a walk that chose nothing.
    pub(super) margin: f64,
    /// Whether the 1,024-cell budget ran out first.
    pub(super) exhausted: bool,
}

/// One f32 ulp at `x`'s magnitude.
pub(super) fn ulp32(x: f64) -> f64 {
    let f = (x.abs() as f32).max(f32::MIN_POSITIVE);
    if !f.is_finite() {
        return f64::INFINITY;
    }
    f64::from(f32::from_bits(f.to_bits() + 1) - f)
}

/// How far apart `a` and `b` are, in f32 ulps of the larger.
fn apart(a: f64, b: f64) -> f64 {
    (a - b).abs() / ulp32(a.abs().max(b.abs()))
}

/// The same ray walked exactly; see the module doc.
pub(super) fn exact(
    space: &BrickTraceSpace,
    far: f32,
    eye: [f32; 3],
    direction: [f32; 3],
    material: impl Fn([i32; 3]) -> u8,
) -> Exact {
    let o = eye.map(f64::from);
    let d = direction.map(f64::from);
    let low = [0, 1, 2].map(|i| f64::from(space.world_min[i]));
    let high = [0, 1, 2].map(|i| low[i] + f64::from(space.pointer_extent[i]) * 8.0);
    let far = f64::from(far);
    let miss = |margin| Exact {
        hit: None,
        margin,
        exhausted: false,
    };
    let (mut enter, mut exit, mut entry) = (0.0f64, far, None);
    for axis in 0..3 {
        if d[axis].abs() < 1e-6 {
            if o[axis] < low[axis] || o[axis] >= high[axis] {
                return miss(f64::INFINITY);
            }
            continue;
        }
        let a = (low[axis] - o[axis]) / d[axis];
        let b = (high[axis] - o[axis]) / d[axis];
        if a.min(b) > enter {
            (enter, entry) = (a.min(b), Some(axis));
        }
        exit = exit.min(a.max(b));
    }
    let mut margin = apart(enter, exit);
    if enter > exit || exit < 0.0 {
        return miss(margin.min(apart(exit, 0.0)));
    }
    let start_t = enter.max(0.0) + 1e-4;
    let start = [0, 1, 2].map(|i| o[i] + d[i] * start_t);
    let mut voxel = start.map(|v| v.floor() as i32);
    // The start voxel is a floor of rounded coordinates, except along the
    // axis the ray entered by: a start rounded back onto that face is
    // stepped off it at once, so only the other axes can choose wrongly.
    for i in (0..3).filter(|i| Some(*i) != entry) {
        let scale = o[i].abs().max(start[i].abs()).max((d[i] * start_t).abs());
        margin = margin.min((start[i] - start[i].round()).abs() / ulp32(scale));
    }
    let step = d.map(|v| if v >= 0.0 { 1 } else { -1 });
    let mut t = start_t;
    let mut normal = [0.0, 1.0, 0.0];
    for _ in 0..1024 {
        let found = material(voxel);
        if found != 0 {
            let hit = Hit {
                voxel,
                material: found,
                t: t as f32,
                normal,
            };
            return Exact {
                hit: Some(hit),
                margin,
                exhausted: false,
            };
        }
        let crossing = [0, 1, 2].map(|i| {
            if d[i] > 1e-6 {
                (f64::from(voxel[i] + 1) - o[i]) / d[i]
            } else if d[i] < -1e-6 {
                (f64::from(voxel[i]) - o[i]) / d[i]
            } else {
                f64::INFINITY
            }
        });
        let axis = if crossing[0] <= crossing[1] && crossing[0] <= crossing[2] {
            0
        } else if crossing[1] <= crossing[2] {
            1
        } else {
            2
        };
        for other in (0..3).filter(|other| *other != axis && crossing[*other].is_finite()) {
            margin = margin.min(apart(crossing[axis], crossing[other]));
        }
        t = crossing[axis];
        voxel[axis] += step[axis];
        normal = [0.0; 3];
        normal[axis] = -step[axis] as f32;
        margin = margin.min(apart(t, exit)).min(apart(t, far));
        if t > exit || t > far {
            return miss(margin);
        }
    }
    Exact {
        hit: None,
        margin,
        exhausted: true,
    }
}
