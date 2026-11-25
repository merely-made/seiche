// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! The graph lane's kernels: a hashed cell grid, the exclusion law over it,
//! weighted springs, pinned integration, and the bounded reads.
//!
//! Every pass takes the slot flags (bit 0 alive, bit 1 pinned): a dead slot
//! is neither binned nor moved and takes a zero force. The law's arithmetic
//! is [`crate::resident::kernels::exclude`]'s, so the two agree.
//!
//! The grid has no bounds. A body's cell is `floor(position / cell)` per
//! axis, hashed to a bucket; a candidate found in a bucket counts only when
//! its own cell is the neighbour cell being visited, so a hash collision
//! costs a skipped candidate and never a false or doubled pair.

use cubecl::prelude::*;

use crate::resident::kernels::STRIDE;

/// The cell coordinate of one axis, clamped to ±1e9 so the cast to `i32` is
/// defined for any finite position. Bodies that far out share a cell and are
/// tested by distance like any others, so the clamp costs no pair.
#[cube]
pub fn cell_coord(value: f32, inv_cell: f32) -> i32 {
    let mut cell = f32::floor(value * inv_cell);
    if cell > 1.0e9f32 {
        cell = 1.0e9f32;
    }
    if cell < -1.0e9f32 {
        cell = -1.0e9f32;
    }
    i32::cast_from(cell)
}

/// The bucket of a cell: two multiplicative hashes mixed, then masked to the
/// table (a power of two, `mask = size - 1`).
#[cube]
pub fn bucket(cx: i32, cy: i32, mask: u32) -> u32 {
    let hx = u32::cast_from(cx) * 73856093u32;
    let hy = u32::cast_from(cy) * 19349663u32;
    let mut h = hx ^ hy;
    h = h ^ (h >> 15u32);
    h = h * 739982445u32;
    h = h ^ (h >> 13u32);
    h & mask
}

/// Device-to-device copy for growth.
#[cube(launch_unchecked)]
pub fn copy_f32(source: &[f32], target: &mut [f32], len: u32) {
    if ABSOLUTE_POS < len as usize {
        target[ABSOLUTE_POS] = source[ABSOLUTE_POS];
    }
}

/// Count the live bodies in each bucket.
#[cube(launch_unchecked)]
pub fn count_buckets(
    positions: &[f32],
    flags: &[u32],
    counts: &mut [Atomic<u32>],
    n: u32,
    inv_cell: f32,
    mask: u32,
) {
    let i = ABSOLUTE_POS;
    if i < n as usize && (flags[i] & 1u32) != 0u32 {
        let base = i * STRIDE as usize;
        let cx = cell_coord(positions[base], inv_cell);
        let cy = cell_coord(positions[base + 1], inv_cell);
        let slot = bucket(cx, cy, mask) as usize;
        Atomic::fetch_add(&counts[slot], 1u32);
    }
}

/// Place each live body's position in the slot an atomic claims in its bucket.
#[cube(launch_unchecked)]
pub fn scatter_buckets(
    positions: &[f32],
    flags: &[u32],
    cursors: &mut [Atomic<u32>],
    sorted: &mut [f32],
    n: u32,
    inv_cell: f32,
    mask: u32,
) {
    let i = ABSOLUTE_POS;
    if i < n as usize && (flags[i] & 1u32) != 0u32 {
        let stride = STRIDE as usize;
        let base = i * stride;
        let cx = cell_coord(positions[base], inv_cell);
        let cy = cell_coord(positions[base + 1], inv_cell);
        let slot = bucket(cx, cy, mask) as usize;
        let out = Atomic::fetch_add(&cursors[slot], 1u32) as usize * stride;
        sorted[out] = positions[base];
        sorted[out + 1] = positions[base + 1];
        sorted[out + 2] = positions[base + 2];
        sorted[out + 3] = positions[base + 3];
    }
}

/// `seiche::NodeExclusion`'s law over the hashed grid: the nine cells around
/// a body, each looked up by bucket and filtered to bodies actually in that
/// cell. Cells are wider than the cutoff, so every pair inside it is found.
#[cube(launch_unchecked)]
pub fn exclude_buckets(
    positions: &[f32],
    sorted: &[f32],
    starts: &[u32],
    flags: &[u32],
    forces: &mut [f32],
    n: u32,
    inv_cell: f32,
    mask: u32,
    strength: f32,
    cutoff_sq: f32,
    min_distance: f32,
) {
    let stride = STRIDE as usize;
    let i = ABSOLUTE_POS;
    if i < n as usize {
        let base = i * stride;
        let mut fx = 0.0f32;
        let mut fy = 0.0f32;
        let mut fz = 0.0f32;
        if (flags[i] & 1u32) != 0u32 {
            let px = positions[base];
            let py = positions[base + 1];
            let pz = positions[base + 2];
            let cx = cell_coord(px, inv_cell);
            let cy = cell_coord(py, inv_cell);
            let mut a = 0u32;
            while a < 3u32 {
                let ncx = cx + i32::cast_from(a) - 1i32;
                let mut b = 0u32;
                while b < 3u32 {
                    let ncy = cy + i32::cast_from(b) - 1i32;
                    let slot = bucket(ncx, ncy, mask) as usize;
                    let mut k = starts[slot] as usize;
                    let end = starts[slot + 1] as usize;
                    while k < end {
                        let other = k * stride;
                        let sx = sorted[other];
                        let sy = sorted[other + 1];
                        if cell_coord(sx, inv_cell) == ncx {
                            if cell_coord(sy, inv_cell) == ncy {
                                let dx = px - sx;
                                let dy = py - sy;
                                let dz = pz - sorted[other + 2];
                                let d2 = dx * dx + dy * dy + dz * dz;
                                if d2 <= cutoff_sq {
                                    let mut dist: f32 = f32::sqrt(d2);
                                    if dist < min_distance {
                                        dist = min_distance;
                                    }
                                    let scale = strength / (dist * dist * dist);
                                    fx += dx * scale;
                                    fy += dy * scale;
                                    fz += dz * scale;
                                }
                            }
                        }
                        k += 1;
                    }
                    b += 1;
                }
                a += 1;
            }
        }
        forces[base] = fx;
        forces[base + 1] = fy;
        forces[base + 2] = fz;
        forces[base + 3] = 0.0f32;
    }
}

/// Weighted springs gathered along CSR rows. An edge carries the generations
/// of both ends as the host knew them; it acts only while both still match,
/// so removing a slot silences its springs with no upload, and a reused
/// index never inherits them.
#[cube(launch_unchecked)]
pub fn springs_weighted(
    positions: &[f32],
    forces: &mut [f32],
    flags: &[u32],
    generations: &[u32],
    offsets: &[u32],
    edge_ids: &[u32],
    edge_weights: &[f32],
    n: u32,
    spring_k: f32,
    rest_length: f32,
) {
    let stride = STRIDE as usize;
    let i = ABSOLUTE_POS;
    if i < n as usize && (flags[i] & 1u32) != 0u32 {
        let base = i * stride;
        let px = positions[base];
        let py = positions[base + 1];
        let pz = positions[base + 2];
        let mut fx = forces[base];
        let mut fy = forces[base + 1];
        let mut fz = forces[base + 2];
        let own = generations[i];

        let mut e = offsets[i] as usize;
        let end = offsets[i + 1] as usize;
        while e < end {
            let target = edge_ids[e * 3] as usize;
            if edge_ids[e * 3 + 1] == generations[target] && edge_ids[e * 3 + 2] == own {
                let at = target * stride;
                let dx = positions[at] - px;
                let dy = positions[at + 1] - py;
                let dz = positions[at + 2] - pz;
                let mut len = f32::sqrt(dx * dx + dy * dy + dz * dz);
                if len < 1.0e-4f32 {
                    len = 1.0e-4f32;
                }
                let scale = spring_k * edge_weights[e] * (len - rest_length) / len;
                fx += dx * scale;
                fy += dy * scale;
                fz += dz * scale;
            }
            e += 1;
        }
        forces[base] = fx;
        forces[base + 1] = fy;
        forces[base + 2] = fz;
    }
}

/// Damped symplectic Euler as [`crate::resident::kernels::integrate`], except
/// that a pinned slot is set to its target with its velocity zeroed, and a
/// dead slot is left alone. Pinned bodies still exerted force this step.
#[cube(launch_unchecked)]
pub fn integrate_pinned(
    positions: &mut [f32],
    velocities: &mut [f32],
    forces: &[f32],
    flags: &[u32],
    pin_targets: &[f32],
    settle: &mut [Atomic<u32>],
    n: u32,
    dt: f32,
    damping: f32,
    centering: f32,
) {
    let stride = STRIDE as usize;
    let i = ABSOLUTE_POS;
    if i < n as usize && (flags[i] & 1u32) != 0u32 {
        let base = i * stride;
        if (flags[i] & 2u32) != 0u32 {
            positions[base] = pin_targets[base];
            positions[base + 1] = pin_targets[base + 1];
            positions[base + 2] = pin_targets[base + 2];
            velocities[base] = 0.0f32;
            velocities[base + 1] = 0.0f32;
            velocities[base + 2] = 0.0f32;
            velocities[base + 3] = 0.0f32;
        } else {
            let px = positions[base];
            let py = positions[base + 1];
            let pz = positions[base + 2];
            let ax = forces[base] - px * centering;
            let ay = forces[base + 1] - py * centering;
            let az = forces[base + 2] - pz * centering;
            let vx = (velocities[base] + ax * dt) * damping;
            let vy = (velocities[base + 1] + ay * dt) * damping;
            let vz = (velocities[base + 2] + az * dt) * damping;
            positions[base] = px + vx * dt;
            positions[base + 1] = py + vy * dt;
            positions[base + 2] = pz + vz * dt;
            velocities[base] = vx;
            velocities[base + 1] = vy;
            velocities[base + 2] = vz;
            velocities[base + 3] = 0.0f32;
            let speed = f32::sqrt(vx * vx + vy * vy + vz * vz);
            Atomic::fetch_max(&settle[0], u32::reinterpret(speed));
        }
    }
}

/// Reset the pick scratch: no candidate yet.
#[cube(launch_unchecked)]
pub fn pick_init(scratch: &mut [Atomic<u32>]) {
    if ABSOLUTE_POS < 2 {
        Atomic::store(&scratch[ABSOLUTE_POS], 4294967295u32);
    }
}

/// Each live slot within `radius_sq` of the point records its squared
/// distance's bits (an ordered key, being non-negative) and joins the
/// minimum; others record the sentinel.
#[cube(launch_unchecked)]
pub fn pick_distance(
    positions: &[f32],
    flags: &[u32],
    distances: &mut [u32],
    scratch: &mut [Atomic<u32>],
    n: u32,
    x: f32,
    y: f32,
    z: f32,
    radius_sq: f32,
) {
    let i = ABSOLUTE_POS;
    if i < n as usize {
        let base = i * STRIDE as usize;
        let mut key = 4294967295u32;
        if (flags[i] & 1u32) != 0u32 {
            let dx = positions[base] - x;
            let dy = positions[base + 1] - y;
            let dz = positions[base + 2] - z;
            let d2 = dx * dx + dy * dy + dz * dz;
            if d2 <= radius_sq {
                key = u32::reinterpret(d2);
                Atomic::fetch_min(&scratch[0], key);
            }
        }
        distances[i] = key;
    }
}

/// Ties go to the lowest index.
#[cube(launch_unchecked)]
pub fn pick_lowest(distances: &[u32], scratch: &mut [Atomic<u32>], n: u32) {
    let i = ABSOLUTE_POS;
    if i < n as usize {
        let best = Atomic::load(&scratch[0]);
        if best != 4294967295u32 && distances[i] == best {
            Atomic::fetch_min(&scratch[1], i as u32);
        }
    }
}

/// The eight bytes the host reads: slot index (all ones for none) and that
/// slot's generation.
#[cube(launch_unchecked)]
pub fn pick_finish(scratch: &[u32], generations: &[u32], out: &mut [u32]) {
    if ABSOLUTE_POS < 1 {
        let index = scratch[1];
        out[0] = index;
        let mut generation = 0u32;
        if index != 4294967295u32 {
            generation = generations[index as usize];
        }
        out[1] = generation;
    }
}

/// Gather the positions of the named slots into a small buffer.
#[cube(launch_unchecked)]
pub fn gather_positions(positions: &[f32], indices: &[u32], out: &mut [f32], count: u32) {
    let k = ABSOLUTE_POS;
    if k < count as usize {
        let from = indices[k] as usize * STRIDE as usize;
        let to = k * STRIDE as usize;
        out[to] = positions[from];
        out[to + 1] = positions[from + 1];
        out[to + 2] = positions[from + 2];
        out[to + 3] = positions[from + 3];
    }
}
