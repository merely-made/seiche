// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! Conatus resident kernels, authored in CubeCL.
//!
//! The three dispatches the field tier advances by (repulsion, springs,
//! integration) plus the settle reduction, and [`exclude`], the node
//! exclusion law a rapier host stages here (see `resident::exclusion`),
//! written once here and
//! compiled by CubeCL to whatever the adapter takes. This is the engine
//! composition ruling's authored lane: our kernels are CubeCL, and the
//! buffers they run over are the same CubeCL allocations Burn
//! addresses, so a tensor pass and a kernel pass meet with no bridge.
//!
//! The force laws are the ones Seiche states on the CPU, and
//! `seiche::repulsion_reference` remains the anchor they are checked
//! against.
//!
//! Positions and velocities are padded 3D throughout, per the spatial
//! compute plan: four floats per body, xyz meaningful, w spare, and a
//! 2D canvas is a constrained case of the same layout rather than a
//! second format. The padding is addressed by stride rather than by a
//! vector type, since CubeCL 0.10 exposes no `vec4` for buffer
//! elements; the bytes on the wire are identical either way.
//!
//! Dialect notes, earned by the seed crystal and paid for in bisection:
//! the `cube` macro cannot expand `==` on floats, and a `let mut` local
//! that is later compared needs its type written, because the
//! comparison expansion erases the back-inference arithmetic keeps.

use cubecl::prelude::*;

use super::binning::cell_index;

/// Threads per cube, and the tile width the repulsion pass stages
/// through shared memory.
pub const CUBE_DIM: u32 = 256;
/// Floats per body: padded 3D.
pub const STRIDE: u32 = 4;

/// Softened inverse-square repulsion over every pair, tiled through
/// shared memory.
///
/// The self term contributes nothing because its displacement is zero,
/// so no diagonal mask is needed. Threads past `n` still load and
/// barrier: a cube that diverges at a barrier is undefined behaviour.
#[cube(launch_unchecked)]
pub fn repulse(
    positions: &[f32],
    forces: &mut [f32],
    n: u32,
    repulsion: f32,
    min_distance: f32,
    #[comptime] tile_floats: usize,
) {
    // Shared-memory extent is comptime, as is the stride: both are
    // shapes the compiler needs before it can lay the cube out.
    let mut tile = Shared::<[f32]>::new_slice(tile_floats);
    let stride = STRIDE as usize;
    let width = CUBE_DIM as usize;
    let count_n = n as usize;
    let i = ABSOLUTE_POS;
    let unit = UNIT_POS as usize;
    let last = count_n - 1;
    let mut mine = i;
    if mine > last {
        mine = last;
    }
    let base_i = mine * stride;
    let px = positions[base_i];
    let py = positions[base_i + 1];
    let pz = positions[base_i + 2];
    let softening = min_distance * min_distance;

    let mut fx = 0.0f32;
    let mut fy = 0.0f32;
    let mut fz = 0.0f32;
    let tiles = count_n.div_ceil(width);
    let mut t = 0usize;
    while t < tiles {
        let mut j = t * width + unit;
        if j > last {
            j = last;
        }
        let src = j * stride;
        let dst = unit * stride;
        tile[dst] = positions[src];
        tile[dst + 1] = positions[src + 1];
        tile[dst + 2] = positions[src + 2];
        sync_cube();

        let start = t * width;
        let mut count = width;
        if count_n - start < width {
            count = count_n - start;
        }
        let mut k = 0usize;
        while k < count {
            let other = k * stride;
            let dx = px - tile[other];
            let dy = py - tile[other + 1];
            let dz = pz - tile[other + 2];
            let d2 = dx * dx + dy * dy + dz * dz + softening;
            let inv = repulsion / (d2 * f32::sqrt(d2));
            fx += dx * inv;
            fy += dy * inv;
            fz += dz * inv;
            k += 1;
        }
        sync_cube();
        t += 1;
    }

    if i < count_n {
        let out = i * stride;
        forces[out] = fx;
        forces[out + 1] = fy;
        forces[out + 2] = fz;
        forces[out + 3] = 0.0f32;
    }
}

/// `seiche::NodeExclusion`'s exact law over every pair, tiled through shared
/// memory: inverse square, a hard `min_distance` floor (not softening), and no
/// interaction past `cutoff` (given squared, the comparison the CPU makes).
///
/// The self pair adds nothing because its displacement is zero, as on the CPU.
/// Threads past `n` still load and barrier, as in [`repulse`].
#[cube(launch_unchecked)]
pub fn exclude(
    positions: &[f32],
    forces: &mut [f32],
    n: u32,
    strength: f32,
    cutoff_sq: f32,
    min_distance: f32,
    #[comptime] tile_floats: usize,
) {
    let mut tile = Shared::<[f32]>::new_slice(tile_floats);
    let stride = STRIDE as usize;
    let width = CUBE_DIM as usize;
    let count_n = n as usize;
    let i = ABSOLUTE_POS;
    let unit = UNIT_POS as usize;
    let last = count_n - 1;
    let mut mine = i;
    if mine > last {
        mine = last;
    }
    let base_i = mine * stride;
    let px = positions[base_i];
    let py = positions[base_i + 1];
    let pz = positions[base_i + 2];

    let mut fx = 0.0f32;
    let mut fy = 0.0f32;
    let mut fz = 0.0f32;
    let tiles = count_n.div_ceil(width);
    let mut t = 0usize;
    while t < tiles {
        let mut j = t * width + unit;
        if j > last {
            j = last;
        }
        let src = j * stride;
        let dst = unit * stride;
        tile[dst] = positions[src];
        tile[dst + 1] = positions[src + 1];
        tile[dst + 2] = positions[src + 2];
        sync_cube();

        let start = t * width;
        let mut count = width;
        if count_n - start < width {
            count = count_n - start;
        }
        let mut k = 0usize;
        while k < count {
            let other = k * stride;
            let dx = px - tile[other];
            let dy = py - tile[other + 1];
            let dz = pz - tile[other + 2];
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
            k += 1;
        }
        sync_cube();
        t += 1;
    }

    if i < count_n {
        let out = i * stride;
        forces[out] = fx;
        forces[out + 1] = fy;
        forces[out + 2] = fz;
        forces[out + 3] = 0.0f32;
    }
}

/// [`exclude`]'s law over the cell list `resident::binning` built: each body
/// visits only the three-by-three block of cells around its own. Cells are at
/// least `cutoff` wide, so every pair inside the cutoff is in that block; the
/// cutoff test itself is the same comparison [`exclude`] makes.
///
/// Forces land in the bodies' submitted order, read from `positions`; the
/// neighbours are read from `sorted`, ranged by `starts`.
// The cube dialect has no `saturating_sub`; the bounds are written out.
#[allow(clippy::implicit_saturating_sub)]
#[cube(launch_unchecked)]
pub fn exclude_cells(
    positions: &[f32],
    sorted: &[f32],
    starts: &[u32],
    forces: &mut [f32],
    n: u32,
    origin_x: f32,
    origin_y: f32,
    inv_cell: f32,
    width: u32,
    height: u32,
    strength: f32,
    cutoff_sq: f32,
    min_distance: f32,
) {
    let stride = STRIDE as usize;
    let i = ABSOLUTE_POS;
    if i < n as usize {
        let base = i * stride;
        let px = positions[base];
        let py = positions[base + 1];
        let pz = positions[base + 2];
        let cell = cell_index(px, py, origin_x, origin_y, inv_cell, width, height);
        let cx = cell % width;
        let cy = cell / width;
        let mut x0 = cx;
        if x0 > 0 {
            x0 -= 1;
        }
        let mut y0 = cy;
        if y0 > 0 {
            y0 -= 1;
        }
        let mut x1 = cx + 1;
        if x1 >= width {
            x1 = width - 1;
        }
        let mut y1 = cy + 1;
        if y1 >= height {
            y1 = height - 1;
        }

        let mut fx = 0.0f32;
        let mut fy = 0.0f32;
        let mut fz = 0.0f32;
        let mut row = y0;
        while row <= y1 {
            let mut col = x0;
            while col <= x1 {
                let c = (row * width + col) as usize;
                let mut k = starts[c] as usize;
                let end = starts[c + 1] as usize;
                while k < end {
                    let other = k * stride;
                    let dx = px - sorted[other];
                    let dy = py - sorted[other + 1];
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
                    k += 1;
                }
                col += 1;
            }
            row += 1;
        }
        forces[base] = fx;
        forces[base + 1] = fy;
        forces[base + 2] = fz;
        forces[base + 3] = 0.0f32;
    }
}

/// Spring forces gathered along CSR adjacency.
///
/// Every write is owned by one invocation, so no float atomics are
/// needed (and WGSL has none).
#[cube(launch_unchecked)]
pub fn springs(
    positions: &[f32],
    forces: &mut [f32],
    offsets: &[u32],
    targets: &[u32],
    n: u32,
    spring_k: f32,
    rest_length: f32,
) {
    let stride = STRIDE as usize;
    let i = ABSOLUTE_POS;
    if i < n as usize {
        let base = i * stride;
        let px = positions[base];
        let py = positions[base + 1];
        let pz = positions[base + 2];
        let mut fx = forces[base];
        let mut fy = forces[base + 1];
        let mut fz = forces[base + 2];

        let mut e = offsets[i] as usize;
        let end = offsets[i + 1] as usize;
        while e < end {
            let target = targets[e] as usize * stride;
            let dx = positions[target] - px;
            let dy = positions[target + 1] - py;
            let dz = positions[target + 2] - pz;
            let mut len = f32::sqrt(dx * dx + dy * dy + dz * dz);
            if len < 1.0e-4f32 {
                len = 1.0e-4f32;
            }
            let scale = spring_k * (len - rest_length) / len;
            fx += dx * scale;
            fy += dy * scale;
            fz += dz * scale;
            e += 1;
        }
        forces[base] = fx;
        forces[base + 1] = fy;
        forces[base + 2] = fz;
    }
}

/// Damped symplectic Euler, a weak centering, and the settle reduction.
///
/// Speed is reinterpreted as `u32` for the atomic maximum: the bit
/// pattern of a non-negative float orders the same way the float does,
/// so one atomic max over the whole grid is the entire convergence
/// probe, and the host reads four bytes rather than the cloud.
#[cube(launch_unchecked)]
pub fn integrate(
    positions: &mut [f32],
    velocities: &mut [f32],
    forces: &[f32],
    settle: &mut [Atomic<u32>],
    n: u32,
    dt: f32,
    damping: f32,
    centering: f32,
) {
    let stride = STRIDE as usize;
    let i = ABSOLUTE_POS;
    if i < n as usize {
        let base = i * stride;
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

/// Reset the settle word before a step, so each step's maximum is its
/// own rather than the running maximum of every step so far.
#[cube(launch_unchecked)]
pub fn clear_settle(settle: &mut [Atomic<u32>]) {
    if ABSOLUTE_POS < 1 {
        Atomic::store(&settle[0], 0u32);
    }
}
