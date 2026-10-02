// SPDX-License-Identifier: Apache-2.0
//
// Ported by hand to CubeCL from Nexus (https://github.com/dimforge/nexus,
// commit 1cfbd76), by Sébastien Crozet / Dimforge, licensed under the Apache
// License, Version 2.0 (`LICENSE-APACHE` in this directory). Sources:
// `src_rbd_shaders/utils/prefix_sum.rs` and `src_rbd/utils/prefix_sum.rs`
// (the scan kernels and their staging), and `src_mpm_shaders/grid/sort.rs`
// (the count / cursor / finalize pattern of the particle sort).
//
// Changes from upstream: rewritten in the CubeCL dialect; the sparse block
// hash map, slab buckets and neighbour "extras" of the MPM sort are replaced
// by a dense 2D grid with one bucket per cell; the finalize pass writes the
// body's padded position rather than its id; batching is dropped.

//! Cell binning: bodies sorted into a dense grid so a pass can visit only
//! neighbouring cells. Count per cell, exclusive scan into cell starts,
//! copy the starts into insertion cursors, then scatter each body into the
//! slot an atomic claims.
//!
//! The scan is the work-efficient (Blelloch) up-sweep / down-sweep over
//! workgroups of [`WORKGROUP_SIZE`], with each workgroup's total written to an
//! auxiliary level that is scanned the same way and added back down. It is
//! the variant whose result has a leading zero, so a counts array with one
//! trailing zero scans into cell starts whose last entry is the total.

use cubecl::client::ComputeClient;
use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::wgpu::WgpuRuntime;

/// Threads per workgroup for every binning pass, and the scan's block width.
pub const WORKGROUP_SIZE: u32 = 256;
/// `log2(WORKGROUP_SIZE)`: the sweep's iteration count.
const SWEEP_STEPS: u32 = 8;
/// Floats per body: padded 3D, as the resident lane lays positions out.
const STRIDE: usize = 4;

/// The dense grid the bodies bin into. Cells are square, `1 / inv_cell`
/// wide, `width` by `height` of them from `origin`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Grid {
    pub origin: [f32; 2],
    pub inv_cell: f32,
    pub width: u32,
    pub height: u32,
}

impl Grid {
    pub fn cells(&self) -> usize {
        self.width as usize * self.height as usize
    }
}

/// The cell a point falls in (row-major), clamped onto the grid.
#[cube]
pub fn cell_index(
    x: f32,
    y: f32,
    origin_x: f32,
    origin_y: f32,
    inv_cell: f32,
    width: u32,
    height: u32,
) -> u32 {
    let mut fx = f32::floor((x - origin_x) * inv_cell);
    let mut fy = f32::floor((y - origin_y) * inv_cell);
    if fx < 0.0f32 {
        fx = 0.0f32;
    }
    if fy < 0.0f32 {
        fy = 0.0f32;
    }
    let mut cx = u32::cast_from(fx);
    let mut cy = u32::cast_from(fy);
    if cx >= width {
        cx = width - 1;
    }
    if cy >= height {
        cy = height - 1;
    }
    cy * width + cx
}

/// Zero a counts array before a count pass.
#[cube(launch_unchecked)]
pub fn clear(data: &mut [u32], len: u32) {
    if ABSOLUTE_POS < len as usize {
        data[ABSOLUTE_POS] = 0u32;
    }
}

/// Count the bodies in each cell (upstream: `gpu_update_block_particle_count`).
#[cube(launch_unchecked)]
pub fn count(
    positions: &[f32],
    counts: &mut [Atomic<u32>],
    n: u32,
    origin_x: f32,
    origin_y: f32,
    inv_cell: f32,
    width: u32,
    height: u32,
) {
    let i = ABSOLUTE_POS;
    if i < n as usize {
        let base = i * STRIDE;
        let cell = cell_index(
            positions[base],
            positions[base + 1],
            origin_x,
            origin_y,
            inv_cell,
            width,
            height,
        );
        let index = cell as usize;
        Atomic::fetch_add(&counts[index], 1u32);
    }
}

/// Exclusive prefix sum of one workgroup's segment, its total written to
/// `aux` (upstream: `gpu_prefix_sum_sweep`). Every thread runs the full
/// power-of-two sweep, zero-padded past `len`, so no barrier sits in
/// non-uniform control flow.
#[cube(launch_unchecked)]
pub fn scan_sweep(data: &mut [u32], aux: &mut [u32], len: u32, #[comptime] width: usize) {
    let mut workspace = Shared::<[u32]>::new_slice(width);
    let block = CUBE_POS as usize;
    let tid = UNIT_POS as usize;
    let data_len = len as usize;
    let element = tid + block * width;

    if element < data_len {
        workspace[tid] = data[element];
    } else {
        workspace[tid] = 0u32;
    }

    // Up-sweep: a tree of partial sums; the root ends at `width - 1`. The
    // loop variables start from literals: one derived from the comptime
    // `width` would expand as a constant CubeCL cannot mutate.
    let mut d = 0usize;
    d += width / 2;
    let mut offset = 1usize;
    let mut step = 0u32;
    while step < SWEEP_STEPS {
        sync_cube();
        if tid < d {
            let ia = tid * 2 * offset + offset - 1;
            let ib = (tid * 2 + 1) * offset + offset - 1;
            workspace[ib] = workspace[ia] + workspace[ib];
        }
        d /= 2;
        offset *= 2;
        step += 1;
    }

    sync_cube();
    if tid == 0 {
        aux[block] = workspace[width - 1];
        workspace[width - 1] = 0u32;
    }

    // Down-sweep: the tree becomes the exclusive scan.
    let mut d = 1usize;
    let mut offset = 0usize;
    offset += width / 2;
    let mut step = 0u32;
    while step < SWEEP_STEPS {
        sync_cube();
        if tid < d {
            let ia = tid * 2 * offset + offset - 1;
            let ib = (tid * 2 + 1) * offset + offset - 1;
            let a = workspace[ia];
            let b = workspace[ib];
            workspace[ia] = b;
            workspace[ib] = a + b;
        }
        d *= 2;
        offset /= 2;
        step += 1;
    }

    sync_cube();
    if element < data_len {
        data[element] = workspace[tid];
    }
}

/// Add each workgroup's scanned offset into its segment
/// (upstream: `gpu_add_data_grp`).
#[cube(launch_unchecked)]
pub fn scan_add(data: &mut [u32], aux: &[u32], len: u32) {
    let element = ABSOLUTE_POS;
    let block = CUBE_POS as usize;
    if element < len as usize {
        data[element] += aux[block];
    }
}

/// Copy the cell starts into insertion cursors
/// (upstream: `gpu_copy_scan_values_to_first_particles`).
#[cube(launch_unchecked)]
pub fn cursors(starts: &[u32], cursors: &mut [u32], cells: u32) {
    if ABSOLUTE_POS < cells as usize {
        cursors[ABSOLUTE_POS] = starts[ABSOLUTE_POS];
    }
}

/// Place each body in the slot an atomic claims in its cell's range
/// (upstream: `gpu_finalize_particles_sort`).
#[cube(launch_unchecked)]
pub fn scatter(
    positions: &[f32],
    cursors: &mut [Atomic<u32>],
    sorted: &mut [f32],
    n: u32,
    origin_x: f32,
    origin_y: f32,
    inv_cell: f32,
    width: u32,
    height: u32,
) {
    let i = ABSOLUTE_POS;
    if i < n as usize {
        let base = i * STRIDE;
        let cell = cell_index(
            positions[base],
            positions[base + 1],
            origin_x,
            origin_y,
            inv_cell,
            width,
            height,
        );
        let index = cell as usize;
        let slot = Atomic::fetch_add(&cursors[index], 1u32) as usize;
        let out = slot * STRIDE;
        sorted[out] = positions[base];
        sorted[out + 1] = positions[base + 1];
        sorted[out + 2] = positions[base + 2];
        sorted[out + 3] = positions[base + 3];
    }
}

/// Exclusive scan of `data` in place (`len` elements), staged as upstream's
/// `PrefixSumWorkspace`: one auxiliary level per factor of
/// [`WORKGROUP_SIZE`], the last of length one.
pub fn exclusive_scan(client: &ComputeClient<WgpuRuntime>, data: &Handle, len: usize) {
    let width = WORKGROUP_SIZE as usize;
    let mut ngroups = vec![len.div_ceil(width).max(1)];
    let mut stages = Vec::new();
    let mut per_level = ngroups[0];
    while per_level != 1 {
        stages.push((client.empty(per_level * 4), per_level));
        per_level = per_level.div_ceil(width);
        ngroups.push(per_level);
    }
    stages.push((client.empty(4), 1));

    let dim = CubeDim::new_1d(WORKGROUP_SIZE);
    let sweep =
        |target: &Handle, target_len: usize, aux: &Handle, aux_len: usize, groups: usize| unsafe {
            scan_sweep::launch_unchecked::<WgpuRuntime>(
                client,
                CubeCount::Static(groups as u32, 1, 1),
                dim,
                BufferArg::from_raw_parts(target.clone(), target_len),
                BufferArg::from_raw_parts(aux.clone(), aux_len),
                target_len as u32,
                width,
            );
        };
    let add = |target: &Handle, target_len: usize, aux: &Handle, aux_len: usize, groups: usize| unsafe {
        scan_add::launch_unchecked::<WgpuRuntime>(
            client,
            CubeCount::Static(groups as u32, 1, 1),
            dim,
            BufferArg::from_raw_parts(target.clone(), target_len),
            BufferArg::from_raw_parts(aux.clone(), aux_len),
            target_len as u32,
        );
    };

    sweep(data, len, &stages[0].0, stages[0].1, ngroups[0]);
    for i in 0..stages.len() - 1 {
        sweep(
            &stages[i].0,
            stages[i].1,
            &stages[i + 1].0,
            stages[i + 1].1,
            ngroups[i + 1],
        );
    }
    if stages.len() > 2 {
        for i in (0..stages.len() - 2).rev() {
            add(
                &stages[i].0,
                stages[i].1,
                &stages[i + 1].0,
                stages[i + 1].1,
                ngroups[i + 1],
            );
        }
    }
    if stages.len() > 1 {
        add(data, len, &stages[0].0, stages[0].1, ngroups[0]);
    }
}

/// Bin `n` padded positions into `grid`. Returns the cell starts
/// (`cells + 1` entries, the last the total) and the positions sorted by
/// cell, both on `client`.
pub fn bin(
    client: &ComputeClient<WgpuRuntime>,
    positions: &Handle,
    n: usize,
    grid: Grid,
) -> (Handle, Handle) {
    let cells = grid.cells();
    let groups = |count: usize| {
        CubeCount::Static(count.div_ceil(WORKGROUP_SIZE as usize).max(1) as u32, 1, 1)
    };
    let dim = CubeDim::new_1d(WORKGROUP_SIZE);
    // One trailing zero past the last cell, so the scan's last entry is
    // the total and every cell's range is `starts[c]..starts[c + 1]`.
    let starts = client.empty((cells + 1) * 4);
    let cursor = client.empty(cells * 4);
    let sorted = client.empty(n * STRIDE * 4);
    unsafe {
        clear::launch_unchecked::<WgpuRuntime>(
            client,
            groups(cells + 1),
            dim,
            BufferArg::from_raw_parts(starts.clone(), cells + 1),
            (cells + 1) as u32,
        );
        count::launch_unchecked::<WgpuRuntime>(
            client,
            groups(n),
            dim,
            BufferArg::from_raw_parts(positions.clone(), n * STRIDE),
            BufferArg::from_raw_parts(starts.clone(), cells + 1),
            n as u32,
            grid.origin[0],
            grid.origin[1],
            grid.inv_cell,
            grid.width,
            grid.height,
        );
    }
    exclusive_scan(client, &starts, cells + 1);
    unsafe {
        cursors::launch_unchecked::<WgpuRuntime>(
            client,
            groups(cells),
            dim,
            BufferArg::from_raw_parts(starts.clone(), cells + 1),
            BufferArg::from_raw_parts(cursor.clone(), cells),
            cells as u32,
        );
        scatter::launch_unchecked::<WgpuRuntime>(
            client,
            groups(n),
            dim,
            BufferArg::from_raw_parts(positions.clone(), n * STRIDE),
            BufferArg::from_raw_parts(cursor, cells),
            BufferArg::from_raw_parts(sorted.clone(), n * STRIDE),
            n as u32,
            grid.origin[0],
            grid.origin[1],
            grid.inv_cell,
            grid.width,
            grid.height,
        );
    }
    (starts, sorted)
}
