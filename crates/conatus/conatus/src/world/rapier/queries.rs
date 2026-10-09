// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! The Rapier side of `BodyWorld`'s refresh, point, voxel and contact queries.

use rapier3d::parry::shape::{VoxelData, Voxels};

use super::{RapierBodyBackend, ivector, vector};
use crate::world::{BodyError, VoxelBox};
use crate::{BodyId, ColliderId, SpatialFilter};

/// Edge of the blocks `VoxelCellWalk` orders by. It must be a multiple of
/// parry's 3D voxel chunk edge (8, private to parry), so each block lies in
/// one chunk and parry yields its cells in x, y, z order; 8 is also T2's
/// brick grain (ruling 331). The canonical-order tests catch a parry change.
const BLOCK: i32 = 8;

impl RapierBodyBackend {
    /// Sync one body's colliders into the query structures. Rapier's own
    /// `detect_collisions` would do this but drops the island bookkeeping of
    /// bodies inserted since the last step, which then never simulate
    /// (measured 2026-10-09), so only the cached pose and the broad-phase
    /// entry are written; the next step still sees every change it owes.
    pub(in crate::world) fn refresh_body(&mut self, id: BodyId) {
        let Some(entry) = self.bodies.get(&id) else {
            return;
        };
        let params = self.physics.integration_parameters;
        let body_pose = *self.physics.bodies[entry.body].position();
        for &handle in &entry.colliders {
            let collider = &self.physics.colliders[handle];
            let current = collider
                .position_wrt_parent()
                .map_or(*collider.position(), |local| body_pose * *local);
            if *collider.position() != current {
                self.physics.colliders[handle].set_position(current);
            }
            let aabb = self.physics.colliders[handle]
                .compute_broad_phase_aabb(&params, &self.physics.bodies);
            self.physics.broad_phase.set_aabb(&params, handle, aabb);
        }
    }

    pub(in crate::world) fn colliders_at_point(
        &self,
        point: [f32; 3],
        filter: SpatialFilter,
    ) -> Result<Vec<ColliderId>, BodyError> {
        let filter = self.query_filter(filter)?;
        let mut hits: Vec<_> = self
            .physics
            .intersect_point(vector(point), filter)
            .filter_map(|(handle, _)| self.collider_id(handle))
            .collect();
        hits.sort_unstable();
        hits.dedup();
        Ok(hits)
    }

    pub(in crate::world) fn voxel_filled(
        &self,
        collider: ColliderId,
        cell: [i32; 3],
    ) -> Result<bool, BodyError> {
        Ok(self
            .voxels(collider)?
            .voxel_state(ivector(cell))
            .is_some_and(|state| !state.is_empty()))
    }

    pub(in crate::world) fn voxel_cells(
        &self,
        collider: ColliderId,
        within: Option<VoxelBox>,
    ) -> Result<VoxelCellWalk<'_>, BodyError> {
        let voxels = self.voxels(collider)?;
        let [mins, maxs] = voxels.domain();
        let mut min = [mins.x, mins.y, mins.z];
        let mut max = [maxs.x, maxs.y, maxs.z];
        if let Some(bounds) = within {
            for axis in 0..3 {
                min[axis] = min[axis].max(bounds.min[axis]);
                max[axis] = max[axis].min(bounds.max[axis]);
            }
        }
        let pending = if (0..3).all(|axis| min[axis] < max[axis]) {
            vec![(min, max)]
        } else {
            Vec::new()
        };
        Ok(VoxelCellWalk {
            voxels,
            pending,
            current: None,
        })
    }

    fn voxels(&self, collider: ColliderId) -> Result<&Voxels, BodyError> {
        let handle = self.collider_handle(collider)?;
        self.physics.colliders[handle]
            .shape()
            .as_voxels()
            .ok_or(BodyError::NotVoxelCollider(collider))
    }
}

/// Walks a cell box by halving it along x, then y, then z down to single
/// blocks, lower half first, skipping halves that hold no voxel. Blocks thus
/// come in ascending (x, y, z) order without sorting or copying any cells.
pub(in crate::world) struct VoxelCellWalk<'a> {
    voxels: &'a Voxels,
    pending: Vec<([i32; 3], [i32; 3])>,
    current: Option<Box<dyn Iterator<Item = VoxelData> + 'a>>,
}

impl Iterator for VoxelCellWalk<'_> {
    type Item = [i32; 3];

    fn next(&mut self) -> Option<[i32; 3]> {
        loop {
            if let Some(cells) = &mut self.current {
                if let Some(voxel) = cells.next() {
                    let cell = voxel.grid_coords;
                    return Some([cell.x, cell.y, cell.z]);
                }
                self.current = None;
            }
            let (min, max) = self.pending.pop()?;
            let mut range = self.voxels.voxels_in_range(ivector(min), ivector(max));
            let Some((axis, middle)) = block_split(min, max) else {
                self.current = Some(Box::new(range));
                continue;
            };
            if range.next().is_none() {
                continue;
            }
            let mut upper_min = min;
            upper_min[axis] = middle;
            let mut lower_max = max;
            lower_max[axis] = middle;
            self.pending.push((upper_min, max));
            self.pending.push((min, lower_max));
        }
    }
}

/// The first axis on which the box spans more than one block, and the block
/// boundary splitting it in half; `None` for a box inside one block.
fn block_split(min: [i32; 3], max: [i32; 3]) -> Option<(usize, i32)> {
    (0..3).find_map(|axis| {
        let first = min[axis].div_euclid(BLOCK);
        let last = (max[axis] - 1).div_euclid(BLOCK);
        (last > first).then(|| (axis, (first + (last - first + 1) / 2) * BLOCK))
    })
}
