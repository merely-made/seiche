// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! Query refresh (ruling 352), point containment, voxel read-back and shape
//! contacts, ruled 2026-10-09 in the Conatus engine plan's BodyWorld queries.

use serde::{Deserialize, Serialize};

use super::rapier::VoxelCellWalk;
use super::{BodyError, BodyWorld, finite3};
use crate::{ColliderId, SpatialFilter};

/// A half-open box of voxel-grid cells: `min` inclusive, `max` exclusive.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VoxelBox {
    pub min: [i32; 3],
    pub max: [i32; 3],
}

/// The occupied cells of one voxel collider, in canonical order: 8-cell
/// blocks in ascending (x, y, z) block order, then cells by x, y, z within
/// each block. The order depends only on which cells are occupied, never on
/// how the grid was built or edited. Borrowed from the world; nothing copied.
pub struct VoxelCells<'a>(VoxelCellWalk<'a>);

impl Iterator for VoxelCells<'_> {
    type Item = [i32; 3];

    fn next(&mut self) -> Option<[i32; 3]> {
        self.0.next()
    }
}

impl BodyWorld {
    /// Make spawns, teleports, kind changes and voxel edits since the last
    /// step visible to queries without stepping. Only the touched colliders'
    /// poses and broad-phase entries are synced: no tick, no events, and the
    /// next step simulates exactly as it would have without the refresh.
    pub fn refresh_queries(&mut self) {
        for id in std::mem::take(&mut self.query_stale) {
            self.backend.refresh_body(id);
        }
    }

    /// The colliders containing `point`, sorted and deduplicated.
    pub fn colliders_at_point(
        &self,
        point: [f32; 3],
        filter: SpatialFilter,
    ) -> Result<Vec<ColliderId>, BodyError> {
        if !finite3(point) {
            return Err(BodyError::InvalidQuery("point must be finite"));
        }
        if !filter.include_sensors && !filter.include_solids {
            return Ok(Vec::new());
        }
        self.backend.colliders_at_point(point, filter)
    }

    /// Whether a voxel collider's own grid holds `cell`; `false` outside it.
    pub fn voxel_filled(&self, collider: ColliderId, cell: [i32; 3]) -> Result<bool, BodyError> {
        self.collider_slot(collider)?;
        self.backend.voxel_filled(collider, cell)
    }

    /// The occupied cells of a voxel collider's own grid, optionally only
    /// those inside `within`.
    pub fn voxel_cells(
        &self,
        collider: ColliderId,
        within: Option<VoxelBox>,
    ) -> Result<VoxelCells<'_>, BodyError> {
        if within.is_some_and(|bounds| (0..3).any(|axis| bounds.min[axis] > bounds.max[axis])) {
            return Err(BodyError::InvalidQuery(
                "voxel box minimum must not exceed its maximum",
            ));
        }
        self.collider_slot(collider)?;
        self.backend.voxel_cells(collider, within).map(VoxelCells)
    }
}
