// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! How large one host's card lets a capacity-fixed atlas grow.

use crate::{ATLAS_SLOTS_X, ATLAS_SLOTS_Z, BRICK_EDGE, MAX_ATLAS_SLOTS_Y};

/// Slots in one atlas row, a 16 by 16 plane.
const SLOTS_PER_ROW: u32 = ATLAS_SLOTS_X * ATLAS_SLOTS_Z;
/// Bytes in one atlas row: 128 KiB.
const ROW_BYTES: u64 = SLOTS_PER_ROW as u64 * BRICK_EDGE.pow(3) as u64;

/// What one host's GPU and memory budget allow a capacity-fixed atlas.
///
/// Plain numbers, so this crate stays GPU-free. A wgpu host fills
/// `max_texture_dimension_3d` from `device.limits()`, the limits its device
/// enforces rather than its adapter's, and `max_atlas_bytes` from its own
/// budget, for which 8 MiB is the recommended default. The atlas grows in
/// whole rows of 16 by 16 slots, 128 KiB each. The texture edge bounds its
/// height and every pointer volume axis, and the atlas stays under 4 GiB,
/// since its texel offsets are 32-bit here and in consumers' uploads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AtlasLimits {
    /// The largest 3D texture edge the device allows, in texels.
    pub max_texture_dimension_3d: u32,
    /// The most bytes the host lets the atlas occupy.
    pub max_atlas_bytes: u64,
}

impl AtlasLimits {
    /// The historical cap, 8 rows in 1 MiB for 2,047 bricks, behind
    /// [`MAX_BRICKS`](crate::MAX_BRICKS) and
    /// [`BrickMap::with_capacity`](crate::BrickMap::with_capacity). Its
    /// 2,048-texel edge is wgpu's default limit.
    pub const DEFAULT: Self = Self {
        max_texture_dimension_3d: 2048,
        max_atlas_bytes: MAX_ATLAS_SLOTS_Y as u64 * ROW_BYTES,
    };

    /// The most bricks one atlas may hold under these limits.
    pub const fn max_bricks(&self) -> usize {
        bricks_in(self.max_rows())
    }

    /// The most whole atlas rows, whichever of the edge and the budget
    /// binds first.
    pub(crate) const fn max_rows(&self) -> u32 {
        let edge = self.max_texture_dimension_3d;
        if edge < ATLAS_SLOTS_X * BRICK_EDGE || edge < ATLAS_SLOTS_Z * BRICK_EDGE {
            return 0;
        }
        let budget = if self.max_atlas_bytes < u32::MAX as u64 {
            self.max_atlas_bytes
        } else {
            u32::MAX as u64
        };
        let by_budget = (budget / ROW_BYTES) as u32;
        let by_edge = edge / BRICK_EDGE;
        if by_budget < by_edge {
            by_budget
        } else {
            by_edge
        }
    }

    /// The whole rows holding `bricks`, if these limits allow them.
    pub(crate) fn rows_for(&self, bricks: usize) -> Option<u32> {
        let spots = (bricks as u64).saturating_add(1);
        let rows = spots.div_ceil(u64::from(SLOTS_PER_ROW));
        (rows <= u64::from(self.max_rows())).then_some(rows as u32)
    }
}

impl Default for AtlasLimits {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// How many bricks `rows` whole atlas rows hold: one slot spot always
/// stays unused, as in every map.
pub(crate) const fn bricks_in(rows: u32) -> usize {
    (rows as usize)
        .saturating_mul(SLOTS_PER_ROW as usize)
        .saturating_sub(1)
}
