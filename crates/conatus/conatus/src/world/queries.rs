// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! Query refresh (ruling 352), point containment, voxel read-back and shape
//! contacts, ruled 2026-10-09 in the Conatus engine plan's BodyWorld queries.

use super::BodyWorld;

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
}
