// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! The Rapier side of `BodyWorld`'s refresh, point, voxel and contact queries.

use super::RapierBodyBackend;
use crate::BodyId;

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
}
