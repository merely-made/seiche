// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

use super::*;
impl Physics {
    /// Clock after the latest accepted tick (latest acknowledged actor snapshot
    /// offloaded). A force/trace reports the clock before its own tick instead.
    pub fn field_clock(&self) -> crate::FieldClock {
        match self {
            Self::Inline(p) => p.sim.field_clock(),
            #[cfg(feature = "actor")]
            Self::Actor(p) => p.field_clock,
        }
    }
    /// Explicit Reset; preserves paused/running state and solver-age counters.
    pub fn reset_field_runtime(&mut self, positions: Vec<(NodeKey, Point2D<f32>)>) {
        match self {
            Self::Inline(p) => {
                p.sim.reset_field_runtime(positions);
                p.pace.forget();
            },
            #[cfg(feature = "actor")]
            Self::Actor(p) => {
                p.field_clock.elapsed = 0.0;
                p.field_clock.epoch = p
                    .field_clock
                    .epoch
                    .checked_add(1)
                    .expect("field clock epoch exhausted");
                p.coupling_trace = None;
                p.coupling_membership = None;
                p.handle
                    .command(PhysicsCommand::ResetFieldRuntime(positions));
                p.barrier();
            },
        }
    }
    /// Explicit owner callback. It runs on the simulation thread, only on a
    /// known outside/non-candidate -> inside transition, never initial matches.
    /// Its returned success/refusal is reported separately from physical writes.
    pub fn set_coupling_entry_handler(&mut self, handler: Option<crate::CouplingEntryHandler>) {
        match self {
            Self::Inline(p) => p.sim.set_coupling_entry_handler(handler),
            #[cfg(feature = "actor")]
            Self::Actor(p) => {
                p.coupling_membership = None;
                p.handle
                    .command(PhysicsCommand::SetCouplingEntryHandler(handler));
                p.barrier();
            },
        }
    }
    /// Historical latest membership frame. Offloaded evidence is received by
    /// normal refresh; reading it never invokes/replays an entry callback.
    pub fn coupling_membership(&self) -> Option<&crate::CouplingMembershipFrame> {
        match self {
            Self::Inline(p) => p.sim.coupling_membership(),
            #[cfg(feature = "actor")]
            Self::Actor(p) => p.coupling_membership.as_ref(),
        }
    }
}
