// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! Accepted simulation time, separate from lagged-result step age (F203).
use crate::{NodeKey, Simulation};
use euclid::default::Point2D;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FieldClock {
    /// Sum of accepted tick intervals, before the tick when in a force context.
    pub elapsed: f64,
    /// Reset generation; does not rewind the solver's monotonic step counter.
    pub epoch: u64,
    pub paused: bool,
}
impl FieldClock {
    /// Numen's AST uses f32. Accumulation uses f64; traces retain the actual
    /// f32 sample time as well as this accumulated clock state.
    pub fn sample_time(self) -> f32 {
        self.elapsed as f32
    }
}
impl Simulation {
    pub fn set_field_clock_paused(&mut self, paused: bool) {
        self.field_clock.paused = paused;
    }
    pub fn set_coupling_entry_handler(&mut self, handler: Option<crate::CouplingEntryHandler>) {
        self.coupling_entry_handler = handler;
        self.coupling_membership_frame = None;
    }
    pub fn coupling_membership(&self) -> Option<&crate::CouplingMembershipFrame> {
        self.coupling_membership_frame.as_ref()
    }

    pub fn field_clock(&self) -> FieldClock {
        self.field_clock
    }

    /// Explicit Reset: restore supplied positions, rewind field time and start
    /// membership reporting afresh without entry actions. Ordinary seeding does
    /// not rewind this clock. No solver-age or binding-generation counter resets.
    pub fn reset_field_runtime(
        &mut self,
        positions: impl IntoIterator<Item = (NodeKey, Point2D<f32>)>,
    ) {
        self.seed_positions(positions);
        self.field_clock.elapsed = 0.0;
        self.field_clock.epoch = self
            .field_clock
            .epoch
            .checked_add(1)
            .expect("field clock epoch exhausted");
        self.coupling_membership.reset();
        self.coupling_membership_frame = None;
        self.coupling_trace = None;
    }
}

// A bounded AST/reference walk for tick demand. Repeated/cyclic references are
// visited once. This does not validate or execute a cyclic field definition.
pub(crate) fn animated(field: &numen::FieldDefinition, registry: &numen::FieldRegistry) -> bool {
    use numen::{FieldDefinition, FieldId, ScalarField, VectorField};
    use std::collections::HashSet;
    fn scalar(
        f: &ScalarField,
        r: &numen::FieldRegistry,
        seen: &mut HashSet<(FieldId, bool)>,
    ) -> bool {
        use ScalarField::*;
        match f {
            Time => true,
            Gaussian { center, .. } | Disk { center, .. } => vector(center, r, seen),
            Linear { normal, .. } => vector(normal, r, seen),
            Add(a, b) | Mul(a, b) => scalar(a, r, seen) || scalar(b, r, seen),
            Scale(a, _) | Negate(a) => scalar(a, r, seen),
            Dot(a, b) => vector(a, r, seen) || vector(b, r, seen),
            Sample(id) => {
                seen.insert((*id, false))
                    && match r.get(*id) {
                        Some(FieldDefinition::Scalar(f)) => scalar(f, r, seen),
                        _ => false,
                    }
            },
            Const(_) | CoordX | CoordY => false,
        }
    }
    fn vector(
        f: &VectorField,
        r: &numen::FieldRegistry,
        seen: &mut HashSet<(FieldId, bool)>,
    ) -> bool {
        use VectorField::*;
        match f {
            Gradient(f) => scalar(f, r, seen),
            Perp(f) | ScaleConst(f, _) => vector(f, r, seen),
            Add(a, b) => vector(a, r, seen) || vector(b, r, seen),
            Scale(v, s) => vector(v, r, seen) || scalar(s, r, seen),
            Sample(id) => {
                seen.insert((*id, true))
                    && match r.get(*id) {
                        Some(FieldDefinition::Vector(f)) => vector(f, r, seen),
                        _ => false,
                    }
            },
            ConstVec { .. } | Coord => false,
        }
    }
    match field {
        FieldDefinition::Scalar(f) => scalar(f, registry, &mut HashSet::new()),
        FieldDefinition::Vector(f) => vector(f, registry, &mut HashSet::new()),
    }
}
