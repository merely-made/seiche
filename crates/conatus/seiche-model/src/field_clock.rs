// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! Accepted simulation time, separate from lagged-result step age (F203).

#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
