// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! Bounded elapsed-time driving of the inline backend. Hosts supply time;
//! this module neither reads a clock nor schedules another frame.

use std::time::Duration;

pub use seiche_model::pace::{ElapsedStepConfig, ElapsedStepReport, TICK_DURATION};

use super::speed::{self, Driver, TICK_UNITS};
use super::{LayoutView, Physics, should_tick};

impl Physics {
    /// Forget fractional elapsed time, for example when the host suspends or
    /// hides a surface. The host must also reset its own last timestamp so the
    /// first resumed call excludes the hidden interval. Does not halt physics,
    /// change positions, or send any command to an independently paced actor.
    /// A deterministic driver's fraction (a slow speed's next tick) stays.
    pub fn reset_elapsed(&mut self) {
        match self {
            Self::Inline(p) => {
                if p.pace.driver == Some(Driver::Elapsed) {
                    p.pace.forget();
                }
            },
            #[cfg(feature = "actor")]
            Self::Actor(_) => {},
        }
    }

    /// Advance inline physics using bounded elapsed time, then publish one
    /// snapshot. Excess whole-step debt is discarded, not queued for later
    /// catch-up bursts. Halted/idle intervals do not accrue simulation time.
    ///
    /// An offloaded backend only drains its latest accepted snapshot; calling
    /// this method never adds ticks or changes the actor's pacing. Existing
    /// [`Self::advance_frame`] remains the deterministic one-step entry point.
    pub fn advance_elapsed(
        &mut self,
        view: &mut LayoutView,
        elapsed: Duration,
        config: ElapsedStepConfig,
    ) -> ElapsedStepReport {
        match self {
            Self::Inline(p) => {
                // Owed time is simulated nanoseconds times a thousand: wall
                // time times the speed in thousandths. Reports are wall time.
                p.pace.drive(Driver::Elapsed);
                let milli = u64::from(p.pace.milli());
                let accepted = elapsed.min(config.max_elapsed);
                p.pace.owe(speed::nanos(accepted));
                let owed_steps = p.pace.owed_steps();
                let cap = p.pace.scaled_cap(config.max_steps);
                // What real time would run in this call: the budget's floor.
                let floor = (speed::nanos(accepted) / speed::TICK_NS)
                    .min(u64::from(config.max_steps)) as u32;
                let clock = p.pace.budget.map(|budget| budget.clock);
                let stepped = speed::step_owed(
                    &mut p.sim,
                    &mut p.pace,
                    &mut p.ticks_remaining,
                    p.dragging,
                    p.halted,
                    cap,
                    floor,
                    clock,
                );
                let settling = should_tick(&p.sim, p.ticks_remaining, p.dragging, p.halted);
                let carried = if settling {
                    p.pace.owed % TICK_UNITS
                } else {
                    0
                };
                let dropped = p.pace.owed - carried;
                speed::settle_meter(p, speed::nanos(elapsed), stepped.steps, settling);
                let report = ElapsedStepReport {
                    steps: stepped.steps,
                    discarded_elapsed: (elapsed - accepted)
                        .saturating_add(Duration::from_nanos(dropped / milli)),
                    carried_elapsed: Duration::from_nanos(carried / milli),
                    settling,
                    owed_steps,
                    budget_bound: stepped.budget_bound,
                    compute: stepped.compute,
                    admitted: stepped.admitted,
                    admitted_until: stepped.admitted_until,
                };
                speed::fold(p, view, settling);
                report
            },
            #[cfg(feature = "actor")]
            Self::Actor(_) => ElapsedStepReport {
                settling: self.advance_frame(view),
                ..ElapsedStepReport::default()
            },
        }
    }
}

#[cfg(test)]
mod tests;
