// Copyright 2026 Mark Alan Boykin
// SPDX-License-Identifier: MPL-2.0

//! Bounded elapsed-time driving of the inline backend. Hosts supply time;
//! this module neither reads a clock nor schedules another frame.

use std::time::Duration;

use super::speed::{self, Driver, TICK_UNITS};
use super::{LayoutView, Physics, should_tick};

/// The nominal 60 Hz interval, rounded to the nearest nanosecond. Each
/// accepted interval still integrates with the existing [`TICK_DT`].
pub const TICK_DURATION: Duration = Duration::from_nanos(16_666_667);

/// Caller-selected limits for one inline advancement. Defaults follow the
/// existing browser practice board's 50 ms catch-up cap. Zero limits are
/// valid: no elapsed contribution or no steps, respectively. `max_steps` is
/// given for real time and scales with the speed (rounded up).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ElapsedStepConfig {
    pub max_elapsed: Duration,
    pub max_steps: u32,
}

impl Default for ElapsedStepConfig {
    fn default() -> Self {
        Self {
            max_elapsed: Duration::from_millis(50),
            max_steps: 3,
        }
    }
}

/// Work performed by this host call. Actor-owned simulation reports zero
/// steps and durations: its independent actor owns time and integration.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ElapsedStepReport {
    pub steps: u32,
    /// Time rejected by either cap, or while idle/halted. Includes a previous
    /// carried fraction when settling ends. Accounting saturates at Duration::MAX.
    pub discarded_elapsed: Duration,
    /// Only a fraction smaller than one tick can carry to the next call.
    pub carried_elapsed: Duration,
    pub settling: bool,
    /// Whole ticks the speed owed this call, before any cap.
    pub owed_steps: u32,
    /// The step budget stopped this call short of what was owed.
    pub budget_bound: bool,
    /// Time spent stepping, when a budget's clock is installed.
    pub compute: Option<Duration>,
}

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
                let milli = u64::from(p.pace.speed.milli());
                let accepted = elapsed.min(config.max_elapsed);
                p.pace.owe(speed::nanos(accepted));
                let owed_steps = p.pace.owed_steps();
                let cap = p.pace.scaled_cap(config.max_steps);
                let clock = p.pace.budget.map(|budget| budget.clock);
                let stepped = speed::step_owed(
                    &mut p.sim,
                    &mut p.pace,
                    &mut p.ticks_remaining,
                    p.dragging,
                    p.halted,
                    cap,
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
