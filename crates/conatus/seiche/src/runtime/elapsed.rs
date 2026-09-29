// Copyright 2026 Mark Alan Boykin
// SPDX-License-Identifier: MPL-2.0

//! Bounded elapsed-time driving of the inline backend. Hosts supply time;
//! this module neither reads a clock nor schedules another frame.

use std::time::Duration;

use super::{LayoutView, Physics, TICK_DT, should_tick};

/// The nominal 60 Hz interval, rounded to the nearest nanosecond. Each
/// accepted interval still integrates with the existing [`TICK_DT`].
pub const TICK_DURATION: Duration = Duration::from_nanos(16_666_667);

/// Caller-selected limits for one inline advancement. Defaults follow the
/// existing browser practice board's 50 ms catch-up cap. Zero limits are
/// valid: no elapsed contribution or no steps, respectively.
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
}

impl Physics {
    /// Forget fractional elapsed time, for example when the host suspends or
    /// hides a surface. The host must also reset its own last timestamp so the
    /// first resumed call excludes the hidden interval. Does not halt physics,
    /// change positions, or send any command to an independently paced actor.
    pub fn reset_elapsed(&mut self) {
        match self {
            Self::Inline(p) => p.elapsed_remainder = Duration::ZERO,
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
                let accepted = elapsed.min(config.max_elapsed);
                let mut available = p.elapsed_remainder.saturating_add(accepted);
                let mut report = ElapsedStepReport {
                    discarded_elapsed: elapsed - accepted,
                    ..ElapsedStepReport::default()
                };
                while available >= TICK_DURATION
                    && report.steps < config.max_steps
                    && should_tick(&p.sim, p.ticks_remaining, p.dragging, p.halted)
                {
                    p.sim.tick(TICK_DT);
                    if p.ticks_remaining > 0 {
                        p.ticks_remaining -= 1;
                    }
                    report.steps += 1;
                    available -= TICK_DURATION;
                }
                report.settling = should_tick(&p.sim, p.ticks_remaining, p.dragging, p.halted);
                report.carried_elapsed = if report.settling {
                    Duration::from_nanos((available.as_nanos() % TICK_DURATION.as_nanos()) as u64)
                } else {
                    Duration::ZERO
                };
                report.discarded_elapsed = report
                    .discarded_elapsed
                    .saturating_add(available - report.carried_elapsed);
                p.elapsed_remainder = report.carried_elapsed;
                p.generation = p.generation.wrapping_add(1);
                view.apply_snapshot(&p.sim.snapshot(p.generation));
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
