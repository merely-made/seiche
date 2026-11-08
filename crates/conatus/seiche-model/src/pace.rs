// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! The pace data of the layout runtime: simulation speed, the per-frame compute
//! budget, what the pace reached, and the elapsed-time step settings. Plain
//! values a host sets and reads; the drivers that spend them stay in `seiche`.

use std::time::Duration;

/// Per-tick timestep handed to the simulation: one 60fps frame. The whole
/// stack integrates at this step, inline or on the actor, so a settle budget
/// in ticks reads directly as seconds.
pub const TICK_DT: f32 = 1.0 / 60.0;

/// Simulated seconds per wall second, in thousandths, clamped to
/// [`Speed::MIN`]..=[`Speed::MAX`], or [`Speed::UNCAPPED`].
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Speed(u32);

impl Speed {
    pub const MIN: Speed = Speed(200);
    pub const MAX: Speed = Speed(50_000);
    pub const REAL_TIME: Speed = Speed(1_000);
    /// As fast as the step budget allows (ruled 2026-10-04, "Target + Max"):
    /// every frame runs ticks until its budget is spent. With no budget
    /// installed it runs at [`Speed::MAX`], so it can never run a living law
    /// away in one frame. Only named, never reached by clamping.
    pub const UNCAPPED: Speed = Speed(u32::MAX);

    pub const fn from_milli(milli: u32) -> Self {
        Speed(if milli < Self::MIN.0 {
            Self::MIN.0
        } else if milli > Self::MAX.0 {
            Self::MAX.0
        } else {
            milli
        })
    }

    /// The nearest thousandth of `factor`; a non-finite factor is real time.
    pub fn from_factor(factor: f32) -> Self {
        if !factor.is_finite() {
            return Self::REAL_TIME;
        }
        Self::from_milli((factor.max(0.0) * 1_000.0).round().min(u32::MAX as f32) as u32)
    }

    pub const fn milli(self) -> u32 {
        self.0
    }

    /// The factor, infinite for [`Speed::UNCAPPED`].
    pub fn factor(self) -> f32 {
        if self == Self::UNCAPPED {
            f32::INFINITY
        } else {
            self.0 as f32 / 1_000.0
        }
    }
}

impl Default for Speed {
    fn default() -> Self {
        Self::REAL_TIME
    }
}

/// A compute budget for one frame's ticks above real time, measured on the
/// host's monotonic clock. The first tick a frame owes always runs; another
/// runs only while the time left covers the forecast tick plus `margin`
/// (ruled 2026-10-04, "Gate keeps a forecast margin"), the host's allowance
/// for a tick reading dearer than the forecast it was admitted on.
#[derive(Clone, Copy, Debug)]
pub struct StepBudget {
    pub per_frame: Duration,
    pub clock: fn() -> Duration,
    pub margin: Duration,
}

/// A [`StepBudget`] as data: the allowance and the forecast margin, without the
/// clock. Whoever runs the step picks the clock: a native actor reads
/// `Instant`, a browser Worker `performance.now`, an inline host the function
/// it gave in its [`StepBudget`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StepBudgetSpec {
    pub per_frame: Duration,
    pub margin: Duration,
}

/// The share of the display's frame period a frame's ticks above real time
/// may spend unless the host asks otherwise (ruled 2026-10-04, "Target + Max,
/// budget as frame share": 50%).
pub const DEFAULT_BUDGET_SHARE: f64 = 0.5;

/// 60 Hz's frame period: what a host that cannot learn its display's rate
/// takes it to be, and the longest a browser's inferred period may be (ruled
/// 2026-10-04, "Known rate, else capped").
pub const FALLBACK_DISPLAY_PERIOD: Duration = Duration::from_nanos(16_666_667);

/// The frame period of a display refreshing at `millihertz`, as winit's
/// `MonitorHandle::refresh_rate_millihertz` reports it; 60 Hz's when the rate
/// is unknown or zero.
pub fn display_period(millihertz: Option<u32>) -> Duration {
    millihertz
        .filter(|&rate| rate > 0)
        .map_or(FALLBACK_DISPLAY_PERIOD, |rate| {
            Duration::from_nanos(1_000_000_000_000 / u64::from(rate))
        })
}

impl StepBudget {
    /// The data half of this budget: the allowance and the margin.
    pub fn spec(&self) -> StepBudgetSpec {
        StepBudgetSpec {
            per_frame: self.per_frame,
            margin: self.margin,
        }
    }

    /// `spec` read on `clock`, the clock of whoever runs the step.
    pub fn from_spec(spec: StepBudgetSpec, clock: fn() -> Duration) -> Self {
        Self {
            per_frame: spec.per_frame,
            clock,
            margin: spec.margin,
        }
    }

    /// `share` of a display's frame `period`, measured on `clock`, the gate
    /// keeping `margin` past the forecast tick.
    pub fn of_period(
        period: Duration,
        share: f64,
        clock: fn() -> Duration,
        margin: Duration,
    ) -> Self {
        Self {
            per_frame: period.mul_f64(share),
            clock,
            margin,
        }
    }

    /// A native host's budget for a display refreshing at `millihertz`
    /// (60 Hz's when unknown): the default share of its period on the
    /// monotonic clock, which reads finely enough to need no margin (ruled
    /// 2026-10-04, "Mere entry point, then turnstone").
    #[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
    pub fn for_display(millihertz: Option<u32>) -> Self {
        Self::of_period(
            display_period(millihertz),
            DEFAULT_BUDGET_SHARE,
            monotonic_clock,
            Duration::ZERO,
        )
    }
}

/// A monotonic clock for native hosts' budgets (not on `wasm32-unknown-unknown`,
/// where the host passes its own, e.g. `performance.now`).
#[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
pub fn monotonic_clock() -> Duration {
    static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    START.get_or_init(std::time::Instant::now).elapsed()
}

/// What the pace reached: ticks run since the backend was built, the
/// effective speed over the last frames while moving (`None` at rest), and
/// whether the budget cut the last frame short.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PaceStats {
    pub ticks: u64,
    pub effective_speed: Option<f32>,
    pub budget_bound: bool,
}

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
    /// Ticks the step budget's gate admitted past the 1x floor, and when the
    /// last of them ended: the time the budget bounds (ruled 2026-10-05,
    /// "Only ticks past the floor"). The floor's own ticks run whatever the
    /// budget, so fast-forward is never slower than 1x.
    pub admitted: u32,
    pub admitted_until: Option<Duration>,
}
