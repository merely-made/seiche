// Copyright 2026 Mark Alan Boykin
// SPDX-License-Identifier: MPL-2.0

//! The simulation-speed dial: simulated seconds per wall second, 0.2x to 50x,
//! at the one fixed [`TICK_DT`] (physics catalog plan, ruled 2026-10-03:
//! "Ticks per frame, fixed dt"). A speed changes how many ticks a frame runs,
//! never the step, so the trajectory is the same at every speed. Fast-forward
//! runs more ticks a frame up to a compute budget and reports the speed it
//! reached; slow motion steps less often and draws between the last two ticks.
//!
//! Owed time is integer: wall nanoseconds times the speed in thousandths, so
//! 0.2x is exactly one tick every five frame-equivalents. Seiche reads no
//! clock; a budget carries the host's.

use std::collections::{HashMap, VecDeque};
use std::time::Duration;

use euclid::default::Point2D;

use super::{InlinePhysics, LayoutView, TICK_DT, TICK_DURATION, should_tick};
use crate::{LayoutSnapshot, NodeKey, Simulation};

/// One tick in nanoseconds, the integer [`TICK_DURATION`].
pub(super) const TICK_NS: u64 = 16_666_667;
/// One tick of owed simulated time, in wall-nanosecond × speed-thousandth units.
pub(super) const TICK_UNITS: u64 = TICK_NS * 1_000;
/// Frames the effective-speed meter averages over.
const METER_FRAMES: usize = 32;

/// Simulated seconds per wall second, in thousandths, clamped to
/// [`Speed::MIN`]..=[`Speed::MAX`], or [`Speed::UNCAPPED`].
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
pub struct PaceStats {
    pub ticks: u64,
    pub effective_speed: Option<f32>,
    pub budget_bound: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Driver {
    Frame,
    Elapsed,
}

/// Effective speed: simulated over wall time across the last frames.
#[derive(Default)]
pub(super) struct Meter {
    frames: VecDeque<(u64, u32)>,
}

impl Meter {
    pub(super) fn record(&mut self, wall_ns: u64, steps: u32) {
        if self.frames.len() == METER_FRAMES {
            self.frames.pop_front();
        }
        self.frames.push_back((wall_ns, steps));
    }

    pub(super) fn clear(&mut self) {
        self.frames.clear();
    }

    pub(super) fn speed(&self) -> Option<f32> {
        let (wall, steps) = self.frames.iter().fold((0u64, 0u64), |(w, s), &(fw, fs)| {
            (w + fw, s + u64::from(fs))
        });
        (wall > 0).then(|| (steps as f64 * TICK_NS as f64 / wall as f64) as f32)
    }
}

/// The inline backend's pacing state.
pub(super) struct Pace {
    pub(super) speed: Speed,
    /// Owed simulated time below one tick between calls, in [`TICK_UNITS`].
    pub(super) owed: u64,
    pub(super) driver: Option<Driver>,
    pub(super) budget: Option<StepBudget>,
    /// The costliest recent tick: the budget's forecast for the next.
    forecast: Duration,
    /// Positions before the last tick, kept below real time to draw between.
    pub(super) previous: Option<HashMap<NodeKey, Point2D<f32>>>,
    pub(super) meter: Meter,
    pub(super) ticks: u64,
    pub(super) budget_bound: bool,
}

impl Default for Pace {
    fn default() -> Self {
        Self {
            speed: Speed::REAL_TIME,
            owed: 0,
            driver: None,
            budget: None,
            forecast: Duration::ZERO,
            previous: None,
            meter: Meter::default(),
            ticks: 0,
            budget_bound: false,
        }
    }
}

impl Pace {
    /// The actor's pace, carried over from the inline backend it replaces.
    #[cfg(feature = "actor")]
    pub(super) fn for_actor(speed: Speed, ticks: u64, per_frame: Option<Duration>) -> Self {
        Self {
            speed,
            ticks,
            budget: Some(actor_budget(per_frame)),
            ..Self::default()
        }
    }

    /// Forget owed time and the tick drawn from (a seed, a halt, a resume).
    pub(super) fn forget(&mut self) {
        self.owed = 0;
        self.previous = None;
    }

    /// Switching drivers drops the other driver's fraction.
    pub(super) fn drive(&mut self, driver: Driver) {
        if self.driver != Some(driver) {
            self.owed = 0;
            self.driver = Some(driver);
        }
    }

    /// Owe `wall_ns` of wall time at the current speed.
    /// The speed's thousandths as stepped: [`Speed::UNCAPPED`] without a
    /// budget runs at [`Speed::MAX`].
    pub(super) fn milli(&self) -> u32 {
        if self.speed == Speed::UNCAPPED && self.budget.is_none() {
            Speed::MAX.milli()
        } else {
            self.speed.milli()
        }
    }

    pub(super) fn owe(&mut self, wall_ns: u64) {
        let units = u128::from(wall_ns) * u128::from(self.milli());
        self.owed = self
            .owed
            .saturating_add(units.min(u128::from(u64::MAX)) as u64);
    }

    pub(super) fn owed_steps(&self) -> u32 {
        (self.owed / TICK_UNITS).min(u64::from(u32::MAX)) as u32
    }

    /// A step cap given for real time, scaled to this speed (rounded up).
    pub(super) fn scaled_cap(&self, cap: u32) -> u32 {
        let scaled = (u64::from(cap) * u64::from(self.milli())).div_ceil(1_000);
        scaled.min(u64::from(u32::MAX)) as u32
    }

    pub(super) fn stats(&self) -> PaceStats {
        PaceStats {
            ticks: self.ticks,
            effective_speed: self.meter.speed(),
            budget_bound: self.budget_bound,
        }
    }
}

/// One call's stepping.
#[derive(Default)]
pub(super) struct Stepped {
    pub(super) steps: u32,
    pub(super) budget_bound: bool,
    pub(super) compute: Option<Duration>,
}

/// Run up to `cap` of the ticks owed, under the budget above real time, and
/// keep the positions before the last tick below it. The budget never stops
/// a call short of `floor`, the ticks real time would run in it, so
/// fast-forward is never slower than 1x. `forecast` decays an eighth a tick
/// so one slow tick does not hold the budget down for good.
#[allow(clippy::too_many_arguments)]
pub(super) fn step_owed(
    sim: &mut Simulation,
    pace: &mut Pace,
    ticks_remaining: &mut u32,
    dragging: bool,
    halted: bool,
    cap: u32,
    floor: u32,
    clock: Option<fn() -> Duration>,
) -> Stepped {
    let mut out = Stepped::default();
    let budget = pace.budget.filter(|_| pace.speed > Speed::REAL_TIME);
    let start = clock.map(|now| now());
    // Below real time the simulation leads by one tick from the start of a
    // motion, so what is drawn between the last two ticks is the speed times
    // wall time from the first frame, not a tick behind it.
    if pace.speed < Speed::REAL_TIME
        && pace.previous.is_none()
        && pace.owed < TICK_UNITS
        && should_tick(sim, *ticks_remaining, dragging, halted)
    {
        pace.owed += TICK_UNITS;
    }
    while pace.owed >= TICK_UNITS
        && out.steps < cap
        && should_tick(sim, *ticks_remaining, dragging, halted)
    {
        if let (Some(budget), Some(start)) = (budget, start)
            && out.steps >= floor.max(1)
            && (budget.clock)().saturating_sub(start) + pace.forecast + budget.margin
                > budget.per_frame
        {
            out.budget_bound = true;
            break;
        }
        if pace.speed < Speed::REAL_TIME {
            pace.previous = Some(sim.positions().collect());
        }
        let before = budget.map(|budget| (budget.clock)());
        sim.tick(TICK_DT);
        if let (Some(budget), Some(before)) = (budget, before) {
            let cost = (budget.clock)().saturating_sub(before);
            pace.forecast = cost.max(pace.forecast - pace.forecast / 8);
        }
        *ticks_remaining = ticks_remaining.saturating_sub(1);
        pace.ticks += 1;
        pace.owed -= TICK_UNITS;
        out.steps += 1;
    }
    out.compute = clock
        .zip(start)
        .map(|(now, start)| now().saturating_sub(start));
    pace.budget_bound = out.budget_bound;
    out
}

/// Fold the simulation into `view`, drawn as [`draw`] says.
pub(super) fn fold(p: &mut InlinePhysics, view: &mut LayoutView, settling: bool) {
    p.generation = p.generation.wrapping_add(1);
    let mut snapshot = p.sim.snapshot(p.generation);
    draw(&mut p.pace, &mut snapshot, settling);
    view.apply_snapshot(&snapshot);
}

/// Below real time and while moving, draw each node between its position
/// before the last tick and its position now, by the fraction of the next
/// tick already owed. Draw only: the simulation is untouched.
pub(super) fn draw(pace: &mut Pace, snapshot: &mut LayoutSnapshot, settling: bool) {
    if !settling || pace.speed >= Speed::REAL_TIME {
        pace.previous = None;
        return;
    }
    let Some(previous) = &pace.previous else {
        return;
    };
    let alpha = pace.owed as f32 / TICK_UNITS as f32;
    for (node, position) in &mut snapshot.positions {
        if let Some(from) = previous.get(node) {
            *position = from.lerp(*position, alpha);
        }
    }
}

/// The actor's interval: owe one [`TICK_DURATION`], run what it pays for under
/// the budget, and meter the `wall_ns` the last interval took.
#[cfg(feature = "actor")]
pub(super) fn actor_interval(
    sim: &mut Simulation,
    pace: &mut Pace,
    ticks_remaining: &mut u32,
    dragging: bool,
    halted: bool,
    wall_ns: u64,
) {
    pace.drive(Driver::Frame);
    pace.owe(TICK_NS);
    let clock = pace.budget.map(|budget| budget.clock);
    let stepped = step_owed(
        sim,
        pace,
        ticks_remaining,
        dragging,
        halted,
        u32::MAX,
        1,
        clock,
    );
    let settling = should_tick(sim, *ticks_remaining, dragging, halted);
    if settling {
        pace.owed %= TICK_UNITS;
        pace.meter.record(wall_ns, stepped.steps);
    } else {
        pace.owed = 0;
        pace.meter.clear();
    }
}

/// The actor's budget: `per_frame`, or its whole interval, on its own clock,
/// which reads `Instant`, fine enough to need no margin.
#[cfg(feature = "actor")]
pub(super) fn actor_budget(per_frame: Option<Duration>) -> StepBudget {
    fn clock() -> Duration {
        static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
        START.get_or_init(std::time::Instant::now).elapsed()
    }
    StepBudget {
        per_frame: per_frame.unwrap_or(TICK_DURATION),
        clock,
        margin: Duration::ZERO,
    }
}

/// One deterministic frame-equivalent ([`TICK_DURATION`]) at the speed: the
/// [`super::Physics::advance_frame`] driver.
pub(super) fn advance_frame(p: &mut InlinePhysics, view: &mut LayoutView) -> bool {
    p.pace.drive(Driver::Frame);
    let moving = should_tick(&p.sim, p.ticks_remaining, p.dragging, p.halted);
    if moving {
        p.pace.owe(TICK_NS);
    }
    let clock = p.pace.budget.map(|budget| budget.clock);
    let stepped = step_owed(
        &mut p.sim,
        &mut p.pace,
        &mut p.ticks_remaining,
        p.dragging,
        p.halted,
        u32::MAX,
        1,
        clock,
    );
    let settling = should_tick(&p.sim, p.ticks_remaining, p.dragging, p.halted);
    settle_meter(p, TICK_NS, stepped.steps, settling);
    fold(p, view, settling);
    settling
}

/// Keep only the fraction while moving and record the frame; at rest, forget.
pub(super) fn settle_meter(p: &mut InlinePhysics, wall_ns: u64, steps: u32, settling: bool) {
    if settling {
        p.pace.owed %= TICK_UNITS;
        p.pace.meter.record(wall_ns, steps);
    } else {
        p.pace.owed = 0;
        p.pace.meter.clear();
    }
}

/// Nanoseconds of `duration`, saturating.
pub(super) fn nanos(duration: Duration) -> u64 {
    duration.as_nanos().min(u128::from(u64::MAX)) as u64
}

const _: () = assert!(TICK_DURATION.as_nanos() == TICK_NS as u128);

#[cfg(test)]
mod tests;
