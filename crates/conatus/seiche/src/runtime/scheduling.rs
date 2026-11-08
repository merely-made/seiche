// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

use super::*;

impl Physics {
    /// A new in-thread backend over `sim`, with an initial settle budget.
    pub fn inline(sim: Simulation, initial_settle: u32) -> Self {
        Physics::Inline(InlinePhysics {
            sim,
            ticks_remaining: initial_settle,
            dragging: false,
            generation: 0,
            halted: false,
            pace: speed::Pace::default(),
        })
    }

    /// Move the simulation onto an [`armillary`] actor thread (always-offload).
    /// A no-op if already offloaded. `wake` pokes the host's event loop when a
    /// snapshot is ready. The actor inherits the current settle budget so an
    /// in-flight first settle continues uninterrupted across the move.
    #[cfg(feature = "actor")]
    pub fn offload(&mut self, wake: Wake) {
        let Physics::Inline(inline) = self else {
            return;
        };
        // Move the real simulation out (swap in a throwaway empty one) so it can
        // be built-into the actor thread; `Simulation: Send` makes this sound.
        let mut sim = std::mem::replace(&mut inline.sim, Simulation::new());
        // An entry handler cannot cross to the actor's thread: it stays here, and
        // the simulation reports entries as events for it (the entry relay).
        let handler = sim.take_coupling_entry_handler();
        sim.set_coupling_entry_relay(handler.is_some());
        let entry = entry_relay::EntryRelay::starting(
            sim.coupling_binding_generation(),
            sim.coupling_bindings(),
            handler,
        );
        let initial_settle = inline.ticks_remaining;
        let dragging = inline.dragging;
        let halted = inline.halted;
        let speed = inline.pace.speed;
        let pace = inline.pace.stats();
        let settling = should_tick(&sim, initial_settle, dragging, halted);
        let state = ActorState {
            ticks_remaining: initial_settle,
            dragging,
            halted,
            command_epoch: 0,
            pace: speed::Pace::for_actor(
                speed,
                pace.ticks,
                inline.pace.budget.map(|budget| budget.spec()),
            ),
        };
        let coupling_trace = sim.coupling_trace().cloned();
        let field_clock = sim.field_clock();
        let coupling_membership = sim.coupling_membership().cloned();
        let (handle, updates) = spawn(wake, move |commands, out| {
            run(sim, state, commands, out);
        });
        *self = Physics::Actor(ActorPhysics {
            coupling_trace,
            field_clock,
            coupling_membership,
            handle,
            updates,
            settling,
            energy: 0.0,
            speed: 0.0,
            command_epoch: 0,
            dial: speed,
            budget: inline.pace.budget,
            pace,
            entry,
        });
    }

    /// Set the simulation speed: ticks per frame at the fixed [`TICK_DT`].
    /// The trajectory does not depend on it.
    pub fn set_speed(&mut self, speed: Speed) {
        match self {
            Physics::Inline(p) => {
                p.pace.speed = speed;
                if speed >= Speed::REAL_TIME {
                    p.pace.previous = None;
                }
            },
            #[cfg(feature = "actor")]
            Physics::Actor(p) => {
                p.dial = speed;
                p.handle.command(PhysicsCommand::SetSpeed(speed));
            },
        }
    }

    pub fn speed(&self) -> Speed {
        match self {
            Physics::Inline(p) => p.pace.speed,
            #[cfg(feature = "actor")]
            Physics::Actor(p) => p.dial,
        }
    }

    /// Bound a frame's ticks above real time (see [`StepBudget`]). Offloaded,
    /// only the allowance and the margin cross to the actor, which measures on
    /// its own clock; the host's clock function is not sent.
    pub fn set_step_budget(&mut self, budget: Option<StepBudget>) {
        match self {
            Physics::Inline(p) => p.pace.budget = budget,
            #[cfg(feature = "actor")]
            Physics::Actor(p) => {
                p.budget = budget;
                p.handle
                    .command(PhysicsCommand::SetStepBudget(budget.map(|b| b.spec())));
            },
        }
    }

    /// The step budget last set, as the host gave it.
    pub fn step_budget(&self) -> Option<StepBudget> {
        match self {
            Physics::Inline(p) => p.pace.budget,
            #[cfg(feature = "actor")]
            Physics::Actor(p) => p.budget,
        }
    }

    /// Ticks run, the effective speed, and whether the budget bound.
    pub fn pace(&self) -> PaceStats {
        match self {
            Physics::Inline(p) => p.pace.stats(),
            #[cfg(feature = "actor")]
            Physics::Actor(p) => p.pace,
        }
    }

    /// Extend the settle budget to at least `ticks` (start / prolong a settle).
    pub fn settle(&mut self, ticks: u32) {
        match self {
            Physics::Inline(p) => {
                p.ticks_remaining = p.ticks_remaining.max(ticks);
                if ticks > 0 {
                    p.halted = false;
                    p.sim.set_field_clock_paused(false);
                }
            },
            #[cfg(feature = "actor")]
            Physics::Actor(p) => {
                if ticks > 0 {
                    p.field_clock.paused = false;
                }
                p.handle.command(PhysicsCommand::Settle(ticks));
                if ticks > 0 {
                    p.settling = true;
                }
            },
        }
    }

    /// Stop current stepping, including perpetual scenes. A later positive
    /// settle resumes; a new drag can temporarily step while held.
    pub fn halt(&mut self) {
        match self {
            Physics::Inline(p) => {
                p.ticks_remaining = 0;
                p.dragging = false;
                p.halted = true;
                p.sim.set_field_clock_paused(true);
                p.pace.forget();
                p.pace.meter.clear();
            },
            #[cfg(feature = "actor")]
            Physics::Actor(p) => {
                p.field_clock.paused = true;
                p.handle.command(PhysicsCommand::Halt);
                p.barrier();
                p.settling = false;
            },
        }
    }

    /// Tell the backend whether a drag is in progress (keep ticking so the
    /// pinned node's neighbors react through the springs).
    pub fn set_dragging(&mut self, dragging: bool) {
        match self {
            Physics::Inline(p) => p.dragging = dragging,
            #[cfg(feature = "actor")]
            Physics::Actor(p) => {
                p.handle.command(PhysicsCommand::SetDragging(dragging));
            },
        }
    }

    /// Whether the layout is still moving (settle in progress or a node dragged).
    pub fn is_settling(&self) -> bool {
        match self {
            Physics::Inline(p) => should_tick(&p.sim, p.ticks_remaining, p.dragging, p.halted),
            #[cfg(feature = "actor")]
            Physics::Actor(p) => p.settling,
        }
    }

    /// Fold the layout into `view` **without stepping** — for a host that
    /// synced bodies and must read or draw them before the next frame.
    /// Offloaded, this drains only what the actor has already emitted.
    pub fn refresh(&mut self, view: &mut LayoutView) {
        match self {
            Physics::Inline(p) => {
                let settling = should_tick(&p.sim, p.ticks_remaining, p.dragging, p.halted);
                speed::fold(p, view, settling);
            },
            #[cfg(feature = "actor")]
            Physics::Actor(p) => p.fold_latest(view),
        }
    }

    /// Advance one frame, folding the latest positions into `view`, and return
    /// whether the layout is still settling.
    ///
    /// - Inline: owe one frame-equivalent ([`TICK_DURATION`]) at the speed, so
    ///   one tick a call at real time, and snapshot. Deterministic callers own
    ///   their tick count: a fraction from elapsed-time driving does not carry.
    /// - Actor: drain the update channel, applying the most recent snapshot
    ///   (older queued ones are superseded — only the freshest layout matters).
    pub fn advance_frame(&mut self, view: &mut LayoutView) -> bool {
        match self {
            Physics::Inline(p) => speed::advance_frame(p, view),
            #[cfg(feature = "actor")]
            Physics::Actor(p) => {
                p.fold_latest(view);
                p.settling
            },
        }
    }

    /// The node bodies' rms speed (see [`Simulation::rms_speed`]): live
    /// inline, the last folded snapshot's offloaded. (G7, F46.)
    pub fn rms_speed(&self) -> f32 {
        match self {
            Physics::Inline(p) => p.sim.rms_speed(),
            #[cfg(feature = "actor")]
            Physics::Actor(p) => p.speed,
        }
    }

    /// Why the layout keeps moving, for a host's diagnostics: whether the
    /// world asks for ticks of its own (a scene, or a force such as a flow
    /// that has not converged), and the settle budget left. The actor reports
    /// neither and answers `(false, 0)`.
    pub fn tick_demand(&self) -> (bool, u32) {
        match self {
            Physics::Inline(p) => (p.sim.wants_continuous_tick(), p.ticks_remaining),
            #[cfg(feature = "actor")]
            Physics::Actor(_) => (false, 0),
        }
    }
}
