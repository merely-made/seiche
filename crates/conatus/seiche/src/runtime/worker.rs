// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! The far side of [`Physics::Remote`](super::Physics::Remote): a simulation
//! driven by bytes.
//!
//! [`PhysicsWorker`] is the actor thread's body without the thread. A Web
//! Worker has no blocking `recv` and no sleep, so its script owns the loop and
//! calls in: [`receive`](PhysicsWorker::receive) for each message the host
//! posted, [`interval`](PhysicsWorker::interval) once per tick of its timer
//! while [`is_busy`](PhysicsWorker::is_busy), posting what they return. The
//! commands are applied by the same function the actor applies them with, and
//! the interval is the actor's (`speed::actor_interval`), so a Worker's
//! trajectory is the actor's, and the inline backend's, bit for bit.
//!
//! The Worker keeps what the actor does not: a [`ChannelCache`], so a law
//! change over an unchanged graph (a descriptor at the same revision) reads
//! PageRank, the colouring and the depths and does not compute them again,
//! as the inline canvas's channel registry does.

use std::{cell::RefCell, rc::Rc, time::Duration};

use seiche_model::{
    DataCommand,
    wire::{FromWorker, PROTOCOL_VERSION, ToWorker, WireError, WireUpdate, decode, encode},
};

use super::{PhysicsCommand, Speed, actor, should_tick, speed};
use crate::{Simulation, catalog};

/// A simulation, the pace state around it, and the cache of its channels.
pub struct PhysicsWorker {
    sim: Simulation,
    state: actor::ActorState,
    cache: Rc<RefCell<catalog::ChannelCache>>,
    clock: fn() -> Duration,
    generation: u64,
    /// The clock's reading at the last interval run, while moving.
    last_interval: Option<Duration>,
    /// A command was applied since the last update (seed and halt must
    /// acknowledge their placement even when no tick follows).
    changed: bool,
}

impl PhysicsWorker {
    /// A worker with an empty simulation. `clock` reads time since any fixed
    /// point (the Worker's `performance.now`); the step budget and the
    /// interval's wall time are measured on it.
    pub fn new(clock: fn() -> Duration) -> Self {
        Self {
            sim: Simulation::new(),
            state: actor::ActorState {
                ticks_remaining: 0,
                dragging: false,
                halted: false,
                command_epoch: 0,
                pace: speed::Pace::for_actor(Speed::REAL_TIME, 0, None, clock),
                clock,
            },
            cache: catalog::ChannelCache::shared(),
            clock,
            generation: 0,
            last_interval: None,
            changed: false,
        }
    }

    /// The channel cache, for counting what a build computed.
    pub fn channel_cache(&self) -> Rc<RefCell<catalog::ChannelCache>> {
        Rc::clone(&self.cache)
    }

    /// Apply one message from the host. Returns the messages to send back
    /// (the answer to `Init`), each one whole.
    ///
    /// # Errors
    /// When the bytes are not a `ToWorker`; nothing was applied.
    pub fn receive(&mut self, bytes: &[u8]) -> Result<Vec<Vec<u8>>, WireError> {
        match decode::<ToWorker>(bytes)? {
            ToWorker::Init { .. } => Ok(vec![encode(&FromWorker::Ready {
                version: PROTOCOL_VERSION,
            })]),
            ToWorker::Command(command) => {
                self.apply(command);
                Ok(Vec::new())
            },
        }
    }

    fn apply(&mut self, command: DataCommand) {
        self.changed = true;
        match command {
            // The descriptor's channels come from the Worker's own cache.
            DataCommand::SetForceSet(descriptor) => self
                .sim
                .set_forces(catalog::build_force_set_cached(&descriptor, &self.cache)),
            command => actor::apply(&mut self.sim, PhysicsCommand::from(command), &mut self.state),
        }
    }

    /// Whether the simulation has ticks to run (a settle, a drag, a scene that
    /// never rests): the script keeps its timer going while true, and parks
    /// when not.
    pub fn is_busy(&self) -> bool {
        should_tick(
            &self.sim,
            self.state.ticks_remaining,
            self.state.dragging,
            self.state.halted,
        )
    }

    /// Run one interval if there is work, and answer with the update it made:
    /// the ticks the speed owes (up to the budget), then one snapshot. Also
    /// answers for commands applied since the last update. `None` when there
    /// was neither. Call it once per 60 Hz timer tick while busy, and once
    /// after the messages that changed something.
    pub fn interval(&mut self) -> Option<Vec<u8>> {
        let moving = self.is_busy();
        if moving {
            let now = (self.clock)();
            let wall = self.last_interval.map_or(speed::TICK_NS, |last| {
                speed::nanos(now.saturating_sub(last))
            });
            self.last_interval = Some(now);
            speed::actor_interval(
                &mut self.sim,
                &mut self.state.pace,
                &mut self.state.ticks_remaining,
                self.state.dragging,
                self.state.halted,
                wall,
            );
        } else {
            self.last_interval = None;
        }
        if !(moving || self.changed) {
            return None;
        }
        self.changed = false;
        self.generation = self.generation.wrapping_add(1);
        let settling = self.is_busy();
        let coupling_entries = self.sim.take_coupling_entry_events();
        let mut snapshot = self.sim.snapshot(self.generation);
        speed::draw(&mut self.state.pace, &mut snapshot, settling);
        Some(encode(&FromWorker::Update(WireUpdate {
            field_clock: self.sim.field_clock(),
            coupling_entries,
            snapshot,
            settling,
            command_epoch: self.state.command_epoch,
            pace: self.state.pace.stats(),
        })))
    }
}
