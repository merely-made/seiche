// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! The lagged seam: a pairwise law computed somewhere that answers later
//! (the host's GPU), applied to rapier a step or more after the positions it
//! was computed from, with the CPU law covering every step it cannot.
//!
//! Rapier keeps every role. A [`LaggedRepulsion`] is handed positions and
//! asked, never blocking, whether an earlier answer has arrived; a
//! [`LaggedLane`] holds the bookkeeping a force needs to use one safely: the
//! body order a submission was made in, the step it answers for, and the
//! staleness limit past which the CPU scan runs instead (physics catalog
//! plan, P5b, ruled 2026-10-02: "Newest result for up to N steps").
//!
//! Up to N submissions are in flight at once, answered in submission order.
//! With one in flight, an answer that takes d steps to arrive would be d to
//! 2d-1 steps old while it serves, so a browser running three steps a frame
//! would cover one step in three; with N in flight every step submits, and
//! the first step of each frame receives an answer one step old.
//!
//! The lane knows nothing of [`crate::NodeExclusion`]: a force that wants a
//! lagged evaluator of its own can hold one the same way.

use std::collections::VecDeque;

use rapier2d::prelude::RigidBodyHandle;

use crate::{RepulsionForces, RepulsionRequest, RepulsionSolver, RepulsionSolverError};

/// A pairwise law evaluated elsewhere and answered later. Neither method may
/// block: a browser cannot wait on its own event loop, and a native actor
/// should not wait on the device.
pub trait LaggedRepulsion: Send {
    /// Start evaluating the law over these positions. `Err` when nothing
    /// could be submitted (no device, a lost device, a failed launch).
    fn submit(
        &mut self,
        xs: &[f32],
        ys: &[f32],
        request: RepulsionRequest,
    ) -> Result<(), RepulsionSolverError>;

    /// The answer to the oldest outstanding submission, once, if it has
    /// arrived. Answers come back in submission order.
    fn poll(&mut self) -> Option<Result<RepulsionForces, RepulsionSolverError>>;

    /// Submissions still waiting for their answers.
    fn in_flight(&self) -> usize;
}

/// Default staleness limit for a host that runs one physics step per frame
/// (a native actor): an answer applies only on the step after its positions.
pub const DEFAULT_MAX_STALE_STEPS: u32 = 1;

/// What a lane did, counted from its installation: the numbers a receipt
/// reads to prove the device ran, and how often the CPU covered for it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LaggedStats {
    /// Steps whose forces came from the lagged evaluator.
    pub device_steps: u64,
    /// Steps at or above the threshold that ran the CPU law instead: the
    /// first step, a late or stale answer, a changed body set, a failure.
    pub cpu_steps: u64,
    /// Submissions accepted.
    pub submissions: u64,
    /// Submissions refused and answers that came back as errors.
    pub failures: u64,
    /// Answers that arrived for a body set that had since changed.
    pub mismatched: u64,
}

struct Answer {
    handles: Vec<RigidBodyHandle>,
    step: u64,
    forces: RepulsionForces,
}

/// A [`LaggedRepulsion`] and the bookkeeping that makes its answers safe to
/// apply. See the module docs.
pub struct LaggedLane {
    solver: Box<dyn LaggedRepulsion>,
    max_stale_steps: u32,
    /// The body order and step of each submission in flight, oldest first.
    submitted: VecDeque<(Vec<RigidBodyHandle>, u64)>,
    latest: Option<Answer>,
    stats: LaggedStats,
}

impl LaggedLane {
    pub fn new(solver: Box<dyn LaggedRepulsion>, max_stale_steps: u32) -> Self {
        Self {
            solver,
            max_stale_steps: max_stale_steps.max(1),
            submitted: VecDeque::new(),
            latest: None,
            stats: LaggedStats::default(),
        }
    }

    pub fn max_stale_steps(&self) -> u32 {
        self.max_stale_steps
    }

    pub fn stats(&self) -> LaggedStats {
        self.stats
    }

    /// One step's use of the lane, at step `now`, for bodies `handles` at
    /// `xs`, `ys`. Collects every answer that has arrived, keeping the newest;
    /// submits these positions while fewer than `max_stale_steps` are in
    /// flight; and returns the newest answer if it is for this same body order
    /// and at most `max_stale_steps` old. `None` means the caller runs its CPU
    /// law for this step.
    pub fn step(
        &mut self,
        now: u64,
        handles: &[RigidBodyHandle],
        xs: &[f32],
        ys: &[f32],
        request: RepulsionRequest,
    ) -> Option<&RepulsionForces> {
        while let Some(answer) = self.solver.poll() {
            let submitted = self.submitted.pop_front();
            match (answer, submitted) {
                (Ok(forces), Some((handles, step))) => {
                    self.latest = Some(Answer {
                        handles,
                        step,
                        forces,
                    });
                },
                (Ok(_), None) => {},
                (Err(error), _) => {
                    self.stats.failures += 1;
                    tracing::warn!(?error, "lagged repulsion failed; the CPU law covers");
                },
            }
        }
        // An evaluator that dropped its queue (a lost device) owes nothing.
        while self.submitted.len() > self.solver.in_flight() {
            self.submitted.pop_front();
        }
        if self.solver.in_flight() < self.max_stale_steps as usize {
            match self.solver.submit(xs, ys, request) {
                Ok(()) => {
                    self.stats.submissions += 1;
                    self.submitted.push_back((handles.to_vec(), now));
                },
                Err(error) => {
                    self.stats.failures += 1;
                    tracing::warn!(?error, "lagged repulsion refused a submission");
                },
            }
        }
        let usable = match &self.latest {
            Some(answer) if answer.handles.as_slice() != handles => {
                self.stats.mismatched += 1;
                self.latest = None;
                false
            },
            Some(answer) => now.saturating_sub(answer.step) <= u64::from(self.max_stale_steps),
            None => false,
        };
        if usable {
            self.stats.device_steps += 1;
            self.latest.as_ref().map(|answer| &answer.forces)
        } else {
            self.stats.cpu_steps += 1;
            None
        }
    }
}

/// How a staged pairwise law is evaluated: a synchronous closure (tests,
/// benches) or a lagged lane (a host's device).
pub enum RepulsionRoute {
    Sync(RepulsionSolver),
    Lagged(LaggedLane),
}

/// The staged evaluator a [`crate::Simulation`] hands its forces, and the
/// node count at or above which they use it.
pub struct Repulsion {
    pub route: RepulsionRoute,
    pub threshold: usize,
}

impl Repulsion {
    /// The lane's counts, when the route is lagged.
    pub fn stats(&self) -> Option<LaggedStats> {
        match &self.route {
            RepulsionRoute::Lagged(lane) => Some(lane.stats()),
            RepulsionRoute::Sync(_) => None,
        }
    }
}
