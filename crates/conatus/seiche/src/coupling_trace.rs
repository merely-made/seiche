// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! Bounded observations of the live coupling slot, before integration.
use std::{num::NonZeroUsize, sync::Arc};

use numen::{CouplingId, CouplingResponse, FieldDefinition, FieldExtent, FieldId, FieldRegistry};
use rapier2d::prelude::*;

use crate::{CouplingEffect, CouplingEvaluation, CouplingForce, ForceContext, NodeKey, Simulation};

/// Opaque process-local simulation identity. Equality compares the owner token,
/// so equal tick/generation numbers from different worlds cannot be conflated.
/// This is not a persistent graph or transport identity.
#[derive(Clone, Debug)]
pub struct CouplingTraceSession(Arc<()>);

impl CouplingTraceSession {
    pub(crate) fn new() -> Self {
        Self(Arc::new(()))
    }
}
impl PartialEq for CouplingTraceSession {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}
impl Eq for CouplingTraceSession {}

/// Authored owner identities supplied by the adapter. The revision is a hint,
/// not a digest: the binding retains the actual definition and registry too.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CouplingSource {
    pub field: FieldId,
    pub coupling: CouplingId,
    pub revision: Option<u64>,
}

/// Stable identity supplied by the body owner. `member` names the owner member,
/// not necessarily a content resource. An occurrence is only claimed when the
/// adapter has one; graph-node bodies must not invent occurrence identities.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CouplingTargetIdentity {
    pub member: u128,
    pub occurrence: Option<u128>,
}

/// The exact immutable input snapshot used by one compiled force. Trace rows
/// retain this Arc across rebuilds; they never look up today's graph by index.
/// Target slots preserve multiplicity and order, including repeated node keys.
#[derive(Clone, Debug, PartialEq)]
pub struct CouplingBinding {
    pub response: CouplingResponse,
    pub strength: f32,
    pub field: FieldDefinition,
    pub extent: FieldExtent,
    /// Syntactic Time dependency through the captured registry.
    pub animated: bool,
    pub registry: FieldRegistry,
    pub targets: Vec<NodeKey>,
    pub selector: Option<numen::NodeSelector>,
    pub source: Option<CouplingSource>,
    pub target_identities: Vec<Option<CouplingTargetIdentity>>,
}

/// Physical state immediately around one coupling call. `force` is Rapier's
/// user-force accumulator, not a post-integration isolated contribution.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CouplingBodyState {
    pub position: [f32; 2],
    pub velocity: [f32; 2],
    pub force: [f32; 2],
    pub dynamic: bool,
    pub sleeping: bool,
}

impl CouplingBodyState {
    fn read(body: &RigidBody) -> Self {
        let p = body.translation();
        let v = body.linvel();
        let f = body.user_force();
        Self {
            position: [p.x, p.y],
            velocity: [v.x, v.y],
            force: [f.x, f.y],
            dynamic: body.is_dynamic(),
            sleeping: body.is_sleeping(),
        }
    }
}

/// A write call can leave a fixed body's state unchanged. Inert evaluations
/// invoke no write; missing bodies have neither an evaluation nor body state.
#[derive(Clone, Debug, PartialEq)]
pub enum CouplingAppliedEffect {
    MissingBody,
    Inert,
    Write {
        before: CouplingBodyState,
        after: CouplingBodyState,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct CouplingTraceRecord {
    pub binding: Arc<CouplingBinding>,
    /// Execution order within the dedicated coupling slot, not an owner ID.
    pub coupling_index: usize,
    pub target_index: usize,
    pub node: NodeKey,
    /// Captured before any writes by this coupling, including duplicates.
    pub sample_position: Option<[f32; 2]>,
    pub evaluation: Option<CouplingEvaluation>,
    pub applied: CouplingAppliedEffect,
}

/// The most recent valid tick only. There is no accumulating history. The
/// caller chooses the row cap; `omitted` discloses truncation without skipping
/// physical work. Generic boxed forces and post-integration contacts are outside
/// this trace. Field samples use the accepted simulation clock.
#[derive(Clone, Debug, PartialEq)]
pub struct CouplingTrace {
    pub session: CouplingTraceSession,
    pub step: u64,
    pub clock: crate::FieldClock,
    pub dt: f32,
    pub binding_generation: u64,
    pub records: Vec<CouplingTraceRecord>,
    pub omitted: usize,
    pub limit: NonZeroUsize,
}

impl CouplingTrace {
    pub(crate) fn has_room(&mut self) -> bool {
        if self.records.len() < self.limit.get() {
            true
        } else {
            self.omitted = self.omitted.saturating_add(1);
            false
        }
    }
}

impl Simulation {
    /// Enable bounded observations of the dedicated coupling slot, or disable
    /// with `None`. Clears the previous trace. Off by default.
    pub fn set_coupling_trace_limit(&mut self, limit: Option<NonZeroUsize>) {
        self.coupling_trace_limit = limit;
        self.coupling_trace = None;
    }

    /// Historical evidence from the most recent tick, not a probe of the current
    /// graph or pose. Replacing the binding set clears it until another tick.
    pub fn coupling_trace(&self) -> Option<&CouplingTrace> {
        self.coupling_trace.as_ref()
    }
}

impl CouplingForce {
    pub(crate) fn apply_traced(
        &self,
        ctx: &mut ForceContext<'_>,
        dt: f32,
        coupling_index: usize,
        trace: &mut CouplingTrace,
        samples: &[(Option<RigidBodyHandle>, Option<[f32; 2]>)],
    ) {
        for (target_index, sample) in samples.iter().copied().enumerate() {
            let capture = trace.has_room();
            let node = self.binding.targets[target_index];
            let mut record = capture.then(|| CouplingTraceRecord {
                binding: Arc::clone(&self.binding),
                coupling_index,
                target_index,
                node,
                sample_position: sample.1,
                evaluation: None,
                applied: CouplingAppliedEffect::MissingBody,
            });
            if let (Some(handle), Some(point)) = sample
                && let Some(body) = ctx.bodies.get_mut(handle)
            {
                let effect = self.evaluate_target(
                    point,
                    body.is_dynamic(),
                    dt,
                    ctx.field_clock.sample_time(),
                );
                let before = capture.then(|| CouplingBodyState::read(body));
                Self::write_effect(body, effect);
                if let Some(record) = &mut record {
                    record.evaluation = Some(CouplingEvaluation {
                        sample_time: ctx.field_clock.sample_time(),
                        effect,
                    });
                    record.applied = if matches!(effect, CouplingEffect::Inert(_)) {
                        CouplingAppliedEffect::Inert
                    } else {
                        CouplingAppliedEffect::Write {
                            before: before.expect("captured state"),
                            after: CouplingBodyState::read(body),
                        }
                    };
                }
            }
            if let Some(record) = record {
                trace.records.push(record);
            }
        }
    }
}

#[cfg(test)]
#[path = "coupling_trace_tests.rs"]
mod tests;
