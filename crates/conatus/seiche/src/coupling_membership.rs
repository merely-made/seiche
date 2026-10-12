// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! Reference-point membership and explicit owner entry callbacks (F204/F205).
use crate::{CouplingBinding, CouplingTargetIdentity, CouplingTraceSession, FieldClock, NodeKey};
use numen::{CouplingId, FieldId};
use rapier2d::prelude::RigidBodyHandle;
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

/// Query candidacy and spatial eligibility remain distinct. An unresolved
/// extent or absent/invalid point must not be reported as outside the field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CouplingEligibility {
    Inside,
    Outside,
    NotCandidate,
    MissingBody,
    UnsupportedExtent,
    InvalidPoint,
}
impl CouplingEligibility {
    fn resolved(self) -> bool {
        matches!(self, Self::Inside | Self::Outside | Self::NotCandidate)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CouplingMembershipChange {
    Initial,
    Entered,
    Exited,
    Unavailable,
    Unchanged,
}

/// Several changes can coincide. These flags describe observed input changes,
/// not a guessed UI gesture; movement is used when binding inputs did not change.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CouplingMembershipCause {
    pub field_edit: bool,
    pub query_change: bool,
    pub source_change: bool,
    pub movement: bool,
    pub reset: bool,
}

/// A callback is explicitly installed by the owner. Unknown applets/responses
/// are not implicitly executed. Completed means the callback returned success;
/// it is separate from a physical body's write and integrator outcome.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CouplingEntryOutcome {
    NotRequested,
    NotRegistered,
    Pending,
    Completed,
    Refused(String),
}
pub type CouplingEntryHandler =
    Arc<dyn Fn(&CouplingMembershipReport) -> Result<(), String> + Send + Sync>;

#[derive(Clone, Debug, PartialEq)]
pub struct CouplingMembershipReport {
    pub session: CouplingTraceSession,
    pub step: u64,
    pub clock: FieldClock,
    pub binding_generation: u64,
    pub binding_identity_ambiguous: bool,
    pub binding: Arc<CouplingBinding>,
    pub node: NodeKey,
    pub identity: Option<CouplingTargetIdentity>,
    /// None for a previously known appearance removed from the candidate set.
    pub target_index: Option<usize>,
    pub reference_point: Option<[f32; 2]>,
    pub previous: Option<CouplingEligibility>,
    pub eligibility: CouplingEligibility,
    pub change: CouplingMembershipChange,
    pub cause: CouplingMembershipCause,
    pub entry_action: CouplingEntryOutcome,
}

/// Latest tick only, bounded by the live coupling/body set. Callbacks run once
/// synchronously at observation, so a slow host cannot replay or lose them by
/// folding/discarding diagnostic frames. Callbacks own their thread/authority.
#[derive(Clone, Debug, PartialEq)]
pub struct CouplingMembershipFrame {
    pub session: CouplingTraceSession,
    pub step: u64,
    pub clock: FieldClock,
    pub binding_generation: u64,
    pub reports: Vec<CouplingMembershipReport>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Owner {
    Named(FieldId, CouplingId),
    Anonymous(usize),
}
fn owner(binding: &Arc<CouplingBinding>) -> Owner {
    binding.source.map_or_else(
        || Owner::Anonymous(Arc::as_ptr(binding) as usize),
        |s| Owner::Named(s.field, s.coupling),
    )
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Appearance {
    node: NodeKey,
    identity: Option<CouplingTargetIdentity>,
}
#[derive(Clone, Copy)]
struct Seen {
    eligibility: CouplingEligibility,
    body: Option<RigidBodyHandle>,
    point: Option<[f32; 2]>,
}
struct Members {
    binding: Arc<CouplingBinding>,
    seen: HashMap<Appearance, Seen>,
}

#[derive(Default)]
pub(crate) struct MembershipRuntime {
    owners: HashMap<Owner, Members>,
    ambiguous: HashSet<Owner>,
    reset: bool,
}
impl MembershipRuntime {
    pub(crate) fn reset(&mut self) {
        self.owners.clear();
        self.reset = true;
    }

    pub(crate) fn retain_bindings(&mut self, bindings: impl Iterator<Item = Arc<CouplingBinding>>) {
        let bindings: Vec<_> = bindings.collect();
        let mut counts = HashMap::new();
        for binding in &bindings {
            *counts.entry(owner(binding)).or_insert(0_usize) += 1;
        }
        self.ambiguous = counts
            .into_iter()
            .filter_map(|(key, n)| (n > 1).then_some(key))
            .collect();
        let owners: HashSet<_> = bindings.iter().map(|b| self.key(b)).collect();
        self.owners.retain(|key, _| owners.contains(key));
    }

    fn key(&self, binding: &Arc<CouplingBinding>) -> Owner {
        let key = owner(binding);
        if self.ambiguous.contains(&key) {
            Owner::Anonymous(Arc::as_ptr(binding) as usize)
        } else {
            key
        }
    }

    pub(crate) fn observe(
        &mut self,
        binding: &Arc<CouplingBinding>,
        samples: &[(Option<RigidBodyHandle>, Option<[f32; 2]>)],
        bodies: &HashMap<NodeKey, RigidBodyHandle>,
        frame: &mut CouplingMembershipFrame,
        handler: Option<&CouplingEntryHandler>,
    ) {
        let identity_ambiguous = self.ambiguous.contains(&owner(binding));
        let key = self.key(binding);
        let members = self.owners.entry(key).or_insert_with(|| Members {
            binding: Arc::clone(binding),
            seen: HashMap::new(),
        });
        let old = &members.binding;
        let mut cause = CouplingMembershipCause {
            field_edit: old.extent != binding.extent
                || old.field != binding.field
                || old.registry != binding.registry
                || old.response != binding.response
                || old.strength != binding.strength,
            source_change: old.source != binding.source,
            reset: self.reset,
            ..Default::default()
        };
        let appearances = |binding: &CouplingBinding| -> HashSet<Appearance> {
            binding
                .targets
                .iter()
                .enumerate()
                .map(|(i, &node)| Appearance {
                    node,
                    identity: binding.target_identities[i],
                })
                .collect()
        };
        let current = appearances(binding);
        cause.query_change = old.selector != binding.selector || appearances(old) != current;
        cause.movement =
            !cause.field_edit && !cause.query_change && !cause.source_change && !cause.reset;
        members.binding = Arc::clone(binding);
        let mut visited = HashSet::new();
        for (i, &node) in binding.targets.iter().enumerate() {
            let appearance = Appearance {
                node,
                identity: binding.target_identities[i],
            };
            // Repeated physical applications of one appearance are still traced
            // separately, but must not multiply its membership/entry action.
            if !visited.insert(appearance) {
                continue;
            }
            let (body, point) = samples[i];
            let eligibility = point.map_or(CouplingEligibility::MissingBody, |p| {
                eligibility(binding, p)
            });
            let previous = members
                .seen
                .get(&appearance)
                .filter(|seen| seen.body == body || body.is_none())
                .map(|s| s.eligibility);
            let mut actual_cause = cause;
            actual_cause.source_change |= members
                .seen
                .keys()
                .any(|old| old.node == node && old.identity != appearance.identity);
            actual_cause.movement = members
                .seen
                .get(&appearance)
                .is_some_and(|s| s.point != point && s.point.is_some() && point.is_some());
            if members
                .seen
                .get(&appearance)
                .is_some_and(|seen| seen.body != body)
            {
                actual_cause.source_change = true;
                actual_cause.movement = false;
            }
            frame.reports.push(report(
                binding,
                appearance,
                Observation {
                    target_index: Some(i),
                    point,
                    previous,
                    eligibility,
                    cause: actual_cause,
                },
                frame,
                handler,
                identity_ambiguous,
            ));
            members.seen.insert(
                appearance,
                Seen {
                    eligibility,
                    body,
                    point,
                },
            );
        }
        // Retain known non-candidates while their actual body remains: a query
        // removing and later restoring one appearance is a later transition.
        let mut departed: Vec<_> = members
            .seen
            .iter()
            .filter(|(a, _)| !current.contains(a))
            .map(|(a, s)| (*a, *s))
            .collect();
        departed.sort_by_key(|(a, _)| {
            (
                a.node.index(),
                a.identity.map(|id| (id.member, id.occurrence)),
            )
        });
        for (appearance, seen) in departed {
            let handle = bodies.get(&appearance.node).copied();
            let eligibility = if handle.is_some() && handle == seen.body {
                CouplingEligibility::NotCandidate
            } else {
                CouplingEligibility::MissingBody
            };
            if seen.eligibility != eligibility {
                let mut departed_cause = cause;
                departed_cause.query_change = true;
                departed_cause.source_change |= eligibility == CouplingEligibility::MissingBody
                    || current
                        .iter()
                        .any(|a| a.node == appearance.node && a.identity != appearance.identity);
                departed_cause.movement = false;
                frame.reports.push(report(
                    binding,
                    appearance,
                    Observation {
                        target_index: None,
                        point: None,
                        previous: Some(seen.eligibility),
                        eligibility,
                        cause: departed_cause,
                    },
                    frame,
                    handler,
                    identity_ambiguous,
                ));
            }
            // An identity replaced on the same body cannot accumulate history.
            if handle.is_none()
                || handle != seen.body
                || current.iter().any(|a| a.node == appearance.node)
            {
                members.seen.remove(&appearance);
            } else {
                members.seen.insert(
                    appearance,
                    Seen {
                        eligibility,
                        body: handle,
                        point: seen.point,
                    },
                );
            }
        }
    }

    pub(crate) fn finish_tick(&mut self) {
        self.reset = false;
    }
}

pub(crate) fn eligibility(binding: &CouplingBinding, point: [f32; 2]) -> CouplingEligibility {
    if !point.iter().all(|p| p.is_finite()) {
        return CouplingEligibility::InvalidPoint;
    }
    match binding.extent.contains(point[0], point[1]) {
        Some(true) => CouplingEligibility::Inside,
        Some(false) => CouplingEligibility::Outside,
        None => CouplingEligibility::UnsupportedExtent,
    }
}
struct Observation {
    target_index: Option<usize>,
    point: Option<[f32; 2]>,
    previous: Option<CouplingEligibility>,
    eligibility: CouplingEligibility,
    cause: CouplingMembershipCause,
}
fn report(
    binding: &Arc<CouplingBinding>,
    appearance: Appearance,
    observation: Observation,
    frame: &CouplingMembershipFrame,
    handler: Option<&CouplingEntryHandler>,
    identity_ambiguous: bool,
) -> CouplingMembershipReport {
    let Observation {
        target_index,
        point: reference_point,
        previous,
        eligibility,
        cause,
    } = observation;
    use CouplingMembershipChange::*;
    let change = match previous {
        None => Initial,
        Some(old) if !old.resolved() && eligibility.resolved() => Initial,
        Some(old) if old == eligibility => Unchanged,
        Some(_) if !eligibility.resolved() => Unavailable,
        Some(old)
            if old != CouplingEligibility::Inside && eligibility == CouplingEligibility::Inside =>
        {
            Entered
        },
        Some(CouplingEligibility::Inside) => Exited,
        Some(_) => Unchanged,
    };
    let mut report = CouplingMembershipReport {
        session: frame.session.clone(),
        step: frame.step,
        clock: frame.clock,
        binding_generation: frame.binding_generation,
        binding_identity_ambiguous: identity_ambiguous,
        binding: Arc::clone(binding),
        node: appearance.node,
        identity: appearance.identity,
        target_index,
        reference_point,
        previous,
        eligibility,
        change,
        cause,
        entry_action: CouplingEntryOutcome::NotRequested,
    };
    if change == Entered && identity_ambiguous {
        report.entry_action =
            CouplingEntryOutcome::Refused("coupling identity has multiple live bindings".into());
    } else if change == Entered {
        report.entry_action = CouplingEntryOutcome::Pending;
        report.entry_action =
            handler.map_or(
                CouplingEntryOutcome::NotRegistered,
                |handler| match handler(&report) {
                    Ok(()) => CouplingEntryOutcome::Completed,
                    Err(reason) => CouplingEntryOutcome::Refused(reason),
                },
            );
    }
    report
}
