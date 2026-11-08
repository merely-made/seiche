// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! Field couplings as data: what a host sends the simulation instead of
//! compiled `CouplingForce`s, and what the simulation sends back when a body
//! enters a coupling's region.
//!
//! A [`CouplingDescriptor`] is every input of a compiled coupling: the
//! response, strength, targets, field definition, registry, extent, selector
//! and provenance. `seiche` builds the force from it (`CouplingForce::from_descriptor`)
//! and can read one back from a force (`CouplingForce::describe`), so the two
//! forms carry the same binding. numen's field types already carry serde.
//!
//! The owner's entry handler is a closure and cannot cross a message boundary.
//! The simulation instead reports each entry as a [`CouplingEntryEvent`] in
//! its update stream, and the owner's side runs the handler when the update
//! lands.

use numen::{
    CouplingId, CouplingResponse, FieldDefinition, FieldExtent, FieldId, FieldRegistry,
    NodeSelector,
};

use crate::{FieldClock, NodeKey};

/// Authored owner identities supplied by the adapter. The revision is a hint,
/// not a digest: the binding retains the actual definition and registry too.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CouplingSource {
    pub field: FieldId,
    pub coupling: CouplingId,
    pub revision: Option<u64>,
}

/// Stable identity supplied by the body owner. `member` names the owner member,
/// not necessarily a content resource. An occurrence is only claimed when the
/// adapter has one; graph-node bodies must not invent occurrence identities.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CouplingTargetIdentity {
    pub member: u128,
    pub occurrence: Option<u128>,
}

/// Query candidacy and spatial eligibility remain distinct. An unresolved
/// extent or absent/invalid point must not be reported as outside the field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CouplingEligibility {
    Inside,
    Outside,
    NotCandidate,
    MissingBody,
    UnsupportedExtent,
    InvalidPoint,
}

impl CouplingEligibility {
    /// Whether the extent and the body both resolved to an answer.
    pub fn resolved(self) -> bool {
        matches!(self, Self::Inside | Self::Outside | Self::NotCandidate)
    }
}

/// Several changes can coincide. These flags describe observed input changes,
/// not a guessed UI gesture; movement is used when binding inputs did not change.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CouplingMembershipCause {
    pub field_edit: bool,
    pub query_change: bool,
    pub source_change: bool,
    pub movement: bool,
    pub reset: bool,
}

/// One field coupling as data: every input of a compiled coupling force.
///
/// `target_identities` runs parallel to `targets` (one slot per target, in the
/// same order); a builder normalises a shorter or longer list to that length.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CouplingDescriptor {
    pub response: CouplingResponse,
    pub strength: f32,
    /// The resolved target set, with multiplicity and order.
    pub targets: Vec<NodeKey>,
    pub field: FieldDefinition,
    /// The registry the field's `Sample` references resolve through.
    pub registry: FieldRegistry,
    pub extent: FieldExtent,
    pub selector: Option<NodeSelector>,
    pub source: Option<CouplingSource>,
    pub target_identities: Vec<Option<CouplingTargetIdentity>>,
}

/// A body entering a coupling's region, as the simulation reports it for the
/// owner's handler to act on. Carries the observation; the binding is the
/// coupling at `coupling` in the list the owner last sent, in the binding
/// generation named here, which the owner's side resolves.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CouplingEntryEvent {
    /// The simulation step the observation was made on.
    pub step: u64,
    pub clock: FieldClock,
    /// The generation of the coupling list the event belongs to: one more
    /// than the previous for every list the owner sends.
    pub binding_generation: u64,
    /// The index of the coupling in that list.
    pub coupling: usize,
    pub node: NodeKey,
    pub identity: Option<CouplingTargetIdentity>,
    pub target_index: Option<usize>,
    pub reference_point: Option<[f32; 2]>,
    pub previous: Option<CouplingEligibility>,
    pub eligibility: CouplingEligibility,
    pub cause: CouplingMembershipCause,
}
