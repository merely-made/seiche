// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! The owner's side of the coupling-entry relay.
//!
//! An entry handler is a closure, so it cannot run on the far side of a
//! message boundary. Offloaded, the simulation reports each entry as a
//! [`CouplingEntryEvent`] in its update stream (`SetCouplingEntryRelay`), and
//! this half runs the owner's handler when the events reach the host: one
//! update later than the inline backend, which runs it inside the tick. The
//! handler receives the same report, built here from the event and the binding
//! the host sent, and the outcome it returns is written into the host's copy of
//! the latest membership frame, where the inline backend has it already.
//!
//! Every event is dispatched once, in observation order, including those in
//! updates a later update supersedes: only the layout is "latest wins", not the
//! entries. An event whose coupling list the host has since replaced still
//! resolves against the list it belonged to.

use std::{collections::VecDeque, sync::Arc};

use seiche_model::CouplingEntryEvent;

use crate::{
    CouplingBinding, CouplingEntryHandler, CouplingEntryOutcome, CouplingMembershipChange,
    CouplingMembershipFrame, CouplingMembershipReport, CouplingTraceSession,
};

/// One coupling list the host sent, and the barrier epoch that acknowledges it.
struct Listing {
    generation: u64,
    epoch: u64,
    bindings: Vec<Arc<CouplingBinding>>,
}

pub(super) struct EntryRelay {
    handler: Option<CouplingEntryHandler>,
    listings: VecDeque<Listing>,
    generation: u64,
    /// The session a report carries when no membership frame is at hand.
    session: CouplingTraceSession,
}

impl Default for EntryRelay {
    fn default() -> Self {
        Self::starting(0, Vec::new(), None)
    }
}

impl EntryRelay {
    /// A relay for a simulation already at `generation` with `bindings` live.
    pub(super) fn starting(
        generation: u64,
        bindings: Vec<Arc<CouplingBinding>>,
        handler: Option<CouplingEntryHandler>,
    ) -> Self {
        Self {
            handler,
            listings: VecDeque::from([Listing {
                generation,
                epoch: 0,
                bindings,
            }]),
            generation,
            session: CouplingTraceSession::new(),
        }
    }

    pub(super) fn set_handler(&mut self, handler: Option<CouplingEntryHandler>) {
        self.handler = handler;
    }

    /// The host sent a new coupling list; the simulation numbers it one past
    /// the last, and `epoch` is the barrier that follows the command.
    pub(super) fn listed(&mut self, epoch: u64, bindings: Vec<Arc<CouplingBinding>>) {
        self.generation = self
            .generation
            .checked_add(1)
            .expect("coupling binding generation exhausted");
        self.listings.push_back(Listing {
            generation: self.generation,
            epoch,
            bindings,
        });
    }

    /// An update acknowledging `epoch` has arrived: lists that a later,
    /// acknowledged list replaced can no longer be named by an event.
    pub(super) fn acknowledged(&mut self, epoch: u64) {
        while self.listings.len() > 1 && self.listings[1].epoch <= epoch {
            self.listings.pop_front();
        }
    }

    /// Run the handler for each event, and write its outcome into `frame`.
    pub(super) fn dispatch(
        &mut self,
        events: Vec<CouplingEntryEvent>,
        mut frame: Option<&mut CouplingMembershipFrame>,
    ) {
        let Some(handler) = self.handler.clone() else {
            return;
        };
        for event in events {
            let Some(binding) = self
                .listings
                .iter()
                .find(|listing| listing.generation == event.binding_generation)
                .and_then(|listing| listing.bindings.get(event.coupling))
            else {
                continue;
            };
            let session = frame
                .as_deref()
                .map_or_else(|| self.session.clone(), |frame| frame.session.clone());
            let report = CouplingMembershipReport {
                session,
                step: event.step,
                clock: event.clock,
                binding_generation: event.binding_generation,
                binding_identity_ambiguous: false,
                binding: Arc::clone(binding),
                node: event.node,
                identity: event.identity,
                target_index: event.target_index,
                reference_point: event.reference_point,
                previous: event.previous,
                eligibility: event.eligibility,
                change: CouplingMembershipChange::Entered,
                cause: event.cause,
                entry_action: CouplingEntryOutcome::Pending,
            };
            let outcome = match handler(&report) {
                Ok(()) => CouplingEntryOutcome::Completed,
                Err(reason) => CouplingEntryOutcome::Refused(reason),
            };
            if let Some(frame) = frame.as_deref_mut() {
                if let Some(row) = frame.reports.iter_mut().find(|row| {
                    row.entry_action == CouplingEntryOutcome::Pending
                        && row.step == event.step
                        && row.binding_generation == event.binding_generation
                        && row.node == event.node
                        && row.identity == event.identity
                        && row.target_index == event.target_index
                        && Arc::ptr_eq(&row.binding, binding)
                }) {
                    row.entry_action = outcome;
                }
            }
        }
    }
}
