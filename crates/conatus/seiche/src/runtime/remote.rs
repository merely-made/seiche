// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! The remote backend's link: commands out as bytes, updates in as bytes.
//!
//! [`Physics::Remote`](super::Physics::Remote) is the actor's contract over a
//! message boundary the host does not own the far side of: a Web Worker, a
//! process, a socket. The host gives it a [`Transport`], two non-blocking byte
//! pipes, and [`RemoteLink`] does the rest: the handshake, the in-order queue of
//! commands issued before the far side said `Ready`, the codec
//! (`seiche_model::wire`, postcard) and the conversion of a [`WireUpdate`] back
//! to the [`PhysicsUpdate`] the host folds. A command whose payload is not data
//! (`SetForces`, a closure handler, a GPU solver) has no wire form and is
//! dropped, with a warning; the host sends the data form (`set_force_set`,
//! the entry relay) to a backend that is not inline (`Physics::runs_inline`).

use seiche_model::{
    DataCommand,
    wire::{FromWorker, PROTOCOL_VERSION, ToWorker, WireUpdate, decode, encode},
};

use super::{PhysicsCommand, PhysicsUpdate};

/// Two non-blocking byte pipes to the far side, one message to a call.
///
/// A message is one whole encoded `ToWorker` or `FromWorker`; a transport that
/// frames or batches them (a `postMessage` of an `ArrayBuffer`, a length-prefixed
/// socket) delivers each one intact and in order.
pub trait Transport {
    /// Hand one message to the far side, without blocking. A transport whose
    /// far side is not open yet buffers it.
    fn send(&mut self, message: Vec<u8>);
    /// The next message that has arrived, or `None` when nothing is waiting.
    fn receive(&mut self) -> Option<Vec<u8>>;
}

/// The handshake and the queue in front of a [`Transport`].
pub(crate) struct RemoteLink {
    transport: Box<dyn Transport>,
    /// The far side said `Ready` at our version.
    ready: bool,
    /// The far side speaks another version, or sent what would not decode:
    /// nothing more is sent, and its updates are dropped.
    failed: bool,
    /// Commands issued before `Ready`, encoded, in order.
    queued: Vec<Vec<u8>>,
    /// Updates read while looking for `Ready`, not yet folded.
    inbox: Vec<PhysicsUpdate>,
}

impl RemoteLink {
    pub(super) fn new(mut transport: Box<dyn Transport>) -> Self {
        transport.send(encode(&ToWorker::Init {
            version: PROTOCOL_VERSION,
        }));
        Self {
            transport,
            ready: false,
            failed: false,
            queued: Vec::new(),
            inbox: Vec::new(),
        }
    }

    /// Whether the far side has said `Ready` (and not failed).
    pub(super) fn is_ready(&self) -> bool {
        self.ready && !self.failed
    }

    /// Send a command, or queue it until the far side is `Ready`.
    pub(super) fn command(&mut self, command: PhysicsCommand) {
        let data = match DataCommand::try_from(command) {
            Ok(data) => data,
            Err(refused) => {
                tracing::warn!(
                    "physics command has no wire form and was dropped by the remote backend: {}",
                    command_name(&refused)
                );
                return;
            },
        };
        if self.failed {
            return;
        }
        self.pump();
        let bytes = encode(&ToWorker::Command(data));
        if self.ready {
            self.transport.send(bytes);
        } else {
            self.queued.push(bytes);
        }
    }

    /// Read every message waiting; `Ready` flushes the queue, in order.
    fn pump(&mut self) {
        while let Some(bytes) = self.transport.receive() {
            match decode::<FromWorker>(&bytes) {
                Ok(FromWorker::Ready { version }) if version == PROTOCOL_VERSION => {
                    self.ready = true;
                    for queued in std::mem::take(&mut self.queued) {
                        self.transport.send(queued);
                    }
                },
                Ok(FromWorker::Ready { version }) => {
                    tracing::warn!(
                        "physics worker speaks protocol {version}, this host {PROTOCOL_VERSION}"
                    );
                    self.failed = true;
                    self.queued.clear();
                },
                Ok(FromWorker::Update(update)) => {
                    if !self.failed {
                        self.inbox.push(update.into());
                    }
                },
                Err(error) => {
                    tracing::warn!("{error}");
                    self.failed = true;
                    self.queued.clear();
                },
            }
        }
    }

    /// Every update that has arrived, in order.
    pub(super) fn drain(&mut self) -> Vec<PhysicsUpdate> {
        self.pump();
        std::mem::take(&mut self.inbox)
    }
}

fn command_name(command: &PhysicsCommand) -> &'static str {
    match command {
        #[cfg(feature = "engine")]
        PhysicsCommand::SetForces(_) => "SetForces",
        PhysicsCommand::SetCouplingEntryHandler(_) => "SetCouplingEntryHandler",
        #[cfg(feature = "engine")]
        PhysicsCommand::SetLaggedRepulsion { .. } => "SetLaggedRepulsion",
        _ => "a data command",
    }
}

impl From<WireUpdate> for PhysicsUpdate {
    fn from(update: WireUpdate) -> Self {
        Self {
            field_clock: update.field_clock,
            #[cfg(feature = "actor")]
            coupling_membership: None,
            #[cfg(feature = "actor")]
            coupling_trace: None,
            coupling_entries: update.coupling_entries,
            snapshot: update.snapshot,
            settling: update.settling,
            command_epoch: update.command_epoch,
            pace: update.pace,
        }
    }
}

impl From<PhysicsUpdate> for WireUpdate {
    /// The update without its diagnostic frames (the coupling membership frame
    /// and trace hold shared bindings and stay on the simulation's side).
    fn from(update: PhysicsUpdate) -> Self {
        Self {
            field_clock: update.field_clock,
            coupling_entries: update.coupling_entries,
            snapshot: update.snapshot,
            settling: update.settling,
            command_epoch: update.command_epoch,
            pace: update.pace,
        }
    }
}
