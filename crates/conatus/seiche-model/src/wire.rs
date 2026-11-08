// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! The messages between a host and a physics Worker, and their codec.
//!
//! A host sends [`ToWorker`]: an [`Init`](ToWorker::Init) carrying the
//! protocol version, then [`DataCommand`]s. The Worker answers [`FromWorker`]:
//! [`Ready`](FromWorker::Ready) once it has started, then a
//! [`WireUpdate`] per interval. All of it is plain data (`serde`), written with
//! `postcard`: a varint format with no field names, so a 200-node layout is
//! about two kilobytes a frame, small beside the cost of the step that made it.
//! (A dense `Float32Array` of positions in `Init`'s order would be smaller
//! still and is the next step if a frame ever shows up in a profile; it needs a
//! stateful codec on both sides, which `postcard` messages do not.)
//!
//! The update's diagnostic frames (the coupling membership frame and the
//! coupling trace) hold shared bindings and are not on the wire; the entry
//! events that the host's handler runs from are.

use crate::{CouplingEntryEvent, DataCommand, FieldClock, LayoutSnapshot, PaceStats};

/// The protocol version a host and a Worker must share. Bump it with any
/// change to a type on the wire; `Ready` echoes the Worker's.
pub const PROTOCOL_VERSION: u32 = 1;

/// Host to Worker.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum ToWorker {
    /// First message: the version the host speaks.
    Init { version: u32 },
    /// One command to apply, in order.
    Command(DataCommand),
}

/// Worker to host.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum FromWorker {
    /// The Worker started and speaks `version`.
    Ready { version: u32 },
    /// One interval's layout.
    Update(WireUpdate),
}

/// One layout the Worker produced: `PhysicsUpdate` without its diagnostic frames.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct WireUpdate {
    pub field_clock: FieldClock,
    pub coupling_entries: Vec<CouplingEntryEvent>,
    pub snapshot: LayoutSnapshot,
    pub settling: bool,
    pub command_epoch: u64,
    pub pace: PaceStats,
}

/// A message that would not decode: wrong bytes, or from another version.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WireError(pub postcard::Error);

impl std::fmt::Display for WireError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "physics wire message: {}", self.0)
    }
}

impl std::error::Error for WireError {}

/// Write a message.
///
/// # Panics
/// Never for the messages in this module; `postcard` writes any of them.
pub fn encode<T: serde::Serialize>(message: &T) -> Vec<u8> {
    postcard::to_allocvec(message).expect("a wire message serialises")
}

/// Read a message.
///
/// # Errors
/// When the bytes are not a whole message of type `T`.
pub fn decode<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Result<T, WireError> {
    postcard::from_bytes(bytes).map_err(WireError)
}

#[cfg(test)]
mod tests {
    use euclid::default::Point2D;

    use super::*;
    use crate::NodeKey;

    fn sample_update() -> WireUpdate {
        WireUpdate {
            field_clock: FieldClock::default(),
            coupling_entries: Vec::new(),
            snapshot: LayoutSnapshot {
                positions: (0..4)
                    .map(|i| (NodeKey::new(i), Point2D::new(i as f32 * 1.5, -(i as f32))))
                    .collect(),
                energy: 0.25,
                speed: 3.0,
                generation: 9,
                ..LayoutSnapshot::default()
            },
            settling: true,
            command_epoch: 4,
            pace: PaceStats {
                ticks: 12,
                effective_speed: Some(1.0),
                budget_bound: false,
            },
        }
    }

    #[test]
    fn an_update_round_trips_bit_for_bit() {
        let update = FromWorker::Update(sample_update());
        let back: FromWorker = decode(&encode(&update)).unwrap();
        assert_eq!(back, update);
    }

    #[test]
    fn commands_and_handshake_round_trip() {
        let init = ToWorker::Init {
            version: PROTOCOL_VERSION,
        };
        assert_eq!(encode(&decode::<ToWorker>(&encode(&init)).unwrap()), encode(&init));
        let command = ToWorker::Command(DataCommand::Pin(NodeKey::new(2), Point2D::new(1.0, 2.0)));
        assert_eq!(
            encode(&decode::<ToWorker>(&encode(&command)).unwrap()),
            encode(&command)
        );
        let ready = FromWorker::Ready {
            version: PROTOCOL_VERSION,
        };
        assert_eq!(decode::<FromWorker>(&encode(&ready)).unwrap(), ready);
    }

    #[test]
    fn a_frame_is_compact() {
        // 4 nodes: far under what the same layout takes as JSON.
        let bytes = encode(&FromWorker::Update(sample_update()));
        let json = serde_json::to_vec(&sample_update()).unwrap();
        assert!(bytes.len() * 3 < json.len(), "{} vs {}", bytes.len(), json.len());
    }

    #[test]
    fn garbage_is_an_error_and_not_a_panic() {
        assert!(decode::<FromWorker>(&[0xff, 0xff, 0xff]).is_err());
    }
}
