// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! The refusals a brick map returns. Split out of `lib.rs` to keep it
//! under the workspace's per-file size ceiling.

use std::{error::Error, fmt};

use crate::BrickKey;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BrickMapError {
    TooManyBricks {
        actual: usize,
        maximum: usize,
    },
    PointerVolumeOverflow {
        pointer_extent: [u32; 3],
    },
    /// A retargeted selection's bounding box outgrew the fixed pointer
    /// volume.
    ExtentExceeded {
        extent: [u32; 3],
        maximum: [u32; 3],
    },
    /// A changed selection was offered without advancing the projection
    /// revision.
    ProjectionNotAdvanced {
        current: u64,
        offered: u64,
    },
    AllocationFailed {
        entries: usize,
    },
    UnknownKey {
        key: BrickKey,
    },
    MissingBrick {
        key: BrickKey,
    },
    InvalidBrickLength {
        key: BrickKey,
        actual: usize,
        expected: usize,
    },
}

impl fmt::Display for BrickMapError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooManyBricks { actual, maximum } => {
                write!(
                    formatter,
                    "brick map has {actual} bricks; maximum is {maximum}"
                )
            },
            Self::PointerVolumeOverflow { pointer_extent } => {
                write!(
                    formatter,
                    "pointer volume overflows usize: {pointer_extent:?}"
                )
            },
            Self::ExtentExceeded { extent, maximum } => {
                write!(
                    formatter,
                    "selection bounds {extent:?} exceed the fixed pointer extent {maximum:?}"
                )
            },
            Self::ProjectionNotAdvanced { current, offered } => {
                write!(
                    formatter,
                    "a changed selection needs a projection revision past {current}; offered {offered}"
                )
            },
            Self::AllocationFailed { entries } => {
                write!(
                    formatter,
                    "pointer volume could not allocate {entries} entries"
                )
            },
            Self::UnknownKey { key } => write!(formatter, "brick key is not selected: {key:?}"),
            Self::MissingBrick { key } => write!(formatter, "selected brick is missing: {key:?}"),
            Self::InvalidBrickLength {
                key,
                actual,
                expected,
            } => write!(
                formatter,
                "brick {key:?} has {actual} bytes; expected {expected}"
            ),
        }
    }
}

impl Error for BrickMapError {}
