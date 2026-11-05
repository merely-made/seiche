// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

use super::*;

impl fmt::Display for ResidentChunkError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyPlaneId => formatter.write_str("resident plane id cannot be empty"),
            Self::EmptyShape { shape } => {
                write!(formatter, "resident plane shape {shape:?} is empty")
            },
            Self::ShapeOverflow { shape } => {
                write!(
                    formatter,
                    "resident plane shape {shape:?} overflows its allocation"
                )
            },
            Self::DuplicatePlane(plane) => {
                write!(formatter, "resident plane {plane} already exists")
            },
            Self::UnknownPlane(plane) => write!(formatter, "resident plane {plane} does not exist"),
            Self::ElementCount {
                plane,
                expected,
                actual,
            } => write!(
                formatter,
                "resident plane {plane} needs {expected} elements, got {actual}"
            ),
            Self::BurnElementType { plane, actual } => write!(
                formatter,
                "resident plane {plane} is {actual:?}, not F32, so it has no Burn float view"
            ),
            Self::StaleStamp { expected, actual } => write!(
                formatter,
                "resident patch expected stamp {expected:?}, but the chunk is at {actual:?}"
            ),
            Self::NonAdvancingStamp {
                previous,
                committed,
            } => write!(
                formatter,
                "resident patch must advance revision and read epoch beyond {previous:?}, got {committed:?}"
            ),
            Self::DirtyRegionOutOfBounds {
                region,
                chunk_extent,
            } => write!(
                formatter,
                "resident dirty region {region:?} does not fit chunk extent {chunk_extent:?}"
            ),
            Self::PatchRequiresSinglePlane { plane_count } => write!(
                formatter,
                "single-plane resident patch cannot restamp a bundle containing {plane_count} planes"
            ),
            Self::EmptyPatchBatch { plane } => {
                write!(formatter, "resident patch batch for plane {plane} is empty")
            },
            Self::OverlappingPatches {
                plane,
                first_index,
                second_index,
            } => write!(
                formatter,
                "resident patches {first_index} and {second_index} overlap in plane {plane}"
            ),
            Self::PatchElementType {
                plane,
                expected,
                actual,
            } => write!(
                formatter,
                "resident plane {plane} stores {expected:?}, not patch element type {actual:?}"
            ),
            Self::PatchRangeOverflow { plane } => {
                write!(
                    formatter,
                    "resident patch range for plane {plane} overflowed"
                )
            },
            Self::PatchRange {
                plane,
                byte_offset,
                byte_len,
                plane_byte_len,
            } => write!(
                formatter,
                "resident patch [{byte_offset}..{}) does not fit plane {plane}'s {plane_byte_len} bytes",
                byte_offset + byte_len
            ),
            Self::UnalignedPatch {
                plane,
                byte_offset,
                byte_len,
                required_alignment,
            } => write!(
                formatter,
                "resident patch for plane {plane} has byte offset {byte_offset} and length {byte_len}; both must be multiples of {required_alignment}"
            ),
            Self::Resource(error) => write!(formatter, "CubeCL resource export failed: {error}"),
        }
    }
}

impl std::error::Error for ResidentChunkError {}
