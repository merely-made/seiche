// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

use super::*;

#[test]
fn negative_world_cells_split_with_euclidean_coordinates() {
    let address = split_voxel_address([-1, -9, 8], [8, 8, 8]).unwrap();
    assert_eq!(address.chunk, [-1, -2, 1]);
    assert_eq!(address.local, [7, 7, 0]);
}

#[test]
fn effective_patch_advances_once_and_publishes_one_dirty_box() {
    let mut chunk = VoxelChunk::new([4, 4, 4], 0u8).unwrap();
    let patch = chunk
        .apply_edits(
            0,
            4,
            [
                VoxelCellEdit {
                    cell: [0, 0, 0],
                    value: 2,
                },
                VoxelCellEdit {
                    cell: [1, 1, 1],
                    value: 3,
                },
                VoxelCellEdit {
                    cell: [0, 0, 0],
                    value: 4,
                },
            ],
        )
        .unwrap();

    assert_eq!(patch.previous_revision, 0);
    assert_eq!(patch.revision, 1);
    assert_eq!(patch.requested, 3);
    assert_eq!(patch.changes.len(), 2);
    assert_eq!(chunk.get([0, 0, 0]), Some(&4));
    assert_eq!(
        patch.dirty_regions,
        [VoxelRegion {
            origin: [0, 0, 0],
            extent: [2, 2, 2],
        }]
    );
    assert_eq!(chunk.dirty_regions(), patch.dirty_regions);
}

#[test]
fn refused_and_noop_patches_leave_authoritative_state_unchanged() {
    let mut chunk = VoxelChunk::new([2, 2, 2], 0u8).unwrap();
    let before = chunk.clone();
    assert!(matches!(
        chunk.apply_edits(
            1,
            1,
            [VoxelCellEdit {
                cell: [0, 0, 0],
                value: 1,
            }]
        ),
        Err(VoxelChunkError::StaleRevision { .. })
    ));
    assert_eq!(chunk, before);
    assert!(matches!(
        chunk.apply_edits(
            0,
            2,
            [
                VoxelCellEdit {
                    cell: [0, 0, 0],
                    value: 1,
                },
                VoxelCellEdit {
                    cell: [2, 0, 0],
                    value: 1,
                },
            ]
        ),
        Err(VoxelChunkError::CellOutOfBounds { .. })
    ));
    assert_eq!(chunk, before);
    assert!(matches!(
        chunk.apply_edits(
            0,
            0,
            [VoxelCellEdit {
                cell: [0, 0, 0],
                value: 1,
            }]
        ),
        Err(VoxelChunkError::BatchTooLarge { .. })
    ));
    assert_eq!(chunk, before);

    let noop = chunk
        .apply_edits(
            0,
            1,
            [VoxelCellEdit {
                cell: [0, 0, 0],
                value: 0,
            }],
        )
        .unwrap();
    assert_eq!(noop.revision, 0);
    assert!(noop.changes.is_empty());
    assert!(chunk.dirty_regions().is_empty());
}

#[test]
fn material_patch_lowers_only_occupancy_changes() {
    let mut chunk = VoxelChunk::new([4, 4, 4], 0u8).unwrap();
    chunk
        .apply_edits(
            0,
            1,
            [VoxelCellEdit {
                cell: [0, 0, 0],
                value: 1,
            }],
        )
        .unwrap();
    let patch = chunk
        .apply_edits(
            1,
            3,
            [
                VoxelCellEdit {
                    cell: [0, 0, 0],
                    value: 2,
                },
                VoxelCellEdit {
                    cell: [1, 0, 0],
                    value: 3,
                },
            ],
        )
        .unwrap();
    let collision_edits = patch.occupancy_edits(|material| *material != 0).unwrap();
    assert_eq!(collision_edits.len(), 1, "material-only edits stay out");
    assert_eq!(collision_edits[0].cell, [1, 0, 0]);
}

#[test]
fn draining_dirt_does_not_change_record_equality() {
    let mut chunk = VoxelChunk::new([2, 2, 2], 0u8).unwrap();
    chunk
        .apply_edits(
            0,
            1,
            [VoxelCellEdit {
                cell: [1, 1, 1],
                value: 1,
            }],
        )
        .unwrap();
    let with_dirt = chunk.clone();
    assert_eq!(chunk.drain_dirty_regions().len(), 1);
    assert_eq!(chunk, with_dirt);
}

#[test]
fn serialization_keeps_facts_and_drops_projection_dirt() {
    let mut chunk = VoxelChunk::new([2, 2, 2], 0u8).unwrap();
    chunk
        .apply_edits(
            0,
            1,
            [VoxelCellEdit {
                cell: [1, 0, 1],
                value: 7,
            }],
        )
        .unwrap();

    let encoded = serde_json::to_vec(&chunk).unwrap();
    let decoded: VoxelChunk<u8> = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(decoded, chunk);
    assert!(decoded.dirty_regions().is_empty());
    assert_eq!(decoded.get([1, 0, 1]), Some(&7));
}
