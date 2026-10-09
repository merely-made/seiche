// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! `colliders_at_point`, `voxel_filled` and `voxel_cells` (rulings P1, O1 to
//! O3 of 2026-10-09).

use conatus::{
    BodyDesc, BodyError, BodyId, BodyWorld, ColliderDesc, ColliderId, ColliderShape, SpatialFilter,
    Transform, VoxelBox, VoxelEdit,
};

fn voxel_world(cell_size: [f32; 3], occupied: Vec<[i32; 3]>) -> (BodyWorld, ColliderId) {
    let mut world = BodyWorld::new([0.0; 3]);
    let body = world
        .spawn(
            BodyDesc::fixed().with_collider(ColliderDesc::new(ColliderShape::VoxelGrid {
                cell_size,
                occupied,
            })),
        )
        .unwrap();
    world.refresh_queries();
    (world, ColliderId::new(body, 0))
}

/// A ragged ground crossing block boundaries, negative cells included.
fn ground() -> Vec<[i32; 3]> {
    let mut cells = Vec::new();
    for x in -13..11i32 {
        for z in -9..14 {
            let height = (x * 7 + z * 13).rem_euclid(11) - 3;
            for y in -4..height {
                cells.push([x, y, z]);
            }
        }
    }
    cells
}

fn canonical(mut cells: Vec<[i32; 3]>) -> Vec<[i32; 3]> {
    cells.sort_by_key(|cell| (cell.map(|v| v.div_euclid(8)), *cell));
    cells
}

#[test]
fn a_point_query_names_the_containing_colliders() {
    let mut world = BodyWorld::new([0.0; 3]);
    let solid = world
        .spawn(BodyDesc::fixed().with_collider(ColliderDesc::new(ColliderShape::cuboid([1.0; 3]))))
        .unwrap();
    let sensor = world
        .spawn(
            BodyDesc::fixed()
                .at(Transform::from_translation([0.5, 0.0, 0.0]))
                .with_collider(ColliderDesc::new(ColliderShape::sphere(1.0)).sensor(true)),
        )
        .unwrap();
    let all = SpatialFilter::default();
    assert!(
        world.colliders_at_point([0.0; 3], all).unwrap().is_empty(),
        "control: not refreshed"
    );
    world.refresh_queries();

    let both = world.colliders_at_point([0.5, 0.0, 0.0], all).unwrap();
    assert_eq!(
        both,
        vec![ColliderId::new(solid, 0), ColliderId::new(sensor, 0)]
    );
    let solids = SpatialFilter {
        include_sensors: false,
        ..all
    };
    assert_eq!(
        world.colliders_at_point([0.5, 0.0, 0.0], solids).unwrap(),
        vec![ColliderId::new(solid, 0)]
    );
    let excluding = SpatialFilter {
        exclude_body: Some(solid),
        ..all
    };
    assert_eq!(
        world
            .colliders_at_point([0.5, 0.0, 0.0], excluding)
            .unwrap(),
        vec![ColliderId::new(sensor, 0)]
    );
    assert!(
        world
            .colliders_at_point([5.0, 0.0, 0.0], all)
            .unwrap()
            .is_empty()
    );
    let neither = SpatialFilter {
        include_sensors: false,
        include_solids: false,
        ..all
    };
    assert!(
        world
            .colliders_at_point([0.5, 0.0, 0.0], neither)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn a_point_query_refuses_like_raycast() {
    let mut world = BodyWorld::new([0.0; 3]);
    let body = world
        .spawn(BodyDesc::fixed().with_collider(ColliderDesc::new(ColliderShape::cuboid([1.0; 3]))))
        .unwrap();
    assert!(matches!(
        world.colliders_at_point([f32::NAN, 0.0, 0.0], SpatialFilter::default()),
        Err(BodyError::InvalidQuery(_))
    ));
    world.despawn(body).unwrap();
    let stale = SpatialFilter {
        exclude_body: Some(body),
        ..SpatialFilter::default()
    };
    assert_eq!(
        world.colliders_at_point([0.0; 3], stale),
        Err(BodyError::UnknownBody(body))
    );
}

#[test]
fn a_point_inside_a_voxel_follows_a_carve() {
    let (mut world, collider) = voxel_world([1.0; 3], vec![[0, 0, 0], [0, 1, 0]]);
    let centre = [0.5, 1.5, 0.5];
    assert_eq!(
        world
            .colliders_at_point(centre, SpatialFilter::default())
            .unwrap(),
        vec![collider]
    );
    world
        .edit_voxels(
            collider,
            [VoxelEdit {
                cell: [0, 1, 0],
                filled: false,
            }],
        )
        .unwrap();
    assert!(
        world
            .colliders_at_point(centre, SpatialFilter::default())
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        world
            .colliders_at_point([0.5, 0.5, 0.5], SpatialFilter::default())
            .unwrap(),
        vec![collider]
    );
}

#[test]
fn voxel_filled_reads_the_collider_grid() {
    let cells = ground();
    let (mut world, collider) = voxel_world([1.0; 3], cells.clone());
    for cell in &cells {
        assert!(world.voxel_filled(collider, *cell).unwrap(), "{cell:?}");
    }
    assert!(
        !world.voxel_filled(collider, [0, 40, 0]).unwrap(),
        "above the grid"
    );
    assert!(
        !world.voxel_filled(collider, [5000, 0, 5000]).unwrap(),
        "outside the domain"
    );
    world
        .edit_voxels(
            collider,
            [VoxelEdit {
                cell: cells[0],
                filled: false,
            }],
        )
        .unwrap();
    assert!(!world.voxel_filled(collider, cells[0]).unwrap());
}

#[test]
fn voxel_reads_refuse_like_edit_voxels() {
    let mut world = BodyWorld::new([0.0; 3]);
    let solid = world
        .spawn(BodyDesc::fixed().with_collider(ColliderDesc::new(ColliderShape::cuboid([1.0; 3]))))
        .unwrap();
    let not_voxels = ColliderId::new(solid, 0);
    assert_eq!(
        world.voxel_filled(not_voxels, [0; 3]),
        Err(BodyError::NotVoxelCollider(not_voxels))
    );
    assert!(matches!(
        world.voxel_cells(not_voxels, None),
        Err(BodyError::NotVoxelCollider(_))
    ));
    let missing = ColliderId::new(solid, 1);
    assert_eq!(
        world.voxel_filled(missing, [0; 3]),
        Err(BodyError::UnknownCollider(missing))
    );
    let inverted = VoxelBox {
        min: [0, 2, 0],
        max: [1, 1, 1],
    };
    assert!(matches!(
        world.voxel_cells(not_voxels, Some(inverted)),
        Err(BodyError::InvalidQuery(_))
    ));
    world.despawn(solid).unwrap();
    assert_eq!(
        world.voxel_filled(not_voxels, [0; 3]),
        Err(BodyError::UnknownBody(solid))
    );
}

#[test]
fn voxel_cells_iterate_in_canonical_order() {
    let cells = ground();
    let (world, collider) = voxel_world([1.0; 3], cells.clone());
    let read: Vec<_> = world.voxel_cells(collider, None).unwrap().collect();
    assert_eq!(read.len(), cells.len());
    assert_eq!(read, canonical(cells));
}

#[test]
fn reversed_and_edit_built_grids_iterate_identically() {
    let cells = ground();
    let (forward, a) = voxel_world([1.0; 3], cells.clone());
    let mut reversed_cells = cells.clone();
    reversed_cells.reverse();
    let (reversed, b) = voxel_world([1.0; 3], reversed_cells.clone());
    let (mut edited, c) = voxel_world([1.0; 3], Vec::new());
    edited
        .edit_voxels(
            c,
            reversed_cells
                .iter()
                .map(|&cell| VoxelEdit { cell, filled: true }),
        )
        .unwrap();
    // A grid built with spare cells then carved back to the same set.
    let mut padded_cells = cells.clone();
    padded_cells.extend((-20..20).map(|x| [x, 30, 3]));
    let (mut carved, d) = voxel_world([1.0; 3], padded_cells);
    carved
        .edit_voxels(
            d,
            (-20..20).map(|x| VoxelEdit {
                cell: [x, 30, 3],
                filled: false,
            }),
        )
        .unwrap();

    let first: Vec<_> = forward.voxel_cells(a, None).unwrap().collect();
    assert_eq!(
        first,
        reversed.voxel_cells(b, None).unwrap().collect::<Vec<_>>()
    );
    assert_eq!(
        first,
        edited.voxel_cells(c, None).unwrap().collect::<Vec<_>>()
    );
    assert_eq!(
        first,
        carved.voxel_cells(d, None).unwrap().collect::<Vec<_>>()
    );
}

#[test]
fn canonical_order_holds_for_uneven_cell_sizes() {
    let cells = ground();
    let mut reversed_cells = cells.clone();
    reversed_cells.reverse();
    let (world, collider) = voxel_world([0.5, 2.0, 1.25], reversed_cells);
    let read: Vec<_> = world.voxel_cells(collider, None).unwrap().collect();
    assert_eq!(read, canonical(cells));
}

#[test]
fn boxed_iteration_reads_only_the_box() {
    let cells = ground();
    let (world, collider) = voxel_world([1.0; 3], cells.clone());
    for bounds in [
        VoxelBox {
            min: [-8, -4, -8],
            max: [0, 0, 0],
        },
        VoxelBox {
            min: [-3, -1, 2],
            max: [5, 6, 11],
        },
        VoxelBox {
            min: [0, 0, 0],
            max: [8, 8, 8],
        },
        VoxelBox {
            min: [1, 1, 1],
            max: [1, 9, 9],
        },
        VoxelBox {
            min: [-100, -100, -100],
            max: [100, 100, 100],
        },
    ] {
        let inside =
            |cell: &[i32; 3]| (0..3).all(|a| bounds.min[a] <= cell[a] && cell[a] < bounds.max[a]);
        let expected = canonical(cells.iter().copied().filter(inside).collect());
        let read: Vec<_> = world.voxel_cells(collider, Some(bounds)).unwrap().collect();
        assert_eq!(read, expected, "{bounds:?}");
    }
}

#[test]
fn an_empty_grid_iterates_nothing() {
    let (mut world, collider) = voxel_world([1.0; 3], Vec::new());
    assert_eq!(world.voxel_cells(collider, None).unwrap().count(), 0);
    world
        .edit_voxels(
            collider,
            [VoxelEdit {
                cell: [3, 3, 3],
                filled: true,
            }],
        )
        .unwrap();
    world
        .edit_voxels(
            collider,
            [VoxelEdit {
                cell: [3, 3, 3],
                filled: false,
            }],
        )
        .unwrap();
    assert_eq!(world.voxel_cells(collider, None).unwrap().count(), 0);
}

#[test]
fn the_probe_cross_check_runs_through_conatus() {
    // The parry-ground probe's exact pass: every source cell per cell, then
    // the iteration finds no extra cell, then per-brick signatures.
    let cells = ground();
    let (world, collider) = voxel_world([1.0; 3], cells.clone());
    let source: std::collections::BTreeSet<_> = cells.iter().copied().collect();
    let mut occupied = 0;
    for x in -16..16 {
        for y in -8..16 {
            for z in -16..16 {
                let filled = world.voxel_filled(collider, [x, y, z]).unwrap();
                assert_eq!(filled, source.contains(&[x, y, z]));
                occupied += filled as usize;
            }
        }
    }
    assert_eq!(occupied, cells.len());
    assert!(
        world
            .voxel_cells(collider, None)
            .unwrap()
            .all(|cell| source.contains(&cell))
    );
    let _: BodyId = collider.body();
}
