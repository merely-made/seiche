// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! `BodyWorld::contacts` (rulings C1 to C4 of 2026-10-09): every shape pair,
//! the prediction honoured at its edges, records on the world collider's side.

use conatus::{
    BodyDesc, BodyError, BodyWorld, ColliderDesc, ColliderId, ColliderShape, ShapeContact,
    SpatialFilter, Transform, VoxelEdit,
};

const ALL: SpatialFilter = SpatialFilter {
    layers: conatus::CollisionLayers::ALL,
    exclude_body: None,
    include_sensors: true,
    include_solids: true,
};

fn block_cells() -> Vec<[i32; 3]> {
    let mut cells = Vec::new();
    for x in 0..3 {
        for y in 0..2 {
            for z in 0..3 {
                cells.push([x, y, z]);
            }
        }
    }
    cells
}

/// World shapes with their top at y = 0, centred on x = z = 0.
fn world_shapes() -> Vec<(&'static str, ColliderShape, [f32; 3])> {
    vec![
        ("sphere", ColliderShape::sphere(0.5), [0.0, -0.5, 0.0]),
        ("box", ColliderShape::cuboid([0.5; 3]), [0.0, -0.5, 0.0]),
        (
            "capsule",
            ColliderShape::capsule_y(0.5, 0.25),
            [0.0, -0.75, 0.0],
        ),
        (
            "cylinder",
            ColliderShape::cylinder_y(0.5, 0.5),
            [0.0, -0.5, 0.0],
        ),
        (
            "voxels",
            ColliderShape::VoxelGrid {
                cell_size: [1.0; 3],
                occupied: block_cells(),
            },
            [-1.5, -2.0, -1.5],
        ),
    ]
}

/// Query shapes with their bottom at y = gap.
fn query_shape(name: &str, gap: f32) -> (ColliderShape, Transform) {
    let at = |y: f32| Transform::from_translation([0.0, y, 0.0]);
    match name {
        "sphere" => (ColliderShape::sphere(0.25), at(gap + 0.25)),
        "box" => (ColliderShape::cuboid([0.25; 3]), at(gap + 0.25)),
        "capsule" => (ColliderShape::capsule_y(0.25, 0.125), at(gap + 0.375)),
        "cylinder" => (ColliderShape::cylinder_y(0.25, 0.25), at(gap + 0.25)),
        "voxels" => (
            ColliderShape::VoxelGrid {
                cell_size: [0.5; 3],
                occupied: vec![[0, 0, 0]],
            },
            Transform::from_translation([-0.25, gap, -0.25]),
        ),
        _ => unreachable!(),
    }
}

const QUERIES: [&str; 5] = ["sphere", "box", "capsule", "cylinder", "voxels"];

fn world_with(shape: ColliderShape, at: [f32; 3]) -> (BodyWorld, ColliderId) {
    let mut world = BodyWorld::new([0.0; 3]);
    let body = world
        .spawn(
            BodyDesc::fixed()
                .at(Transform::from_translation(at))
                .with_collider(ColliderDesc::new(shape)),
        )
        .unwrap();
    world.refresh_queries();
    (world, ColliderId::new(body, 0))
}

/// Every pair, at gaps either side of three predictions: found exactly when
/// the gap is within the prediction, every record within it, on the world
/// collider's top face and pointing up out of it.
#[test]
fn the_prediction_means_what_it_says_for_every_pair() {
    let edge = 1.0 / 512.0;
    let cases: [(f32, &[f32], &[f32]); 3] = [
        (0.0, &[-0.0625, 0.0], &[edge]),
        (
            0.125,
            &[-0.0625, 0.0, 0.0625, 0.125 - edge, 0.125],
            &[0.125 + edge, 0.25],
        ),
        // The gaps measured missing before compensation (M5, 2026-10-09).
        (0.1, &[0.04, 0.05, 0.09], &[0.11]),
    ];
    let mut checked = 0;
    for (world_name, world_shape, at) in world_shapes() {
        let (world, collider) = world_with(world_shape, at);
        for query_name in QUERIES {
            for (prediction, inside, outside) in cases {
                for (&gap, expected) in inside
                    .iter()
                    .map(|gap| (gap, true))
                    .chain(outside.iter().map(|gap| (gap, false)))
                {
                    let pair =
                        format!("{world_name} <- {query_name}, prediction {prediction}, gap {gap}");
                    let (shape, transform) = query_shape(query_name, gap);
                    let found = world.contacts(transform, &shape, prediction, ALL).unwrap();
                    assert_eq!(!found.is_empty(), expected, "{pair}: {found:?}");
                    for contact in &found {
                        assert_eq!(contact.collider, collider, "{pair}");
                        assert!(contact.distance <= prediction, "{pair}: {contact:?}");
                        assert!(
                            contact.point[1].abs() < 1e-3,
                            "{pair}: point off the top face {contact:?}"
                        );
                        assert!(
                            contact.normal[1] > 0.999,
                            "{pair}: normal not out of the collider {contact:?}"
                        );
                    }
                    if gap < 0.0 {
                        assert!(
                            found
                                .iter()
                                .all(|c| c.distance < 0.0 && c.distance > -0.0626),
                            "{pair}: {found:?}"
                        );
                    }
                    checked += 1;
                }
            }
        }
    }
    assert_eq!(checked, 25 * 14);
}

#[test]
fn the_probe_case_reports_one_point_on_the_voxel_top() {
    let (mut world, collider) = world_with(
        ColliderShape::VoxelGrid {
            cell_size: [1.0; 3],
            occupied: block_cells(),
        },
        [0.0; 3],
    );
    let ball = ColliderShape::sphere(0.2);
    let centre = Transform::from_translation([1.5, 1.9, 1.5]);
    let found = world.contacts(centre, &ball, 0.0, ALL).unwrap();
    assert_eq!(found.len(), 1, "{found:?}");
    let ShapeContact {
        collider: hit,
        point,
        normal,
        distance,
    } = found[0];
    assert_eq!(hit, collider);
    assert_eq!(point, [1.5, 2.0, 1.5]);
    assert_eq!(normal, [0.0, 1.0, 0.0]);
    assert!((distance + 0.3).abs() < 1e-6, "{distance}");

    world
        .edit_voxels(
            collider,
            [VoxelEdit {
                cell: [1, 1, 1],
                filled: false,
            }],
        )
        .unwrap();
    assert!(
        world.contacts(centre, &ball, 0.0, ALL).unwrap().is_empty(),
        "carved"
    );
}

#[test]
fn a_point_beyond_the_prediction_is_dropped() {
    // Parry returned a capsule point at +0.3 for prediction 0 (M5).
    let (world, _) = world_with(
        ColliderShape::VoxelGrid {
            cell_size: [1.0; 3],
            occupied: block_cells(),
        },
        [0.0; 3],
    );
    let capsule = ColliderShape::capsule_y(0.3, 0.2);
    let at = Transform::from_translation([1.5, 2.2, 1.5]);
    let at_zero = world.contacts(at, &capsule, 0.0, ALL).unwrap();
    assert!(!at_zero.is_empty());
    assert!(at_zero.iter().all(|c| c.distance <= 0.0), "{at_zero:?}");
    let control = world.contacts(at, &capsule, 1.0, ALL).unwrap();
    assert!(
        control.iter().any(|c| c.distance > 0.0),
        "control keeps it: {control:?}"
    );
}

#[test]
fn records_follow_a_moved_and_turned_collider() {
    let half = std::f32::consts::FRAC_1_SQRT_2;
    for (translation, rotation) in [
        ([4096.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
        ([0.0, 0.0, 0.0], [0.0, 0.0, half, half]),
        ([-300.0, 20.0, 77.0], [half, 0.0, 0.0, half]),
    ] {
        let mut world = BodyWorld::new([0.0; 3]);
        world
            .spawn(
                BodyDesc::fixed()
                    .at(Transform {
                        translation,
                        rotation,
                    })
                    .with_collider(ColliderDesc::new(ColliderShape::cuboid([0.5, 1.0, 0.5]))),
            )
            .unwrap();
        world.refresh_queries();
        // A rotated box's top is at 0.5 (quarter turn about z or x) or 1.0.
        let top = if rotation[3] == 1.0 { 1.0 } else { 0.5 };
        let [x, y, z] = translation;
        let ball = ColliderShape::sphere(0.25);
        let found = world
            .contacts(
                Transform::from_translation([x, y + top + 0.125, z]),
                &ball,
                0.0,
                ALL,
            )
            .unwrap();
        assert_eq!(found.len(), 1, "{found:?}");
        let contact = found[0];
        assert!((contact.point[0] - x).abs() < 1e-3 && (contact.point[1] - (y + top)).abs() < 1e-3);
        assert!(contact.normal[1] > 0.999, "{contact:?}");
        assert!((contact.distance + 0.125).abs() < 1e-3, "{contact:?}");
    }
}

#[test]
fn records_do_not_depend_on_how_the_grid_was_built() {
    // Steps of one to three cells across the grid's internal 8-cell chunk
    // boundaries, so a box pressed in meets several faces in several chunks.
    let mut steps = Vec::new();
    for x in 5..11 {
        for z in 5..11 {
            for y in 0..((x + z) % 3 + 1) {
                steps.push([x, y, z]);
            }
        }
    }
    let read = |cells: Vec<[i32; 3]>| {
        let (world, _) = world_with(
            ColliderShape::VoxelGrid {
                cell_size: [1.0; 3],
                occupied: cells,
            },
            [0.0; 3],
        );
        world
            .contacts(
                Transform::from_translation([8.0, 2.6, 8.0]),
                &ColliderShape::cuboid([2.6, 0.5, 2.6]),
                0.0,
                ALL,
            )
            .unwrap()
    };
    let forward = read(steps.clone());
    assert!(forward.len() > 4, "{forward:?}");
    steps.reverse();
    assert_eq!(forward, read(steps));
}

#[test]
fn queries_see_only_what_refresh_published() {
    let mut world = BodyWorld::new([0.0; 3]);
    world
        .spawn(BodyDesc::fixed().with_collider(ColliderDesc::new(ColliderShape::cuboid([0.5; 3]))))
        .unwrap();
    let ball = ColliderShape::sphere(0.25);
    let at = Transform::from_translation([0.0, 0.7, 0.0]);
    assert!(
        world.contacts(at, &ball, 0.0, ALL).unwrap().is_empty(),
        "unrefreshed"
    );
    world.refresh_queries();
    assert_eq!(world.contacts(at, &ball, 0.0, ALL).unwrap().len(), 1);
}

#[test]
fn contacts_filter_and_refuse_like_overlaps() {
    let (mut world, collider) = world_with(ColliderShape::cuboid([0.5; 3]), [0.0, -0.5, 0.0]);
    let ball = ColliderShape::sphere(0.25);
    let at = Transform::from_translation([0.0, 0.2, 0.0]);
    let body = collider.body();

    for prediction in [-0.1, f32::NAN, f32::INFINITY] {
        assert!(matches!(
            world.contacts(at, &ball, prediction, ALL),
            Err(BodyError::InvalidQuery(_))
        ));
    }
    let nan = Transform::from_translation([f32::NAN, 0.0, 0.0]);
    assert!(matches!(
        world.contacts(nan, &ball, 0.0, ALL),
        Err(BodyError::InvalidQuery(_))
    ));
    let flat = ColliderShape::sphere(0.0);
    assert!(matches!(
        world.contacts(at, &flat, 0.0, ALL),
        Err(BodyError::InvalidQuery(_))
    ));

    let neither = SpatialFilter {
        include_sensors: false,
        include_solids: false,
        ..ALL
    };
    assert!(world.contacts(at, &ball, 0.0, neither).unwrap().is_empty());
    let excluding = SpatialFilter {
        exclude_body: Some(body),
        ..ALL
    };
    assert!(
        world
            .contacts(at, &ball, 0.0, excluding)
            .unwrap()
            .is_empty()
    );
    assert_eq!(world.contacts(at, &ball, 0.0, ALL).unwrap().len(), 1);

    world.despawn(body).unwrap();
    assert_eq!(
        world.contacts(at, &ball, 0.0, excluding),
        Err(BodyError::UnknownBody(body))
    );
}
