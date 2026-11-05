// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

use super::*;
use crate::{ColliderDesc, ColliderShape};

fn ball(position: [f32; 3]) -> BodyDesc {
    BodyDesc::dynamic()
        .at(Transform::from_translation(position))
        .with_collider(ColliderDesc::new(ColliderShape::sphere(0.5)))
}

#[test]
fn ids_are_stable_and_stale_ids_do_not_alias_replacements() {
    let mut world = BodyWorld::default();
    let first = world.spawn(ball([0.0, 2.0, 0.0])).unwrap();
    world.despawn(first).unwrap();
    let replacement = world.spawn(ball([0.0, 2.0, 0.0])).unwrap();

    assert_eq!(first.slot(), replacement.slot());
    assert_ne!(first.generation(), replacement.generation());
    assert!(!world.contains(first));
    assert!(world.contains(replacement));
}

#[test]
fn a_dynamic_body_falls_and_reports_changed_state() {
    let mut world = BodyWorld::default();
    let body = world.spawn(ball([0.0, 2.0, 0.0])).unwrap();
    let update = world.step(1.0 / 60.0).unwrap();

    assert_eq!(update.changed.len(), 1);
    assert_eq!(update.changed[0].id, body);
    assert!(update.changed[0].transform.translation[1] < 2.0);
}

#[test]
fn raycast_returns_conatus_collider_identity() {
    let mut world = BodyWorld::new([0.0; 3]);
    let body = world
        .spawn(
            BodyDesc::fixed()
                .with_collider(ColliderDesc::new(ColliderShape::cuboid([1.0, 1.0, 1.0]))),
        )
        .unwrap();
    world.step(1.0 / 60.0).unwrap();

    let hit = world
        .raycast(
            [0.0, 4.0, 0.0],
            [0.0, -1.0, 0.0],
            10.0,
            true,
            SpatialFilter::default(),
        )
        .unwrap()
        .unwrap();
    assert_eq!(hit.collider, ColliderId::new(body, 0));
    assert!((hit.distance - 3.0).abs() < 0.001);
    assert_eq!(hit.normal, [0.0, 1.0, 0.0]);
}

#[test]
fn sensor_entries_and_exits_are_frame_events() {
    let mut world = BodyWorld::new([0.0; 3]);
    let sensor =
        world
            .spawn(BodyDesc::fixed().with_collider(
                ColliderDesc::new(ColliderShape::cuboid([1.0, 1.0, 1.0])).sensor(true),
            ))
            .unwrap();
    let mover = world.spawn(ball([0.0, 0.0, 0.0])).unwrap();
    let entered = world.step(1.0 / 60.0).unwrap();
    assert!(entered.interactions.iter().any(|event| {
        event.state == InteractionState::Started
            && event.interaction == Interaction::Sensor
            && [event.a.body(), event.b.body()].contains(&sensor)
            && [event.a.body(), event.b.body()].contains(&mover)
    }));

    world
        .set_transform(mover, Transform::from_translation([10.0, 0.0, 0.0]), true)
        .unwrap();
    let exited = world.step(1.0 / 60.0).unwrap();
    assert!(exited.interactions.iter().any(|event| {
        event.state == InteractionState::Stopped && event.interaction == Interaction::Sensor
    }));
}

#[test]
fn voxel_colliders_edit_in_place_and_revise() {
    let mut world = BodyWorld::new([0.0; 3]);
    let body = world
        .spawn(
            BodyDesc::fixed().with_collider(ColliderDesc::new(ColliderShape::VoxelGrid {
                cell_size: [1.0; 3],
                occupied: vec![[0, 0, 0]],
            })),
        )
        .unwrap();
    let collider = ColliderId::new(body, 0);
    let before = world.collider_revision(collider).unwrap();
    let edited = world
        .edit_voxels(
            collider,
            [
                VoxelEdit {
                    cell: [0, 0, 0],
                    filled: false,
                },
                VoxelEdit {
                    cell: [1, 0, 0],
                    filled: true,
                },
            ],
        )
        .unwrap();

    assert_eq!(edited.previous_revision, before);
    assert_eq!(edited.changed, 2);
    assert!(edited.revision > before);
    assert_eq!(world.collider_revision(collider).unwrap(), edited.revision);
}

#[test]
fn character_movement_stops_at_world_collision() {
    let mut world = BodyWorld::new([0.0; 3]);
    let wall = world
        .spawn(
            BodyDesc::fixed()
                .at(Transform::from_translation([2.0, 0.0, 0.0]))
                .with_collider(ColliderDesc::new(ColliderShape::cuboid([0.5, 2.0, 2.0]))),
        )
        .unwrap();
    let character = world
        .spawn(
            BodyDesc::kinematic_position()
                .with_collider(ColliderDesc::new(ColliderShape::sphere(0.5))),
        )
        .unwrap();
    world.step(1.0 / 60.0).unwrap();

    let movement = world
        .move_character(
            ColliderId::new(character, 0),
            [5.0, 0.0, 0.0],
            1.0 / 60.0,
            CharacterConfig {
                snap_to_ground: None,
                ..CharacterConfig::default()
            },
        )
        .unwrap();
    assert!(movement.applied[0] < 1.1, "applied {:?}", movement.applied);
    assert!(
        movement
            .collisions
            .iter()
            .any(|collision| collision.collider.body() == wall)
    );

    world.step(1.0 / 60.0).unwrap();
    let x = world.state(character).unwrap().transform.translation[0];
    assert!((x - movement.applied[0]).abs() < 0.001);
}
