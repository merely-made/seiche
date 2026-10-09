// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! `BodyWorld::refresh_queries` (ruling 352). Each case runs beside the same
//! case without the refresh, its control.

use conatus::{
    BodyDesc, BodyId, BodyState, BodyWorld, ColliderDesc, ColliderId, ColliderShape, SpatialFilter,
    Transform, VoxelEdit,
};

const DT: f32 = 1.0 / 60.0;

fn ray_down(world: &BodyWorld, x: f32, z: f32) -> Option<f32> {
    world
        .raycast(
            [x, 30.0, z],
            [0.0, -1.0, 0.0],
            100.0,
            true,
            SpatialFilter::default(),
        )
        .unwrap()
        .map(|hit| hit.distance)
}

fn falling_ball(world: &mut BodyWorld) -> BodyId {
    world
        .spawn(
            BodyDesc::dynamic()
                .at(Transform::from_translation([0.0, 2.0, 0.0]))
                .with_collider(ColliderDesc::new(ColliderShape::sphere(0.5))),
        )
        .unwrap()
}

fn trajectory(world: &mut BodyWorld, body: BodyId, steps: usize) -> Vec<BodyState> {
    (0..steps)
        .map(|_| {
            world.step(DT).unwrap();
            world.state(body).unwrap()
        })
        .collect()
}

#[test]
fn a_body_refreshed_straight_after_insert_still_simulates() {
    let mut control = BodyWorld::default();
    let control_ball = falling_ball(&mut control);
    assert_eq!(
        ray_down(&control, 0.0, 0.0),
        None,
        "unrefreshed spawn is invisible"
    );

    let mut world = BodyWorld::default();
    let ball = falling_ball(&mut world);
    world.refresh_queries();
    assert_eq!(
        ray_down(&world, 0.0, 0.0),
        Some(27.5),
        "refresh makes the spawn visible"
    );

    let expected = trajectory(&mut control, control_ball, 10);
    let refreshed = trajectory(&mut world, ball, 10);
    assert_eq!(
        refreshed, expected,
        "refresh leaves the simulation bit-identical"
    );
    let y = refreshed[9].transform.translation[1];
    assert!(
        (y - 1.8603).abs() < 1e-4,
        "the ball falls to 1.8603, got {y}"
    );
}

#[test]
fn refresh_neither_ticks_nor_emits_events() {
    let mut world = BodyWorld::default();
    falling_ball(&mut world);
    world.step(DT).unwrap();
    let (tick, revision) = (world.tick(), world.revision());
    falling_ball(&mut world);
    let revision_after_spawn = world.revision();
    world.refresh_queries();
    assert_eq!(world.tick(), tick);
    assert_eq!(world.revision(), revision_after_spawn);
    assert!(revision_after_spawn > revision);
    assert!(world.drain_events().is_empty());
}

#[test]
fn a_teleport_is_visible_after_refresh() {
    let spawn_and_move = || {
        let mut world = BodyWorld::new([0.0; 3]);
        let body = world
            .spawn(
                BodyDesc::fixed().with_collider(ColliderDesc::new(ColliderShape::cuboid([1.0; 3]))),
            )
            .unwrap();
        world.step(DT).unwrap();
        world
            .set_transform(body, Transform::from_translation([10.0, 0.0, 0.0]), true)
            .unwrap();
        world
    };

    let control = spawn_and_move();
    assert_eq!(
        ray_down(&control, 0.0, 0.0),
        Some(29.0),
        "control answers at the old place"
    );
    assert_eq!(
        ray_down(&control, 10.0, 0.0),
        None,
        "control misses the new place"
    );

    let mut world = spawn_and_move();
    world.refresh_queries();
    assert_eq!(ray_down(&world, 0.0, 0.0), None);
    assert_eq!(ray_down(&world, 10.0, 0.0), Some(29.0));
}

#[test]
fn a_teleported_dynamic_body_simulates_as_its_control() {
    let run = |refresh: bool| {
        let mut world = BodyWorld::default();
        let ball = falling_ball(&mut world);
        world.step(DT).unwrap();
        world
            .set_transform(ball, Transform::from_translation([3.0, 5.0, 0.0]), true)
            .unwrap();
        if refresh {
            world.refresh_queries();
        }
        trajectory(&mut world, ball, 10)
    };
    assert_eq!(run(true), run(false));
}

#[test]
fn a_voxel_added_outside_the_grid_bounds_is_visible_after_refresh() {
    let grown = || {
        let mut world = BodyWorld::new([0.0; 3]);
        let body = world
            .spawn(
                BodyDesc::fixed().with_collider(ColliderDesc::new(ColliderShape::VoxelGrid {
                    cell_size: [1.0; 3],
                    occupied: vec![[0, 0, 0], [1, 0, 0]],
                })),
            )
            .unwrap();
        world.step(DT).unwrap();
        world
            .edit_voxels(
                ColliderId::new(body, 0),
                [VoxelEdit {
                    cell: [40, 0, 40],
                    filled: true,
                }],
            )
            .unwrap();
        world
    };

    let control = grown();
    assert_eq!(
        ray_down(&control, 40.5, 40.5),
        None,
        "control cannot see the growth"
    );

    let mut world = grown();
    world.refresh_queries();
    assert_eq!(ray_down(&world, 40.5, 40.5), Some(29.0));
    assert_eq!(
        ray_down(&world, 0.5, 0.5),
        Some(29.0),
        "the old cells still answer"
    );
}

#[test]
fn a_despawned_body_needs_no_refresh() {
    let mut world = BodyWorld::new([0.0; 3]);
    let body = world
        .spawn(BodyDesc::fixed().with_collider(ColliderDesc::new(ColliderShape::cuboid([1.0; 3]))))
        .unwrap();
    world.refresh_queries();
    assert_eq!(ray_down(&world, 0.0, 0.0), Some(29.0));
    world.despawn(body).unwrap();
    assert_eq!(ray_down(&world, 0.0, 0.0), None);
    world.refresh_queries();
    assert_eq!(ray_down(&world, 0.0, 0.0), None);
}
