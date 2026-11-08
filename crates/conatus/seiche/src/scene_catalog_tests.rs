// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

use crate::NodeKey;
use euclid::default::Point2D;

use crate::*;

/// Every catalog scene loads its bodies (under the cap) without disturbing a node, and reports
/// the perpetual flag the spec declared.
#[test]
fn catalog_scenes_load_and_report_perpetual() {
    let cases: [(SceneSpec, bool); 11] = [
        (drop_bowl_scene(), false),
        (pyramid_scene(), false),
        (domino_scene(), false),
        (galton_scene(), false),
        (funnel_scene(), false),
        (drift_scene(), true),
        (chain_scene(), false),
        (cradle_scene(), true),
        (bridge_scene(), false),
        (ball_and_chain_scene(), false),
        (mixer_scene(), true),
    ];
    for (spec, perpetual) in cases {
        let want = spec.bodies.len();
        let mut sim = Simulation::new();
        let node = NodeKey::new(0);
        sim.sync_nodes([(node, Point2D::new(0.0, 0.0))]);
        sim.load_scene(&spec);
        assert!(
            want <= 200,
            "scene must stay under SCENE_BODY_CAP (was {want})"
        );
        assert_eq!(
            sim.scene_body_count(),
            want,
            "every spec body became a scene body"
        );
        assert_eq!(
            sim.body_count(),
            1,
            "the node is not counted as a scene body"
        );
        assert_eq!(
            sim.scene_perpetual(),
            perpetual,
            "scene reports its perpetual flag"
        );
    }
}

/// A settling scene comes to rest (its dynamic bodies stop moving); a perpetual scene keeps
/// drifting. Position deltas over a window stand in for "still moving".
#[test]
fn perpetual_drifts_while_settling_scene_comes_to_rest() {
    // Pyramid: settles. Sample positions, run, then confirm a later window is near-still.
    let mut settle = Simulation::new();
    settle.load_scene(&pyramid_scene());
    for _ in 0..600 {
        settle.tick(1.0 / 60.0);
    }
    let a: Vec<_> = settle.scene_bodies().map(|b| b.position).collect();
    for _ in 0..30 {
        settle.tick(1.0 / 60.0);
    }
    let b: Vec<_> = settle.scene_bodies().map(|b| b.position).collect();
    let settled_motion: f32 = a.iter().zip(&b).map(|(p, q)| (*p - *q).length()).sum();
    assert!(
        settled_motion < 5.0,
        "pyramid should be at rest (moved {settled_motion})"
    );

    // Drift: still moving after the same long run (perpetual + near-zero damping).
    let mut drift = Simulation::new();
    drift.load_scene(&drift_scene());
    for _ in 0..600 {
        drift.tick(1.0 / 60.0);
    }
    let a: Vec<_> = drift.scene_bodies().map(|b| b.position).collect();
    for _ in 0..30 {
        drift.tick(1.0 / 60.0);
    }
    let b: Vec<_> = drift.scene_bodies().map(|b| b.position).collect();
    let drift_motion: f32 = a.iter().zip(&b).map(|(p, q)| (*p - *q).length()).sum();
    assert!(
        drift_motion > 5.0,
        "drift orbs should still be moving (moved {drift_motion})"
    );
}

/// The rope joints anchor the chain: instead of the eight links falling away forever under
/// gravity, they hang ~chain-length below the fixed anchor. (Without joints the links would be
/// thousands of px down after this run.)
#[test]
fn chain_joints_anchor_the_links() {
    let mut sim = Simulation::new();
    sim.load_scene(&chain_scene());
    for _ in 0..300 {
        sim.tick(1.0 / 60.0);
    }
    let max_y = sim
        .scene_bodies()
        .map(|b| b.position.y)
        .fold(f32::NEG_INFINITY, f32::max);
    assert!(
        max_y < 500.0,
        "rope joints should anchor the chain, not let it fall away (max_y {max_y})"
    );
    // ...and it hangs below the anchor (it swung down from the horizontal start).
    assert!(
        max_y > 0.0,
        "the chain should hang below the anchor (max_y {max_y})"
    );
}

/// The mixer's revolute motor actually turns the paddle: its rotation advances over a run
/// (proving the motor drives it, not just gravity / contacts). `scene_bodies` walks a HashMap so
/// it is unordered; the paddle is the scene's only Hull body, so pick it by shape.
#[test]
fn mixer_motor_spins_the_paddle() {
    let mut sim = Simulation::new();
    sim.load_scene(&mixer_scene());
    let paddle_rot = |sim: &Simulation| {
        sim.scene_bodies()
            .find(|b| matches!(b.collider, crate::NodeCollider::Hull { .. }))
            .map(|b| b.rotation)
            .expect("the mixer has a paddle")
    };
    let start = paddle_rot(&sim);
    for _ in 0..120 {
        sim.tick(1.0 / 60.0);
    }
    let after = paddle_rot(&sim);
    assert!(
        (after - start).abs() > 0.5,
        "the motor should have rotated the paddle (start {start}, after {after})"
    );
}

/// The bridge bows under load: the planks are the Hull bodies, and the loaded span sags so the
/// lowest plank ends up well below the deck line (y=0) it started on, while the end posts keep it
/// from dropping away. (A jointless plank row would just fall.)
#[test]
fn bridge_sags_under_the_dropped_weight() {
    let mut sim = Simulation::new();
    sim.load_scene(&bridge_scene());
    for _ in 0..240 {
        sim.tick(1.0 / 60.0);
    }
    let lowest_plank_y = sim
        .scene_bodies()
        .filter(|b| matches!(b.collider, crate::NodeCollider::Hull { .. }))
        .map(|b| b.position.y)
        .fold(f32::NEG_INFINITY, f32::max);
    assert!(
        lowest_plank_y > 20.0,
        "the loaded bridge should sag below the deck (was {lowest_plank_y})"
    );
    assert!(
        lowest_plank_y < 600.0,
        "the posts should still hold the span, not drop it (was {lowest_plank_y})"
    );
}
