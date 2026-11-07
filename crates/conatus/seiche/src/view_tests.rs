// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! The `LayoutView` parity tests: they build a real `Simulation`, so they stay
//! with the engine. `LayoutView` itself lives in `seiche-model`.

use euclid::default::{Box2D, Point2D};

use crate::{NodeKey, Simulation};

/// a@(0,0), b@(100,0), one edge a–b — built through a real `Simulation` so
/// the view-vs-sim parity assertions share one source of truth.
fn sim_with_edge() -> (Simulation, NodeKey, NodeKey) {
    let a = NodeKey::new(0);
    let b = NodeKey::new(1);
    let mut sim = Simulation::new();
    sim.sync_nodes([(a, Point2D::new(0.0, 0.0)), (b, Point2D::new(100.0, 0.0))]);
    sim.sync_edges([(a, b)]);
    sim.refresh_spatial_index();
    (sim, a, b)
}

#[test]
fn view_node_picks_and_cull_match_the_simulation() {
    let (sim, a, b) = sim_with_edge();
    let view = sim.view();

    // Node point-pick parity (well-separated, so rapier and the view agree).
    assert_eq!(
        view.hit_test(Point2D::new(0.0, 0.0)),
        sim.hit_test(Point2D::new(0.0, 0.0))
    );
    assert_eq!(view.hit_test(Point2D::new(0.0, 0.0)), Some(a));
    assert_eq!(view.hit_test(Point2D::new(100.0, 0.0)), Some(b));
    assert!(view.hit_test(Point2D::new(5000.0, 5000.0)).is_none());

    // Cull parity over a window around the origin (catches a, not b).
    let region = Box2D::new(Point2D::new(-30.0, -30.0), Point2D::new(30.0, 30.0));
    let mut from_view = view.cull_aabb(region);
    let mut from_sim = sim.cull_aabb(region);
    from_view.sort();
    from_sim.sort();
    assert_eq!(from_view, from_sim);
    assert_eq!(from_view, vec![a]);
}

#[test]
fn view_edge_picks_match_the_simulation() {
    let (sim, a, b) = sim_with_edge();
    let view = sim.view();

    assert_eq!(
        view.edge_hit_test(Point2D::new(50.0, 0.0), 5.0),
        Some((a, b))
    );
    assert_eq!(
        view.edge_hit_test(Point2D::new(50.0, 0.0), 5.0),
        sim.edge_hit_test(Point2D::new(50.0, 0.0), 5.0)
    );
    assert!(view.edge_hit_test(Point2D::new(50.0, 50.0), 5.0).is_none());

    let near_origin = view.rect_select(Box2D::new(
        Point2D::new(-10.0, -10.0),
        Point2D::new(10.0, 10.0),
    ));
    assert_eq!(near_origin.nodes, vec![a]);
    assert_eq!(near_origin.edges, vec![(a, b)]);
}

#[test]
fn snapshot_round_trips_and_drag_override_sticks() {
    let (sim, a, _b) = sim_with_edge();
    let snapshot = sim.snapshot(7);
    assert_eq!(snapshot.generation, 7);
    assert_eq!(snapshot.positions.len(), 2);

    let mut view = crate::LayoutView::new();
    view.apply_snapshot(&snapshot);
    assert_eq!(view.len(), 2);
    assert!((view.position_of(a).unwrap() - Point2D::new(0.0, 0.0)).length() < 0.5);

    // A local drag override moves the node without a new snapshot.
    view.set_position(a, Point2D::new(250.0, 250.0));
    assert_eq!(view.position_of(a), Some(Point2D::new(250.0, 250.0)));
    // ...and a later snapshot replaces it (the actor caught up).
    view.apply_snapshot(&snapshot);
    assert!((view.position_of(a).unwrap() - Point2D::new(0.0, 0.0)).length() < 0.5);
}

#[test]
fn per_node_radius_grows_the_pick_and_cull() {
    let (sim, a, b) = sim_with_edge(); // a@(0,0), b@(100,0)
    let mut view = sim.view();

    // A point 40px from a's center misses at the default radius (18)...
    let near_a = Point2D::new(40.0, 0.0);
    assert!(view.hit_test(near_a).is_none());

    // ...but grow a to radius 60 and the same point now grabs it.
    view.set_radii([(a, 60.0)]);
    assert_eq!(view.hit_test(near_a), Some(a));
    // b, left at the default, is unaffected — a point 30px off b (and well clear of
    // grown a) still misses.
    assert!(view.hit_test(Point2D::new(100.0, 30.0)).is_none());

    // Cull grows with the radius too: a tight window that excludes a's center's
    // default circle still catches it once a is large enough to reach in.
    let window = Box2D::new(Point2D::new(45.0, -10.0), Point2D::new(55.0, 10.0));
    assert!(view.cull_aabb(window).contains(&a));

    // Clearing a's override (the new set names only b) reverts a to the default:
    // a point 40px straight up off a misses again (and grown b, off on the x-axis,
    // can't reach it).
    view.set_radii([(b, 60.0)]);
    assert!(view.hit_test(Point2D::new(0.0, 40.0)).is_none());
}
