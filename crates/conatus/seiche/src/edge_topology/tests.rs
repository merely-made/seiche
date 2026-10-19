// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

use euclid::default::{Box2D, Point2D};

use super::*;
use crate::{Force, ForceContext, Simulation};

fn rows() -> Vec<(NodeKey, NodeKey)> {
    let [a, b, c, missing] = [0, 1, 2, 99].map(NodeKey::new);
    vec![
        (a, b),
        (a, b),
        (a, b),
        (b, a),
        (b, c),
        (b, c),
        (a, b),
        (a, a),
        (a, missing),
    ]
}

#[test]
fn runs_preserve_direction_interleaving_self_rows_and_multiplicity() {
    let original = rows();
    let topology: EdgeTopology = original.iter().copied().collect();
    assert_eq!(topology.len(), 9);
    assert_eq!(topology.run_count(), 6);
    assert_eq!(
        topology
            .runs()
            .iter()
            .map(EdgeRun::count)
            .collect::<Vec<_>>(),
        [3, 1, 2, 1, 1, 1]
    );
    assert_eq!(topology.iter().copied().collect::<Vec<_>>(), original);
    let mut iter = topology.iter();
    for (i, row) in original.iter().enumerate() {
        assert_eq!(iter.len(), original.len() - i);
        assert_eq!(iter.next(), Some(row));
    }
    assert_eq!(iter.len(), 0);
    assert_eq!(iter.next(), None);
    assert_eq!(iter.next(), None);
    assert!(EdgeTopology::default().is_empty());
}

fn world() -> Simulation {
    let mut sim = Simulation::new();
    sim.sync_nodes([
        (NodeKey::new(0), Point2D::new(-195.25, 33.5)),
        (NodeKey::new(1), Point2D::new(145.75, -22.25)),
        (NodeKey::new(2), Point2D::new(21.5, 216.75)),
    ]);
    sim
}

/// Independent input control: force receives the original uncompressed fixture
/// slice even though its simulation has no synced edges. This detects omitted,
/// reordered, deduplicated and reversed rows at the compatibility boundary.
struct OriginalRowsForce {
    inner: Box<dyn Force>,
    rows: Vec<(NodeKey, NodeKey)>,
}

impl crate::Declared for OriginalRowsForce {
    fn terms(&self) -> Vec<crate::Term> {
        self.inner.terms()
    }
}

impl Force for OriginalRowsForce {
    fn apply(&self, ctx: &mut ForceContext<'_>, dt: f32) {
        let mut original = ForceContext {
            bodies: &mut *ctx.bodies,
            colliders: ctx.colliders,
            joints: &mut *ctx.joints,
            bodies_by_node: ctx.bodies_by_node,
            edges: &self.rows,
            repulsion: None,
            step: ctx.step,
            field_clock: ctx.field_clock,
        };
        self.inner.apply(&mut original, dt);
    }
}

fn state(sim: &Simulation) -> Vec<[u32; 6]> {
    (0..3)
        .map(|i| {
            let body = &sim.bodies[sim.body_for(NodeKey::new(i)).unwrap()];
            let p = body.translation();
            let v = body.linvel();
            let f = body.user_force();
            [p.x, p.y, v.x, v.y, f.x, f.y].map(f32::to_bits)
        })
        .collect()
}

fn parity(make: impl Fn() -> Box<dyn Force>) {
    let mut counted = world();
    counted.sync_edge_topology(rows().into_iter().collect());
    counted.set_forces(vec![make()]);
    let mut original = world();
    original.set_forces(vec![Box::new(OriginalRowsForce {
        inner: make(),
        rows: rows(),
    })]);
    for tick in 0..120 {
        counted.tick(crate::TICK_DT);
        original.tick(crate::TICK_DT);
        assert_eq!(
            state(&counted),
            state(&original),
            "bitwise force/motion parity at tick {tick}"
        );
    }
    assert_eq!(counted.edge_count(), 9);
    assert_eq!(counted.edge_topology().run_count(), 6);
}

#[test]
fn spring_force_and_motion_match_original_rows_bit_for_bit() {
    parity(|| Box::new(crate::EdgeSpring::default()));
}

#[test]
fn linlog_degree_and_attraction_match_original_rows_bit_for_bit() {
    parity(|| Box::new(crate::LinLogForce::default()));
}

#[test]
fn magnetic_direction_matches_original_rows_bit_for_bit() {
    parity(|| Box::new(crate::MagneticSpring::default()));
}

#[test]
fn boids_duplicate_neighbors_match_original_rows_bit_for_bit() {
    parity(|| Box::new(crate::Boids::default()));
}

#[test]
fn anneal_ordered_neighbors_match_original_rows_bit_for_bit() {
    parity(|| Box::new(crate::Anneal::seeded(87)));
}

#[test]
fn kuramoto_degrees_and_phase_updates_match_original_rows_bit_for_bit() {
    parity(|| Box::new(crate::Kuramoto::new([])));
}

#[test]
fn degree_overlays_match_original_rows_bit_for_bit() {
    parity(|| Box::new(crate::overlays::DegreeRepulsion::default()));
    parity(|| Box::new(crate::overlays::HubGravity::default()));
}

#[test]
fn counted_queries_preserve_rows_missing_endpoints_and_pick_order() {
    let mut sim = world();
    sim.sync_edge_topology(rows().into_iter().collect());
    let expected: Vec<_> = rows()
        .into_iter()
        .filter(|(a, b)| sim.position_of(*a).is_some() && sim.position_of(*b).is_some())
        .collect();
    let view = sim.view();
    assert_eq!(
        sim.edge_segments()
            .map(|(a, b, _, _)| (a, b))
            .collect::<Vec<_>>(),
        expected
    );
    assert_eq!(
        view.edge_segments()
            .map(|(a, b, _, _)| (a, b))
            .collect::<Vec<_>>(),
        expected
    );
    let a = sim.position_of(NodeKey::new(0)).unwrap();
    let b = sim.position_of(NodeKey::new(1)).unwrap();
    let midpoint = Point2D::new((a.x + b.x) / 2.0, (a.y + b.y) / 2.0);
    assert_eq!(
        sim.edge_hit_test(midpoint, 0.01),
        Some((NodeKey::new(0), NodeKey::new(1)))
    );
    assert_eq!(
        view.edge_hit_test(midpoint, 0.01),
        sim.edge_hit_test(midpoint, 0.01)
    );
    let region = Box2D::new(Point2D::new(-1000.0, -1000.0), Point2D::new(1000.0, 1000.0));
    assert_eq!(sim.rect_select(region).edges, expected);
    assert_eq!(view.rect_select(region).edges, expected);
}
