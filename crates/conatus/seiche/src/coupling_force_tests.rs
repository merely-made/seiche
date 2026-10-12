// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

use super::*;
use crate::{Boundary, NodeKey, Simulation};
use euclid::default::Point2D;

/// Kernel-free node-list builder (seiche drives physics from `(NodeKey,
/// position)` pairs).
#[derive(Default)]
struct Nodes {
    list: Vec<(NodeKey, Point2D<f32>)>,
}

impl Nodes {
    fn at(&mut self, x: f32, y: f32) -> NodeKey {
        let key = NodeKey::new(self.list.len());
        self.list.push((key, Point2D::new(x, y)));
        key
    }
    fn sync(&self, sim: &mut Simulation) {
        sim.sync_nodes(self.list.iter().copied());
    }
}

fn radius(sim: &Simulation, k: NodeKey) -> f32 {
    sim.position_of(k).unwrap().to_vector().length()
}

/// `½(x² + y²)` — gradient `(x, y)`, the potential whose `AttractToMin`
/// coupling reproduces `Boundary`.
pub(super) fn paraboloid() -> ScalarField {
    ScalarField::Scale(
        Box::new(ScalarField::Add(
            Box::new(ScalarField::Mul(
                Box::new(ScalarField::CoordX),
                Box::new(ScalarField::CoordX),
            )),
            Box::new(ScalarField::Mul(
                Box::new(ScalarField::CoordY),
                Box::new(ScalarField::CoordY),
            )),
        )),
        0.5,
    )
}

#[test]
fn attract_to_min_centers_like_boundary() {
    // Two sims, same seed: Boundary vs an AttractToMin coupling on the
    // paraboloid. The paraboloid's gradient is (x, y), so the coupling force
    // is -pos·strength — Boundary exactly, up to numen's finite-difference
    // step. The bodies track each other across the whole settle (a ~580-unit
    // journey leaves them within a couple of units).
    let mut g = Nodes::default();
    let a = g.at(500.0, -300.0);

    let mut boundary_sim = Simulation::new();
    g.sync(&mut boundary_sim);
    boundary_sim.add_force(Boundary { strength: 1.5 });

    let mut coupling_sim = Simulation::new();
    g.sync(&mut coupling_sim);
    coupling_sim.add_force(CouplingForce::new(
        CouplingResponse::AttractToMin,
        1.5,
        [a],
        FieldDefinition::Scalar(paraboloid()),
    ));

    for _ in 0..120 {
        boundary_sim.tick(1.0 / 60.0);
        coupling_sim.tick(1.0 / 60.0);
    }

    let pb = boundary_sim.position_of(a).unwrap();
    let pc = coupling_sim.position_of(a).unwrap();
    assert!(
        radius(&coupling_sim, a) < 500.0,
        "coupling should pull inward"
    );
    assert!(
        (pb - pc).length() < 5.0,
        "coupling diverged from Boundary: {pb:?} vs {pc:?}"
    );
}

#[test]
fn open_response_applies_no_force() {
    // An open-tail response (e.g. a visual coupling) is not a force: the
    // integrator must leave the body where it is, even on a field whose
    // AttractToMin response would otherwise pull it to the origin.
    let mut g = Nodes::default();
    let a = g.at(300.0, 0.0);

    let mut sim = Simulation::new();
    g.sync(&mut sim);
    sim.add_force(CouplingForce::new(
        CouplingResponse::open("https://mere.computer/ns/coupling#visual/highlight".to_string()),
        1.0,
        [a],
        FieldDefinition::Scalar(paraboloid()),
    ));
    let p0 = sim.position_of(a).unwrap();
    for _ in 0..120 {
        sim.tick(1.0 / 60.0);
    }
    assert!(
        (sim.position_of(a).unwrap() - p0).length() < 1.0e-3,
        "an open-tail response must apply no force"
    );
}

#[test]
fn flow_advect_leaves_a_pinned_body_at_its_kinematic_target() {
    // FlowAdvect writes positions; a pinned (kinematic) body belongs to
    // the drag, so it stays on its target while a free one is advected.
    let mut g = Nodes::default();
    let pinned = g.at(0.0, 0.0);
    let free = g.at(100.0, 0.0);
    let mut sim = Simulation::new();
    g.sync(&mut sim);
    sim.add_force(CouplingForce::new(
        CouplingResponse::FlowAdvect,
        1.0,
        [pinned, free],
        FieldDefinition::Vector(VectorField::ConstVec { x: 120.0, y: 0.0 }),
    ));
    let target = Point2D::new(-200.0, 50.0);
    sim.pin(pinned, target);
    let free_start = sim.position_of(free).unwrap();
    for _ in 0..60 {
        sim.tick(1.0 / 60.0);
    }
    let held = sim.position_of(pinned).unwrap();
    assert!(
        (held - target).length() < 1e-3,
        "the pinned body left its target: {held:?}"
    );
    assert!(
        sim.position_of(free).unwrap().x > free_start.x + 10.0,
        "the free body is still advected"
    );
}
