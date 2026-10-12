// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! A kernel-free point probe followed by the real integrator. The evaluation
//! proposes a write; the simulation's subsequent tick is separate evidence.
use euclid::default::Point2D;
use numen::{CouplingResponse, FieldDefinition, ScalarField, VectorField};
use seiche::{CouplingEffect, CouplingForce, NodeKey, Simulation};

fn main() {
    let node = NodeKey::new(0);
    for (name, response, field) in [
        (
            "wall",
            CouplingResponse::ContainmentWall,
            FieldDefinition::Scalar(ScalarField::CoordX),
        ),
        (
            "damping",
            CouplingResponse::DampenInside { factor: 0.5 },
            FieldDefinition::Scalar(ScalarField::CoordX),
        ),
        (
            "alignment",
            CouplingResponse::AlignVelocity,
            FieldDefinition::Vector(VectorField::ConstVec { x: 20.0, y: -10.0 }),
        ),
        (
            "advection",
            CouplingResponse::FlowAdvect,
            FieldDefinition::Vector(VectorField::ConstVec { x: 20.0, y: -10.0 }),
        ),
    ] {
        let mut simulation = Simulation::new();
        simulation.sync_nodes([(node, Point2D::new(10.0, 5.0))]);
        // Seed a moving body through the normal coupling path, so damping's
        // actual result is observable rather than scaling an already-zero speed.
        simulation.set_coupling_forces(vec![CouplingForce::new(
            CouplingResponse::AlignVelocity,
            1.0,
            [node],
            FieldDefinition::Vector(VectorField::ConstVec { x: 20.0, y: -10.0 }),
        )]);
        simulation.tick(1.0 / 60.0);
        let before = simulation.position_of(node).unwrap();
        let force = CouplingForce::new(response, 2.0, [node], field);
        let evaluation = force.evaluate_at(node, [before.x, before.y], true, 1.0 / 60.0);
        assert!(!matches!(evaluation.effect, CouplingEffect::Inert(_)));
        println!(
            "{name}: configured sample time {}; proposed {:?}",
            evaluation.sample_time, evaluation.effect
        );
        println!(
            "{name}: before position {before:?}, velocity {:?}",
            simulation.velocity_of(node).unwrap()
        );
        simulation.set_coupling_forces(vec![force]);
        for _ in 0..60 {
            simulation.tick(1.0 / 60.0);
        }
        println!(
            "{name}: after 60 actual ticks position {:?}, velocity {:?}",
            simulation.position_of(node).unwrap(),
            simulation.velocity_of(node).unwrap()
        );
    }
}
