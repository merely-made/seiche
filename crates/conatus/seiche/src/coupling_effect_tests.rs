// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

use super::*;
use crate::Simulation;
use euclid::default::Point2D;
use std::collections::BTreeMap;

fn trajectory(name: &str, response: CouplingResponse, field: FieldDefinition) -> u64 {
    let keys = [NodeKey::new(0), NodeKey::new(1), NodeKey::new(2)];
    let mut sim = Simulation::new();
    sim.sync_nodes([
        (keys[0], Point2D::new(40.0, -30.0)),
        (keys[1], Point2D::new(80.0, 45.0)),
        (keys[2], Point2D::new(700.0, 900.0)),
    ]);
    for key in keys {
        let handle = sim.bodies_by_node[&key];
        sim.bodies
            .get_mut(handle)
            .unwrap()
            .set_linvel(Vector::new(20.0, -7.0), true);
    }
    sim.pin(keys[0], Point2D::new(40.0, -30.0));
    // The third body is deliberately outside the captured target set.
    let targets = if name.starts_with("duplicate") {
        vec![keys[0], keys[1], keys[1]]
    } else {
        vec![keys[0], keys[1]]
    };
    sim.add_force(CouplingForce::new(response, 0.75, targets, field));
    let mut hash = 0xcbf29ce484222325_u64;
    for tick in 0..120 {
        sim.tick(if tick < 60 { 1.0 / 60.0 } else { 1.0 / 120.0 });
        for key in keys {
            let position = sim.position_of(key).unwrap();
            let velocity = sim.velocity_of(key).unwrap();
            for value in [position.x, position.y, velocity.x, velocity.y] {
                assert!(
                    value.is_finite(),
                    "{name}: nonfinite trajectory at tick {tick}"
                );
                for byte in value.to_bits().to_le_bytes() {
                    hash = (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3);
                }
            }
        }
    }
    hash
}

/// A black-box receipt over the complete trajectory, captured on the original
/// executor before the effect evaluation refactor. This detects altered float
/// operations, duplicate-target behavior, pin handling and integration order.
#[test]
fn execution_trajectory_receipt() {
    let scalar = || FieldDefinition::Scalar(super::tests::paraboloid());
    let positive = || FieldDefinition::Scalar(ScalarField::CoordX);
    let vector = || FieldDefinition::Vector(VectorField::ConstVec { x: 15.0, y: -12.0 });
    let fixtures = [
        ("attract", CouplingResponse::AttractToMin, scalar()),
        ("repel", CouplingResponse::RepelFromMax, scalar()),
        ("wall", CouplingResponse::ContainmentWall, positive()),
        (
            "dampen",
            CouplingResponse::DampenInside { factor: 0.85 },
            positive(),
        ),
        ("align", CouplingResponse::AlignVelocity, vector()),
        ("advect", CouplingResponse::FlowAdvect, vector()),
        (
            "open",
            CouplingResponse::open("https://mere.computer/ns/coupling#visual/highlight"),
            scalar(),
        ),
        ("wrong-scalar", CouplingResponse::AttractToMin, vector()),
        ("wrong-vector", CouplingResponse::AlignVelocity, scalar()),
        ("duplicate-advect", CouplingResponse::FlowAdvect, vector()),
        (
            "duplicate-attract",
            CouplingResponse::AttractToMin,
            scalar(),
        ),
        (
            "duplicate-dampen",
            CouplingResponse::DampenInside { factor: 0.85 },
            positive(),
        ),
        (
            "outside-dampen",
            CouplingResponse::DampenInside { factor: 0.85 },
            FieldDefinition::Scalar(ScalarField::Negate(Box::new(ScalarField::CoordX))),
        ),
    ];
    let measured: BTreeMap<_, _> = fixtures
        .into_iter()
        .map(|(name, response, field)| (name, trajectory(name, response, field)))
        .collect();
    for (name, digest) in &measured {
        println!("coupling_trajectory {name} {digest:016x}");
    }
    assert_eq!(
        measured,
        BTreeMap::from([
            ("advect", 0x54b5cd56e7b80b83),
            ("align", 0xbf3c47d11ff264bf),
            ("attract", 0xc717ab2701156915),
            ("dampen", 0x26b550ad4cdbd84f),
            ("duplicate-advect", 0x1798dfbd94938d15),
            ("duplicate-attract", 0xcac4854c803f022f),
            ("duplicate-dampen", 0x4ba4c04b15203996),
            ("open", 0x22f06774732f2065),
            ("outside-dampen", 0x22f06774732f2065),
            ("repel", 0x79743f9bd576ab4e),
            ("wall", 0x5978bb7ef06dc2bf),
            ("wrong-scalar", 0x22f06774732f2065),
            ("wrong-vector", 0x22f06774732f2065),
        ])
    );
}

#[test]
fn gradient_and_wall_reports_are_forces_not_position_exclusions() {
    let node = NodeKey::new(4);
    let field = || {
        FieldDefinition::Scalar(ScalarField::Linear {
            normal: Box::new(VectorField::ConstVec { x: 2.0, y: -1.0 }),
            offset: 0.0,
        })
    };
    for (response, force) in [
        (CouplingResponse::AttractToMin, [-6.0, 3.0]),
        (CouplingResponse::RepelFromMax, [6.0, -3.0]),
    ] {
        let coupling = CouplingForce::new(response, 3.0, [node], field());
        assert_eq!(
            coupling.evaluate_at(node, [5.0, 8.0], true, 0.25).effect,
            CouplingEffect::AddForce(force)
        );
    }
    let wall = CouplingForce::new(
        CouplingResponse::ContainmentWall,
        3.0,
        [node],
        FieldDefinition::Scalar(ScalarField::CoordX),
    );
    assert_eq!(
        wall.evaluate_at(node, [2.0, -1.0], true, 0.25).effect,
        CouplingEffect::AddForce([6.0, 0.0])
    );
    for x in [-2.0, 0.0] {
        assert_eq!(
            wall.evaluate_at(node, [x, -1.0], true, 0.25).effect,
            CouplingEffect::Inert(CouplingInertReason::OutsidePositiveRegion)
        );
    }
}

#[test]
fn damping_reports_its_factor_without_inventing_strength_scaling() {
    let node = NodeKey::new(4);
    for strength in [0.0, 3.0, 100.0] {
        let coupling = CouplingForce::new(
            CouplingResponse::DampenInside { factor: 0.85 },
            strength,
            [node],
            FieldDefinition::Scalar(ScalarField::CoordX),
        );
        assert_eq!(
            coupling.evaluate_at(node, [2.0, -1.0], true, 0.25).effect,
            CouplingEffect::ScaleVelocity(0.85)
        );
        assert_eq!(
            coupling.evaluate_at(node, [0.0, -1.0], true, 0.25).effect,
            CouplingEffect::Inert(CouplingInertReason::OutsidePositiveRegion)
        );
    }
}

#[test]
fn velocity_assignment_and_advection_have_distinct_hold_behavior() {
    let node = NodeKey::new(4);
    let field = || FieldDefinition::Vector(VectorField::ConstVec { x: 2.0, y: -1.0 });
    let align = CouplingForce::new(CouplingResponse::AlignVelocity, 3.0, [node], field());
    let advect = CouplingForce::new(CouplingResponse::FlowAdvect, 3.0, [node], field());
    // The existing velocity writer does not use the advection hold test.
    for dynamic in [true, false] {
        assert_eq!(
            align.evaluate_at(node, [5.0, 8.0], dynamic, 0.25).effect,
            CouplingEffect::SetVelocity([6.0, -3.0])
        );
    }
    assert_eq!(
        advect.evaluate_at(node, [5.0, 8.0], true, 0.25).effect,
        CouplingEffect::Displace([1.5, -0.75])
    );
    assert_eq!(
        advect.evaluate_at(node, [5.0, 8.0], false, 0.25).effect,
        CouplingEffect::Inert(CouplingInertReason::NonDynamicBody)
    );
}

#[test]
fn unsupported_response_wrong_field_and_unmatched_target_stay_distinct() {
    let node = NodeKey::new(4);
    let open = CouplingForce::new(
        CouplingResponse::open("urn:test:visual"),
        3.0,
        [node],
        FieldDefinition::Scalar(ScalarField::CoordX),
    );
    assert_eq!(
        open.evaluate_at(node, [5.0, 8.0], true, 0.25).effect,
        CouplingEffect::Inert(CouplingInertReason::UnsupportedResponse)
    );
    assert_eq!(
        open.evaluate_at(NodeKey::new(5), [5.0, 8.0], true, 0.25)
            .effect,
        CouplingEffect::Inert(CouplingInertReason::NotInResolvedTargets)
    );
    for response in [
        CouplingResponse::AlignVelocity,
        CouplingResponse::FlowAdvect,
    ] {
        let wrong = CouplingForce::new(
            response,
            3.0,
            [node],
            FieldDefinition::Scalar(ScalarField::CoordX),
        );
        assert_eq!(
            wrong.evaluate_at(node, [5.0, 8.0], false, 0.25).effect,
            CouplingEffect::Inert(CouplingInertReason::FieldTypeMismatch)
        );
    }
}

#[test]
fn sampling_time_and_registry_fallback_match_the_existing_executor() {
    let node = NodeKey::new(4);
    let timed = CouplingForce::new(
        CouplingResponse::AlignVelocity,
        3.0,
        [node],
        FieldDefinition::Vector(VectorField::Scale(
            Box::new(VectorField::ConstVec { x: 2.0, y: -1.0 }),
            Box::new(ScalarField::Time),
        )),
    );
    assert_eq!(
        timed.evaluate_at(node, [5.0, 8.0], true, 0.25),
        CouplingEvaluation {
            sample_time: 0.0,
            effect: CouplingEffect::SetVelocity([0.0, -0.0])
        }
    );
    let mut registry = FieldRegistry::new();
    let field = registry.insert_vector("held vector", VectorField::ConstVec { x: 2.0, y: -1.0 });
    let unresolved = CouplingForce::new(
        CouplingResponse::AlignVelocity,
        3.0,
        [node],
        FieldDefinition::Vector(VectorField::Sample(field)),
    );
    assert_eq!(
        unresolved.evaluate_at(node, [5.0, 8.0], true, 0.25).effect,
        CouplingEffect::SetVelocity([0.0, 0.0])
    );
    let resolved = unresolved.with_registry(registry);
    assert_eq!(
        resolved.evaluate_at(node, [5.0, 8.0], true, 0.25).effect,
        CouplingEffect::SetVelocity([6.0, -3.0])
    );
}

#[test]
fn querying_an_installed_coupling_does_not_advance_or_mutate_the_simulation() {
    let node = NodeKey::new(4);
    let make = || {
        let mut sim = Simulation::new();
        sim.sync_nodes([(node, Point2D::new(100.0, -30.0))]);
        sim.set_coupling_forces(vec![CouplingForce::new(
            CouplingResponse::FlowAdvect,
            2.0,
            [node],
            FieldDefinition::Vector(VectorField::ConstVec { x: 8.0, y: -3.0 }),
        )]);
        sim
    };
    let mut queried = make();
    let mut control = make();
    for _ in 0..120 {
        let before = queried.position_of(node).unwrap();
        let velocity = queried.velocity_of(node).unwrap();
        let step = queried.steps;
        for _ in 0..20 {
            assert_eq!(
                queried.coupling_forces[0]
                    .evaluate_at(node, [before.x, before.y], true, 0.25)
                    .effect,
                CouplingEffect::Displace([4.0, -1.5])
            );
        }
        assert_eq!(queried.position_of(node), Some(before));
        assert_eq!(queried.velocity_of(node), Some(velocity));
        assert_eq!(queried.steps, step);
        queried.tick(1.0 / 60.0);
        control.tick(1.0 / 60.0);
        assert_eq!(queried.position_of(node), control.position_of(node));
        assert_eq!(queried.velocity_of(node), control.velocity_of(node));
    }
}
