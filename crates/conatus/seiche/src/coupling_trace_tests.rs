// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

use super::*;
use crate::{CouplingInertReason, Physics};
use euclid::default::Point2D;
use numen::{ScalarField, VectorField};
use uuid::Uuid;

fn source(id: u128) -> CouplingSource {
    CouplingSource {
        field: FieldId::from_uuid(Uuid::from_u128(100)),
        coupling: CouplingId::from_uuid(Uuid::from_u128(id)),
        revision: Some(7),
    }
}
fn vector(x: f32) -> FieldDefinition {
    FieldDefinition::Vector(VectorField::ConstVec { x, y: 0.0 })
}
fn force(
    id: u128,
    response: CouplingResponse,
    targets: Vec<NodeKey>,
    field: FieldDefinition,
) -> CouplingForce {
    CouplingForce::new(response, 1.0, targets, field).with_provenance(source(id), |slot, node| {
        Some(CouplingTargetIdentity {
            member: node.index() as u128 + 10,
            occurrence: Some(slot as u128 + 20),
        })
    })
}
fn sim() -> (Simulation, NodeKey) {
    let node = NodeKey::new(0);
    let mut sim = Simulation::new();
    sim.sync_nodes([(node, Point2D::new(50.0, 20.0))]);
    sim.set_coupling_trace_limit(NonZeroUsize::new(16));
    (sim, node)
}
fn states(row: &CouplingTraceRecord) -> (CouplingBodyState, CouplingBodyState) {
    match row.applied {
        CouplingAppliedEffect::Write { before, after } => (before, after),
        _ => panic!("expected actual write"),
    }
}

#[test]
fn duplicate_slots_keep_sample_points_and_actual_accumulated_writes() {
    let (mut sim, node) = sim();
    sim.set_coupling_forces(vec![force(
        1,
        CouplingResponse::FlowAdvect,
        vec![node, node],
        vector(8.0),
    )]);
    sim.tick(0.25);
    let trace = sim.coupling_trace().unwrap();
    assert_eq!(
        (trace.step, trace.dt, trace.binding_generation),
        (0, 0.25, 1)
    );
    assert_eq!(trace.records.len(), 2);
    let [a, b] = trace.records.as_slice() else {
        panic!()
    };
    assert_eq!(a.sample_position, Some([50.0, 20.0]));
    assert_eq!(b.sample_position, a.sample_position);
    assert_eq!((a.target_index, b.target_index), (0, 1));
    assert_ne!(
        a.binding.target_identities[0],
        b.binding.target_identities[1]
    );
    assert!(Arc::ptr_eq(&a.binding, &b.binding));
    let (before_a, after_a) = states(a);
    let (before_b, after_b) = states(b);
    assert_eq!(before_a.position, [50.0, 20.0]);
    assert_eq!(after_a.position, [52.0, 20.0]);
    assert_eq!(before_b.position, after_a.position);
    assert_eq!(after_b.position, [54.0, 20.0]);
    assert_eq!(sim.position_of(node).unwrap(), Point2D::new(54.0, 20.0));
}

#[test]
fn equal_looking_couplings_keep_ids_and_sequential_velocity_effects() {
    let (mut sim, node) = sim();
    let response = CouplingResponse::DampenInside { factor: 0.5 };
    let field = FieldDefinition::Scalar(ScalarField::Const(1.0));
    let handle = sim.bodies_by_node[&node];
    sim.bodies
        .get_mut(handle)
        .unwrap()
        .set_linvel(Vector::new(12.0, -8.0), true);
    sim.set_coupling_forces(vec![
        force(1, response.clone(), vec![node], field.clone()),
        force(2, response, vec![node], field),
    ]);
    sim.tick(0.25);
    let rows = &sim.coupling_trace().unwrap().records;
    assert_eq!(rows[0].binding.source, Some(source(1)));
    assert_eq!(rows[1].binding.source, Some(source(2)));
    assert_eq!((rows[0].coupling_index, rows[1].coupling_index), (0, 1));
    assert_eq!(states(&rows[0]).1.velocity, [6.0, -4.0]);
    assert_eq!(states(&rows[1]).0.velocity, [6.0, -4.0]);
    assert_eq!(states(&rows[1]).1.velocity, [3.0, -2.0]);
    // Integration/damping occurs later: the trace must retain pre-integration state.
    assert_ne!(sim.velocity_of(node).unwrap().x, 3.0);
}

#[test]
fn force_accumulation_is_observed_separately_from_velocity_and_integration() {
    let (mut sim, node) = sim();
    sim.set_coupling_forces(vec![
        force(
            1,
            CouplingResponse::RepelFromMax,
            vec![node],
            FieldDefinition::Scalar(ScalarField::CoordX),
        ),
        force(
            2,
            CouplingResponse::RepelFromMax,
            vec![node],
            FieldDefinition::Scalar(ScalarField::CoordX),
        ),
    ]);
    sim.tick(0.25);
    let rows = &sim.coupling_trace().unwrap().records;
    let (a, b) = states(&rows[0]);
    let (c, d) = states(&rows[1]);
    assert_eq!(a.force, [0.0, 0.0]);
    assert_eq!(b.force, [1.0, 0.0]);
    assert_eq!(c.force, b.force);
    assert_eq!(d.force, [2.0, 0.0]);
    assert_eq!(d.velocity, [0.0, 0.0]);
    assert!(sim.velocity_of(node).unwrap().x > 0.0);
}

#[test]
fn fixed_body_write_call_is_distinct_from_advection_guard() {
    let (mut sim, node) = sim();
    sim.pin(node, Point2D::new(50.0, 20.0));
    sim.set_coupling_forces(vec![
        force(
            1,
            CouplingResponse::RepelFromMax,
            vec![node],
            FieldDefinition::Scalar(ScalarField::CoordX),
        ),
        force(2, CouplingResponse::FlowAdvect, vec![node], vector(8.0)),
    ]);
    sim.tick(0.25);
    let rows = &sim.coupling_trace().unwrap().records;
    assert_eq!(
        states(&rows[0]).0,
        states(&rows[0]).1,
        "Rapier ignored the non-dynamic force write"
    );
    assert_eq!(rows[1].applied, CouplingAppliedEffect::Inert);
    assert_eq!(
        rows[1].evaluation.unwrap().effect,
        CouplingEffect::Inert(CouplingInertReason::NonDynamicBody)
    );
}

#[test]
fn missing_body_and_inert_evaluation_remain_distinct() {
    let (mut sim, node) = sim();
    sim.set_coupling_forces(vec![force(
        1,
        CouplingResponse::AlignVelocity,
        vec![NodeKey::new(9), node],
        FieldDefinition::Scalar(ScalarField::Const(1.0)),
    )]);
    sim.tick(0.25);
    let rows = &sim.coupling_trace().unwrap().records;
    assert_eq!(rows[0].applied, CouplingAppliedEffect::MissingBody);
    assert_eq!((rows[0].sample_position, rows[0].evaluation), (None, None));
    assert_eq!(rows[1].applied, CouplingAppliedEffect::Inert);
    assert_eq!(
        rows[1].evaluation.unwrap().effect,
        CouplingEffect::Inert(CouplingInertReason::FieldTypeMismatch)
    );
}

#[test]
fn capped_latest_tick_does_not_skip_physics_or_accumulate_history() {
    let (mut sim, node) = sim();
    sim.set_coupling_trace_limit(NonZeroUsize::new(1));
    sim.set_coupling_forces(vec![force(
        1,
        CouplingResponse::FlowAdvect,
        vec![node, node, node],
        vector(8.0),
    )]);
    sim.tick(0.25);
    let first = sim.coupling_trace().unwrap().clone();
    assert_eq!((first.records.len(), first.omitted), (1, 2));
    assert_eq!(sim.position_of(node).unwrap().x, 56.0);
    sim.tick(f32::NAN);
    assert_eq!(sim.coupling_trace(), Some(&first));
    sim.tick(0.25);
    let second = sim.coupling_trace().unwrap();
    assert_eq!(
        (second.step, second.records.len(), second.omitted),
        (1, 1, 2)
    );
    assert_eq!(sim.position_of(node).unwrap().x, 62.0);
    sim.set_coupling_trace_limit(None);
    assert!(sim.coupling_trace().is_none());
    sim.tick(0.25);
    assert!(sim.coupling_trace().is_none());
    assert_eq!(sim.position_of(node).unwrap().x, 68.0);
}

#[test]
fn rebound_ids_keep_old_registry_and_target_identity_in_retained_evidence() {
    let (mut sim, node) = sim();
    let mut registry = FieldRegistry::new();
    registry.insert_with_id(source(1).field, vector(8.0));
    sim.set_coupling_forces(vec![
        force(1, CouplingResponse::FlowAdvect, vec![node], vector(8.0)).with_registry(registry),
    ]);
    sim.tick(0.25);
    let old = sim.coupling_trace().unwrap().clone();
    let next = CouplingForce::new(CouplingResponse::FlowAdvect, 1.0, [node], vector(16.0))
        .with_provenance(source(1), |_, _| {
            Some(CouplingTargetIdentity {
                member: 999,
                occurrence: None,
            })
        });
    sim.set_coupling_forces(vec![next]);
    assert!(sim.coupling_trace().is_none());
    sim.tick(0.25);
    let new = sim.coupling_trace().unwrap();
    assert_eq!(new.binding_generation, old.binding_generation + 1);
    assert_eq!(
        old.records[0].binding.registry.get(source(1).field),
        Some(&vector(8.0))
    );
    assert_eq!(
        old.records[0].binding.target_identities[0].unwrap().member,
        10
    );
    assert_eq!(
        new.records[0].binding.target_identities[0].unwrap().member,
        999
    );
    assert!(!Arc::ptr_eq(
        &old.records[0].binding,
        &new.records[0].binding
    ));
    assert_ne!(old.records[0].evaluation, new.records[0].evaluation);
}

#[test]
fn anonymous_force_does_not_invent_owner_ids_or_capture_generic_force_slot() {
    let (mut sim, node) = sim();
    sim.add_force(CouplingForce::new(
        CouplingResponse::FlowAdvect,
        1.0,
        [node],
        vector(8.0),
    ));
    sim.tick(0.25);
    assert!(sim.coupling_trace().unwrap().records.is_empty());
    sim.set_coupling_forces(vec![CouplingForce::new(
        CouplingResponse::FlowAdvect,
        1.0,
        [node],
        vector(8.0),
    )]);
    sim.tick(0.25);
    let binding = &sim.coupling_trace().unwrap().records[0].binding;
    assert_eq!(binding.source, None);
    assert_eq!(binding.target_identities, [None]);
}

#[test]
fn inline_runtime_exposes_and_clears_actual_trace_without_requesting_ticks() {
    let (mut sim, node) = sim();
    sim.set_coupling_forces(vec![force(
        1,
        CouplingResponse::FlowAdvect,
        vec![node],
        vector(8.0),
    )]);
    let mut view = sim.view();
    let mut physics = Physics::inline(sim, 1);
    assert!(physics.coupling_trace().is_none());
    physics.advance_frame(&mut view);
    assert_eq!(physics.coupling_trace().unwrap().records.len(), 1);
    physics.set_coupling_forces(vec![]);
    assert!(physics.coupling_trace().is_none());
    physics.halt();
    physics.set_coupling_trace_limit(NonZeroUsize::new(1));
    assert!(!physics.is_settling());
}

#[test]
fn equal_tick_numbers_from_separate_worlds_keep_distinct_runtime_owners() {
    let (mut first, _) = sim();
    let (mut second, _) = sim();
    first.tick(0.25);
    second.tick(0.25);
    let a = first.coupling_trace().unwrap();
    let b = second.coupling_trace().unwrap();
    assert_eq!(
        (a.step, a.binding_generation),
        (b.step, b.binding_generation)
    );
    assert_ne!(a.session, b.session);
}

#[test]
fn velocity_write_discloses_whether_it_actually_woke_the_body() {
    let (mut sim, node) = sim();
    sim.bodies
        .get_mut(sim.bodies_by_node[&node])
        .unwrap()
        .sleep();
    sim.set_coupling_forces(vec![force(
        1,
        CouplingResponse::AlignVelocity,
        vec![node],
        vector(0.0),
    )]);
    sim.tick(0.25);
    let (before, after) = states(&sim.coupling_trace().unwrap().records[0]);
    assert_eq!(before.velocity, after.velocity);
    assert!(before.sleeping);
    assert!(
        after.sleeping,
        "Rapier ignores an identical velocity assignment"
    );
    sim.set_coupling_forces(vec![force(
        1,
        CouplingResponse::AlignVelocity,
        vec![node],
        vector(8.0),
    )]);
    sim.tick(0.25);
    let (before, after) = states(&sim.coupling_trace().unwrap().records[0]);
    assert!(before.sleeping);
    assert!(!after.sleeping);
    assert_eq!(after.velocity, [8.0, 0.0]);
}
