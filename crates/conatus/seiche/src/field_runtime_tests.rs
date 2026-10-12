// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

use crate::*;
use euclid::default::Point2D;
use numen::{
    CouplingId, CouplingResponse, FieldDefinition, FieldExtent, FieldId, ScalarField, VectorField,
};
use std::{
    num::NonZeroUsize,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};
use uuid::Uuid;

fn region(min: f32, max: f32) -> FieldExtent {
    FieldExtent::Region {
        min_x: min,
        min_y: -100.0,
        max_x: max,
        max_y: 100.0,
    }
}
fn force(
    targets: &[NodeKey],
    extent: FieldExtent,
    field: FieldDefinition,
    response: CouplingResponse,
) -> CouplingForce {
    CouplingForce::new(response, 1.0, targets.iter().copied(), field)
        .with_extent(extent)
        .with_provenance(
            CouplingSource {
                field: FieldId::from_uuid(Uuid::from_u128(100)),
                coupling: CouplingId::from_uuid(Uuid::from_u128(101)),
                revision: Some(7),
            },
            |_, node| {
                Some(CouplingTargetIdentity {
                    member: node.index() as u128 + 1000,
                    occurrence: None,
                })
            },
        )
}
fn observed(targets: &[NodeKey], extent: FieldExtent) -> CouplingForce {
    force(
        targets,
        extent,
        FieldDefinition::Scalar(ScalarField::Const(1.0)),
        CouplingResponse::open("mere://test/owner-entry"),
    )
}
fn timed(node: NodeKey) -> CouplingForce {
    force(
        &[node],
        FieldExtent::Global,
        FieldDefinition::Vector(VectorField::Scale(
            Box::new(VectorField::const_vec(8.0, 0.0)),
            Box::new(ScalarField::Time),
        )),
        CouplingResponse::FlowAdvect,
    )
}
fn sim(points: &[(f32, f32)]) -> (Simulation, Vec<NodeKey>) {
    let nodes: Vec<_> = (0..points.len()).map(NodeKey::new).collect();
    let mut sim = Simulation::new();
    sim.sync_nodes(
        nodes
            .iter()
            .zip(points)
            .map(|(&n, &(x, y))| (n, Point2D::new(x, y))),
    );
    sim.set_coupling_trace_limit(NonZeroUsize::new(16));
    (sim, nodes)
}
fn count_entries(sim: &mut Simulation) -> Arc<AtomicUsize> {
    let count = Arc::new(AtomicUsize::new(0));
    let actual = Arc::clone(&count);
    sim.set_coupling_entry_handler(Some(Arc::new(move |report| {
        assert_eq!(report.change, CouplingMembershipChange::Entered);
        assert_eq!(report.entry_action, CouplingEntryOutcome::Pending);
        assert!(report.binding.source.is_some());
        actual.fetch_add(1, Ordering::SeqCst);
        Ok(())
    })));
    count
}

#[test]
fn field_runtime_clock_controls_real_motion_and_reset_does_not_rewind_solver_age() {
    let (mut sim, nodes) = sim(&[(50.0, 20.0)]);
    let n = nodes[0];
    sim.set_coupling_forces(vec![timed(n)]);
    sim.tick(0.25);
    assert_eq!(sim.position_of(n).unwrap().x, 50.0);
    sim.tick(0.25);
    let old = sim.coupling_trace().unwrap().clone();
    assert_eq!(old.clock.elapsed, 0.25);
    assert_eq!(old.records[0].evaluation.unwrap().sample_time, 0.25);
    assert_eq!(
        old.records[0].evaluation.unwrap().effect,
        CouplingEffect::Displace([0.5, 0.0])
    );
    assert_eq!(sim.position_of(n).unwrap().x, 50.5);
    assert_eq!(sim.field_clock().elapsed, 0.5);
    sim.seed_positions([(n, Point2D::new(100.0, 20.0))]);
    assert_eq!(
        sim.field_clock().elapsed,
        0.5,
        "ordinary seeding preserves time"
    );
    sim.reset_field_runtime([(n, Point2D::new(50.0, 20.0))]);
    assert_eq!(sim.field_clock().elapsed, 0.0);
    assert_eq!(sim.field_clock().epoch, 1);
    assert!(sim.coupling_trace().is_none());
    assert!(sim.coupling_membership().is_none());
    sim.tick(0.25);
    let new = sim.coupling_trace().unwrap();
    assert_eq!(
        (
            new.step,
            new.clock.epoch,
            new.records[0].evaluation.unwrap().sample_time
        ),
        (2, 1, 0.0)
    );
    assert_eq!(sim.position_of(n).unwrap().x, 50.0);
    assert_eq!(old.records[0].evaluation.unwrap().sample_time, 0.25);
}

#[test]
fn field_runtime_pause_freezes_time_even_when_a_drag_requests_body_steps() {
    let (mut sim, nodes) = sim(&[(50.0, 20.0)]);
    sim.set_coupling_forces(vec![timed(nodes[0])]);
    let mut view = sim.view();
    let mut physics = Physics::inline(sim, 1);
    physics.advance_frame(&mut view);
    let elapsed = physics.field_clock().elapsed;
    assert!(
        physics.is_settling(),
        "the animated field requests continued ticks"
    );
    physics.halt();
    for _ in 0..3 {
        assert!(!physics.advance_frame(&mut view));
    }
    physics.set_dragging(true);
    for _ in 0..3 {
        physics.advance_frame(&mut view);
        assert_eq!(physics.field_clock().elapsed, elapsed);
    }
    assert!(physics.field_clock().paused);
    physics.set_dragging(false);
    physics.reset_field_runtime(vec![(nodes[0], Point2D::new(50.0, 20.0))]);
    assert_eq!(physics.field_clock().elapsed, 0.0);
    assert!(physics.field_clock().paused);
    assert!(!physics.is_settling());
    physics.settle(1);
    physics.advance_frame(&mut view);
    assert_eq!(physics.field_clock().elapsed, elapsed);
    assert!(!physics.field_clock().paused);
}

#[test]
fn field_runtime_speed_and_elapsed_caps_count_only_accepted_fixed_ticks() {
    for speed in [1.0, 4.0] {
        let (mut sim, nodes) = sim(&[(50.0, 20.0)]);
        sim.set_coupling_forces(vec![timed(nodes[0])]);
        let mut view = sim.view();
        let mut physics = Physics::inline(sim, 100);
        physics.set_speed(Speed::from_factor(speed));
        let report = physics.advance_elapsed(
            &mut view,
            Duration::from_secs(10),
            ElapsedStepConfig {
                max_elapsed: Duration::from_millis(50),
                max_steps: 2,
            },
        );
        assert!(report.steps > 0 && report.discarded_elapsed > Duration::from_secs(9));
        assert_eq!(
            physics.field_clock().elapsed,
            f64::from(report.steps) * f64::from(TICK_DT)
        );
        let trace = physics.coupling_trace().unwrap();
        assert_eq!(
            trace.records[0].evaluation.unwrap().sample_time,
            trace.clock.sample_time()
        );
        assert_eq!(
            trace.clock.elapsed + f64::from(TICK_DT),
            physics.field_clock().elapsed
        );
    }
}

#[test]
fn field_runtime_repeated_appearances_are_independent_and_initial_matches_do_not_act() {
    let (mut sim, nodes) = sim(&[(0.0, 0.0), (1000.0, 0.0)]);
    let counter = count_entries(&mut sim);
    let bound = observed(&[nodes[0], nodes[1], nodes[1]], region(-100.0, 100.0)).with_provenance(
        CouplingSource {
            field: FieldId::from_uuid(Uuid::from_u128(100)),
            coupling: CouplingId::from_uuid(Uuid::from_u128(101)),
            revision: Some(7),
        },
        |_, n| {
            Some(CouplingTargetIdentity {
                member: 42,
                occurrence: Some(n.index() as u128 + 1),
            })
        },
    );
    sim.set_coupling_forces(vec![bound]);
    sim.tick(0.25);
    let first = sim.coupling_membership().unwrap().clone();
    assert_eq!(first.reports.len(), 2);
    assert_eq!(sim.coupling_trace().unwrap().records.len(), 3);
    assert_eq!(first.reports[0].eligibility, CouplingEligibility::Inside);
    assert_eq!(first.reports[1].eligibility, CouplingEligibility::Outside);
    assert!(
        first
            .reports
            .iter()
            .all(|r| r.change == CouplingMembershipChange::Initial)
    );
    assert_eq!(counter.load(Ordering::SeqCst), 0);
    sim.seed_positions([
        (nodes[0], Point2D::new(2000.0, 0.0)),
        (nodes[1], Point2D::new(0.0, 0.0)),
    ]);
    sim.tick(0.25);
    let next = sim.coupling_membership().unwrap();
    assert_eq!(next.reports[0].change, CouplingMembershipChange::Exited);
    assert_eq!(next.reports[1].change, CouplingMembershipChange::Entered);
    assert_eq!(next.reports[1].identity.unwrap().occurrence, Some(2));
    assert!(next.reports[1].cause.movement);
    assert_eq!(
        next.reports[1].entry_action,
        CouplingEntryOutcome::Completed
    );
    assert_eq!(
        counter.load(Ordering::SeqCst),
        1,
        "duplicate target slots do not multiply entry actions"
    );
    for _ in 0..3 {
        let _ = sim.coupling_membership();
    }
    assert_eq!(
        counter.load(Ordering::SeqCst),
        1,
        "reading evidence cannot replay an action"
    );
    sim.reset_field_runtime([
        (nodes[0], Point2D::new(2000.0, 0.0)),
        (nodes[1], Point2D::new(0.0, 0.0)),
    ]);
    sim.tick(0.25);
    assert_eq!(counter.load(Ordering::SeqCst), 1);
    assert!(
        sim.coupling_membership()
            .unwrap()
            .reports
            .iter()
            .all(|r| r.change == CouplingMembershipChange::Initial && r.cause.reset)
    );
    assert_eq!(
        first.reports[1].eligibility,
        CouplingEligibility::Outside,
        "held evidence is immutable"
    );
}

#[test]
fn field_runtime_inclusive_reference_points_filter_actual_writes() {
    let outside = f32::from_bits(100.0_f32.to_bits() + 1);
    let (mut sim, nodes) = sim(&[(-100.0, -100.0), (100.0, 100.0), (outside, 0.0)]);
    sim.set_coupling_forces(vec![force(
        &nodes,
        region(-100.0, 100.0),
        FieldDefinition::Vector(VectorField::const_vec(4.0, 0.0)),
        CouplingResponse::FlowAdvect,
    )]);
    sim.tick(0.25);
    let rows = &sim.coupling_trace().unwrap().records;
    for row in &rows[..2] {
        assert!(matches!(row.applied, CouplingAppliedEffect::Write { .. }));
    }
    assert_eq!(
        rows[2].evaluation.unwrap().effect,
        CouplingEffect::Inert(CouplingInertReason::OutsideExtent)
    );
    assert_eq!(rows[2].applied, CouplingAppliedEffect::Inert);
    assert_eq!(
        sim.coupling_membership().unwrap().reports[2].eligibility,
        CouplingEligibility::Outside
    );
    assert_eq!(sim.position_of(nodes[2]).unwrap().x, outside);
}

#[test]
fn field_runtime_field_and_query_changes_preserve_known_members_and_exact_backing() {
    let (mut sim, nodes) = sim(&[(1000.0, 0.0)]);
    let counter = count_entries(&mut sim);
    sim.set_coupling_forces(vec![observed(&nodes, region(-100.0, 100.0))]);
    sim.tick(0.25);
    let old = sim.coupling_membership().unwrap().clone();
    sim.set_coupling_forces(vec![observed(&nodes, region(900.0, 1100.0))]);
    sim.tick(0.25);
    let edited = &sim.coupling_membership().unwrap().reports[0];
    assert_eq!(edited.change, CouplingMembershipChange::Entered);
    assert!(edited.cause.field_edit && !edited.cause.movement && !edited.cause.query_change);
    assert_eq!(counter.load(Ordering::SeqCst), 1);
    assert_ne!(old.reports[0].binding.extent, edited.binding.extent);
    sim.set_coupling_forces(vec![observed(&[], region(900.0, 1100.0))]);
    sim.tick(0.25);
    let gone = &sim.coupling_membership().unwrap().reports[0];
    assert_eq!(gone.change, CouplingMembershipChange::Exited);
    assert!(gone.cause.query_change);
    assert_eq!(gone.eligibility, CouplingEligibility::NotCandidate);
    sim.set_coupling_forces(vec![observed(&nodes, region(900.0, 1100.0))]);
    sim.tick(0.25);
    let returned = &sim.coupling_membership().unwrap().reports[0];
    assert_eq!(returned.change, CouplingMembershipChange::Entered);
    assert!(returned.cause.query_change);
    assert_eq!(counter.load(Ordering::SeqCst), 2);
}

#[test]
fn field_runtime_new_candidates_and_reused_body_keys_start_initial_reporting() {
    let (mut sim, nodes) = sim(&[(0.0, 0.0), (1000.0, 0.0)]);
    let counter = count_entries(&mut sim);
    sim.set_coupling_forces(vec![observed(&[nodes[0]], FieldExtent::Global)]);
    sim.tick(0.25);
    sim.set_coupling_forces(vec![observed(&nodes, FieldExtent::Global)]);
    sim.tick(0.25);
    let reports = &sim.coupling_membership().unwrap().reports;
    assert_eq!(reports[1].change, CouplingMembershipChange::Initial);
    assert_eq!(counter.load(Ordering::SeqCst), 0);
    sim.sync_nodes([]);
    sim.sync_nodes([(nodes[0], Point2D::zero())]);
    sim.tick(0.25);
    let report = &sim.coupling_membership().unwrap().reports[0];
    assert_eq!(report.change, CouplingMembershipChange::Initial);
    assert!(report.cause.source_change);
    assert_eq!(counter.load(Ordering::SeqCst), 0);
}

#[test]
fn field_runtime_trace_caps_do_not_suppress_entry_callbacks_or_physics() {
    let (mut sim, nodes) = sim(&[(1000.0, 0.0), (2000.0, 0.0)]);
    let counter = count_entries(&mut sim);
    sim.set_coupling_trace_limit(NonZeroUsize::new(1));
    sim.set_coupling_forces(vec![observed(&nodes, region(-100.0, 100.0))]);
    sim.tick(0.25);
    sim.set_coupling_forces(vec![force(
        &nodes,
        FieldExtent::Global,
        FieldDefinition::Vector(VectorField::const_vec(4.0, 0.0)),
        CouplingResponse::FlowAdvect,
    )]);
    sim.tick(0.25);
    assert_eq!(counter.load(Ordering::SeqCst), 2);
    assert_eq!(sim.coupling_trace().unwrap().omitted, 1);
    assert_eq!(sim.position_of(nodes[0]).unwrap().x, 1001.0);
    assert_eq!(sim.position_of(nodes[1]).unwrap().x, 2001.0);
}

#[test]
fn field_runtime_unresolved_extent_and_unregistered_or_refused_actions_are_explicit() {
    let (mut sim, nodes) = sim(&[(1000.0, 0.0)]);
    sim.set_coupling_forces(vec![observed(
        &nodes,
        FieldExtent::AttachedToNode(Uuid::from_u128(42)),
    )]);
    sim.tick(0.25);
    assert_eq!(
        sim.coupling_membership().unwrap().reports[0].eligibility,
        CouplingEligibility::UnsupportedExtent
    );
    assert_eq!(
        sim.coupling_trace().unwrap().records[0]
            .evaluation
            .unwrap()
            .effect,
        CouplingEffect::Inert(CouplingInertReason::UnsupportedExtent)
    );
    sim.set_coupling_forces(vec![observed(&nodes, FieldExtent::Global)]);
    sim.tick(0.25);
    assert_eq!(
        sim.coupling_membership().unwrap().reports[0].change,
        CouplingMembershipChange::Initial,
        "unavailable input was not an observed outside point"
    );
    sim.set_coupling_forces(vec![observed(&nodes, region(-100.0, 100.0))]);
    sim.tick(0.25);
    sim.set_coupling_forces(vec![observed(&nodes, FieldExtent::Global)]);
    sim.tick(0.25);
    assert_eq!(
        sim.coupling_membership().unwrap().reports[0].entry_action,
        CouplingEntryOutcome::NotRegistered
    );
    sim.set_coupling_entry_handler(Some(Arc::new(|_| Err("owner declined the edit".into()))));
    sim.set_coupling_forces(vec![observed(&nodes, region(-100.0, 100.0))]);
    sim.tick(0.25);
    sim.set_coupling_forces(vec![observed(&nodes, FieldExtent::Global)]);
    sim.tick(0.25);
    assert_eq!(
        sim.coupling_membership().unwrap().reports[0].entry_action,
        CouplingEntryOutcome::Refused("owner declined the edit".into())
    );
}

#[test]
fn field_runtime_invalid_ticks_preserve_clock_and_observation_state() {
    let (mut sim, nodes) = sim(&[(50.0, 20.0)]);
    sim.set_coupling_forces(vec![timed(nodes[0])]);
    sim.tick(0.25);
    let trace = sim.coupling_trace().cloned();
    let membership = sim.coupling_membership().cloned();
    let clock = sim.field_clock();
    for dt in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        sim.tick(dt);
    }
    assert_eq!(sim.field_clock(), clock);
    assert_eq!(sim.coupling_trace(), trace.as_ref());
    assert_eq!(sim.coupling_membership(), membership.as_ref());
}

#[test]
fn field_runtime_animated_registry_references_request_ticks_without_timeless_energy_claims() {
    let n = NodeKey::new(0);
    let mut registry = numen::FieldRegistry::new();
    let field = registry.insert_scalar("time", ScalarField::Time);
    let bound = force(
        &[n],
        FieldExtent::Global,
        FieldDefinition::Scalar(ScalarField::Mul(
            Box::new(ScalarField::Sample(field)),
            Box::new(ScalarField::CoordX),
        )),
        CouplingResponse::RepelFromMax,
    )
    .with_registry(registry);
    assert!(bound.wants_tick());
    assert_eq!(bound.terms()[0].class, Class::N);
    let (mut sim, _) = sim(&[(50.0, 20.0)]);
    sim.set_coupling_forces(vec![bound]);
    sim.tick(0.25);
    sim.tick(0.25);
    let CouplingEffect::AddForce([fx, fy]) = sim.coupling_trace().unwrap().records[0]
        .evaluation
        .unwrap()
        .effect
    else {
        panic!()
    };
    // Numen's Mul/Sample gradient uses finite differences at this scale.
    assert!(
        (fx - 0.25).abs() < 1.0e-3 && fy == 0.0,
        "actual gradient {fx}, {fy}"
    );
}

#[test]
fn field_runtime_selector_edits_with_equal_matches_remain_visible_in_provenance() {
    let (mut sim, nodes) = sim(&[(0.0, 0.0)]);
    let counter = count_entries(&mut sim);
    sim.set_coupling_forces(vec![
        observed(&nodes, FieldExtent::Global).with_selector(numen::NodeSelector::All),
    ]);
    sim.tick(0.25);
    let old = sim.coupling_membership().unwrap().clone();
    sim.set_coupling_forces(vec![
        observed(&nodes, FieldExtent::Global)
            .with_selector(numen::NodeSelector::Tagged("lane".into())),
    ]);
    sim.tick(0.25);
    let report = &sim.coupling_membership().unwrap().reports[0];
    assert!(report.cause.query_change);
    assert_eq!(report.change, CouplingMembershipChange::Unchanged);
    assert_eq!(
        old.reports[0].binding.selector,
        Some(numen::NodeSelector::All)
    );
    assert_eq!(counter.load(Ordering::SeqCst), 0);
}

#[test]
fn field_runtime_duplicate_named_bindings_do_not_fabricate_shared_membership_or_action_authority() {
    let (mut sim, nodes) = sim(&[(1000.0, 0.0), (2000.0, 0.0)]);
    let counter = count_entries(&mut sim);
    sim.set_coupling_forces(vec![
        observed(&[nodes[0]], region(-100.0, 100.0)),
        observed(&[nodes[1]], region(-100.0, 100.0)),
    ]);
    sim.tick(0.25);
    assert_eq!(sim.coupling_membership().unwrap().reports.len(), 2);
    sim.seed_positions([
        (nodes[0], Point2D::zero()),
        (nodes[1], Point2D::new(50.0, 0.0)),
    ]);
    sim.tick(0.25);
    let reports = &sim.coupling_membership().unwrap().reports;
    assert_eq!(reports.len(), 2);
    assert!(reports.iter().all(|r| r.binding_identity_ambiguous));
    assert!(
        reports
            .iter()
            .all(|r| r.change == CouplingMembershipChange::Entered
                && matches!(r.entry_action, CouplingEntryOutcome::Refused(_)))
    );
    assert_eq!(counter.load(Ordering::SeqCst), 0);
}

#[test]
fn field_runtime_missing_bodies_are_unavailable_and_replacement_identities_are_initial() {
    let (mut sim, nodes) = sim(&[(0.0, 0.0)]);
    let counter = count_entries(&mut sim);
    sim.set_coupling_forces(vec![observed(&nodes, FieldExtent::Global)]);
    sim.tick(0.25);
    sim.sync_nodes([]);
    sim.tick(0.25);
    let missing = &sim.coupling_membership().unwrap().reports[0];
    assert_eq!(missing.change, CouplingMembershipChange::Unavailable);
    assert_eq!(missing.eligibility, CouplingEligibility::MissingBody);
    assert!(missing.cause.source_change);
    sim.sync_nodes([(nodes[0], Point2D::zero())]);
    sim.tick(0.25);
    let new = &sim.coupling_membership().unwrap().reports[0];
    assert_eq!(new.change, CouplingMembershipChange::Initial);
    assert_eq!(counter.load(Ordering::SeqCst), 0);
    let probe = timed(nodes[0]).evaluate_at_time(nodes[0], [0.0, 0.0], true, 0.25, f32::INFINITY);
    assert_eq!(
        probe.effect,
        CouplingEffect::Inert(CouplingInertReason::InvalidSampleTime)
    );
}

#[test]
fn field_runtime_scoped_gradients_do_not_claim_a_global_autonomous_potential() {
    let n = NodeKey::new(0);
    let bound = force(
        &[n],
        region(-100.0, 100.0),
        FieldDefinition::Scalar(ScalarField::CoordX),
        CouplingResponse::RepelFromMax,
    );
    assert_eq!(bound.terms()[0].class, Class::N);
    assert_eq!(
        bound
            .evaluate_at_time(n, [1000.0, 0.0], true, 0.25, 0.0)
            .effect,
        CouplingEffect::Inert(CouplingInertReason::OutsideExtent)
    );
}
#[test]
fn field_runtime_timed_motion_is_identical_with_trace_disabled_and_capped() {
    let mut receipts = Vec::new();
    for limit in [None, NonZeroUsize::new(1)] {
        let (mut sim, nodes) = sim(&[(50.0, 20.0)]);
        sim.set_coupling_forces(vec![timed(nodes[0])]);
        sim.set_coupling_trace_limit(limit);
        let mut poses = Vec::new();
        for dt in [0.25, 0.125, 0.25, 0.125] {
            sim.tick(dt);
            poses.push((sim.position_of(nodes[0]).unwrap(), sim.field_clock()));
        }
        receipts.push(poses);
    }
    assert_eq!(receipts[0], receipts[1]);
    assert!(receipts[0].last().unwrap().0.x > 50.0);
}
