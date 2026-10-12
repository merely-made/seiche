// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! `cargo run -p seiche --example field_runtime`: real clock, scope and entries.
use euclid::default::Point2D;
use numen::{
    CouplingId, CouplingResponse, FieldDefinition, FieldExtent, FieldId, ScalarField, VectorField,
};
use seiche::{CouplingForce, CouplingSource, CouplingTargetIdentity, NodeKey, Simulation};
use std::{
    num::NonZeroUsize,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};
use uuid::Uuid;

fn bound(node: NodeKey, extent: FieldExtent) -> CouplingForce {
    CouplingForce::new(
        CouplingResponse::FlowAdvect,
        1.0,
        [node],
        FieldDefinition::Vector(VectorField::Scale(
            Box::new(VectorField::const_vec(8.0, 0.0)),
            Box::new(ScalarField::Time),
        )),
    )
    .with_extent(extent)
    .with_provenance(
        CouplingSource {
            field: FieldId::from_uuid(Uuid::from_u128(100)),
            coupling: CouplingId::from_uuid(Uuid::from_u128(101)),
            revision: None,
        },
        |_, _| {
            Some(CouplingTargetIdentity {
                member: 42,
                occurrence: Some(1),
            })
        },
    )
}
fn show(sim: &Simulation, label: &str, node: NodeKey) {
    let trace = sim.coupling_trace().unwrap();
    let report = &sim.coupling_membership().unwrap().reports[0];
    println!(
        "{label}: step={} epoch={} sample_time={} clock_after={} x={} eligibility={:?} change={:?} cause={:?} entry={:?} effect={:?}",
        trace.step,
        trace.clock.epoch,
        trace.records[0].evaluation.unwrap().sample_time,
        sim.field_clock().elapsed,
        sim.position_of(node).unwrap().x,
        report.eligibility,
        report.change,
        report.cause,
        report.entry_action,
        trace.records[0].evaluation.unwrap().effect
    );
}
fn main() {
    let node = NodeKey::new(0);
    let mut sim = Simulation::new();
    sim.sync_nodes([(node, Point2D::new(50.0, 0.0))]);
    let counter = Arc::new(AtomicUsize::new(0));
    let actual = Arc::clone(&counter);
    sim.set_coupling_entry_handler(Some(Arc::new(move |_| {
        actual.fetch_add(1, Ordering::SeqCst);
        Ok(())
    })));
    sim.set_coupling_trace_limit(NonZeroUsize::new(1));
    sim.set_coupling_forces(vec![bound(
        node,
        FieldExtent::Region {
            min_x: -100.0,
            min_y: -100.0,
            max_x: 100.0,
            max_y: 100.0,
        },
    )]);
    sim.tick(0.25);
    show(&sim, "initial", node);
    assert_eq!(counter.load(Ordering::SeqCst), 0);
    sim.seed_positions([(node, Point2D::new(1000.0, 0.0))]);
    sim.tick(0.25);
    show(&sim, "outside", node);
    sim.set_coupling_forces(vec![bound(node, FieldExtent::Global)]);
    sim.tick(0.25);
    show(&sim, "field edit", node);
    assert_eq!(counter.load(Ordering::SeqCst), 1);
    assert_eq!(sim.position_of(node).unwrap().x, 1001.0);
    sim.set_field_clock_paused(true);
    let clock = sim.field_clock();
    sim.tick(0.25);
    assert_eq!(sim.field_clock().elapsed, clock.elapsed);
    sim.reset_field_runtime([(node, Point2D::new(50.0, 0.0))]);
    sim.set_field_clock_paused(false);
    sim.tick(0.25);
    show(&sim, "reset", node);
    assert_eq!(sim.position_of(node).unwrap().x, 50.0);
    assert_eq!(counter.load(Ordering::SeqCst), 1);
    println!("owner_entry_calls={}", counter.load(Ordering::SeqCst));
}
