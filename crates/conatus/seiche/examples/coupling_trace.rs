// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! `cargo run -p seiche --example coupling_trace`: actual writes, no host graph.
use std::num::NonZeroUsize;

use euclid::default::Point2D;
use numen::{CouplingId, CouplingResponse, FieldDefinition, FieldId, VectorField};
use seiche::{CouplingForce, CouplingSource, CouplingTargetIdentity, NodeKey, Simulation};
use uuid::Uuid;

fn main() {
    let node = NodeKey::new(0);
    let field = FieldId::from_uuid(Uuid::from_u128(100));
    let mut sim = Simulation::new();
    sim.sync_nodes([(node, Point2D::new(50.0, 20.0))]);
    sim.set_coupling_forces(
        [1, 2]
            .into_iter()
            .map(|id| {
                CouplingForce::new(
                    CouplingResponse::FlowAdvect,
                    1.0,
                    [node, node],
                    FieldDefinition::Vector(VectorField::ConstVec { x: 8.0, y: 0.0 }),
                )
                .with_provenance(
                    CouplingSource {
                        field,
                        coupling: CouplingId::from_uuid(Uuid::from_u128(id)),
                        revision: None,
                    },
                    |slot, _| {
                        Some(CouplingTargetIdentity {
                            member: 42,
                            occurrence: Some(slot as u128 + 1),
                        })
                    },
                )
            })
            .collect(),
    );
    sim.set_coupling_trace_limit(NonZeroUsize::new(3));
    sim.tick(0.25);
    let trace = sim.coupling_trace().unwrap();
    println!(
        "step={} dt={} generation={} recorded={} omitted={}",
        trace.step,
        trace.dt,
        trace.binding_generation,
        trace.records.len(),
        trace.omitted
    );
    for row in &trace.records {
        println!(
            "coupling={:?} target_slot={} identity={:?} sample={:?} evaluation={:?} applied={:?}",
            row.binding.source.unwrap().coupling,
            row.target_index,
            row.binding.target_identities[row.target_index],
            row.sample_position,
            row.evaluation,
            row.applied
        );
    }
    let position = sim.position_of(node).unwrap();
    println!("integrated_position=({}, {})", position.x, position.y);
    assert_eq!(trace.omitted, 1);
    assert_eq!(position, Point2D::new(58.0, 20.0));
}
