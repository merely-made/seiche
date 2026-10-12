// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

use super::*;
use std::{
    num::NonZeroUsize,
    sync::{Arc, mpsc},
};

fn force(node: NodeKey) -> CouplingForce {
    CouplingForce::new(
        numen::CouplingResponse::RepelFromMax,
        1.0,
        [node],
        numen::FieldDefinition::Scalar(numen::ScalarField::CoordX),
    )
}

#[test]
fn actor_commands_publish_actual_capped_trace_and_clear_on_rebind() {
    let mut sim = Simulation::new();
    let node = NodeKey::new(0);
    sim.sync_nodes([(node, Point2D::new(50.0, 20.0))]);
    let (handle, updates) = spawn(Arc::new(|| {}), move |commands, out| {
        run(sim, ActorState::fresh(0, false, true), commands, out);
    });
    handle.command(PhysicsCommand::SetCouplingTraceLimit(NonZeroUsize::new(1)));
    handle.command(PhysicsCommand::SetCouplingForces(vec![
        force(node),
        force(node),
    ]));
    handle.command(PhysicsCommand::Settle(1));
    handle.command(PhysicsCommand::Barrier(1));
    let receive = |epoch| {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            let update = updates
                .recv_timeout(deadline.saturating_duration_since(std::time::Instant::now()))
                .unwrap();
            if update.command_epoch == epoch {
                break update;
            }
        }
    };
    let tick = receive(1);
    let trace = tick
        .coupling_trace
        .unwrap()
        .read()
        .expect("latest actor tick");
    assert_eq!((trace.records.len(), trace.omitted), (1, 1));
    let crate::CouplingAppliedEffect::Write { before, after } = trace.records[0].applied else {
        panic!()
    };
    assert_eq!(before.force, [0.0, 0.0]);
    assert_eq!(after.force, [1.0, 0.0]);
    handle.command(PhysicsCommand::SetCouplingForces(vec![]));
    handle.command(PhysicsCommand::Barrier(2));
    assert!(receive(2).coupling_trace.is_none());
}

#[test]
fn stale_actor_trace_cannot_reappear_after_configuration_barrier() {
    let (handle, _) = spawn(
        Arc::new(|| {}),
        |commands, _out: Emitter<PhysicsUpdate>| {
            while commands.recv().is_ok() {}
        },
    );
    let (tx, updates) = mpsc::channel();
    let mut physics = Physics::Actor(ActorPhysics {
        handle,
        updates,
        coupling_trace: None,
        settling: false,
        energy: 0.0,
        speed: 0.0,
        command_epoch: 0,
        dial: Speed::REAL_TIME,
        budget: None,
        pace: PaceStats::default(),
    });
    let mut sim = Simulation::new();
    let node = NodeKey::new(0);
    sim.sync_nodes([(node, Point2D::new(50.0, 20.0))]);
    sim.set_coupling_forces(vec![force(node)]);
    sim.set_coupling_trace_limit(NonZeroUsize::new(1));
    sim.tick(TICK_DT);
    let mut view = sim.view();
    tx.send(PhysicsUpdate {
        snapshot: sim.snapshot(1),
        coupling_trace: CouplingTraceDelivery::fixture(1, sim.coupling_trace()),
        settling: false,
        command_epoch: 0,
        pace: PaceStats::default(),
    })
    .unwrap();
    physics.refresh(&mut view);
    assert!(physics.coupling_trace().is_some());
    physics.set_coupling_forces(vec![]);
    assert!(physics.coupling_trace().is_none());
    tx.send(PhysicsUpdate {
        snapshot: sim.snapshot(2),
        coupling_trace: CouplingTraceDelivery::fixture(1, sim.coupling_trace()),
        settling: false,
        command_epoch: 0,
        pace: PaceStats::default(),
    })
    .unwrap();
    physics.refresh(&mut view);
    assert!(physics.coupling_trace().is_none());
    physics.set_coupling_trace_limit(None);
    tx.send(PhysicsUpdate {
        snapshot: sim.snapshot(3),
        coupling_trace: CouplingTraceDelivery::fixture(1, sim.coupling_trace()),
        settling: false,
        command_epoch: 1,
        pace: PaceStats::default(),
    })
    .unwrap();
    physics.refresh(&mut view);
    assert!(physics.coupling_trace().is_none());
}
