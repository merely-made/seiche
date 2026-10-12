// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

use super::*;
use numen::{CouplingResponse, FieldDefinition, FieldExtent, ScalarField};
use std::{
    num::NonZeroUsize,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

fn receive(updates: &Receiver<PhysicsUpdate>, epoch: u64) -> PhysicsUpdate {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        let update = updates
            .recv_timeout(deadline.saturating_duration_since(std::time::Instant::now()))
            .unwrap();
        if update.command_epoch == epoch {
            return update;
        }
    }
}

#[test]
fn field_runtime_actor_executes_later_entries_once_and_reset_acknowledges_zero_clock() {
    let node = NodeKey::new(0);
    let mut sim = Simulation::new();
    sim.sync_nodes([(node, Point2D::new(1000.0, 0.0))]);
    let force = CouplingForce::new(
        CouplingResponse::open("mere://test/owner-entry"),
        1.0,
        [node],
        FieldDefinition::Scalar(ScalarField::Const(1.0)),
    )
    .with_extent(FieldExtent::Region {
        min_x: -100.0,
        min_y: -100.0,
        max_x: 100.0,
        max_y: 100.0,
    });
    let count = Arc::new(AtomicUsize::new(0));
    let actual = Arc::clone(&count);
    let (handle, updates) = spawn(Arc::new(|| {}), move |commands, out| {
        run(sim, ActorState::fresh(0, false, true), commands, out)
    });
    handle.command(PhysicsCommand::SetCouplingForces(vec![force]));
    handle.command(PhysicsCommand::SetCouplingTraceLimit(NonZeroUsize::new(1)));
    handle.command(PhysicsCommand::SetCouplingEntryHandler(Some(Arc::new(
        move |report| {
            assert_eq!(report.change, crate::CouplingMembershipChange::Entered);
            actual.fetch_add(1, Ordering::SeqCst);
            Ok(())
        },
    ))));
    handle.command(PhysicsCommand::Settle(1));
    handle.command(PhysicsCommand::Barrier(1));
    let first = receive(&updates, 1);
    assert_eq!(first.field_clock.elapsed, f64::from(TICK_DT));
    assert_eq!(
        first.coupling_membership.unwrap().read().unwrap().reports[0].change,
        crate::CouplingMembershipChange::Initial
    );
    assert_eq!(count.load(Ordering::SeqCst), 0);
    handle.command(PhysicsCommand::Seed(vec![(node, Point2D::zero())]));
    handle.command(PhysicsCommand::Settle(1));
    handle.command(PhysicsCommand::Barrier(2));
    let next = receive(&updates, 2);
    let delivery = next.coupling_membership.unwrap();
    let frame = delivery.read().unwrap();
    assert_eq!(
        frame.reports[0].change,
        crate::CouplingMembershipChange::Entered
    );
    assert_eq!(
        frame.reports[0].entry_action,
        crate::CouplingEntryOutcome::Completed
    );
    assert_eq!(count.load(Ordering::SeqCst), 1);
    let _ = delivery.read();
    let _ = delivery.read();
    assert_eq!(count.load(Ordering::SeqCst), 1);
    handle.command(PhysicsCommand::Halt);
    handle.command(PhysicsCommand::Barrier(3));
    let paused = receive(&updates, 3);
    assert!(paused.field_clock.paused);
    handle.command(PhysicsCommand::ResetFieldRuntime(vec![(
        node,
        Point2D::zero(),
    )]));
    handle.command(PhysicsCommand::Barrier(4));
    let reset = receive(&updates, 4);
    assert_eq!(
        (
            reset.field_clock.elapsed,
            reset.field_clock.epoch,
            reset.field_clock.paused
        ),
        (0.0, 1, true)
    );
    assert!(reset.coupling_trace.is_none());
    assert!(reset.coupling_membership.is_none());
    handle.command(PhysicsCommand::Settle(1));
    handle.command(PhysicsCommand::Barrier(5));
    let replay = receive(&updates, 5);
    assert_eq!(replay.coupling_trace.unwrap().read().unwrap().step, 2);
    let membership = replay.coupling_membership.unwrap().read().unwrap();
    assert_eq!(
        membership.reports[0].change,
        crate::CouplingMembershipChange::Initial
    );
    assert!(membership.reports[0].cause.reset);
    assert_eq!(count.load(Ordering::SeqCst), 1);
}

#[test]
fn field_runtime_actor_reset_rejects_delayed_clock_and_membership_evidence() {
    use std::sync::mpsc;
    let (handle, _) = spawn(
        Arc::new(|| {}),
        |commands, _out: Emitter<PhysicsUpdate>| {
            while commands.recv().is_ok() {}
        },
    );
    let (tx, updates) = mpsc::channel();
    let mut sim = Simulation::new();
    let node = NodeKey::new(0);
    sim.sync_nodes([(node, Point2D::zero())]);
    sim.tick(0.5);
    let mut view = sim.view();
    let mut physics = Physics::Actor(ActorPhysics {
        handle,
        updates,
        field_clock: sim.field_clock(),
        coupling_membership: sim.coupling_membership().cloned(),
        coupling_trace: None,
        settling: false,
        energy: 0.0,
        speed: 0.0,
        command_epoch: 0,
        dial: Speed::REAL_TIME,
        budget: None,
        pace: PaceStats::default(),
    });
    physics.reset_field_runtime(vec![(node, Point2D::zero())]);
    let mailbox = super::membership_delivery::MembershipMailbox::default();
    tx.send(PhysicsUpdate {
        field_clock: sim.field_clock(),
        coupling_membership: mailbox.publish(1, sim.coupling_membership()),
        coupling_trace: None,
        snapshot: sim.snapshot(1),
        settling: false,
        command_epoch: 0,
        pace: PaceStats::default(),
    })
    .unwrap();
    physics.refresh(&mut view);
    assert_eq!(
        (physics.field_clock().elapsed, physics.field_clock().epoch),
        (0.0, 1)
    );
    assert!(physics.coupling_membership().is_none());
}

#[test]
fn field_runtime_membership_delivery_keeps_one_latest_frame_and_reading_does_not_dispatch() {
    let mailbox = super::membership_delivery::MembershipMailbox::default();
    let mut sim = Simulation::new();
    sim.tick(0.25);
    let first = mailbox.publish(1, sim.coupling_membership()).unwrap();
    sim.tick(0.25);
    let second = mailbox.publish(2, sim.coupling_membership()).unwrap();
    assert!(first.read().is_none());
    assert_eq!(second.read().unwrap().clock.elapsed, 0.25);
    mailbox.publish(3, None);
    assert!(second.read().is_none());
}
