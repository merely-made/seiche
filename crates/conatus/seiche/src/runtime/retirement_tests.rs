// Copyright 2026 Mark Alan Boykin
// SPDX-License-Identifier: MPL-2.0

use super::*;
use std::sync::{Arc, mpsc};

#[test]
fn retirement_rejects_stale_actor_layouts_but_ordinary_sync_keeps_the_epoch() {
    let (commands_tx, commands_rx) = mpsc::channel();
    let (handle, _) = spawn(
        Arc::new(|| {}),
        move |commands, _out: armillary::Emitter<PhysicsUpdate>| {
            while let Ok(cmd) = commands.recv() {
                commands_tx.send(cmd).unwrap();
            }
        },
    );
    let (updates_tx, updates) = mpsc::channel();
    let mut physics = Physics::Actor(ActorPhysics {
        field_clock: Default::default(),
        coupling_membership: None,
        coupling_trace: None,
        handle,
        updates,
        settling: false,
        energy: 0.,
        speed: 0.,
        command_epoch: 0,
        dial: Speed::REAL_TIME,
        budget: None,
        pace: PaceStats::default(),
    });
    let node = NodeKey::new(0);
    let replacement = Point2D::new(10., 20.);
    let mut sim = Simulation::new();
    sim.sync_nodes([(node, replacement)]);
    let mut view = sim.view();
    physics.sync_nodes_retiring(vec![(node, replacement)], vec![]);
    let receive = || {
        commands_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .unwrap()
    };
    assert!(matches!(receive(), PhysicsCommand::SyncNodes(_)));
    physics.sync_nodes_retiring(vec![(node, replacement)], vec![node]);
    assert!(
        matches!(receive(), PhysicsCommand::SyncNodesRetiring { nodes, retired }
        if nodes == vec![(node, replacement)] && retired == vec![node])
    );
    assert!(matches!(receive(), PhysicsCommand::Barrier(1)));
    let mut send = |epoch, at, generation| {
        sim.seed_positions([(node, at)]);
        updates_tx
            .send(PhysicsUpdate {
                field_clock: sim.field_clock(),
                coupling_membership: None,
                coupling_trace: None,
                snapshot: sim.snapshot(generation),
                settling: false,
                command_epoch: epoch,
                pace: PaceStats::default(),
            })
            .unwrap();
    };
    send(0, Point2D::new(-900., -800.), 100);
    physics.refresh(&mut view);
    assert_eq!(view.position_of(node), Some(replacement));
    let landed = Point2D::new(11., 21.);
    send(1, landed, 1);
    physics.refresh(&mut view);
    assert_eq!(view.position_of(node), Some(landed));
}
