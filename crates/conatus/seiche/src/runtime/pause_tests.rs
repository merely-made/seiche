// Copyright 2026 Mark Alan Boykin
// SPDX-License-Identifier: MPL-2.0

use super::*;

fn perpetual_pair() -> Simulation {
    let mut sim = Simulation::new();
    sim.sync_nodes(vec![
        (NodeKey::new(0), Point2D::new(-80.0, 0.0)),
        (NodeKey::new(1), Point2D::new(80.0, 0.0)),
    ]);
    sim.add_force(crate::NodeExclusion::default());
    sim.set_scene_field(Some(SceneField::Vortex {
        center: (0.0, 0.0),
        strength: 1.0,
        inward: 0.0,
    }));
    assert!(sim.wants_continuous_tick());
    sim
}

#[test]
fn halt_stops_perpetual_inline_motion_until_settle() {
    let sim = perpetual_pair();
    let mut view = sim.view();
    let node = NodeKey::new(0);
    let before = view.position_of(node).unwrap();
    let mut physics = Physics::inline(sim, 0);
    for _ in 0..12 {
        assert!(physics.advance_frame(&mut view));
    }
    let moving = view.position_of(node).unwrap();
    assert_ne!(moving, before, "perpetual motion was actually running");
    physics.halt();
    physics.settle(0);
    assert!(!physics.is_settling());
    for _ in 0..12 {
        assert!(!physics.advance_frame(&mut view));
        assert_eq!(view.position_of(node), Some(moving));
    }
    physics.settle(1);
    assert!(physics.advance_frame(&mut view));
    assert_ne!(view.position_of(node), Some(moving));
}

#[cfg(feature = "actor")]
#[test]
fn stale_actor_snapshots_cannot_undo_reseed_or_halt() {
    use std::sync::{Arc, mpsc};

    // A controlled producer gives the host delayed layouts deterministically,
    // including an old epoch with a larger ordinary snapshot generation.
    let (handle, _) = spawn(
        Arc::new(|| {}),
        |commands, _out: Emitter<PhysicsUpdate>| {
            while commands.recv().is_ok() {}
        },
    );
    let (updates_tx, updates) = mpsc::channel();
    let mut physics = Physics::Actor(ActorPhysics {
        handle,
        updates,
        settling: true,
        energy: 0.0,
        speed: 0.0,
        command_epoch: 0,
    });
    let mut sim = Simulation::new();
    let node = NodeKey::new(0);
    let frozen = Point2D::new(300.0, 200.0);
    sim.sync_nodes(vec![(node, frozen)]);
    let mut view = sim.view();
    physics.seed(vec![(node, frozen)]);
    physics.halt();
    physics.seed(vec![(node, frozen)]);
    physics.settle(1);

    sim.seed_positions(vec![(node, Point2D::new(-900.0, -800.0))]);
    updates_tx
        .send(PhysicsUpdate {
            snapshot: sim.snapshot(100),
            settling: false,
            command_epoch: 2,
        })
        .unwrap();
    assert!(physics.advance_frame(&mut view));
    assert_eq!(view.position_of(node), Some(frozen));

    let resumed = Point2D::new(301.0, 200.0);
    sim.seed_positions(vec![(node, resumed)]);
    updates_tx
        .send(PhysicsUpdate {
            snapshot: sim.snapshot(101),
            settling: true,
            command_epoch: 3,
        })
        .unwrap();
    physics.refresh(&mut view);
    assert_eq!(view.position_of(node), Some(resumed));

    physics.halt();
    physics.settle(0);
    assert!(
        !physics.is_settling(),
        "zero ticks must not restart a halted actor"
    );
    sim.seed_positions(vec![(node, Point2D::zero())]);
    updates_tx
        .send(PhysicsUpdate {
            snapshot: sim.snapshot(102),
            settling: true,
            command_epoch: 3,
        })
        .unwrap();
    physics.refresh(&mut view);
    assert_eq!(view.position_of(node), Some(resumed));
    assert!(!physics.is_settling());
}

#[cfg(feature = "actor")]
fn receive_epoch(updates: &Receiver<PhysicsUpdate>, epoch: u64) -> PhysicsUpdate {
    let deadline = std::time::Instant::now() + Duration::from_secs(2);
    loop {
        let remaining = deadline
            .checked_duration_since(std::time::Instant::now())
            .expect("actor acknowledgement deadline exceeded");
        let update = updates
            .recv_timeout(remaining)
            .expect("actor acknowledgement");
        if update.command_epoch == epoch {
            return update;
        }
    }
}

#[cfg(feature = "actor")]
#[test]
fn actor_acknowledges_seed_and_halt_without_a_tick() {
    use std::sync::Arc;

    let sim = perpetual_pair();
    let (handle, updates) = spawn(Arc::new(|| {}), move |commands, out| {
        run(sim, 0, false, true, commands, out);
    });
    let node = NodeKey::new(0);
    let restored = Point2D::new(10.0, 20.0);
    // The public Seed command remains usable without a host-specific wrapper.
    handle.command(PhysicsCommand::Seed(vec![(node, restored)]));
    handle.command(PhysicsCommand::Halt);
    handle.command(PhysicsCommand::Barrier(9));
    let paused = receive_epoch(&updates, 9);
    assert!(!paused.settling);
    assert_eq!(
        paused
            .snapshot
            .positions
            .iter()
            .find(|(key, _)| *key == node)
            .unwrap()
            .1,
        restored
    );

    handle.command(PhysicsCommand::Settle(1));
    handle.command(PhysicsCommand::Barrier(10));
    let update = receive_epoch(&updates, 10);
    assert!(
        update.settling,
        "perpetual motion resumes after explicit settle"
    );
    assert_ne!(
        update
            .snapshot
            .positions
            .iter()
            .find(|(key, _)| *key == node)
            .unwrap()
            .1,
        restored
    );
}
