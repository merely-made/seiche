// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

use super::*;
use numen::{CouplingResponse, FieldDefinition, FieldExtent, ScalarField, VectorField};

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

fn seeded_field_physics() -> (Physics, LayoutView, [NodeKey; 2]) {
    let nodes = [NodeKey::new(0), NodeKey::new(1)];
    let mut sim = Simulation::new();
    sim.sync_nodes(vec![
        (nodes[0], Point2D::new(-80.0, 0.0)),
        (nodes[1], Point2D::new(80.0, 0.0)),
    ]);
    sim.sync_edges([(nodes[0], nodes[1])]);
    sim.add_force(crate::Anneal::seeded(0x5eed));
    sim.set_coupling_forces(vec![
        crate::CouplingForce::new(
            CouplingResponse::FlowAdvect,
            1.0,
            [nodes[0]],
            FieldDefinition::Vector(VectorField::Scale(
                Box::new(VectorField::const_vec(8.0, 0.0)),
                Box::new(ScalarField::Time),
            )),
        )
        .with_extent(FieldExtent::Global),
    ]);
    let view = sim.view();
    (Physics::inline(sim, 0), view, nodes)
}

fn state_bits(physics: &Physics, nodes: [NodeKey; 2]) -> Vec<(NodeKey, [u32; 4])> {
    let Physics::Inline(inline) = physics else {
        panic!("this deterministic regression uses the inline backend")
    };
    nodes
        .into_iter()
        .map(|node| {
            let position = inline.sim.position_of(node).unwrap();
            let velocity = inline.sim.velocity_of(node).unwrap();
            (
                node,
                [
                    position.x.to_bits(),
                    position.y.to_bits(),
                    velocity.x.to_bits(),
                    velocity.y.to_bits(),
                ],
            )
        })
        .collect()
}

#[derive(Debug, PartialEq, Eq)]
struct TickWitness {
    labelled_state: Vec<(NodeKey, [u32; 4])>,
    field_elapsed: u64,
}

fn advance_and_record(
    physics: &mut Physics,
    view: &mut LayoutView,
    nodes: [NodeKey; 2],
    ticks: usize,
) -> Vec<TickWitness> {
    let mut witnesses = Vec::with_capacity(ticks);
    for _ in 0..ticks {
        assert!(physics.advance_frame(view));
        witnesses.push(TickWitness {
            labelled_state: state_bits(physics, nodes),
            field_elapsed: physics.field_clock().elapsed.to_bits(),
        });
    }
    witnesses
}

#[test]
fn halt_no_tick_resume_matches_uninterrupted_seeded_law_and_animated_field() {
    const PREFIX: usize = 11;
    const SUFFIX: usize = 17;
    const TOTAL: usize = PREFIX + SUFFIX;

    let (mut uninterrupted, mut uninterrupted_view, nodes) = seeded_field_physics();
    let (mut split, mut split_view, split_nodes) = seeded_field_physics();
    assert_eq!(nodes, split_nodes);

    let uninterrupted_ticks =
        advance_and_record(&mut uninterrupted, &mut uninterrupted_view, nodes, TOTAL);
    let split_prefix = advance_and_record(&mut split, &mut split_view, nodes, PREFIX);
    assert_eq!(&split_prefix, &uninterrupted_ticks[..PREFIX]);
    let paused_state = state_bits(&split, nodes);
    let paused_clock = split.field_clock();
    assert!(paused_clock.elapsed > 0.0);
    assert!(!paused_clock.paused);

    split.halt();
    let halt_clock = split.field_clock();
    assert_eq!(halt_clock.elapsed.to_bits(), paused_clock.elapsed.to_bits());
    assert!(halt_clock.paused);
    let paused = split.advance_elapsed(
        &mut split_view,
        std::time::Duration::from_secs(5),
        ElapsedStepConfig {
            max_elapsed: std::time::Duration::from_secs(5),
            max_steps: 300,
        },
    );
    assert_eq!(paused.steps, 0, "halted elapsed time cannot add ticks");
    assert_eq!(paused.discarded_elapsed, std::time::Duration::from_secs(5));
    assert_eq!(state_bits(&split, nodes), paused_state);
    assert_eq!(
        split.field_clock().elapsed.to_bits(),
        halt_clock.elapsed.to_bits()
    );
    assert!(!split.is_settling());

    split.settle(1);
    assert!(split.is_settling());
    let resumed_ticks = advance_and_record(&mut split, &mut split_view, nodes, SUFFIX);
    for (suffix_tick, resumed) in resumed_ticks.iter().enumerate() {
        assert_eq!(
            resumed,
            &uninterrupted_ticks[PREFIX + suffix_tick],
            "state and field time diverged at resumed suffix tick {}",
            suffix_tick + 1
        );
    }

    assert_eq!(state_bits(&split, nodes), state_bits(&uninterrupted, nodes));
    assert_eq!(
        split.field_clock().elapsed.to_bits(),
        uninterrupted.field_clock().elapsed.to_bits(),
        "the field clock must match after the same accepted tick count"
    );
    assert!(!split.field_clock().paused);

    // A one-tick-short control witnesses that the equality above detects a
    // lost or extra body tick instead of merely comparing a stationary state.
    let (mut short, mut short_view, _) = seeded_field_physics();
    let short_ticks = advance_and_record(&mut short, &mut short_view, nodes, TOTAL - 1);
    let short_final = short_ticks.last().expect("TOTAL is positive");
    assert_ne!(state_bits(&split, nodes), short_final.labelled_state);
    assert_ne!(
        split.field_clock().elapsed.to_bits(),
        short_final.field_elapsed
    );
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
        field_clock: Default::default(),
        coupling_membership: None,
        coupling_trace: None,
        handle,
        updates,
        settling: true,
        energy: 0.0,
        speed: 0.0,
        command_epoch: 0,
        dial: Speed::REAL_TIME,
        budget: None,
        pace: PaceStats::default(),
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
            field_clock: sim.field_clock(),
            coupling_membership: None,
            coupling_trace: None,
            snapshot: sim.snapshot(100),
            settling: false,
            command_epoch: 2,
            pace: PaceStats::default(),
        })
        .unwrap();
    assert!(physics.advance_frame(&mut view));
    assert_eq!(view.position_of(node), Some(frozen));

    let resumed = Point2D::new(301.0, 200.0);
    sim.seed_positions(vec![(node, resumed)]);
    updates_tx
        .send(PhysicsUpdate {
            field_clock: sim.field_clock(),
            coupling_membership: None,
            coupling_trace: None,
            snapshot: sim.snapshot(101),
            settling: true,
            command_epoch: 3,
            pace: PaceStats::default(),
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
            field_clock: sim.field_clock(),
            coupling_membership: None,
            coupling_trace: None,
            snapshot: sim.snapshot(102),
            settling: true,
            command_epoch: 3,
            pace: PaceStats::default(),
        })
        .unwrap();
    physics.refresh(&mut view);
    assert_eq!(view.position_of(node), Some(resumed));
    assert!(!physics.is_settling());
}

#[cfg(feature = "actor")]
fn receive_epoch(updates: &Receiver<PhysicsUpdate>, epoch: u64) -> PhysicsUpdate {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
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
        run(sim, ActorState::fresh(0, false, true), commands, out);
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
