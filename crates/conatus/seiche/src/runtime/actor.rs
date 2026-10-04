// Copyright 2026 Mark Alan Boykin
// SPDX-License-Identifier: MPL-2.0

//! The actor half of the runtime: the thread body that owns an offloaded
//! [`Simulation`], applies [`PhysicsCommand`]s and paces its intervals.

use std::sync::mpsc::{Receiver, TryRecvError};

use armillary::Emitter;

use super::{PhysicsCommand, PhysicsUpdate, Speed, TICK_DURATION, should_tick, speed};
use crate::Simulation;

/// The actor's settle/drag state beside its simulation.
pub(super) struct ActorState {
    pub(super) ticks_remaining: u32,
    pub(super) dragging: bool,
    pub(super) halted: bool,
    pub(super) command_epoch: u64,
    pub(super) pace: speed::Pace,
}

#[cfg(test)]
impl ActorState {
    /// A real-time actor state with no pace carried over (tests).
    pub(super) fn fresh(ticks_remaining: u32, dragging: bool, halted: bool) -> Self {
        Self {
            ticks_remaining,
            dragging,
            halted,
            command_epoch: 0,
            pace: speed::Pace::for_actor(Speed::REAL_TIME, 0, None),
        }
    }
}

/// The actor thread body: own the simulation, apply commands, and run one
/// interval of [`TICK_DURATION`] at a time while there is work: the ticks the
/// speed owes (up to the budget), one snapshot, then sleep out the interval.
/// Parks on `recv` when idle (no busy-spin), and ends when the command channel
/// closes (the handle drops).
pub(super) fn run(
    mut sim: Simulation,
    mut state: ActorState,
    commands: Receiver<PhysicsCommand>,
    out: Emitter<PhysicsUpdate>,
) {
    let mut generation: u64 = 0;
    let mut last_interval: Option<std::time::Instant> = None;

    loop {
        let mut changed = false;
        // Idle: block for the next command so the thread parks. A closed channel
        // (the host dropped the handle) ends the actor. A perpetual scene (a drifting
        // backdrop) is never idle, so the actor keeps ticking instead of parking.
        if !should_tick(&sim, state.ticks_remaining, state.dragging, state.halted) {
            last_interval = None;
            match commands.recv() {
                Ok(cmd) => {
                    apply(&mut sim, cmd, &mut state);
                    changed = true;
                },
                Err(_) => return,
            }
        }
        // Drain any further pending commands without blocking. A disconnect means
        // the host is gone: stop accepting commands, but still run out whatever
        // settle is already queued before exiting (so a drop right after a
        // `Settle` does not throw away the layout work).
        let mut disconnected = false;
        loop {
            match commands.try_recv() {
                Ok(cmd) => {
                    apply(&mut sim, cmd, &mut state);
                    changed = true;
                },
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    disconnected = true;
                    break;
                },
            }
        }
        // While there is work (a settle, a drag, or a perpetual scene), run the
        // interval's ticks and emit the layout, drawn between ticks below real time.
        let interval = std::time::Instant::now();
        let moving = should_tick(&sim, state.ticks_remaining, state.dragging, state.halted);
        if moving {
            let wall = last_interval.map_or(speed::TICK_NS, |last| {
                speed::nanos(interval.saturating_duration_since(last))
            });
            last_interval = Some(interval);
            speed::actor_interval(
                &mut sim,
                &mut state.pace,
                &mut state.ticks_remaining,
                state.dragging,
                state.halted,
                wall,
            );
        }
        // Seed/Halt must acknowledge their placement even when no tick follows.
        if moving || changed {
            generation = generation.wrapping_add(1);
            let settling = should_tick(&sim, state.ticks_remaining, state.dragging, state.halted);
            let mut snapshot = sim.snapshot(generation);
            speed::draw(&mut state.pace, &mut snapshot, settling);
            out.emit(PhysicsUpdate {
                snapshot,
                settling,
                command_epoch: state.command_epoch,
                pace: state.pace.stats(),
            });
        }
        if moving {
            std::thread::sleep(TICK_DURATION.saturating_sub(interval.elapsed()));
        }
        // Host gone and the settle budget spent: wind down.
        if disconnected && state.ticks_remaining == 0 {
            return;
        }
    }
}

/// Fold one command into the actor's simulation + settle/drag state.
fn apply(sim: &mut Simulation, cmd: PhysicsCommand, state: &mut ActorState) {
    match cmd {
        PhysicsCommand::SyncNodes(nodes) => sim.sync_nodes(nodes),
        PhysicsCommand::SyncEdges(edges) => sim.sync_edges(edges),
        PhysicsCommand::Seed(positions) => {
            sim.seed_positions(positions);
            state.pace.forget();
        },
        PhysicsCommand::Barrier(epoch) => state.command_epoch = state.command_epoch.max(epoch),
        PhysicsCommand::Pin(node, position) => sim.pin(node, position),
        PhysicsCommand::Unpin(node) => sim.unpin(node),
        PhysicsCommand::Settle(n) => {
            state.ticks_remaining = state.ticks_remaining.max(n);
            if n > 0 {
                state.halted = false;
            }
        },
        PhysicsCommand::Halt => {
            state.ticks_remaining = 0;
            state.dragging = false;
            state.halted = true;
            state.pace.forget();
            state.pace.meter.clear();
        },
        PhysicsCommand::SetDragging(d) => state.dragging = d,
        PhysicsCommand::SetSpeed(speed) => {
            state.pace.speed = speed;
            if speed >= Speed::REAL_TIME {
                state.pace.previous = None;
            }
        },
        PhysicsCommand::SetStepBudget(per_frame) => {
            state.pace.budget = Some(speed::actor_budget(per_frame));
        },
        PhysicsCommand::SetCouplingForces(forces) => sim.set_coupling_forces(forces),
        PhysicsCommand::SetAffinityForce(force) => sim.set_affinity_force(force),
        PhysicsCommand::SetAnchorForce(force) => sim.set_anchor_force(force),
        PhysicsCommand::SetForces(forces) => sim.set_forces(forces),
        PhysicsCommand::SetLaggedRepulsion {
            solver,
            threshold,
            max_stale_steps,
        } => sim.set_lagged_repulsion(solver, threshold, max_stale_steps),
        PhysicsCommand::SetLinearDamping(damping) => sim.set_linear_damping(damping),
        PhysicsCommand::SetNodeColliders(colliders) => sim.set_node_colliders(colliders),
        PhysicsCommand::SetNodeMaterials(materials) => sim.set_node_materials(materials),
        PhysicsCommand::SetAxisLocks(locks) => sim.set_axis_locks(locks),
        PhysicsCommand::AddSceneBody(collider, position, velocity) => {
            sim.add_scene_body(collider, position, velocity);
        },
        PhysicsCommand::SetNodesTangible(tangible) => sim.set_nodes_tangible(tangible),
        PhysicsCommand::LoadScene(spec) => sim.load_scene(&spec),
        PhysicsCommand::ClearScene => sim.clear_scene(),
        PhysicsCommand::LoadFluid {
            params,
            basin,
            origin,
            cols,
            rows,
            spacing,
        } => sim.load_fluid(params, basin, origin, cols, rows, spacing),
        PhysicsCommand::ClearFluid => sim.clear_fluid(),
        PhysicsCommand::SetSceneField(field) => sim.set_scene_field(field),
        PhysicsCommand::AddEmitter(spec) => sim.add_emitter(spec),
        PhysicsCommand::ClearEmitters => sim.clear_emitters(),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use armillary::{Wake, spawn};
    use euclid::default::Point2D;

    use super::*;
    use crate::NodeKey;

    /// The actor processes a sync + settle and emits layout snapshots, then ends
    /// cleanly when the handle drops. (The physics math itself is seiche's concern;
    /// this is a protocol smoke test of the run loop.)
    #[test]
    fn actor_syncs_settles_and_emits_snapshots() {
        let (a, b) = (NodeKey::new(0), NodeKey::new(1));

        let mut sim = Simulation::new();
        sim.add_force(crate::NodeExclusion::default());
        let wake: Wake = Arc::new(|| {});
        let (handle, updates) = spawn(wake, move |commands, out| {
            run(sim, ActorState::fresh(0, false, false), commands, out)
        });

        handle.command(PhysicsCommand::SyncNodes(vec![
            (a, Point2D::new(0.0, 0.0)),
            (b, Point2D::new(1.0, 0.0)),
        ]));
        handle.command(PhysicsCommand::SyncEdges(vec![(a, b)]));
        handle.command(PhysicsCommand::Settle(4));
        // Dropping the handle closes the command channel; the actor finishes its
        // settle, then ends. `iter` collects every emitted update to completion.
        drop(handle);

        let emitted: Vec<PhysicsUpdate> = updates.iter().collect();
        assert!(
            !emitted.is_empty(),
            "the actor emitted at least one layout snapshot"
        );
        let last = emitted.last().unwrap();
        assert_eq!(
            last.snapshot.positions.len(),
            2,
            "both bodies are in the snapshot"
        );
    }
}
