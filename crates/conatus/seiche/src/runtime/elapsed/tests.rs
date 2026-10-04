// Copyright 2026 Mark Alan Boykin
// SPDX-License-Identifier: MPL-2.0

use super::*;
use crate::{NodeExclusion, NodeKey, Simulation};
use euclid::default::Point2D;

fn running() -> (Physics, LayoutView) {
    let mut sim = Simulation::new();
    sim.sync_nodes(vec![
        (NodeKey::new(0), Point2D::new(-80.0, 0.0)),
        (NodeKey::new(1), Point2D::new(80.0, 0.0)),
    ]);
    sim.add_force(NodeExclusion::default());
    let view = sim.view();
    (Physics::inline(sim, 10_000), view)
}

fn config() -> ElapsedStepConfig {
    ElapsedStepConfig {
        max_elapsed: TICK_DURATION * 4,
        max_steps: 4,
    }
}

fn run_schedule(schedule: impl IntoIterator<Item = Duration>) -> (u32, Vec<Point2D<f32>>) {
    let (mut physics, mut view) = running();
    let mut steps = 0;
    for elapsed in schedule {
        let report = physics.advance_elapsed(&mut view, elapsed, config());
        assert_eq!(report.discarded_elapsed, Duration::ZERO);
        steps += report.steps;
    }
    let positions = [NodeKey::new(0), NodeKey::new(1)]
        .map(|key| view.position_of(key).unwrap())
        .to_vec();
    (steps, positions)
}

#[test]
fn different_render_rates_integrate_the_same_elapsed_ticks() {
    let full = run_schedule(std::iter::repeat_n(TICK_DURATION, 60));
    let slow = run_schedule(std::iter::repeat_n(TICK_DURATION * 2, 30));
    let half = TICK_DURATION / 2;
    // Alternate the two integer halves so rounding cannot lose 60 ns.
    let fast = run_schedule((0..60).flat_map(|_| [half, TICK_DURATION - half]));
    assert_eq!(full.0, 60);
    assert_eq!(full, slow);
    assert_eq!(full, fast);
}

#[test]
fn stalls_drop_excess_time_and_never_queue_whole_step_debt() {
    let (mut physics, mut view) = running();
    let report = physics.advance_elapsed(
        &mut view,
        Duration::from_secs(5),
        ElapsedStepConfig {
            max_elapsed: TICK_DURATION * 3,
            max_steps: 2,
        },
    );
    assert_eq!(report.steps, 2);
    assert_eq!(
        report.discarded_elapsed,
        Duration::from_secs(5) - TICK_DURATION * 2
    );
    assert_eq!(report.carried_elapsed, Duration::ZERO);
    assert_eq!(
        physics
            .advance_elapsed(&mut view, Duration::ZERO, config())
            .steps,
        0
    );
}

#[test]
fn zero_limits_drop_whole_steps_and_keep_only_fractional_time() {
    let (mut physics, mut view) = running();
    let half = TICK_DURATION / 2;
    let report = physics.advance_elapsed(
        &mut view,
        TICK_DURATION * 5 + half,
        ElapsedStepConfig {
            max_elapsed: Duration::from_secs(1),
            max_steps: 0,
        },
    );
    assert_eq!(report.steps, 0);
    assert_eq!(report.discarded_elapsed, TICK_DURATION * 5);
    assert_eq!(report.carried_elapsed, half);
    let report = physics.advance_elapsed(
        &mut view,
        Duration::from_secs(9),
        ElapsedStepConfig {
            max_elapsed: Duration::ZERO,
            max_steps: 4,
        },
    );
    assert_eq!(report.steps, 0);
    assert_eq!(report.discarded_elapsed, Duration::from_secs(9));
    assert_eq!(report.carried_elapsed, half);
    assert_eq!(
        physics
            .advance_elapsed(&mut view, TICK_DURATION - half, config())
            .steps,
        1
    );
}

#[test]
fn nominal_sixty_hz_rounding_is_explicit_at_nanosecond_boundary() {
    assert_eq!(TICK_DURATION.as_nanos(), 16_666_667);
    assert_eq!(
        TICK_DURATION * 60,
        Duration::from_secs(1) + Duration::from_nanos(20)
    );
    let (mut physics, mut view) = running();
    let almost = TICK_DURATION - Duration::from_nanos(1);
    assert_eq!(
        physics.advance_elapsed(&mut view, almost, config()).steps,
        0
    );
    assert_eq!(
        physics
            .advance_elapsed(&mut view, Duration::from_nanos(1), config())
            .steps,
        1
    );
}

#[test]
fn halt_seed_suspend_and_deterministic_mode_clear_fractional_debt() {
    let half = TICK_DURATION / 2;
    for reset in 0..4 {
        let (mut physics, mut view) = running();
        assert_eq!(physics.advance_elapsed(&mut view, half, config()).steps, 0);
        match reset {
            0 => {
                physics.halt();
                let frozen = view.position_of(NodeKey::new(0));
                let paused = physics.advance_elapsed(&mut view, Duration::from_secs(10), config());
                assert_eq!(paused.discarded_elapsed, Duration::from_secs(10));
                assert_eq!(paused.carried_elapsed, Duration::ZERO);
                assert_eq!(paused.steps, 0);
                assert_eq!(view.position_of(NodeKey::new(0)), frozen);
                physics.settle(10_000);
            },
            1 => physics.seed(vec![(NodeKey::new(0), Point2D::new(-60.0, 0.0))]),
            2 => physics.reset_elapsed(),
            _ => {
                physics.advance_frame(&mut view);
            },
        }
        assert_eq!(
            physics
                .advance_elapsed(&mut view, TICK_DURATION - half, config())
                .steps,
            0
        );
    }
}

#[test]
fn settling_end_drops_unused_elapsed_including_fraction() {
    let (mut physics, mut view) = running();
    physics.halt();
    physics.settle(1);
    let elapsed = TICK_DURATION * 3 + TICK_DURATION / 2;
    let report = physics.advance_elapsed(&mut view, elapsed, config());
    assert_eq!(report.steps, 1);
    assert!(!report.settling);
    assert_eq!(report.carried_elapsed, Duration::ZERO);
    assert_eq!(report.discarded_elapsed, elapsed - TICK_DURATION);
    physics.settle(1);
    assert_eq!(
        physics
            .advance_elapsed(&mut view, Duration::ZERO, config())
            .steps,
        0
    );
}

#[test]
fn deterministic_calls_still_integrate_exactly_one_step() {
    let (mut physics, mut view) = running();
    for _ in 0..60 {
        assert!(physics.advance_frame(&mut view));
    }
    let positions = [NodeKey::new(0), NodeKey::new(1)]
        .map(|key| view.position_of(key).unwrap())
        .to_vec();
    assert_eq!(
        positions,
        run_schedule(std::iter::repeat_n(TICK_DURATION, 60)).1
    );
}

#[cfg(feature = "actor")]
#[test]
fn actor_elapsed_calls_only_drain_and_send_no_commands() {
    use crate::runtime::{ActorPhysics, PhysicsCommand, PhysicsUpdate};
    use armillary::{Emitter, spawn};
    use std::sync::{Arc, mpsc};

    let (observed_tx, observed) = mpsc::channel();
    let (handle, _) = spawn(
        Arc::new(|| {}),
        move |commands, _out: Emitter<PhysicsUpdate>| {
            while let Ok(command) = commands.recv() {
                observed_tx.send(command).unwrap();
            }
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
        dial: crate::Speed::REAL_TIME,
        pace: crate::PaceStats::default(),
    });
    let mut sim = Simulation::new();
    let node = NodeKey::new(0);
    let position = Point2D::new(123.0, 456.0);
    sim.sync_nodes(vec![(node, position)]);
    let mut view = LayoutView::new();
    updates_tx
        .send(PhysicsUpdate {
            snapshot: sim.snapshot(1),
            settling: false,
            command_epoch: 0,
            pace: crate::PaceStats::default(),
        })
        .unwrap();
    for _ in 0..3 {
        let report = physics.advance_elapsed(&mut view, Duration::from_secs(60), config());
        assert_eq!(report, ElapsedStepReport::default());
        assert_eq!(view.position_of(node), Some(position));
        physics.reset_elapsed();
    }
    // FIFO marker acknowledges that every earlier command would be observed.
    if let Physics::Actor(p) = &physics {
        p.handle.command(PhysicsCommand::Barrier(123));
    }
    assert!(matches!(
        observed.recv_timeout(Duration::from_secs(2)).unwrap(),
        PhysicsCommand::Barrier(123)
    ));
    assert!(observed.try_recv().is_err());
}
