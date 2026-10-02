// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! The lagged seam's receipts, on evaluators whose timing the test controls:
//! when an answer applies, when it is too stale, what a failure does, and
//! that a refused device leaves exactly the CPU run.

use super::*;

/// Sequential node keys and their seed positions.
#[derive(Default)]
struct Nodes {
    list: Vec<(NodeKey, Point2D<f32>)>,
}

impl Nodes {
    fn at(&mut self, x: f32, y: f32) {
        self.list
            .push((NodeKey::new(self.list.len()), Point2D::new(x, y)));
    }

    fn keys(&self) -> Vec<NodeKey> {
        self.list.iter().map(|(k, _)| *k).collect()
    }

    fn sync(&self, sim: &mut Simulation) {
        sim.sync_nodes(self.list.iter().copied());
    }
}

/// The step clock the mocks read: the test advances it after every tick, so
/// during tick k it reads k, the step the lane stamps on submissions.
type Clock = std::sync::Arc<std::sync::atomic::AtomicU64>;

fn now(clock: &Clock) -> u64 {
    clock.load(std::sync::atomic::Ordering::SeqCst)
}

/// A lagged evaluator that computes the CPU law at submission and answers
/// `delay` steps later (1 = on the next step), in submission order. `push` replaces the law with
/// a +x shove the symmetric law can never produce, to prove forces landed.
/// `refuse` fails every submission; `fail_answers` answers with an error.
struct Mock {
    clock: Clock,
    delay: u64,
    push: bool,
    refuse: bool,
    fail_answers: bool,
    waiting: std::collections::VecDeque<(u64, RepulsionForces)>,
}

impl Mock {
    fn after(clock: &Clock, delay: u64) -> Self {
        Self {
            clock: clock.clone(),
            delay,
            push: false,
            refuse: false,
            fail_answers: false,
            waiting: Default::default(),
        }
    }
}

impl LaggedRepulsion for Mock {
    fn submit(
        &mut self,
        xs: &[f32],
        ys: &[f32],
        request: RepulsionRequest,
    ) -> Result<(), RepulsionSolverError> {
        if self.refuse {
            return Err(RepulsionSolverError::Backend("no adapter".into()));
        }
        let (fx, fy) = if self.push {
            (vec![1_000_000.0; xs.len()], vec![0.0; xs.len()])
        } else {
            node_exclusion_reference(
                xs,
                ys,
                NodeExclusionParams {
                    strength: request.strength,
                    cutoff: request.cutoff,
                    min_distance: request.min_distance,
                },
            )
            .unwrap()
        };
        let ready = now(&self.clock) + self.delay;
        self.waiting
            .push_back((ready, RepulsionForces::new(xs.len(), fx, fy)?));
        Ok(())
    }

    fn poll(&mut self) -> Option<Result<RepulsionForces, RepulsionSolverError>> {
        let (ready, _) = self.waiting.front()?;
        if *ready > now(&self.clock) {
            return None;
        }
        let (_, forces) = self.waiting.pop_front()?;
        if self.fail_answers {
            return Some(Err(RepulsionSolverError::Backend("device lost".into())));
        }
        Some(Ok(forces))
    }

    fn in_flight(&self) -> usize {
        self.waiting.len()
    }
}

/// Fifty unlinked bodies under the Springs trio, seeded in a spiral.
fn spiral(count: usize) -> Nodes {
    let mut nodes = Nodes::default();
    for i in 0..count {
        let angle = i as f32 * 2.399_963;
        let radius = 40.0 * (i as f32).sqrt();
        nodes.at(radius * angle.cos(), radius * angle.sin());
    }
    nodes
}

fn trio(nodes: &Nodes) -> Simulation {
    let mut sim = Simulation::new();
    sim.add_force(NodeExclusion::default());
    sim.add_force(EdgeSpring::default());
    sim.add_force(Boundary::default());
    nodes.sync(&mut sim);
    sim
}

fn run(sim: &mut Simulation, clock: &Clock, ticks: usize) {
    for _ in 0..ticks {
        sim.tick(1.0 / 60.0);
        clock.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }
}

#[test]
fn an_answer_applies_from_the_step_after_its_positions() {
    let nodes = spiral(50);
    let clock = Clock::default();
    let mut sim = trio(&nodes);
    sim.set_lagged_repulsion(Some(Box::new(Mock::after(&clock, 1))), 0, 1);
    run(&mut sim, &clock, 60);
    let stats = sim.repulsion_stats().unwrap();
    // The first step has nothing to apply yet; every later one has the
    // previous step's answer, exactly one step old.
    assert_eq!(stats.cpu_steps, 1, "{stats:?}");
    assert_eq!(stats.device_steps, 59, "{stats:?}");
    assert_eq!(stats.submissions, 60, "{stats:?}");
    assert_eq!(stats.failures + stats.mismatched, 0, "{stats:?}");
}

#[test]
fn an_answer_older_than_the_limit_leaves_the_step_to_the_cpu() {
    // Answers take three steps to arrive. With a limit of one step none is
    // ever fresh enough; with a limit of three they all are.
    let nodes = spiral(50);
    let clock = Clock::default();
    let mut strict = trio(&nodes);
    strict.set_lagged_repulsion(Some(Box::new(Mock::after(&clock, 3))), 0, 1);
    run(&mut strict, &clock, 60);
    let stats = strict.repulsion_stats().unwrap();
    assert_eq!(
        stats.device_steps, 0,
        "a three-step-old answer applied: {stats:?}"
    );
    assert_eq!(stats.cpu_steps, 60, "{stats:?}");

    let clock = Clock::default();
    let mut patient = trio(&nodes);
    patient.set_lagged_repulsion(Some(Box::new(Mock::after(&clock, 3))), 0, 3);
    run(&mut patient, &clock, 60);
    let stats = patient.repulsion_stats().unwrap();
    // Three in flight: every step submits, every step from the third on
    // holds an answer exactly three steps old, and only the first three
    // steps wait on the CPU.
    assert_eq!(stats.cpu_steps, 3, "{stats:?}");
    assert_eq!(stats.device_steps, 57, "{stats:?}");
    assert_eq!(stats.submissions, 60, "{stats:?}");
}

#[test]
fn a_refused_device_leaves_the_cpu_run() {
    // The forced adapter failure: every submission refused. No device force
    // may apply, and the layout must be the CPU law's. Two CPU runs are not
    // bit-identical (NodeExclusion sums in a HashMap's per-instance order),
    // so the bar is the CPU-against-CPU spread measured here, with a floor.
    let nodes = spiral(50);
    let clock = Clock::default();
    let mut cpu = trio(&nodes);
    run(&mut cpu, &Clock::default(), 120);
    let mut again = trio(&nodes);
    run(&mut again, &Clock::default(), 120);

    let mut refused = trio(&nodes);
    let mut mock = Mock::after(&clock, 1);
    mock.refuse = true;
    refused.set_lagged_repulsion(Some(Box::new(mock)), 0, 1);
    run(&mut refused, &clock, 120);
    let stats = refused.repulsion_stats().unwrap();
    assert_eq!(stats.device_steps, 0, "{stats:?}");
    assert_eq!(stats.cpu_steps, 120, "{stats:?}");
    assert_eq!(stats.failures, 120, "{stats:?}");

    let gap = |other: &Simulation| {
        nodes
            .keys()
            .iter()
            .map(|&key| (cpu.position_of(key).unwrap() - other.position_of(key).unwrap()).length())
            .fold(0.0f32, f32::max)
    };
    let noise = gap(&again);
    let refused_gap = gap(&refused);
    assert!(
        refused_gap <= (noise * 10.0).max(0.01),
        "the refused-device run left the CPU path: {refused_gap} px against CPU noise {noise} px"
    );
}

#[test]
fn an_answer_that_fails_is_counted_and_covered() {
    let nodes = spiral(50);
    let clock = Clock::default();
    let mut sim = trio(&nodes);
    let mut mock = Mock::after(&clock, 1);
    mock.fail_answers = true;
    sim.set_lagged_repulsion(Some(Box::new(mock)), 0, 1);
    run(&mut sim, &clock, 30);
    let stats = sim.repulsion_stats().unwrap();
    assert_eq!(stats.device_steps, 0, "{stats:?}");
    assert_eq!(stats.cpu_steps, 30, "{stats:?}");
    assert_eq!(stats.failures, 29, "{stats:?}");
}

#[test]
fn an_answer_for_a_changed_body_set_is_discarded() {
    let mut nodes = spiral(50);
    let clock = Clock::default();
    let mut sim = trio(&nodes);
    sim.set_lagged_repulsion(Some(Box::new(Mock::after(&clock, 1))), 0, 1);
    run(&mut sim, &clock, 10);
    nodes.at(5_000.0, 5_000.0);
    nodes.sync(&mut sim);
    run(&mut sim, &clock, 10);
    let stats = sim.repulsion_stats().unwrap();
    // Step 11 holds an answer for the fifty bodies; with fifty-one it must
    // not apply, so that step runs the CPU law.
    assert_eq!(stats.mismatched, 1, "{stats:?}");
    assert_eq!(stats.cpu_steps, 2, "{stats:?}");
    assert_eq!(stats.device_steps, 18, "{stats:?}");
}

#[test]
fn lagged_forces_reach_the_bodies_and_respect_the_threshold() {
    let nodes = spiral(50);
    let clock = Clock::default();
    let mut below = trio(&nodes);
    let mut mock = Mock::after(&clock, 1);
    mock.push = true;
    below.set_lagged_repulsion(Some(Box::new(mock)), 51, 1);
    run(&mut below, &clock, 10);
    let stats = below.repulsion_stats().unwrap();
    assert_eq!(stats, LaggedStats::default(), "used below its threshold");

    let clock = Clock::default();
    let mut above = trio(&nodes);
    let mut mock = Mock::after(&clock, 1);
    mock.push = true;
    above.set_lagged_repulsion(Some(Box::new(mock)), 50, 1);
    run(&mut above, &clock, 30);
    let keys = nodes.keys();
    let mean_x = keys
        .iter()
        .filter_map(|&k| above.position_of(k).map(|p| p.x))
        .sum::<f32>()
        / keys.len() as f32;
    assert!(above.repulsion_stats().unwrap().device_steps > 0);
    assert!(
        mean_x > 100.0,
        "the lagged +x push never reached the bodies: mean x {mean_x}"
    );
}
