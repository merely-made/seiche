// Copyright 2026 Mark Alan Boykin
// SPDX-License-Identifier: MPL-2.0

//! The dial's receipts: one trajectory at every speed (two 1x runs agreeing
//! first, and a different-dt run as the positive control), settle budgets in
//! ticks, slow motion drawn between ticks, and fast-forward under a budget
//! (with a cheaper tick the budget does not bind as the control).
//!
//! Bit identity needs reproducible forces. `NodeExclusion` and
//! `BarnesHutRepulsion` summed in `HashMap` order until 2026-10-04 ("Sum in
//! key order"); [`every_force_is_reproducible_run_to_run`] holds every force
//! set to it now. The trajectory receipts run Springs and Charge (the two
//! that were not), LinLog, and Anneal, whose seeded walk and per-tick cooling
//! change on any extra or missing tick.

use std::cell::Cell;
use std::time::Duration;

use euclid::default::Point2D;

use super::super::{ElapsedStepConfig, LayoutView, Physics};
use super::{Speed, StepBudget, TICK_DT, TICK_DURATION};
use crate::{
    Anneal, BarnesHutRepulsion, Boundary, EdgeSpring, Force, LinLogForce, NodeExclusion, NodeKey,
    Simulation,
};

const NODES: u32 = 24;
const TICKS: u32 = 600;

#[derive(Clone, Copy, Debug)]
enum Set {
    Springs,
    Charge,
    LinLog,
    Anneal,
}

fn forces(set: Set) -> Vec<Box<dyn Force>> {
    match set {
        Set::Springs => vec![
            Box::new(NodeExclusion::default()),
            Box::new(EdgeSpring::default()),
            Box::new(Boundary::default()),
        ],
        Set::Charge => vec![
            Box::new(BarnesHutRepulsion::default()),
            Box::new(EdgeSpring::default()),
            Box::new(Boundary::default()),
        ],
        Set::LinLog => vec![
            Box::new(LinLogForce::default()),
            Box::new(EdgeSpring::default()),
            Box::new(Boundary::default()),
        ],
        Set::Anneal => vec![Box::new(Anneal::seeded(7))],
    }
}

/// A golden-angle spiral with ring and chord edges, inserted in key order.
fn bodies() -> Simulation {
    let mut sim = Simulation::new();
    sim.sync_nodes(
        (0..NODES)
            .map(|i| {
                let a = i as f32 * 2.399_963;
                let r = 30.0 * (i as f32 + 0.5).sqrt();
                (
                    NodeKey::new(i as usize),
                    Point2D::new(r * a.cos(), r * a.sin()),
                )
            })
            .collect::<Vec<_>>(),
    );
    sim.sync_edges(
        (0..NODES)
            .flat_map(|i| {
                let a = NodeKey::new(i as usize);
                [
                    (a, NodeKey::new(((i + 1) % NODES) as usize)),
                    (a, NodeKey::new(((i * 7 + 3) % NODES) as usize)),
                ]
            })
            .collect::<Vec<_>>(),
    );
    sim
}

fn sim(set: Set) -> Simulation {
    let mut sim = bodies();
    sim.set_forces(forces(set));
    sim
}

/// Positions sorted by key, as bits.
fn bits(view: &LayoutView) -> Vec<(usize, u32, u32)> {
    let mut out: Vec<_> = view
        .positions()
        .map(|(k, p)| (k.index(), p.x.to_bits(), p.y.to_bits()))
        .collect();
    out.sort_unstable();
    out
}

/// Settle `TICKS` at `speed` frame by frame until the budget is spent;
/// returns the frames it took, the ticks run, and the final positions.
fn settle_at(set: Set, speed: Speed) -> (u32, u64, Vec<(usize, u32, u32)>) {
    let sim = sim(set);
    let mut view = sim.view();
    let mut physics = Physics::inline(sim, TICKS);
    physics.set_speed(speed);
    let mut frames = 1;
    while physics.advance_frame(&mut view) {
        frames += 1;
        assert!(frames < 10 * TICKS, "the settle never ended");
    }
    (frames, physics.pace().ticks, bits(&view))
}

/// The same through the elapsed driver at 60 Hz frames.
fn settle_elapsed_at(set: Set, speed: Speed) -> (u64, Vec<(usize, u32, u32)>) {
    let sim = sim(set);
    let mut view = sim.view();
    let mut physics = Physics::inline(sim, TICKS);
    physics.set_speed(speed);
    let config = ElapsedStepConfig {
        max_elapsed: TICK_DURATION * 4,
        max_steps: 4,
    };
    let mut frames = 0;
    while physics
        .advance_elapsed(&mut view, TICK_DURATION, config)
        .settling
    {
        frames += 1;
        assert!(frames < 10 * TICKS, "the settle never ended");
    }
    (physics.pace().ticks, bits(&view))
}

#[test]
fn speeds_clamp_to_the_ruled_range_in_thousandths() {
    assert_eq!(Speed::from_factor(0.2).milli(), 200);
    assert_eq!(Speed::from_factor(50.0).milli(), 50_000);
    assert_eq!(Speed::from_factor(0.05), Speed::MIN);
    assert_eq!(Speed::from_factor(400.0), Speed::MAX);
    assert_eq!(Speed::from_factor(f32::NAN), Speed::REAL_TIME);
    assert_eq!(Speed::default(), Speed::REAL_TIME);
}

#[test]
fn one_trajectory_at_every_speed_and_a_different_dt_differs() {
    for set in [Set::Springs, Set::Charge, Set::LinLog, Set::Anneal] {
        // Two real-time runs agree bit for bit, or no comparison across
        // speeds would mean anything.
        let (frames_1x, ticks_1x, at_1x) = settle_at(set, Speed::REAL_TIME);
        assert_eq!(
            settle_at(set, Speed::REAL_TIME).2,
            at_1x,
            "{set:?}: 1x twice"
        );
        let (frames_slow, ticks_slow, slow) = settle_at(set, Speed::from_factor(0.2));
        let (frames_fast, ticks_fast, fast) = settle_at(set, Speed::from_factor(50.0));
        // Settle budgets count ticks: every speed runs exactly the budget,
        // over five times the frames at 0.2x and a fiftieth at 50x.
        assert_eq!(
            (ticks_1x, ticks_slow, ticks_fast),
            (600, 600, 600),
            "{set:?}"
        );
        assert_eq!(
            (frames_1x, frames_slow, frames_fast),
            (600, 2995, 12),
            "{set:?}"
        );
        assert_eq!(slow, at_1x, "{set:?}: 0.2x left the 1x trajectory");
        assert_eq!(fast, at_1x, "{set:?}: 50x left the 1x trajectory");
        for speed in [0.2, 3.7, 50.0] {
            let (ticks, positions) = settle_elapsed_at(set, Speed::from_factor(speed));
            assert_eq!(ticks, 600, "{set:?} elapsed at {speed}x");
            assert_eq!(positions, at_1x, "{set:?}: elapsed at {speed}x left it");
        }

        // Positive control: the same simulated ten seconds at dt = 1/30 must
        // not land on the same bits, or the comparisons above could not fail.
        let mut other = sim(set);
        for _ in 0..TICKS / 2 {
            other.tick(1.0 / 30.0);
        }
        let differs = bits(&other.view())
            .iter()
            .zip(&at_1x)
            .filter(|(a, b)| a != b)
            .count();
        assert!(differs > 0, "{set:?}: a different dt matched");
        println!("{set:?}: dt 1/30 differs on {differs} of {NODES} bodies");
    }
}

#[test]
fn slow_motion_draws_every_frame_and_steps_every_fifth() {
    let sim = sim(Set::LinLog);
    let mut view = sim.view();
    let mut physics = Physics::inline(sim, TICKS);
    physics.set_speed(Speed::from_factor(0.2));
    let mut last = bits(&view);
    let mut stepped = Vec::new();
    for frame in 1..=50 {
        let before = physics.pace().ticks;
        assert!(physics.advance_frame(&mut view));
        if physics.pace().ticks > before {
            stepped.push(frame);
        }
        let drawn = bits(&view);
        assert_ne!(drawn, last, "frame {frame} drew nothing new");
        last = drawn;
    }
    // The lead tick on the first frame, then one every fifth.
    assert_eq!(stepped, vec![1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]);
    // Draw only: the simulation under the drawing is the 1x trajectory at the
    // same tick, here the eleventh, and the drawing sits a tick behind it.
    let mut reference = self::sim(Set::LinLog);
    for _ in 0..11 {
        reference.tick(TICK_DT);
    }
    let simulated = match &physics {
        Physics::Inline(inline) => bits(&inline.sim.view()),
        #[cfg(feature = "actor")]
        Physics::Actor(_) => unreachable!(),
    };
    assert_eq!(simulated, bits(&reference.view()));
    let mut tenth = self::sim(Set::LinLog);
    for _ in 0..10 {
        tenth.tick(TICK_DT);
    }
    assert_eq!(
        bits(&view),
        bits(&tenth.view()),
        "alpha 0 draws the tick before"
    );
    let reached = physics.pace().effective_speed.unwrap();
    assert!((0.18..=0.22).contains(&reached), "reached {reached}");
}

/// A host that resets elapsed time before each deterministic frame (the
/// canvas's `frame`) keeps the slow speed's fraction.
#[test]
fn resetting_elapsed_time_keeps_a_deterministic_fraction() {
    let sim = sim(Set::LinLog);
    let mut view = sim.view();
    let mut physics = Physics::inline(sim, TICKS);
    physics.set_speed(Speed::from_factor(0.2));
    let mut stepped = Vec::new();
    for frame in 1..=10 {
        physics.reset_elapsed();
        let before = physics.pace().ticks;
        physics.advance_frame(&mut view);
        if physics.pace().ticks > before {
            stepped.push(frame);
        }
    }
    assert_eq!(stepped, vec![1, 5, 10]);
}

#[test]
fn at_real_time_nothing_is_drawn_between() {
    let sim = sim(Set::LinLog);
    let mut view = sim.view();
    let mut physics = Physics::inline(sim, TICKS);
    let mut reference = self::sim(Set::LinLog);
    for _ in 0..30 {
        physics.advance_frame(&mut view);
        reference.tick(TICK_DT);
        assert_eq!(bits(&view), bits(&reference.view()));
    }
}

thread_local! {
    static NOW_US: Cell<u64> = const { Cell::new(0) };
}

/// Virtual time that only a tick advances: `Costly` adds its cost to it.
fn virtual_clock() -> Duration {
    Duration::from_micros(NOW_US.with(Cell::get))
}

/// A force that costs `us` microseconds of virtual time a tick.
struct Costly {
    us: u64,
}

impl crate::Declared for Costly {
    fn terms(&self) -> Vec<crate::Term> {
        Vec::new()
    }
}

impl Force for Costly {
    fn apply(&self, _: &mut crate::ForceContext<'_>, _: f32) {
        NOW_US.with(|now| now.set(now.get() + self.us));
    }
}

fn costly(us: u64, budget: Duration, speed: f32) -> (Physics, LayoutView) {
    let mut sim = sim(Set::LinLog);
    sim.add_force(Costly { us });
    let view = sim.view();
    let mut physics = Physics::inline(sim, TICKS);
    physics.set_speed(Speed::from_factor(speed));
    physics.set_step_budget(Some(StepBudget {
        per_frame: budget,
        clock: virtual_clock,
    }));
    (physics, view)
}

#[test]
fn fast_forward_stops_at_the_budget_and_reports_the_speed_it_reached() {
    let config = ElapsedStepConfig {
        max_elapsed: TICK_DURATION * 4,
        max_steps: 3,
    };
    let budget = Duration::from_millis(8);
    // A 1 ms tick against an 8 ms budget at 50x: eight ticks a frame.
    let (mut physics, mut view) = costly(1_000, budget, 50.0);
    for _ in 0..8 {
        let report = physics.advance_elapsed(&mut view, TICK_DURATION, config);
        assert_eq!(report.owed_steps, 50);
        assert!(report.budget_bound);
        assert_eq!(report.steps, 8);
        assert!(report.compute.unwrap() <= budget, "{:?}", report.compute);
    }
    let reached = physics.pace().effective_speed.unwrap();
    assert!((reached - 8.0).abs() < 1e-3, "reached {reached}");
    assert!(physics.pace().budget_bound);

    // Control: a 0.1 ms tick fits fifty in the same budget; nothing binds and
    // the speed reached is the speed asked.
    let (mut physics, mut view) = costly(100, budget, 50.0);
    for _ in 0..8 {
        let report = physics.advance_elapsed(&mut view, TICK_DURATION, config);
        assert!(!report.budget_bound);
        assert_eq!(report.steps, 50);
        assert!(report.compute.unwrap() <= budget);
    }
    assert!((physics.pace().effective_speed.unwrap() - 50.0).abs() < 1e-3);

    // A tick dearer than the whole budget: 50x still runs what 1x would in a
    // three-tick frame, never fewer.
    let (mut physics, mut view) = costly(20_000, budget, 50.0);
    let report = physics.advance_elapsed(&mut view, TICK_DURATION * 3, config);
    assert_eq!((report.steps, report.budget_bound), (3, true));
    let (mut physics, mut view) = costly(20_000, budget, 1.0);
    let report = physics.advance_elapsed(&mut view, TICK_DURATION * 3, config);
    assert_eq!((report.steps, report.budget_bound), (3, false));

    // At real time the budget does not apply: the 1x caps govern alone.
    let (mut physics, mut view) = costly(1_000, Duration::ZERO, 1.0);
    let report = physics.advance_elapsed(&mut view, TICK_DURATION * 3, config);
    assert_eq!((report.steps, report.budget_bound), (3, false));
}

/// Max runs as many ticks as the budget allows, past 50x when the ticks are
/// cheap; with no budget it runs at 50x (the control: it cannot run away).
#[test]
fn uncapped_runs_until_the_budget_is_spent() {
    let config = ElapsedStepConfig {
        max_elapsed: TICK_DURATION * 4,
        max_steps: 3,
    };
    let budget = Duration::from_millis(8);
    let (mut fifty, mut view) = costly(100, budget, 50.0);
    let capped = fifty.advance_elapsed(&mut view, TICK_DURATION, config);
    assert_eq!((capped.steps, capped.budget_bound), (50, false));

    let (mut max, mut view) = costly(100, budget, 50.0);
    max.set_speed(Speed::UNCAPPED);
    // Six frames of 80 stay inside the 600-tick settle.
    for _ in 0..6 {
        let report = max.advance_elapsed(&mut view, TICK_DURATION, config);
        assert_eq!((report.steps, report.budget_bound), (80, true));
        assert!(report.compute.unwrap() <= budget, "{:?}", report.compute);
    }
    let reached = max.pace().effective_speed.unwrap();
    assert!((reached - 80.0).abs() < 1e-3, "reached {reached}");
    assert!(Speed::UNCAPPED.factor().is_infinite());

    // Control: no budget installed, Max steps at 50x.
    let sim = sim(Set::LinLog);
    let mut view = sim.view();
    let mut unbudgeted = Physics::inline(sim, TICKS);
    unbudgeted.set_speed(Speed::UNCAPPED);
    let report = unbudgeted.advance_elapsed(&mut view, TICK_DURATION, config);
    assert_eq!((report.steps, report.budget_bound), (50, false));
    assert!(unbudgeted.advance_frame(&mut view));
    assert_eq!(unbudgeted.pace().ticks, 100, "the deterministic driver too");
}

#[test]
fn a_seed_snaps_instead_of_gliding_in_slow_motion() {
    let sim = sim(Set::LinLog);
    let mut view = sim.view();
    let mut physics = Physics::inline(sim, TICKS);
    physics.set_speed(Speed::from_factor(0.2));
    for _ in 0..3 {
        physics.advance_frame(&mut view);
    }
    let node = NodeKey::new(0);
    let target = Point2D::new(900.0, -900.0);
    physics.seed(vec![(node, target)]);
    physics.refresh(&mut view);
    assert_eq!(view.position_of(node), Some(target));
}

#[cfg(feature = "actor")]
#[test]
fn the_actor_keeps_the_trajectory_at_fifty_times() {
    use std::sync::Arc;

    let (_, _, at_1x) = settle_at(Set::LinLog, Speed::REAL_TIME);
    let sim = sim(Set::LinLog);
    let mut view = sim.view();
    let mut physics = Physics::inline(sim, TICKS);
    physics.set_speed(Speed::from_factor(50.0));
    physics.offload(Arc::new(|| {}));
    let deadline = std::time::Instant::now() + Duration::from_secs(60);
    let mut fastest = 0.0f32;
    loop {
        let settling = physics.advance_frame(&mut view);
        if let Some(speed) = physics.pace().effective_speed {
            fastest = fastest.max(speed);
        }
        if !settling && physics.pace().ticks == u64::from(TICKS) {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "the actor never finished"
        );
        std::thread::sleep(Duration::from_millis(2));
    }
    assert_eq!(bits(&view), at_1x, "the actor left the 1x trajectory");
    assert!(
        fastest > 1.0,
        "the actor never ran above real time ({fastest})"
    );
}

#[cfg(feature = "actor")]
#[test]
fn the_actor_draws_between_ticks_in_slow_motion() {
    use std::sync::Arc;

    let sim = sim(Set::LinLog);
    let mut view = sim.view();
    let mut physics = Physics::inline(sim, TICKS);
    physics.set_speed(Speed::from_factor(0.2));
    physics.offload(Arc::new(|| {}));
    let mut drawings = 0u32;
    let mut last = bits(&view);
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while physics.pace().ticks < 8 && std::time::Instant::now() < deadline {
        physics.advance_frame(&mut view);
        let now = bits(&view);
        if now != last {
            drawings += 1;
            last = now;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    let ticks = physics.pace().ticks;
    assert!(ticks >= 8, "only {ticks} ticks in ten seconds");
    // Five intervals a tick, each its own drawing: well over two per tick.
    assert!(
        u64::from(drawings) > 2 * ticks,
        "{drawings} drawings over {ticks} ticks"
    );
}

type Make = Box<dyn Fn() -> Vec<Box<dyn Force>>>;

/// `term` beside the springs set's edge springs and boundary.
fn with_springs(term: impl Fn() -> Box<dyn Force> + 'static) -> Make {
    Box::new(move || {
        vec![
            term(),
            Box::new(EdgeSpring::default()),
            Box::new(Boundary::default()),
        ]
    })
}

fn keys() -> impl Iterator<Item = NodeKey> {
    (0..NODES).map(|i| NodeKey::new(i as usize))
}

/// Every force set is bit-reproducible run to run: five 1x runs of 600 ticks
/// each, against the first. Before key order, `NodeExclusion` moved all 24
/// bodies by up to 33,559 ULP here and `BarnesHutRepulsion` by up to 69,959.
#[test]
fn every_force_is_reproducible_run_to_run() {
    let sets: Vec<(&str, Make)> = vec![
        (
            "EdgeSpring+Boundary",
            Box::new(|| {
                vec![
                    Box::new(EdgeSpring::default()),
                    Box::new(Boundary::default()),
                ]
            }),
        ),
        (
            "NodeExclusion",
            with_springs(|| Box::new(NodeExclusion::default())),
        ),
        (
            "BarnesHutRepulsion (Charge)",
            with_springs(|| Box::new(crate::BarnesHutRepulsion::default())),
        ),
        (
            "StressSpring (Stress)",
            with_springs(|| {
                Box::new(crate::StressSpring::from_distances(
                    keys().map(|k| (k, NodeKey::new((k.index() + 1) % NODES as usize), 1)),
                    170.0,
                ))
            }),
        ),
        (
            "LinLog (Energy)",
            with_springs(|| Box::new(LinLogForce::default())),
        ),
        (
            "Gravity (Orbit)",
            with_springs(|| {
                Box::new(crate::Gravity::new(
                    keys().map(|k| (k, 1.0 + k.index() as f32)),
                    crate::CounterDamping::Tangential,
                ))
            }),
        ),
        (
            "ParticleLife (Kinds)",
            with_springs(|| {
                Box::new(crate::ParticleLife::seeded(
                    keys().map(|k| (k, (k.index() % 3) as u8)),
                    3,
                    7,
                ))
            }),
        ),
        (
            "Boids (Flock)",
            with_springs(|| Box::new(crate::Boids::default())),
        ),
        (
            "Kuramoto (Sync)",
            with_springs(|| Box::new(crate::Kuramoto::new(keys().map(|k| (k, 240.0))))),
        ),
        (
            "MagneticSpring (Flow)",
            with_springs(|| Box::new(crate::MagneticSpring::default())),
        ),
        ("Anneal", Box::new(|| vec![Box::new(Anneal::seeded(7))])),
        ("Hold (Still)", Box::new(|| vec![Box::new(crate::Hold)])),
    ];
    for (name, make) in &sets {
        let runs: Vec<_> = (0..5)
            .map(|_| {
                let mut sim = bodies();
                sim.set_forces(make());
                for _ in 0..TICKS {
                    sim.tick(TICK_DT);
                }
                bits(&sim.view())
            })
            .collect();
        let (differing, worst_ulp) = runs[1..]
            .iter()
            .map(|run| {
                run.iter().zip(&runs[0]).filter(|(a, b)| a != b).fold(
                    (0usize, 0u32),
                    |(n, worst), (a, b)| {
                        (n + 1, worst.max(a.1.abs_diff(b.1)).max(a.2.abs_diff(b.2)))
                    },
                )
            })
            .max()
            .unwrap();
        println!("{name}: {differing} of {NODES} bodies differ, worst {worst_ulp} ulp");
        assert_eq!(differing, 0, "{name} is not reproducible ({worst_ulp} ulp)");
    }
}
