// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

use super::*;

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

/// A native host's budget from its display's refresh rate (ruled 2026-10-04,
/// "Mere entry point, then turnstone"): half the period winit reports, in
/// millihertz, and 60 Hz's when the rate is unknown; the physics hands back
/// the budget it was given, inline or offloaded.
#[test]
fn a_budget_from_the_display_refresh_rate() {
    let us = |duration: Duration| duration.as_micros();
    assert_eq!(us(display_period(Some(60_000))), 16_666);
    assert_eq!(us(display_period(Some(120_000))), 8_333);
    assert_eq!(us(display_period(Some(144_000))), 6_944);
    assert_eq!(us(display_period(Some(165_000))), 6_060);
    assert_eq!(display_period(Some(59_940)).as_nanos(), 16_683_350);
    assert_eq!(display_period(None), FALLBACK_DISPLAY_PERIOD);
    assert_eq!(display_period(Some(0)), FALLBACK_DISPLAY_PERIOD);
    assert_eq!(DEFAULT_BUDGET_SHARE, 0.5);

    let budget = StepBudget::for_display(Some(165_000));
    assert_eq!(us(budget.per_frame), 3_030);
    assert_eq!(budget.margin, Duration::ZERO);
    assert_eq!(us(StepBudget::for_display(None).per_frame), 8_333);
    let quarter = StepBudget::of_period(
        display_period(Some(144_000)),
        0.25,
        virtual_clock,
        Duration::from_micros(200),
    );
    assert_eq!((us(quarter.per_frame), us(quarter.margin)), (1_736, 200));

    let sim = sim(Set::LinLog);
    let mut physics = Physics::inline(sim, TICKS);
    assert!(physics.step_budget().is_none());
    physics.set_step_budget(Some(budget));
    assert_eq!(physics.step_budget().map(|b| us(b.per_frame)), Some(3_030));
    #[cfg(feature = "actor")]
    {
        physics.offload(std::sync::Arc::new(|| {}));
        assert_eq!(physics.step_budget().map(|b| us(b.per_frame)), Some(3_030));
        physics.set_step_budget(Some(StepBudget::for_display(Some(60_000))));
        assert_eq!(physics.step_budget().map(|b| us(b.per_frame)), Some(8_333));
    }
}

/// "Only ticks past the floor" (ruled 2026-10-05): the report names the
/// ticks the gate admitted past the 1x floor and when the last ended, the
/// time the budget bounds. A floor dearer than the budget admits none, its
/// overrun the floor's own; a tick the gate admitted that runs long ends past
/// the budget and shows there (the planted control).
#[test]
fn the_report_bounds_only_the_ticks_the_gate_admitted() {
    let config = ElapsedStepConfig {
        max_elapsed: TICK_DURATION * 4,
        max_steps: 3,
    };
    let budget = Duration::from_millis(8);
    // 1 ms ticks: the floor's one, then seven the gate admits, ending at 8 ms.
    let (mut physics, mut view) = costly(1_000, budget, 50.0);
    let report = physics.advance_elapsed(&mut view, TICK_DURATION, config);
    assert_eq!((report.steps, report.admitted), (8, 7));
    assert_eq!(report.admitted_until, Some(budget));

    // A floor of three 20 ms ticks: 60 ms of stepping, none of it admitted.
    let (mut physics, mut view) = costly(20_000, budget, 50.0);
    let report = physics.advance_elapsed(&mut view, TICK_DURATION * 3, config);
    assert_eq!((report.steps, report.admitted), (3, 0));
    assert_eq!(report.admitted_until, None);
    assert_eq!(report.compute, Some(Duration::from_millis(60)));

    // Planted: the fifth tick costs 5 ms. The gate admits it with 4 ms left
    // and a 1 ms forecast, and it ends at 9 ms, a millisecond past.
    NOW_US.with(|now| now.set(0));
    let mut sim = sim(Set::LinLog);
    sim.add_force(Uneven {
        base: 1_000,
        dear: 5_000,
        every: 5,
        ticks: Cell::new(0),
    });
    let mut view = sim.view();
    let mut physics = Physics::inline(sim, TICKS);
    physics.set_speed(Speed::from_factor(50.0));
    physics.set_step_budget(Some(StepBudget {
        per_frame: budget,
        clock: virtual_clock,
        margin: Duration::ZERO,
    }));
    let report = physics.advance_elapsed(&mut view, TICK_DURATION, config);
    assert_eq!((report.steps, report.admitted), (5, 4));
    assert_eq!(report.admitted_until, Some(Duration::from_millis(9)));
}
