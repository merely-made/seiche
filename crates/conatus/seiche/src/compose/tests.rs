// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! The currency rule's receipts: a law's currency from its terms, one test
//! per currency pair, and the conversion a kinematic law makes, against the
//! same force integrated as force.

use euclid::default::Point2D;

use super::*;
use crate::{
    Anneal, Boundary, Class, CounterDamping, Declared, Density, EdgeSpring, ForceContext, Gravity,
    GravityLocus, Hold, Kernel, NodeExclusion, NodeKey, Observable, Simulation, State, Topology,
    Weighted, terms::CONTACTS,
};

/// A law that integrates on the GPU, for the resident pairs: no catalog law
/// is resident yet.
struct ResidentLaw;

impl Force for ResidentLaw {
    fn apply(&self, _ctx: &mut ForceContext<'_>, _dt: f32) {}
}

impl Declared for ResidentLaw {
    fn terms(&self) -> Vec<Term> {
        vec![
            Term::force(
                "resident",
                Topology::AllPairs { cutoff: None },
                Kernel::Repulsion { exponent: -2.0 },
                Class::E,
                Observable::Overlaps,
            )
            .moving(State::Position, Currency::Resident),
        ]
    }
}

fn currency(forces: Vec<Box<dyn Force>>) -> Currency {
    currency_of(&forces)
}

#[test]
fn every_law_kind_reports_its_currency_from_its_terms() {
    let springs: Vec<Box<dyn Force>> = vec![
        Box::new(NodeExclusion::default()),
        Box::new(EdgeSpring::default()),
        Box::new(Boundary::default()),
    ];
    assert_eq!(currency(springs), Currency::Force);
    assert_eq!(currency(vec![Box::new(Hold)]), Currency::Kinematic);
    assert_eq!(
        currency(vec![Box::new(Anneal::seeded(1))]),
        Currency::Kinematic
    );
    assert_eq!(
        currency(vec![Box::new(Density::new(Vec::new(), 16))]),
        Currency::Kinematic
    );
    assert_eq!(currency(vec![Box::new(ResidentLaw)]), Currency::Resident);
    // Orbit's kick writes velocity once: an initial condition, so Orbit
    // stays a force law. Declared every tick, the same term would make it
    // kinematic, which is the control on `once`.
    let orbit = Gravity::new(Vec::new(), CounterDamping::Tangential);
    assert_eq!(currency(vec![Box::new(orbit)]), Currency::Force);
    let mut every_tick = Gravity::new(Vec::new(), CounterDamping::Tangential).terms();
    for term in &mut every_tick {
        term.initial = false;
    }
    assert_eq!(law_currency(&every_tick), Currency::Kinematic);
}

/// One row per currency pair: the law's currency, the term's, and what the
/// rule does.
#[test]
fn a_test_per_currency_pair() {
    let force_term = EdgeSpring::default().terms()[0];
    let resident_force = NodeExclusion::default();
    let kinematic_term = Hold.terms()[0];
    let resident_term = ResidentLaw.terms()[0];
    let rows = [
        (Currency::Force, force_term, false, Admission::Compose),
        (Currency::Force, kinematic_term, false, Admission::Compose),
        (Currency::Force, resident_term, false, Admission::Compose),
        (Currency::Kinematic, force_term, false, Admission::Convert),
        (
            Currency::Kinematic,
            kinematic_term,
            false,
            Admission::Refuse(TWO_WRITERS),
        ),
        (
            Currency::Kinematic,
            resident_term,
            false,
            Admission::Refuse(TWO_WRITERS),
        ),
        (
            Currency::Resident,
            resident_force.terms()[0],
            resident_force.resident(0),
            Admission::Compose,
        ),
        (
            Currency::Resident,
            force_term,
            EdgeSpring::default().resident(0),
            Admission::Refuse(NOT_RESIDENT),
        ),
        (
            Currency::Resident,
            kinematic_term,
            false,
            Admission::Refuse(RESIDENT_WRITE),
        ),
        (Currency::Resident, resident_term, false, Admission::Compose),
    ];
    for (law, term, resident, expected) in rows {
        assert_eq!(
            admit(law, &term, resident),
            expected,
            "{law:?} law, {} ({:?})",
            term.name,
            term.currency
        );
    }
    // The integrator's terms and an initial condition never decide.
    for law in [Currency::Force, Currency::Kinematic, Currency::Resident] {
        assert_eq!(admit(law, &CONTACTS, false), Admission::Compose);
        let kick = Gravity::new(Vec::new(), CounterDamping::Off).terms()[2];
        assert!(kick.initial);
        assert_eq!(admit(law, &kick, false), Admission::Compose);
    }
    // A whole force takes its most restrictive term.
    assert_eq!(
        admit_force(Currency::Resident, &resident_force),
        Admission::Compose
    );
    assert_eq!(
        admit_force(Currency::Kinematic, &GravityLocus::at((0.0, 0.0))),
        Admission::Convert
    );
}

#[test]
fn a_weighted_mix_takes_force_laws_only() {
    assert_eq!(mix_refusal(&[Currency::Force, Currency::Force]), None);
    assert_eq!(
        mix_refusal(&[Currency::Force, Currency::Kinematic]),
        Some(UNWEIGHTED)
    );
    assert_eq!(mix_refusal(&[Currency::Resident]), Some(UNWEIGHTED));
    assert!(Weighted::new(Box::new(Hold), 0.5).is_err());
    assert!(Weighted::new(Box::new(EdgeSpring::default()), 0.5).is_ok());
}

/// One body 170 from a centre pull, ticked at 60 Hz for 30 ticks: after each
/// tick, its speed and the speed one step from rest gives at its distance,
/// `F·dt / (m·(1 + c·dt))`, `c` the damping.
fn speeds(law: Option<Box<dyn Force>>) -> Vec<(f32, f32)> {
    let key = NodeKey::new(0);
    let locus = GravityLocus::at((0.0, 0.0));
    let mass = std::f32::consts::PI * crate::NODE_BODY_RADIUS.powi(2) * crate::NODE_BODY_DENSITY;
    let dt = 1.0 / 60.0;
    let mut sim = Simulation::new();
    let (strength, damping) = (locus.strength, sim.linear_damping());
    let one_step = |distance: f32| strength * distance * dt / (mass * (1.0 + damping * dt));
    sim.sync_nodes([(key, Point2D::new(170.0, 0.0))]);
    sim.sync_edges(Vec::<(NodeKey, NodeKey)>::new());
    let mut forces: Vec<Box<dyn Force>> = law.into_iter().collect();
    forces.push(Box::new(locus));
    sim.set_forces(forces);
    (0..30)
        .map(|_| {
            sim.tick(dt);
            let at = sim.position_of(key).unwrap();
            (
                sim.velocity_of(key).unwrap().length(),
                one_step(at.to_vector().length()),
            )
        })
        .collect()
}

/// A kinematic law takes a force converted: after each tick the body moves
/// at the speed one step from rest gives, so the speed never builds. Hold
/// always did; Anneal, cooling and spent, now does too. The control: the
/// same pull as a force law's term builds speed tick on tick.
#[test]
fn a_kinematic_law_takes_forces_converted_and_a_force_law_integrates_them() {
    // A floor above its starting temperature: the schedule is spent at once.
    let mut spent = Anneal::seeded(1);
    spent.floor = f32::INFINITY;
    for (name, law) in [
        ("still", Box::new(Hold) as Box<dyn Force>),
        ("anneal, cooling", Box::new(Anneal::seeded(1))),
        ("anneal, spent", Box::new(spent)),
    ] {
        let v = speeds(Some(law));
        println!(
            "{name}: speed {:.4} then {:.4}, one step from rest {:.4} then {:.4}",
            v[0].0, v[29].0, v[0].1, v[29].1
        );
        for (speed, one_step) in &v {
            assert!(
                (speed - one_step).abs() <= 0.02 * one_step,
                "{name}: speed {speed} against {one_step}"
            );
        }
    }
    let inertial = speeds(None);
    println!(
        "force law: speed {:.4} then {:.4}, one step from rest {:.4} then {:.4}",
        inertial[0].0, inertial[29].0, inertial[0].1, inertial[29].1
    );
    assert!((inertial[0].0 - inertial[0].1).abs() <= 0.02 * inertial[0].1);
    assert!(inertial[29].0 > 10.0 * inertial[29].1, "{inertial:?}");
}
