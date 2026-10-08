// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! The scale receipt (dynamics grammar plan, G3, F5): every term on the
//! common scale, rebuilt at weight 1 and measured at its reference in a
//! probe, pushes with unit force within 1e-3; at its default it pushes with
//! the weight it declares; and a term whose declared weight does not match
//! its force fails the same check (the positive control).

use super::*;
use crate::instruments::{Probe, Vector, isolated};
use crate::{
    AffinitySpring, AnchorSpring, BarnesHutRepulsion, Boids, Boundary, CounterDamping, Declared,
    DegreeRepulsion, DepthGravity, DomainCluster, EdgeSpring, Force, ForceContext, Gravity,
    GravityLocus, GridSnap, HubGravity, Kuramoto, LinLogForce, MagneticSpring, NodeExclusion,
    NodeKey, ParticleLife, StressSpring, Term, Topology,
};

const TOLERANCE: f64 = 1e-3;

fn a() -> NodeKey {
    NodeKey::new(0)
}

fn b() -> NodeKey {
    NodeKey::new(1)
}

/// One scaled term: its force over its fixture, which term, and how the
/// fixture places the bodies for its reference (the measured body last).
struct Case {
    name: &'static str,
    term: usize,
    make: Box<dyn Fn() -> Box<dyn Force>>,
}

fn cases() -> Vec<Case> {
    let unit = |key| (key, 1.0f32);
    vec![
        Case {
            name: "exclusion",
            term: 0,
            make: Box::new(|| Box::new(NodeExclusion::default())),
        },
        Case {
            name: "charge",
            term: 0,
            make: Box::new(|| {
                Box::new(BarnesHutRepulsion {
                    strength: 6_000.0,
                    ..BarnesHutRepulsion::default()
                })
            }),
        },
        Case {
            name: "linlog repulsion",
            term: 0,
            make: Box::new(|| Box::new(LinLogForce::default())),
        },
        Case {
            name: "magnetic repulsion",
            term: 0,
            make: Box::new(|| Box::new(MagneticSpring::default())),
        },
        Case {
            name: "boids separation",
            term: 0,
            make: Box::new(|| Box::new(Boids::default())),
        },
        Case {
            name: "hub room",
            term: 0,
            make: Box::new(move || {
                Box::new(DegreeRepulsion::default().with_weights([unit(a()), unit(b())]))
            }),
        },
        Case {
            name: "edge spring",
            term: 0,
            make: Box::new(|| Box::new(EdgeSpring::default())),
        },
        Case {
            name: "stress",
            term: 0,
            make: Box::new(|| Box::new(StressSpring::from_distances([(a(), b(), 1)], 170.0))),
        },
        Case {
            name: "magnetic spring",
            term: 1,
            make: Box::new(|| Box::new(MagneticSpring::default())),
        },
        Case {
            name: "affinity",
            term: 0,
            make: Box::new(|| Box::new(AffinitySpring::new([(a(), b(), 1.0)]))),
        },
        Case {
            name: "centring",
            term: 0,
            make: Box::new(|| Box::new(Boundary::default())),
        },
        Case {
            name: "centre",
            term: 0,
            make: Box::new(|| Box::new(GravityLocus::at((0.0, 0.0)))),
        },
        Case {
            name: "tide",
            term: 0,
            make: Box::new(|| Box::new(GravityLocus::tidal((0.0, 0.0), 240.0, 24.0))),
        },
        Case {
            name: "group pull",
            term: 0,
            make: Box::new(|| Box::new(DomainCluster::new([(a(), 0), (b(), 0)]))),
        },
        Case {
            name: "depth",
            term: 0,
            make: Box::new(|| Box::new(DepthGravity::new([(a(), 0), (b(), 0)]))),
        },
        Case {
            name: "anchor",
            term: 0,
            make: Box::new(|| Box::new(AnchorSpring::new([(a(), (0.0, 0.0)), (b(), (0.0, 0.0))]))),
        },
        Case {
            name: "linlog centring",
            term: 2,
            make: Box::new(|| Box::new(LinLogForce::default())),
        },
        Case {
            name: "boids centring",
            term: 4,
            make: Box::new(|| Box::new(Boids::default())),
        },
        Case {
            name: "kinds centring",
            term: 1,
            make: Box::new(|| Box::new(ParticleLife::seeded([(a(), 0), (b(), 0)], 1, 1))),
        },
        Case {
            name: "ring draw",
            term: 1,
            make: Box::new(|| Box::new(Kuramoto::new([(a(), 200.0), (b(), 200.0)]))),
        },
        // F79: Grid at its half cell.
        Case {
            name: "grid",
            term: 0,
            make: Box::new(|| Box::new(GridSnap::default())),
        },
        // F80: where distance matters, at contact.
        Case {
            name: "linlog attraction",
            term: 1,
            make: Box::new(|| Box::new(LinLogForce::default())),
        },
        Case {
            name: "hub pull",
            term: 0,
            make: Box::new(move || {
                Box::new(HubGravity::default().with_weights([unit(a()), unit(b())]))
            }),
        },
        Case {
            name: "gravitation",
            term: 0,
            make: Box::new(|| Box::new(Gravity::new([], CounterDamping::Off))),
        },
        Case {
            name: "boids cohesion",
            term: 2,
            make: Box::new(|| Box::new(Boids::default())),
        },
    ]
}

/// The force on the measured body (`b`) with the fixture placed for
/// `reference`. The other body, where there is one, sits where it adds
/// nothing to the reading: at the origin for a pair, mirrored for a group,
/// and at its own target for a unary term.
fn measured(force: &dyn Force, term: &Term, reference: Reference, name: &str) -> f64 {
    let (keys, edges) = (vec![a(), b()], vec![(a(), b())]);
    let place = |probe: &mut Probe, at_b: Vector, at_a: Vector| {
        probe.place(&[at_a, at_b]);
    };
    let joined = matches!(reference, Reference::Stretch { .. }) || term.topology == Topology::Edges;
    let mut probe = if joined {
        Probe::new(&keys, &edges)
    } else {
        Probe::new(&keys, &[])
    };
    let l = UNIT_LENGTH;
    match reference {
        Reference::Contact => place(&mut probe, Vector::new(CONTACT, 0.0), Vector::ZERO),
        Reference::Stretch { rest } => {
            place(&mut probe, Vector::new(2.0 * rest, 0.0), Vector::ZERO)
        },
        // Half a cell from the grid point at the origin, on the x axis.
        Reference::HalfCell { cell } => place(
            &mut probe,
            Vector::new(cell / 2.0, 0.0),
            Vector::new(-cell / 2.0, 0.0),
        ),
        Reference::Offset => match name {
            "group pull" => place(&mut probe, Vector::new(l, 0.0), Vector::new(-l, 0.0)),
            // Depth pulls along its direction, toward depth 0 at the origin.
            "depth" => place(&mut probe, Vector::new(0.0, l), Vector::new(0.0, -l)),
            // The ring's target for key 1 is its seeded phase's point.
            "ring draw" => {
                let phase = 2.399_963f32 % std::f32::consts::TAU;
                let (c, s) = (phase.cos(), phase.sin());
                place(
                    &mut probe,
                    Vector::new((200.0 + l) * c, (200.0 + l) * s),
                    Vector::new(200.0, 0.0),
                )
            },
            _ => place(&mut probe, Vector::new(l, 0.0), Vector::new(-l, 0.0)),
        },
    }
    let f = probe.forces(force);
    f64::from(f[1].length())
}

#[test]
fn every_scaled_term_pushes_with_unit_force_at_its_reference() {
    let mut table = Vec::new();
    for case in cases() {
        let force = (case.make)();
        let term = force.terms()[case.term];
        let scale = force
            .scale(case.term)
            .unwrap_or_else(|| panic!("{} declares a scale", case.name));
        assert_eq!(
            Some(scale.reference.family()),
            family(&term),
            "{}: F5's reference for its kernel",
            case.name
        );
        let at_default = measured(
            isolated((case.make)(), case.term).as_ref(),
            &term,
            scale.reference,
            case.name,
        );
        let unit = isolated(force.reweighted(case.term, 1.0).unwrap(), case.term);
        let at_unit = measured(unit.as_ref(), &term, scale.reference, case.name);
        table.push(format!(
            "{:<18} {:?}: weight {:.3}, measured {:.3} at default; {:.6} at weight 1",
            case.name, scale.reference, scale.weight, at_default, at_unit
        ));
        assert!(
            (at_unit - 1.0).abs() <= TOLERANCE,
            "{}: unit force at its reference, read {at_unit}",
            case.name
        );
        assert!(
            (at_default - scale.weight).abs() <= TOLERANCE * scale.weight.max(1.0),
            "{}: its default pushes with its declared weight {} (read {at_default})",
            case.name,
            scale.weight
        );
    }
    println!("{}", table.join("\n"));
}

/// The control: a term that declares the inverse-square weight but pushes
/// as `1/d` fails the same reading, by the ratio of the two at contact.
#[test]
fn a_term_whose_declared_weight_is_wrong_fails_the_receipt() {
    struct Mislabelled(BarnesHutRepulsion);
    impl Force for Mislabelled {
        fn apply(&self, ctx: &mut ForceContext<'_>, dt: f32) {
            self.0.apply(ctx, dt);
        }
    }
    impl Declared for Mislabelled {
        fn terms(&self) -> Vec<Term> {
            NodeExclusion::default().terms()
        }
        fn scale(&self, _term: usize) -> Option<Scale> {
            Some(Scale {
                reference: Reference::Contact,
                weight: at_contact(self.0.strength, -2.0),
            })
        }
        fn reweighted(&self, _term: usize, weight: f64) -> Option<Box<dyn Force>> {
            Some(Box::new(Mislabelled(BarnesHutRepulsion {
                strength: strength_at_contact(weight, -2.0),
                ..self.0
            })))
        }
    }
    let force = Mislabelled(BarnesHutRepulsion::default());
    let term = force.terms()[0];
    let unit = force.reweighted(0, 1.0).unwrap();
    let read = measured(unit.as_ref(), &term, Reference::Contact, "mislabelled");
    println!(
        "mislabelled: {read:.3} at weight 1 (36 expected, the ratio of 1/d to 1/d² at contact)"
    );
    assert!((read - 1.0).abs() > 10.0 * TOLERANCE, "{read}");
    assert!((read - f64::from(CONTACT)).abs() < 1e-2, "{read}");
}

/// F80's probe for Boids' cohesion: it steers toward the mates' centre in
/// proportion to the offset, so at contact and at twice contact it reads
/// different pulls, and it belongs on the scale there.
#[test]
fn boids_cohesion_varies_with_offset_at_contact() {
    let cohesion = isolated(Box::new(Boids::default()), 2);
    let mut probe = Probe::new(&[a(), b()], &[(a(), b())]);
    let mut read = |d: f32| {
        probe.place(&[Vector::ZERO, Vector::new(d, 0.0)]);
        f64::from(probe.forces(cohesion.as_ref())[1].length())
    };
    let (near, far) = (read(CONTACT), read(2.0 * CONTACT));
    println!("boids cohesion: {near:.3} at contact, {far:.3} at twice contact");
    assert!((far / near - 2.0).abs() < TOLERANCE, "{near} {far}");
}

/// The tent read at contact (F80, held): 36 is inside particle life's
/// repulsive core (`core · radius`, 66 at the defaults), so the reading is
/// the core's push, the same for every kind rule.
#[test]
fn the_tent_at_contact_reads_its_core_whatever_the_rule() {
    let mut readings = Vec::new();
    for rule in [-1.0f32, 0.0, 1.0] {
        let law = ParticleLife::new([(a(), 0), (b(), 0)], 1, vec![rule]);
        let core =
            f64::from(law.strength) * (1.0 - f64::from(CONTACT) / f64::from(law.core * law.radius));
        let tent = isolated(Box::new(law), 0);
        let mut probe = Probe::new(&[a(), b()], &[]);
        probe.place(&[Vector::ZERO, Vector::new(CONTACT, 0.0)]);
        let read = f64::from(probe.forces(tent.as_ref())[1].length());
        readings.push(format!("rule {rule}: {read:.3}"));
        assert!(
            (read - core).abs() < 1e-2,
            "rule {rule}: {read} against {core}"
        );
    }
    println!("tent at contact: {}", readings.join(", "));
}
