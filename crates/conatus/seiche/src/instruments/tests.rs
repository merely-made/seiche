// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! The instruments' own receipts: each says yes to a term built to pass and
//! no to one built to fail, in the same run.

use super::*;
use crate::{
    Boundary, Class, Declared, EdgeSpring, LinLogForce, NodeExclusion, ParticleLife, Term,
};

/// The P2 fixture's topology (Graphshell's reference host,
/// `ports/graphshell/src/mere_host_fixture.rs`): eleven nodes in two
/// components, ten relations.
fn p2() -> (Vec<NodeKey>, Vec<(NodeKey, NodeKey)>) {
    let keys: Vec<NodeKey> = (0..11).map(NodeKey::new).collect();
    let edges = [
        (0, 1),
        (0, 2),
        (3, 0),
        (3, 2),
        (4, 3),
        (5, 6),
        (5, 7),
        (6, 8),
        (9, 5),
        (10, 9),
    ]
    .iter()
    .map(|&(a, b)| (keys[a], keys[b]))
    .collect();
    (keys, edges)
}

/// The P2 fixture's kinds by site, folded to eight as the catalog folds them.
fn p2_kinds(keys: &[NodeKey]) -> Vec<(NodeKey, u8)> {
    let site = [0u32, 1, 2, 3, 4, 5, 6, 6, 7, 8, 9];
    keys.iter()
        .zip(site)
        .map(|(&k, s)| (k, (s % 8) as u8))
        .collect()
}

/// The catalog's seed for Kinds' matrix (`physics_catalog.rs`, `LAW_SEED`).
const LAW_SEED: u64 = 0x5EED_CA7A_1064;

/// A rotational push about the origin: a curl, the plainest N.
struct Swirl;

impl Force for Swirl {
    fn apply(&self, ctx: &mut ForceContext<'_>, _dt: f32) {
        for &handle in ctx.bodies_by_node.values() {
            if let Some(body) = ctx.bodies.get_mut(handle) {
                let p = body.translation();
                body.add_force(Vector::new(-p.y, p.x), true);
            }
        }
    }
}

impl Declared for Swirl {
    fn terms(&self) -> Vec<Term> {
        vec![Term::force(
            "swirl",
            crate::Topology::Unary,
            crate::Kernel::Vortex,
            Class::N,
            crate::Observable::Energy,
        )]
    }
}

fn starts(probe: &Probe) -> Vec<Vec<Vector>> {
    (1..=3).map(|seed| probe.scatter(seed, 150.0)).collect()
}

#[test]
fn a_gradient_passes_and_a_curl_fails_every_conservativeness_candidate() {
    let (keys, edges) = p2();
    let mut probe = Probe::new(&keys, &edges);
    let starts = starts(&probe);
    let spring = read(&mut probe, &|| Box::new(EdgeSpring::default()), 0, &starts);
    let swirl = read(&mut probe, &|| Box::new(Swirl), 0, &starts);
    for candidate in Candidate::ALL {
        assert!(spring.conservative(candidate), "{candidate:?}: {spring:?}");
        assert!(!swirl.conservative(candidate), "{candidate:?}: {swirl:?}");
        spring.agrees(candidate).unwrap();
        swirl.agrees(candidate).unwrap();
    }
}

#[test]
fn the_energy_instruments_pass_a_true_energy_and_catch_a_wrong_one() {
    let (keys, edges) = p2();
    let mut probe = Probe::new(&keys, &edges);
    let at = probe.scatter(2, 150.0);
    let exclusion = NodeExclusion::default();
    let fall = descent(&mut probe, &exclusion, 0, &at, 300, 0.5).unwrap();
    assert!(
        fall.end < fall.start && fall.relative_rise() <= tolerance::RISE,
        "{fall:?}"
    );
    let error = gradient_error(&mut probe, &exclusion, 0, &at, 0.05).unwrap();
    assert!(error <= tolerance::GRADIENT, "{error}");
    // Boundary's energy read against Springs' forces: not their potential.
    struct Mislabelled;
    impl Force for Mislabelled {
        fn apply(&self, ctx: &mut ForceContext<'_>, dt: f32) {
            EdgeSpring::default().apply(ctx, dt);
        }
    }
    impl Declared for Mislabelled {
        fn terms(&self) -> Vec<Term> {
            Boundary::default().terms()
        }
        fn energy(&self, term: usize, layout: &Layout<'_>) -> Option<f64> {
            Boundary::default().energy(term, layout)
        }
    }
    let error = gradient_error(&mut probe, &Mislabelled, 0, &at, 0.05).unwrap();
    assert!(error > 0.5, "{error}");
}

/// True LinLog as a tuning: at attraction exponent 0 each term still
/// descends its energy and is its gradient. The cusp of `a·d` at contact makes
/// a fixed step chatter, so this run takes half the instruments' usual step.
#[test]
fn linlog_at_exponent_zero_runs_and_descends_its_own_energy() {
    let (keys, edges) = p2();
    let mut probe = Probe::new(&keys, &edges);
    let at = probe.scatter(3, 150.0);
    let linlog = LinLogForce {
        attraction_exponent: 0.0,
        ..LinLogForce::default()
    };
    for term in 0..3 {
        let only = isolated(Box::new(linlog), term);
        let fall = descent(&mut probe, only.as_ref(), term, &at, 300, 0.25).unwrap();
        assert!(
            fall.relative_rise() <= tolerance::RISE,
            "term {term}: {fall:?}"
        );
        let error = gradient_error(&mut probe, only.as_ref(), term, &at, 0.05).unwrap();
        assert!(error <= tolerance::GRADIENT, "term {term}: {error}");
    }
}

/// The positive control on the P2 fixture, the catalog's seed and kinds:
/// Kinds' seeded matrix reads non-conservative by every candidate and does
/// not balance; its symmetrized twin is E and agrees on everything.
#[test]
fn kinds_seeded_fails_and_symmetrized_passes_on_the_p2_fixture() {
    let (keys, edges) = p2();
    let mut probe = Probe::new(&keys, &edges);
    let starts = starts(&probe);
    let seeded = ParticleLife::seeded(p2_kinds(&keys), 8, LAW_SEED);
    let symmetric = seeded.symmetrized();
    let fails = {
        let law = seeded.clone();
        read(&mut probe, &move || Box::new(law.clone()), 0, &starts)
    };
    let passes = {
        let law = symmetric.clone();
        read(&mut probe, &move || Box::new(law.clone()), 0, &starts)
    };
    assert_eq!((fails.term.class, passes.term.class), (Class::N, Class::E));
    assert!(fails.balance.unwrap() > tolerance::BALANCE, "{fails:?}");
    for candidate in Candidate::ALL {
        assert!(!fails.conservative(candidate), "{candidate:?}: {fails:?}");
        assert!(passes.conservative(candidate), "{candidate:?}: {passes:?}");
        fails.agrees(candidate).unwrap();
        passes.agrees(candidate).unwrap();
    }
}
