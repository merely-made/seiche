// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

use super::*;

/// The diagnosis: the catalog's Orbit on both fixtures, then each term
/// removed in turn, then centring (which Orbit lacks) added; the instrument's
/// positive control is exclusion alone at no damping, whose booked work must
/// equal its energy drop.
#[test]
#[ignore = "diagnostic: prints Orbit's term ledger"]
fn diag_orbit_terms() {
    let control = run(
        "instrument control: exclusion alone, no damping",
        &Fixture::p2(),
        19.0,
        0.0,
        vec![Part::new("exclusion", NodeExclusion::default())],
        6,
    );
    let (w, du) = control.exclusion_check.unwrap();
    assert!(
        (w - du).abs() < 0.02 * du.abs(),
        "the ledger books exclusion's work as its energy drop: {w:.0} against {du:.0}"
    );
    for fixture in [Fixture::p2(), Fixture::generated()] {
        run(
            "catalog Orbit",
            &fixture,
            19.0,
            2.5,
            catalog_parts(&fixture, None),
            120,
        );
        for drop in ["exclusion", "gravitation", "counter-damping", "kick"] {
            run(
                &format!("without {drop}"),
                &fixture,
                19.0,
                2.5,
                catalog_parts(&fixture, Some(drop)),
                120,
            );
        }
        let mut with_centring = catalog_parts(&fixture, None);
        with_centring.push(Part::new("centring", Boundary::default()));
        run(
            "with centring (Boundary 0.08)",
            &fixture,
            19.0,
            2.5,
            with_centring,
            120,
        );
    }
}

/// The candidates: exclusion's reach, the counter-damping's form, centring
/// and softening, alone and together, on both fixtures.
#[test]
#[ignore = "diagnostic: prints the candidate fixes' readings"]
fn diag_orbit_candidates() {
    let catalog = Recipe::catalog();
    let candidates: Vec<(&str, Recipe)> = vec![
        ("catalog", catalog),
        (
            "brake only",
            Recipe {
                counter: Counter::Brake,
                ..catalog
            },
        ),
        ("reach 2 only", catalog.reach(2.0)),
        (
            "reach 2 + brake",
            Recipe {
                counter: Counter::Brake,
                ..catalog.reach(2.0)
            },
        ),
        (
            "reach 1.5 + brake",
            Recipe {
                counter: Counter::Brake,
                ..catalog.reach(1.5)
            },
        ),
        (
            "reach 3 + brake",
            Recipe {
                counter: Counter::Brake,
                ..catalog.reach(3.0)
            },
        ),
        (
            "reach 2 + cap",
            Recipe {
                counter: Counter::Cap,
                ..catalog.reach(2.0)
            },
        ),
        (
            "reach 2 + centring 0.02",
            Recipe {
                centring: Some(0.02),
                ..catalog.reach(2.0)
            },
        ),
        (
            "reach 2 + softening 72",
            Recipe {
                softening: 72.0,
                ..catalog.reach(2.0)
            },
        ),
        (
            "no exclusion + brake",
            Recipe {
                exclusion: None,
                counter: Counter::Brake,
                ..catalog
            },
        ),
        (
            "tangential only",
            Recipe {
                counter: Counter::Tangential,
                ..catalog
            },
        ),
        (
            "reach 2 + tangential",
            Recipe {
                counter: Counter::Tangential,
                ..catalog.reach(2.0)
            },
        ),
        (
            "reach 3 + tangential",
            Recipe {
                counter: Counter::Tangential,
                ..catalog.reach(3.0)
            },
        ),
        (
            "no exclusion + tangential",
            Recipe {
                exclusion: None,
                counter: Counter::Tangential,
                ..catalog
            },
        ),
    ];
    for fixture in [Fixture::p2(), Fixture::generated()] {
        for (label, recipe) in &candidates {
            run(label, &fixture, 19.0, 2.5, recipe.parts(&fixture), 120);
        }
    }
}

/// The leading candidate's robustness: exclusion's reach, the host's damping
/// (2.5 the canvas default, 0.82 the old page, 0.7 the saved-scene default),
/// a looser seed, and the law's own radial damping beside the host's.
#[test]
#[ignore = "diagnostic: prints the leading candidate's readings"]
fn diag_orbit_leading() {
    let catalog = Recipe::catalog();
    let tangential = |reach: f32| Recipe {
        counter: Counter::Tangential,
        ..catalog.reach(reach)
    };
    let own = Recipe {
        counter: Counter::Own(2.5),
        ..catalog.reach(2.0)
    };
    for fixture in [Fixture::p2(), Fixture::generated()] {
        run(
            "reach 1.5 + tangential",
            &fixture,
            19.0,
            2.5,
            tangential(1.5).parts(&fixture),
            120,
        );
        for damping in [0.82, 0.7] {
            run(
                "reach 2 + tangential",
                &fixture,
                19.0,
                damping,
                tangential(2.0).parts(&fixture),
                120,
            );
        }
        run(
            "reach 2 + tangential",
            &fixture,
            40.0,
            2.5,
            tangential(2.0).parts(&fixture),
            120,
        );
        for damping in [0.82, 0.0] {
            run(
                "reach 2 + own radial damping 2.5",
                &fixture,
                19.0,
                damping,
                own.parts(&fixture),
                120,
            );
        }
    }
}

/// The ruled Orbit as the catalog builds it (exclusion to two diameters,
/// `CounterDamping::Tangential`, centring 0.02) beside the test force it was
/// built from, over the sweep; then each fixture at no damping, the
/// question the ruling left open.
#[test]
#[ignore = "diagnostic: prints the ruled Orbit over the sweep and at no damping"]
fn diag_orbit_ruled() {
    let catalog = Recipe::catalog();
    let ruled = Recipe {
        counter: Counter::Law,
        centring: Some(0.02),
        ..catalog.reach(2.0)
    };
    let tested = Recipe {
        counter: Counter::Tangential,
        ..ruled
    };
    sweep("ruled (Gravity's Tangential)", &ruled);
    sweep("tested (the diagnostic's Tangential)", &tested);
    for fixture in [Fixture::p2(), Fixture::generated()] {
        run(
            "ruled, no damping",
            &fixture,
            19.0,
            0.0,
            ruled.parts(&fixture),
            120,
        );
        run(
            "control: today's Orbit at no damping",
            &fixture,
            19.0,
            0.0,
            catalog.parts(&fixture),
            120,
        );
    }
}

/// At no host damping, the open question: the ruled Orbit over the sweep,
/// and the same with a floor under its radial settling (the law's own radial
/// damping at the old page's 0.82 and the saved scenes' 0.7).
#[test]
#[ignore = "diagnostic: prints the ruled Orbit and floored variants at no damping"]
fn diag_orbit_no_damping() {
    let ruled = Recipe {
        counter: Counter::Law,
        centring: Some(0.02),
        ..Recipe::catalog().reach(2.0)
    };
    sweep_at("ruled, no damping", &ruled, &[0.0]);
    for floor in [0.82, 0.7] {
        sweep_at(
            &format!("ruled + radial floor {floor}, no damping"),
            &Recipe {
                counter: Counter::Own(floor),
                ..ruled
            },
            &[0.0],
        );
    }
}

/// The ruled Orbit with its radial floor (0.82, ruled 2026-10-04) at no
/// damping, the saved scenes' 0.7, the old page's 0.82 and the tree page's
/// 2.5, against the bar; the same law without the floor at no damping as the
/// control.
#[test]
#[ignore = "diagnostic: prints the floored Orbit at four dampings"]
fn diag_orbit_floor() {
    let ruled = Recipe {
        counter: Counter::Law,
        centring: Some(0.02),
        ..Recipe::catalog().reach(2.0)
    };
    sweep_at("ruled with its floor", &ruled, &[0.0, 0.7, 0.82, 2.5]);
    sweep_at(
        "control: no floor",
        &Recipe {
            radial_floor: 0.0,
            ..ruled
        },
        &[0.0],
    );
}
