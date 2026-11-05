// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

use super::*;

/// [`sweep`] at the given dampings.
pub(in super::super) fn sweep_at(label: &str, recipe: &Recipe, dampings: &[f32]) {
    for fixture in [Fixture::p2(), Fixture::generated()] {
        for &damping in dampings {
            let reports: Vec<Report> = SPACINGS
                .iter()
                .map(|&spacing| {
                    run_quietly(
                        label,
                        &fixture,
                        spacing,
                        damping,
                        recipe.parts(&fixture),
                        120,
                        false,
                    )
                })
                .collect();
            let mut max: Vec<f32> = reports.iter().map(|r| r.max_ratio).collect();
            max.sort_by(f32::total_cmp);
            let mean = |f: &dyn Fn(&Report) -> f32| {
                reports.iter().map(f).sum::<f32>() / reports.len() as f32
            };
            let least =
                |f: &dyn Fn(&Report) -> f32| reports.iter().map(f).fold(f32::INFINITY, f32::min);
            let most = |f: &dyn Fn(&Report) -> f32| {
                reports.iter().map(f).fold(f32::NEG_INFINITY, f32::max)
            };
            println!(
                "{label:<40} {:<6} d {damping:<4}: least seed tangential {:.2}, coherence {:.2}, revolutions {:.2}, KE end {:.0}; most mass~radius {:.2}",
                fixture.name,
                least(&|r| r.mean_tangential),
                least(&|r| r.mean_coherence),
                least(&|r| r.revolutions),
                least(&|r| r.end_energy),
                most(&|r| r.mean_mass_radius),
            );
            println!(
                "{label:<40} {:<6} d {damping:<4}: max x worst {:.2} median {:.2} (each {}); end x worst {:.2}; KE min {:.0}; tangential {:.2}, coherence {:.2}, mass~radius {:.2}, revolutions {:.2}; overlaps max {}",
                fixture.name,
                max[max.len() - 1],
                max[max.len() / 2],
                reports
                    .iter()
                    .map(|r| format!("{:.2}", r.max_ratio))
                    .collect::<Vec<_>>()
                    .join(" "),
                reports.iter().map(|r| r.end_ratio).fold(0.0, f32::max),
                reports
                    .iter()
                    .map(|r| r.min_energy)
                    .fold(f32::INFINITY, f32::min),
                mean(&|r| r.mean_tangential),
                mean(&|r| r.mean_coherence),
                mean(&|r| r.mean_mass_radius),
                mean(&|r| r.revolutions),
                reports.iter().map(|r| r.max_overlaps).max().unwrap_or(0),
            );
        }
    }
}

/// The candidates over the seed sweep.
#[test]
#[ignore = "diagnostic: prints the candidates over five seeds and both dampings"]
fn diag_orbit_sweep() {
    let catalog = Recipe::catalog();
    let tangential = Recipe {
        counter: Counter::Tangential,
        ..catalog.reach(2.0)
    };
    for (label, recipe) in [
        ("catalog", catalog),
        ("reach 2 + tangential", tangential),
        (
            "reach 3 + tangential",
            Recipe {
                counter: Counter::Tangential,
                ..catalog.reach(3.0)
            },
        ),
        (
            "reach 2 + tangential + centring 0.005",
            Recipe {
                centring: Some(0.005),
                ..tangential
            },
        ),
        (
            "reach 2 + tangential + centring 0.02",
            Recipe {
                centring: Some(0.02),
                ..tangential
            },
        ),
        (
            "reach 2 + own radial damping 2.5",
            Recipe {
                counter: Counter::Own(2.5),
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
    ] {
        sweep(label, &recipe);
    }
}

/// The P2 seeds in full for the tangential candidates: where the kinetic
/// energy dips, what the motion is doing.
#[test]
#[ignore = "diagnostic: prints P2's per-seed readings for the tangential candidates"]
fn diag_orbit_p2_detail() {
    let catalog = Recipe::catalog();
    let fixture = Fixture::p2();
    for reach in [2.0, 3.0] {
        let recipe = Recipe {
            counter: Counter::Tangential,
            ..catalog.reach(reach)
        };
        for damping in DAMPINGS {
            for spacing in SPACINGS {
                run(
                    &format!("reach {reach} + tangential"),
                    &fixture,
                    spacing,
                    damping,
                    recipe.parts(&fixture),
                    120,
                );
            }
        }
    }
}

/// The refinements over the seed sweep, with each seed's worst character.
#[test]
#[ignore = "diagnostic: prints the refinements over five seeds and both dampings"]
fn diag_orbit_sweep_refined() {
    let catalog = Recipe::catalog();
    let tangential = Recipe {
        counter: Counter::Tangential,
        ..catalog.reach(2.0)
    };
    let prograde = Recipe {
        counter: Counter::Prograde,
        ..catalog.reach(2.0)
    };
    for (label, recipe) in [
        (
            "reach 2 + tangential + centring 0.02",
            Recipe {
                centring: Some(0.02),
                ..tangential
            },
        ),
        ("reach 2 + prograde", prograde),
        (
            "reach 3 + prograde",
            Recipe {
                counter: Counter::Prograde,
                ..catalog.reach(3.0)
            },
        ),
        (
            "reach 2 + prograde + centring 0.02",
            Recipe {
                centring: Some(0.02),
                ..prograde
            },
        ),
    ] {
        sweep(label, &recipe);
    }
}
