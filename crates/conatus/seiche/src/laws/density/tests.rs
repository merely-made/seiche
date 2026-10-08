// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! Density's grid against an analytic reference, and the law's claims: no
//! self flow, mass rides the flow, a pinned body is never moved.

use super::*;
use crate::Simulation;
use euclid::default::Point2D;

/// The exact solution of `(I − α∇²) u = b` with a mirror edge, by the cosine
/// transform the mirror Laplacian is diagonal in: the analytic reference the
/// Jacobi sweeps must reach.
fn spectral_reference(b: &[f32], n: usize, alpha: f64) -> Vec<f64> {
    let basis =
        |k: usize, i: usize| (std::f64::consts::PI * k as f64 * (i as f64 + 0.5) / n as f64).cos();
    let norm = |k: usize| {
        if k == 0 {
            1.0 / n as f64
        } else {
            2.0 / n as f64
        }
    };
    let eigen = |k: usize| 2.0 - 2.0 * (std::f64::consts::PI * k as f64 / n as f64).cos();
    let mut out = vec![0.0f64; n * n];
    for ky in 0..n {
        for kx in 0..n {
            let mut coeff = 0.0;
            for y in 0..n {
                for x in 0..n {
                    coeff += b[y * n + x] as f64 * basis(kx, x) * basis(ky, y);
                }
            }
            coeff *= norm(kx) * norm(ky) / (1.0 + alpha * (eigen(kx) + eigen(ky)));
            for y in 0..n {
                for x in 0..n {
                    out[y * n + x] += coeff * basis(kx, x) * basis(ky, y);
                }
            }
        }
    }
    out
}

fn unit_domain(n: usize) -> DensityDomain {
    DensityDomain {
        min: Vector::ZERO,
        side: n as f32,
    }
}

/// A known splat: one mass at a cell centre lands in that cell alone, one at
/// a cell corner splits in quarters, and mass is conserved.
#[test]
fn the_splat_is_cloud_in_cell_and_conserves_mass() {
    let mut grid = DensityGrid::new(8);
    let domain = unit_domain(8);
    grid.splat(domain, &[(Vector::new(2.5, 3.5), 4.0)]);
    assert_eq!(grid.source()[3 * 8 + 2], 4.0);
    grid.splat(domain, &[(Vector::new(3.0, 3.0), 4.0)]);
    for cell in [2 * 8 + 2, 2 * 8 + 3, 3 * 8 + 2, 3 * 8 + 3] {
        assert!((grid.source()[cell] - 1.0).abs() < 1e-6);
    }
    let total: f32 = grid.source().iter().sum();
    assert!((total - 4.0).abs() < 1e-5);
}

/// The diffused field matches the analytic (spectral) solution of the
/// implicit step on a known splat, to 1e-4 relative, and the walls conserve
/// mass.
#[test]
fn jacobi_diffusion_matches_the_spectral_reference() {
    let n = 24;
    let alpha = 6.0;
    let mut grid = DensityGrid::new(n);
    grid.splat(
        unit_domain(n),
        &[
            (Vector::new(7.3, 9.6), 3.0),
            (Vector::new(16.0, 15.0), 1.0),
            (Vector::new(0.2, 23.9), 2.0),
        ],
    );
    grid.diffuse(alpha, 3_000);
    let reference = spectral_reference(grid.source(), n, alpha as f64);
    let peak = reference.iter().cloned().fold(0.0f64, f64::max);
    let worst = grid
        .field()
        .iter()
        .zip(&reference)
        .map(|(u, r)| (*u as f64 - r).abs())
        .fold(0.0f64, f64::max);
    assert!(
        worst / peak < 1e-4,
        "jacobi vs spectral: {worst:e} of {peak:e}"
    );
    let before: f32 = grid.source().iter().sum();
    let after: f32 = grid.field().iter().sum();
    assert!(
        (before - after).abs() / before < 1e-4,
        "walls conserve mass"
    );
    let src_peak = grid.source().iter().cloned().fold(0.0, f32::max);
    assert!((peak as f32) < src_peak * 0.5, "a blur, not a copy");
}

/// An evolving field conserves mass step after step and evens out.
#[test]
fn the_evolving_field_conserves_mass_and_evens() {
    let n = 32;
    let mut grid = DensityGrid::new(n);
    grid.begin(unit_domain(n), &[(Vector::new(8.0, 8.0), 10.0)], 1.0, 0.01);
    let total: f32 = grid.field().iter().sum();
    let spread = |g: &DensityGrid| {
        let (lo, hi) = g
            .field()
            .iter()
            .fold((f32::MAX, 0.0f32), |(lo, hi), u| (lo.min(*u), hi.max(*u)));
        hi / lo
    };
    let early = spread(&grid);
    for _ in 0..400 {
        grid.advance(2.0);
    }
    let after: f32 = grid.field().iter().sum();
    assert!(
        (total - after).abs() / total < 1e-3,
        "mass {total} -> {after}"
    );
    let late = spread(&grid);
    assert!(late < 1.01, "even: {late} (from {early})");
}

/// A lone particle feels no flow of its own, wherever it sits in a cell.
#[test]
fn a_lone_particle_feels_no_self_flow() {
    let n = 32;
    for (x, y) in [(16.0, 16.0), (15.8, 16.3), (16.5, 16.5)] {
        let mut grid = DensityGrid::new(n);
        let p = Vector::new(x, y);
        grid.begin(unit_domain(n), &[(p, 1.0)], 3.0, 1e-4);
        let mut out = Vec::new();
        grid.flow(&[p], &mut out);
        assert!(
            out[0].length() < 1e-4,
            "self flow at ({x}, {y}): {:?}",
            out[0]
        );
    }
}

fn crowd(n: usize) -> Vec<(NodeKey, Point2D<f32>)> {
    (0..n)
        .map(|i| {
            let a = i as f32 * 2.399_963;
            let r = 6.0 * (i as f32).sqrt();
            (NodeKey::new(i), Point2D::new(r * a.cos(), r * a.sin()))
        })
        .collect()
}

/// Shares a law with the simulation so the test can read its flow state.
struct Shared(std::sync::Arc<Density>);

impl Declared for Shared {
    fn terms(&self) -> Vec<Term> {
        self.0.terms()
    }

    fn metric(&self, term: usize, layout: &Layout<'_>) -> Option<Vec<f64>> {
        self.0.metric(term, layout)
    }
}

impl Force for Shared {
    fn apply(&self, ctx: &mut ForceContext<'_>, dt: f32) {
        self.0.apply(ctx, dt);
    }

    fn wants_tick(&self) -> bool {
        self.0.wants_tick()
    }
}

/// The law's claim at small scale: a crowd spreads out and the heavy node
/// ends with the most room (its nearest neighbour farthest away), the CFL
/// bound held on every tick and the flow spent.
#[test]
fn a_crowd_spreads_and_the_heavy_node_gets_room() {
    let nodes = crowd(24);
    let heavy = NodeKey::new(0);
    let mut sim = Simulation::new();
    sim.sync_nodes(nodes.clone());
    sim.sync_edges(Vec::<(NodeKey, NodeKey)>::new());
    let masses = nodes
        .iter()
        .map(|(k, _)| (*k, if *k == heavy { 8.0 } else { 1.0 }));
    let law = std::sync::Arc::new(Density::new(masses, 64));
    sim.set_forces(vec![Box::new(Shared(law.clone()))]);
    for _ in 0..360 {
        sim.tick(1.0 / 60.0);
    }
    let at: Vec<(NodeKey, Point2D<f32>)> = sim.positions().collect();
    let nearest = |k: NodeKey| {
        let p = at.iter().find(|(q, _)| *q == k).unwrap().1;
        at.iter()
            .filter(|(q, _)| *q != k)
            .map(|(_, o)| (*o - p).length())
            .fold(f32::MAX, f32::min)
    };
    let heavy_room = nearest(heavy);
    let mut light: Vec<f32> = at
        .iter()
        .filter(|(k, _)| *k != heavy)
        .map(|(k, _)| nearest(*k))
        .collect();
    light.sort_by(f32::total_cmp);
    let median = light[light.len() / 2];
    let flow = law.flow_state().expect("the flow began");
    assert!(
        median > 36.0,
        "the crowd spread: median nearest {median:.0}"
    );
    assert!(
        heavy_room > median * 1.4,
        "heavy {heavy_room:.0} vs median {median:.0}"
    );
    assert_eq!(flow.capped_ticks, 0, "the CFL bound held: {flow:?}");
    assert!(flow.last_speed < 2.0, "the flow is spent: {flow:?}");
}

/// A pinned (kinematic) body stays at its target while the rest flow.
/// Positive control: without the dynamic-body skip this fails, since the
/// advection writes the pinned body's translation.
#[test]
fn a_pinned_body_is_never_moved() {
    let nodes = crowd(16);
    let pinned = NodeKey::new(3);
    let mut sim = Simulation::new();
    sim.sync_nodes(nodes.clone());
    sim.sync_edges(Vec::<(NodeKey, NodeKey)>::new());
    sim.set_forces(vec![Box::new(Density::new(
        nodes.iter().map(|(k, _)| (*k, 1.0)),
        64,
    ))]);
    let target = Point2D::new(30.0, -20.0);
    sim.pin(pinned, target);
    let free_start = sim.position_of(NodeKey::new(9)).unwrap();
    for _ in 0..120 {
        sim.tick(1.0 / 60.0);
        let held = sim.position_of(pinned).unwrap();
        assert!(
            (held - target).length() < 1e-3,
            "the pinned body left its target: {held:?}"
        );
    }
    assert!(
        (sim.position_of(NodeKey::new(9)).unwrap() - free_start).length() > 5.0,
        "the free bodies still flow"
    );
}

/// The stop: under a shift test the passes end and the law stops asking for
/// ticks; the pass cap ends them when the test never would; and a drag after
/// the stop re-arms them, which end again once the node is let go.
#[test]
fn passes_stop_on_the_test_or_the_cap_and_a_drag_rearms_them() {
    let nodes = crowd(16);
    let run = |law: Density, ticks: usize| {
        let law = std::sync::Arc::new(law);
        let mut sim = Simulation::new();
        sim.sync_nodes(nodes.clone());
        sim.sync_edges(Vec::<(NodeKey, NodeKey)>::new());
        sim.set_forces(vec![Box::new(Shared(law.clone()))]);
        for _ in 0..ticks {
            sim.tick(1.0 / 60.0);
        }
        (sim, law)
    };
    let masses = || nodes.iter().map(|(k, _)| (*k, 1.0)).collect::<Vec<_>>();
    // The test stops the passes well before the cap.
    let mut converging = Density::new(masses(), 64);
    converging.stop = DensityStop::Shift(0.05);
    converging.patience = 2;
    converging.max_passes = 100;
    let (mut sim, law) = run(converging, 60 * 30);
    let flow = law.flow_state().unwrap();
    assert!(flow.converged, "the test stopped the passes: {flow:?}");
    assert!(!law.wants_tick() && !sim.wants_continuous_tick());
    let history = law.pass_history();
    let stopped = history.len();
    assert!(stopped < 30, "stopped after {stopped} passes");
    assert!(history.iter().rev().take(2).all(|p| p.shift < 0.05));
    // A drag re-arms the passes, and they stop again after the release.
    sim.pin(NodeKey::new(5), Point2D::new(40.0, 40.0));
    sim.tick(1.0 / 60.0);
    assert!(law.wants_tick(), "a held node re-arms the flow");
    assert!(sim.wants_continuous_tick(), "and the host keeps ticking");
    for _ in 0..30 {
        sim.tick(1.0 / 60.0);
    }
    sim.unpin(NodeKey::new(5));
    for _ in 0..60 * 30 {
        sim.tick(1.0 / 60.0);
    }
    assert!(law.flow_state().unwrap().converged, "stopped again");
    assert!(law.pass_history().len() > stopped, "the drag ran passes");
    // The cap is the fallback: a test that never passes stops at it.
    let mut capped = Density::new(masses(), 64);
    capped.stop = DensityStop::Shift(0.0);
    capped.max_passes = 3;
    let (_, law) = run(capped, 60 * 6);
    assert!(law.flow_state().unwrap().converged);
    assert_eq!(
        law.pass_history().len(),
        3,
        "the cap ended it at three passes"
    );
}

/// Why Density's conversion is off until it takes a force (dynamics grammar
/// plan, G3): converting changes Density alone on a crowded start. Each
/// tick the moving bodies start at rest, so the contacts that push the
/// crowd's overlapping bodies apart no longer carry over, and the crowd
/// spreads less in the same 360 ticks. Both medians are printed.
#[test]
fn converting_changes_density_alone_on_a_crowded_start() {
    let median = |converts: bool| {
        let nodes = crowd(24);
        let mut sim = Simulation::new();
        sim.sync_nodes(nodes.clone());
        sim.sync_edges(Vec::<(NodeKey, NodeKey)>::new());
        let mut law = Density::new(nodes.iter().map(|(k, _)| (*k, 1.0)), 64);
        law.converts = converts;
        sim.set_forces(vec![Box::new(law)]);
        for _ in 0..360 {
            sim.tick(1.0 / 60.0);
        }
        let at: Vec<Point2D<f32>> = sim.positions().map(|(_, p)| p).collect();
        let mut nearest: Vec<f32> = at
            .iter()
            .enumerate()
            .map(|(i, p)| {
                at.iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .map(|(_, o)| (*o - *p).length())
                    .fold(f32::MAX, f32::min)
            })
            .collect();
        nearest.sort_by(f32::total_cmp);
        nearest[nearest.len() / 2]
    };
    let (off, on) = (median(false), median(true));
    println!("crowd of 24 after 360 ticks: median nearest {off:.2} as built, {on:.2} converting");
    assert!(off > 36.0, "as built the crowd spreads: {off}");
    assert!(on < off, "converting, it spreads less: {on} against {off}");
}
