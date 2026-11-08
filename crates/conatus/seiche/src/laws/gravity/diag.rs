// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! Orbit's diagnosis (orbit-retune lane, 2026-10-04). The catalog's force set
//! on the P2 fixture and on G1's generated 40-node graph, metered term by
//! term: each term's work on the kinetic energy and its outward push, per
//! window, beside the extent, the energy and the motion's character; then
//! each term removed in turn.
//!
//! The candidate fixes (`Counter`'s forms, exclusion's reach, centring) are
//! test forces here; the ruled one ("Frictionless orbits + centring",
//! 2026-10-04) is the law's own `CounterDamping::Tangential`, which
//! `Counter::Law` runs beside the test force it was built from.
//! `diag_orbit_terms` is the diagnosis and its controls, `diag_orbit_candidates`
//! and `diag_orbit_leading` one seed each, `diag_orbit_sweep` and
//! `diag_orbit_sweep_refined` five seeds at both pages' dampings,
//! `diag_orbit_p2_detail` P2's seeds in full, `diag_orbit_ruled` the ruled law
//! over the sweep, and `diag_orbit_no_damping` the law at no host damping.
//! Gravity's default test `tangential_counter_damping_keeps_the_orbits_bound`
//! reuses this instrument. Logs: `Code/testing/mere/orbit/`.
//!
//! `cargo test -p seiche --lib laws::gravity::diag -- --ignored --nocapture`

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use euclid::default::Point2D;
#[cfg(feature = "engine")]
use rapier2d::prelude::*;
#[cfg(not(feature = "engine"))]
use crate::vector::Vector;

use super::{CounterDamping, Gravity};
use crate::{Boundary, Declared, Force, ForceContext, NodeExclusion, NodeKey, Simulation, TICK_DT};

/// The P2 fixture's ten relations over its eleven nodes (G1's `P2_EDGES`).
const P2_EDGES: [(usize, usize); 10] = [
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
];

/// Below this centre distance two node bodies overlap (two radii).
const TOUCH: f32 = 2.0 * crate::NODE_BODY_RADIUS;

pub(super) struct Fixture {
    pub name: &'static str,
    pub keys: Vec<NodeKey>,
    pub edges: Vec<(NodeKey, NodeKey)>,
    /// `1 + degree`, the catalog's Degree source (multiplicity counted).
    pub masses: Vec<f32>,
}

impl Fixture {
    fn new(name: &'static str, n: usize, pairs: &[(usize, usize)]) -> Self {
        let keys: Vec<NodeKey> = (0..n).map(NodeKey::new).collect();
        let mut masses = vec![1.0; n];
        for &(a, b) in pairs {
            masses[a] += 1.0;
            masses[b] += 1.0;
        }
        Self {
            name,
            edges: pairs.iter().map(|&(a, b)| (keys[a], keys[b])).collect(),
            keys,
            masses,
        }
    }

    pub fn p2() -> Self {
        Self::new("p2", 11, &P2_EDGES)
    }

    /// G1's generated graph: a seeded random tree plus twenty chords.
    pub fn generated() -> Self {
        let mut state = 7u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as usize
        };
        let n = 40;
        let mut pairs: Vec<(usize, usize)> = (1..n).map(|i| (i, next() % i)).collect();
        for _ in 0..n / 2 {
            let (a, b) = (next() % n, next() % n);
            if a != b {
                pairs.push((a, b));
            }
        }
        Self::new("gen-40", n, &pairs)
    }

    pub fn mass_map(&self) -> Vec<(NodeKey, f32)> {
        self.keys
            .iter()
            .copied()
            .zip(self.masses.iter().copied())
            .collect()
    }

    /// The boot Spiral's shape: golden-angle phyllotaxis, `spacing · √i`.
    pub fn spiral(&self, spacing: f32) -> Vec<(NodeKey, Point2D<f32>)> {
        self.keys
            .iter()
            .enumerate()
            .map(|(i, &k)| {
                let (r, t) = (spacing * (i as f32).sqrt(), i as f32 * 2.399_963);
                (k, Point2D::new(r * t.cos(), r * t.sin()))
            })
            .collect()
    }
}

/// What one metered term did this tick: its push per body, and the kinetic
/// energy its velocity writes added.
#[derive(Default)]
struct Ledger {
    push: HashMap<RigidBodyHandle, Vector>,
    written: f64,
}

/// A term whose pushes and velocity writes are booked. `writes_only` keeps
/// its velocity writes and takes back its pushes (the kick without the
/// gravitation it is bundled with).
pub(super) struct Metered {
    inner: Box<dyn Force>,
    ledger: Arc<Mutex<Ledger>>,
    writes_only: bool,
}

impl std::fmt::Debug for Metered {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Metered")
    }
}

impl Declared for Metered {
    fn terms(&self) -> Vec<crate::Term> {
        self.inner.terms()
    }
}

impl Force for Metered {
    #[cfg(feature = "engine")]
    fn apply(&self, ctx: &mut ForceContext<'_>, dt: f32) {
        let handles: Vec<RigidBodyHandle> = ctx.bodies_by_node.values().copied().collect();
        let before: Vec<(Vector, Vector, f32)> = handles
            .iter()
            .map(|h| {
                ctx.bodies
                    .get(*h)
                    .map_or((Vector::ZERO, Vector::ZERO, 0.0), |b| {
                        (b.user_force(), b.linvel(), b.mass())
                    })
            })
            .collect();
        self.inner.apply(ctx, dt);
        let mut ledger = self.ledger.lock().unwrap();
        ledger.push.clear();
        for (h, (f0, v0, m)) in handles.iter().zip(before) {
            let Some(body) = ctx.bodies.get_mut(*h) else {
                continue;
            };
            let v1 = body.linvel();
            ledger.written +=
                0.5 * f64::from(m) * f64::from(v1.length_squared() - v0.length_squared());
            if self.writes_only {
                body.reset_forces(false);
                body.add_force(f0, false);
            } else {
                ledger.push.insert(*h, body.user_force() - f0);
            }
        }
    }
}

/// Records every body's velocity after the force pass, before the step.
#[derive(Debug)]
struct Snapshot(Arc<Mutex<HashMap<RigidBodyHandle, Vector>>>);

impl Declared for Snapshot {
    fn terms(&self) -> Vec<crate::Term> {
        Vec::new()
    }
}

impl Force for Snapshot {
    #[cfg(feature = "engine")]
    fn apply(&self, ctx: &mut ForceContext<'_>, _dt: f32) {
        let mut out = self.0.lock().unwrap();
        out.clear();
        for &h in ctx.bodies_by_node.values() {
            if let Some(b) = ctx.bodies.get(h) {
                out.insert(h, b.linvel());
            }
        }
    }
}

/// One term of a run: a label and the force that applies it.
pub(super) struct Part {
    pub label: &'static str,
    pub force: Box<dyn Force>,
    pub writes_only: bool,
}

impl Part {
    pub fn new(label: &'static str, force: impl Force + 'static) -> Self {
        Self {
            label,
            force: Box::new(force),
            writes_only: false,
        }
    }
}

/// How the drive that cancels the host's damping behaves.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Counter {
    /// Today's: every body's damping cancelled.
    Full,
    /// Bound bodies' damping cancelled; an unbound body's motion relative to
    /// the system is damped until it is bound again.
    Brake,
    /// Every body's damping cancelled in the ratio of the kick's kinetic
    /// energy to the present one, when the present one is larger.
    Cap,
    /// Only the tangential part of each body's motion about the system's
    /// mass centre is frictionless; radial motion and drift are damped.
    Tangential,
    /// Every body's damping cancelled, and the law's own damping, at this
    /// rate, on radial motion and drift: Tangential without the host's say.
    Own(f32),
    /// The law's own [`CounterDamping::Tangential`], to check it against the
    /// test force it was built from.
    Law,
    /// Tangential, prograde only: motion in the kick's sense is frictionless;
    /// radial, retrograde and drift motion are damped.
    Prograde,
}

/// One Orbit composition, term by term.
#[derive(Clone, Copy, Debug)]
pub(super) struct Recipe {
    pub exclusion: Option<NodeExclusion>,
    pub strength: f32,
    pub softening: f32,
    pub kick: bool,
    pub gravitation: bool,
    pub counter: Counter,
    /// `Gravity::radial_floor`, read under `Counter::Law` only.
    pub radial_floor: f32,
    pub centring: Option<f32>,
}

/// The catalog's Orbit, split into its terms: exclusion, kick, gravitation,
/// counter-damping. `drop` names the term left out.
pub(super) fn catalog_parts(fixture: &Fixture, drop: Option<&str>) -> Vec<Part> {
    let mut parts = Recipe::catalog().parts(fixture);
    parts.retain(|p| Some(p.label) != drop);
    parts
}

/// Counter-damping with an escape brake: a body whose specific energy about
/// the system (`½|v − v̄|² − φ`, gravitation's Plummer potential) is below zero
/// is frictionless; an unbound one keeps only its share of the system's drift.
#[derive(Debug)]
struct Brake {
    masses: HashMap<NodeKey, f32>,
    strength: f32,
    softening: f32,
}

impl Declared for Brake {
    fn terms(&self) -> Vec<crate::Term> {
        Vec::new()
    }
}

impl Force for Brake {
    #[cfg(feature = "engine")]
    fn apply(&self, ctx: &mut ForceContext<'_>, _dt: f32) {
        let nodes = super::node_positions(ctx);
        let m: Vec<f32> = nodes
            .iter()
            .map(|(k, _, _)| self.masses.get(k).copied().unwrap_or(1.0).max(0.1))
            .collect();
        let v: Vec<Vector> = nodes
            .iter()
            .map(|(_, h, _)| ctx.bodies.get(*h).map_or(Vector::ZERO, |b| b.linvel()))
            .collect();
        let total: f32 = m.iter().sum();
        let drift = v.iter().zip(&m).fold(Vector::ZERO, |a, (v, m)| a + *v * *m) / total;
        let soft2 = self.softening * self.softening;
        for i in 0..nodes.len() {
            let mut phi = 0.0;
            for j in 0..nodes.len() {
                if i != j {
                    let d2 = (nodes[j].2 - nodes[i].2).length_squared() + soft2;
                    phi += self.strength * m[j] / d2.sqrt();
                }
            }
            let bound = 0.5 * (v[i] - drift).length_squared() < phi;
            if let Some(body) = ctx.bodies.get_mut(nodes[i].1) {
                let drive = if bound { v[i] } else { drift };
                let inertial = body.mass();
                body.add_force(drive * (body.linear_damping() * inertial), true);
            }
        }
    }
}

/// Counter-damping on the tangential motion alone: about the gravitational
/// mass centre, moving with it, each body's tangential velocity is driven
/// back against the host's damping; its radial velocity and the drift are not.
#[derive(Debug)]
struct Tangential {
    masses: HashMap<NodeKey, f32>,
    /// `Some(rate)`: instead, damp the radial motion and drift at this rate.
    own: Option<f32>,
    /// Only motion in the kick's (counter-clockwise) sense is driven.
    prograde: bool,
}

impl Declared for Tangential {
    fn terms(&self) -> Vec<crate::Term> {
        Vec::new()
    }
}

impl Force for Tangential {
    #[cfg(feature = "engine")]
    fn apply(&self, ctx: &mut ForceContext<'_>, _dt: f32) {
        let nodes = super::node_positions(ctx);
        let m: Vec<f32> = nodes
            .iter()
            .map(|(k, _, _)| self.masses.get(k).copied().unwrap_or(1.0).max(0.1))
            .collect();
        let v: Vec<Vector> = nodes
            .iter()
            .map(|(_, h, _)| ctx.bodies.get(*h).map_or(Vector::ZERO, |b| b.linvel()))
            .collect();
        let total: f32 = m.iter().sum();
        let weigh = |xs: &dyn Fn(usize) -> Vector| {
            (0..nodes.len()).fold(Vector::ZERO, |a, i| a + xs(i) * m[i]) / total
        };
        let centre = weigh(&|i| nodes[i].2);
        let drift = weigh(&|i| v[i]);
        for i in 0..nodes.len() {
            let r = nodes[i].2 - centre;
            let rl = r.length();
            if rl < 1e-3 {
                continue;
            }
            let t = Vector::new(-r.y, r.x) / rl;
            let u = v[i] - drift;
            let along = u.x * t.x + u.y * t.y;
            let tangential = t * if self.prograde { along.max(0.0) } else { along };
            if let Some(body) = ctx.bodies.get_mut(nodes[i].1) {
                let inertial = body.mass();
                let push = match self.own {
                    None => tangential * (body.linear_damping() * inertial),
                    Some(rate) => (tangential - v[i]) * (rate * inertial),
                };
                body.add_force(push, true);
            }
        }
    }
}

/// Counter-damping capped at the kick's kinetic energy.
#[derive(Debug, Default)]
struct Cap {
    budget: Mutex<Option<f32>>,
}

impl Declared for Cap {
    fn terms(&self) -> Vec<crate::Term> {
        Vec::new()
    }
}

impl Force for Cap {
    #[cfg(feature = "engine")]
    fn apply(&self, ctx: &mut ForceContext<'_>, _dt: f32) {
        let handles: Vec<RigidBodyHandle> = ctx.bodies_by_node.values().copied().collect();
        let energy: f32 = handles
            .iter()
            .filter_map(|h| ctx.bodies.get(*h))
            .map(|b| 0.5 * b.mass() * b.linvel().length_squared())
            .sum();
        let budget = *self.budget.lock().unwrap().get_or_insert(energy.max(1.0));
        let share = if energy > budget {
            budget / energy
        } else {
            1.0
        };
        for h in handles {
            if let Some(body) = ctx.bodies.get_mut(h) {
                let push = body.linvel() * (body.linear_damping() * body.mass() * share);
                body.add_force(push, true);
            }
        }
    }
}

/// The motion's character and size at one instant.
struct Shape {
    extent: f32,
    overlaps: usize,
    /// Share of kinetic energy (about the centroid) in tangential motion.
    tangential: f32,
    /// Net angular momentum over its maximum: 1 all co-rotating circles.
    coherence: f32,
    /// Spearman of mass against radius: below zero, hubs sit inside.
    mass_radius: f32,
}

fn spearman(a: &[f32], b: &[f32]) -> f32 {
    let rank = |v: &[f32]| {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_by(|&i, &j| v[i].total_cmp(&v[j]));
        let mut r = vec![0.0f32; v.len()];
        let mut i = 0;
        while i < idx.len() {
            let mut j = i;
            while j + 1 < idx.len() && v[idx[j + 1]] == v[idx[i]] {
                j += 1;
            }
            let mean = (i + j) as f32 / 2.0;
            for &k in &idx[i..=j] {
                r[k] = mean;
            }
            i = j + 1;
        }
        r
    };
    let (ra, rb) = (rank(a), rank(b));
    let n = a.len() as f32;
    let (ma, mb) = (ra.iter().sum::<f32>() / n, rb.iter().sum::<f32>() / n);
    let (mut c, mut va, mut vb) = (0.0, 0.0, 0.0);
    for i in 0..a.len() {
        c += (ra[i] - ma) * (rb[i] - mb);
        va += (ra[i] - ma).powi(2);
        vb += (rb[i] - mb).powi(2);
    }
    if va <= 0.0 || vb <= 0.0 {
        0.0
    } else {
        c / (va * vb).sqrt()
    }
}

fn centroid(points: &[Vector]) -> Vector {
    points.iter().fold(Vector::ZERO, |a, p| a + *p) / points.len() as f32
}

fn shape(at: &[Vector], vel: &[Vector], masses: &[f32]) -> Shape {
    let (mut lo, mut hi) = (at[0], at[0]);
    for p in at {
        lo = Vector::new(lo.x.min(p.x), lo.y.min(p.y));
        hi = Vector::new(hi.x.max(p.x), hi.y.max(p.y));
    }
    let mut overlaps = 0;
    for i in 0..at.len() {
        for j in (i + 1)..at.len() {
            overlaps += usize::from((at[i] - at[j]).length() < TOUCH);
        }
    }
    let (c, vc) = (centroid(at), centroid(vel));
    let (mut tan, mut all, mut l, mut lmax) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let mut radii = Vec::with_capacity(at.len());
    for (p, v) in at.iter().zip(vel) {
        let (r, u) = (*p - c, *v - vc);
        let rl = r.length();
        radii.push(rl);
        if rl < 1e-3 {
            continue;
        }
        let cross = r.x * u.y - r.y * u.x;
        tan += (cross / rl).powi(2);
        all += u.length_squared();
        l += cross;
        lmax += rl * u.length();
    }
    Shape {
        extent: (hi.x - lo.x).max(hi.y - lo.y),
        overlaps,
        tangential: if all > 0.0 { tan / all } else { 0.0 },
        coherence: if lmax > 0.0 { l / lmax } else { 0.0 },
        mass_radius: spearman(masses, &radii),
    }
}

/// One window's booked work per term and the mean outward push per term.
#[derive(Default, Clone)]
struct Window {
    work: Vec<f64>,
    outward: Vec<f64>,
    written: f64,
    damping: f64,
    contacts: f64,
    ticks: u32,
}

/// A run's readings at each sample time, printed as it goes.
pub(super) struct Report {
    pub extent_1s: f32,
    pub max_ratio: f32,
    pub end_ratio: f32,
    pub min_energy: f32,
    pub end_energy: f32,
    pub mean_tangential: f32,
    pub mean_coherence: f32,
    pub mean_mass_radius: f32,
    pub revolutions: f32,
    pub exclusion_check: Option<(f64, f64)>,
    pub max_overlaps: usize,
}

pub(super) const SAMPLES: [u32; 9] = [1, 2, 6, 10, 20, 30, 60, 90, 120];

/// The seeds a sweep runs: the boot Spiral's spacing and four others.
const SPACINGS: [f32; 5] = [16.0, 19.0, 24.0, 32.0, 40.0];
/// The two pages' dampings: the tree page's canvas default, the old page's.
const DAMPINGS: [f32; 2] = [2.5, 0.82];

/// Every seed and both dampings on both fixtures, 120 s each, one line per
/// fixture and damping: the worst and median of the extent's largest
/// multiple of its first second, and the character's means.
pub(super) fn sweep(label: &str, recipe: &Recipe) {
    sweep_at(label, recipe, &DAMPINGS);
}

#[path = "diag/controls.rs"]
mod controls;

#[path = "diag/sweeps.rs"]
mod sweeps;

#[path = "diag/instrument.rs"]
mod instrument;

pub(super) use instrument::run;

pub(super) use instrument::run_quietly;

#[path = "diag/recipes.rs"]
mod recipes;

pub(super) use sweeps::sweep_at;
