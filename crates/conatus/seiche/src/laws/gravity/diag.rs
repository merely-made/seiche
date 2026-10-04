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
use rapier2d::prelude::*;

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

impl Recipe {
    /// The catalog's Orbit as built today.
    pub fn catalog() -> Self {
        let g = Gravity::new([], CounterDamping::Full);
        Self {
            exclusion: Some(NodeExclusion::default()),
            strength: g.strength,
            softening: g.softening,
            kick: true,
            gravitation: true,
            counter: Counter::Full,
            radial_floor: g.radial_floor,
            centring: None,
        }
    }

    /// Exclusion whose reach ends at `diameters` node diameters.
    pub fn reach(self, diameters: f32) -> Self {
        Self {
            exclusion: Some(NodeExclusion {
                cutoff: diameters * TOUCH,
                ..NodeExclusion::default()
            }),
            ..self
        }
    }

    pub fn parts(&self, fixture: &Fixture) -> Vec<Part> {
        let gravity = |strength: f32, kick: f32, counter: CounterDamping| Gravity {
            strength,
            softening: self.softening,
            orbital_kick: kick,
            counter_damping: counter,
            radial_floor: self.radial_floor,
            ..Gravity::new(fixture.mass_map(), counter)
        };
        let mut parts = Vec::new();
        if let Some(exclusion) = self.exclusion {
            parts.push(Part::new("exclusion", exclusion));
        }
        if self.kick {
            parts.push(Part {
                label: "kick",
                force: Box::new(gravity(self.strength, 1.0, CounterDamping::Off)),
                writes_only: true,
            });
        }
        if self.gravitation {
            parts.push(Part::new(
                "gravitation",
                gravity(self.strength, 0.0, CounterDamping::Off),
            ));
        }
        match self.counter {
            Counter::Full => parts.push(Part::new(
                "counter-damping",
                gravity(0.0, 0.0, CounterDamping::Full),
            )),
            Counter::Law => parts.push(Part::new(
                "counter-damping",
                gravity(0.0, 0.0, CounterDamping::Tangential),
            )),
            Counter::Brake => parts.push(Part::new(
                "counter-damping",
                Brake {
                    masses: fixture.mass_map().into_iter().collect(),
                    strength: self.strength,
                    softening: self.softening,
                },
            )),
            Counter::Cap => parts.push(Part::new("counter-damping", Cap::default())),
            Counter::Tangential => parts.push(Part::new(
                "counter-damping",
                Tangential {
                    masses: fixture.mass_map().into_iter().collect(),
                    own: None,
                    prograde: false,
                },
            )),
            Counter::Prograde => parts.push(Part::new(
                "counter-damping",
                Tangential {
                    masses: fixture.mass_map().into_iter().collect(),
                    own: None,
                    prograde: true,
                },
            )),
            Counter::Own(rate) => {
                parts.push(Part::new(
                    "counter-damping",
                    gravity(0.0, 0.0, CounterDamping::Full),
                ));
                parts.push(Part::new(
                    "radial damping",
                    Tangential {
                        masses: fixture.mass_map().into_iter().collect(),
                        own: Some(rate),
                        prograde: false,
                    },
                ));
            },
        }
        if let Some(strength) = self.centring {
            parts.push(Part::new("centring", Boundary { strength }));
        }
        parts
    }
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

/// Run `parts` on `fixture` from a spiral seed for `seconds`, printing one
/// line per sample time: shape, energy and the window's term ledger.
pub(super) fn run(
    title: &str,
    fixture: &Fixture,
    spacing: f32,
    damping: f32,
    parts: Vec<Part>,
    seconds: u32,
) -> Report {
    run_quietly(title, fixture, spacing, damping, parts, seconds, true)
}

pub(super) fn run_quietly(
    title: &str,
    fixture: &Fixture,
    spacing: f32,
    damping: f32,
    parts: Vec<Part>,
    seconds: u32,
    verbose: bool,
) -> Report {
    let mut sim = Simulation::new();
    sim.set_linear_damping(damping);
    sim.sync_nodes(fixture.spiral(spacing));
    sim.sync_edges(fixture.edges.clone());
    let labels: Vec<&str> = parts.iter().map(|p| p.label).collect();
    let mut ledgers = Vec::new();
    let mut forces: Vec<Box<dyn Force>> = Vec::new();
    for part in parts {
        let ledger = Arc::new(Mutex::new(Ledger::default()));
        ledgers.push(ledger.clone());
        forces.push(Box::new(Metered {
            inner: part.force,
            ledger,
            writes_only: part.writes_only,
        }));
    }
    let snap = Arc::new(Mutex::new(HashMap::new()));
    forces.push(Box::new(Snapshot(snap.clone())));
    sim.set_forces(forces);
    let handles: Vec<RigidBodyHandle> =
        fixture.keys.iter().map(|k| sim.bodies_by_node[k]).collect();
    let n = handles.len();
    let exclusion_at = labels.iter().position(|l| *l == "exclusion");
    let counter_at = labels.iter().position(|l| *l == "counter-damping");
    let exclusion_energy = |sim: &Simulation| {
        let nodes: Vec<(NodeKey, Vector)> = fixture
            .keys
            .iter()
            .map(|k| {
                let p = sim.position_of(*k).unwrap();
                (*k, Vector::new(p.x, p.y))
            })
            .collect();
        NodeExclusion::default()
            .energy(
                0,
                &crate::Layout {
                    nodes: &nodes,
                    edges: &fixture.edges,
                },
            )
            .unwrap()
    };
    let u_exclusion_0 = exclusion_energy(&sim);
    let mut exclusion_work_total = 0.0;

    if verbose {
        println!(
            "{title} [{} {} nodes, spiral {spacing}, damping {damping}]",
            fixture.name, n
        );
        println!(
            "  {:>5} {:>7} {:>6} {:>9} {:>4} {:>5} {:>5} {:>6} | work in the window (share): {} written damping contacts; counter-damping net of damping | mean outward push: {}",
            "t s",
            "extent",
            "x1s",
            "KE",
            "ovl",
            "tang",
            "coh",
            "m~r",
            labels.join(" "),
            labels.join(" "),
        );
    }
    let k = 1.0 / (1.0 + f64::from(damping) * f64::from(TICK_DT));
    let mut window = Window {
        work: vec![0.0; labels.len()],
        outward: vec![0.0; labels.len()],
        ..Window::default()
    };
    let mut written_seen = vec![0.0f64; labels.len()];
    let mut angle: Vec<f32> = vec![0.0; n];
    let mut turned: Vec<f32> = vec![0.0; n];
    let mut extent_1s = 0.0f32;
    let (mut max_ratio, mut end_ratio) = (0.0f32, 0.0f32);
    let (mut min_energy, mut end_energy) = (f32::INFINITY, 0.0f32);
    let (mut tangential, mut coherence, mut mass_radius, mut counted) = (0.0, 0.0, 0.0, 0);
    let mut max_overlaps = 0usize;
    let read = |sim: &Simulation| -> (Vec<Vector>, Vec<Vector>) {
        let at = handles
            .iter()
            .map(|h| sim.bodies[*h].translation())
            .collect();
        let vel = handles.iter().map(|h| sim.bodies[*h].linvel()).collect();
        (at, vel)
    };
    {
        let (at, _) = read(&sim);
        let c = centroid(&at);
        for i in 0..n {
            let r = at[i] - c;
            angle[i] = r.y.atan2(r.x);
        }
    }
    let ticks = seconds * 60;
    for tick in 1..=ticks {
        let (at0, _) = read(&sim);
        let c0 = centroid(&at0);
        sim.tick(TICK_DT);
        let (at1, v1) = read(&sim);
        let v0 = snap.lock().unwrap().clone();
        let mut tick_work = 0.0;
        for (t, ledger) in ledgers.iter().enumerate() {
            let ledger = ledger.lock().unwrap();
            let fresh = ledger.written - written_seen[t];
            written_seen[t] = ledger.written;
            window.written += fresh;
            for (i, h) in handles.iter().enumerate() {
                let Some(push) = ledger.push.get(h) else {
                    continue;
                };
                let vbar = (v0[h] + v1[i]) * 0.5;
                let w = f64::from(push.x * vbar.x + push.y * vbar.y) * f64::from(TICK_DT) * k;
                window.work[t] += w;
                tick_work += w;
                if Some(t) == exclusion_at {
                    exclusion_work_total += w;
                }
                let r = at0[i] - c0;
                let rl = r.length();
                if rl > 1e-3 {
                    window.outward[t] += f64::from((push.x * r.x + push.y * r.y) / rl) / n as f64;
                }
            }
        }
        let (mut ke0, mut ke1, mut damp) = (0.0f64, 0.0f64, 0.0f64);
        for (i, h) in handles.iter().enumerate() {
            let m = f64::from(sim.bodies[*h].mass());
            let (a, b) = (v0[h], v1[i]);
            let vbar = (a + b) * 0.5;
            ke0 += 0.5 * m * f64::from(a.length_squared());
            ke1 += 0.5 * m * f64::from(b.length_squared());
            damp -= m
                * f64::from(damping)
                * f64::from(TICK_DT)
                * k
                * f64::from(a.x * vbar.x + a.y * vbar.y);
        }
        window.damping += damp;
        window.contacts += (ke1 - ke0) - tick_work - damp;
        window.ticks += 1;
        let c1 = centroid(&at1);
        for i in 0..n {
            let r = at1[i] - c1;
            let a = r.y.atan2(r.x);
            let mut d = a - angle[i];
            while d > std::f32::consts::PI {
                d -= std::f32::consts::TAU;
            }
            while d < -std::f32::consts::PI {
                d += std::f32::consts::TAU;
            }
            turned[i] += d;
            angle[i] = a;
        }
        let second = tick / 60;
        if tick % 60 != 0 || !SAMPLES.contains(&second) || second > seconds {
            continue;
        }
        let s = shape(&at1, &v1, &fixture.masses);
        let energy = sim.kinetic_energy();
        if second >= 1 {
            max_overlaps = max_overlaps.max(s.overlaps);
        }
        if second == 1 {
            extent_1s = s.extent;
        }
        let ratio = s.extent / extent_1s.max(1.0);
        if second >= 1 {
            max_ratio = max_ratio.max(ratio);
            end_ratio = ratio;
            end_energy = energy;
            min_energy = min_energy.min(energy);
        }
        if second >= 6 {
            tangential += s.tangential;
            coherence += s.coherence;
            mass_radius += s.mass_radius;
            counted += 1;
        }
        let total: f64 = window.work.iter().map(|w| w.abs()).sum::<f64>()
            + window.written.abs()
            + window.damping.abs()
            + window.contacts.abs();
        let share = |w: f64| {
            if total > 0.0 {
                100.0 * w.abs() / total
            } else {
                0.0
            }
        };
        let works: Vec<String> = window
            .work
            .iter()
            .map(|w| format!("{w:.0}({:.0}%)", share(*w)))
            .collect();
        let outward: Vec<String> = window
            .outward
            .iter()
            .map(|o| format!("{:.1}", o / f64::from(window.ticks)))
            .collect();
        if verbose {
            println!(
                "  {second:>5} {:>7.0} {ratio:>6.2} {energy:>9.0} {:>4} {:>5.2} {:>5.2} {:>6.2} | {} {:.0}({:.0}%) {:.0}({:.0}%) {:.0}({:.0}%); {:.0} | {}",
                s.extent,
                s.overlaps,
                s.tangential,
                s.coherence,
                s.mass_radius,
                works.join(" "),
                window.written,
                share(window.written),
                window.damping,
                share(window.damping),
                window.contacts,
                share(window.contacts),
                counter_at.map_or(0.0, |c| window.work[c] + window.damping),
                outward.join(" "),
            );
        }
        window = Window {
            work: vec![0.0; labels.len()],
            outward: vec![0.0; labels.len()],
            ..Window::default()
        };
    }
    let revolutions =
        turned.iter().map(|t| t.abs()).sum::<f32>() / n as f32 / std::f32::consts::TAU;
    let exclusion_check =
        exclusion_at.map(|_| (exclusion_work_total, u_exclusion_0 - exclusion_energy(&sim)));
    if let Some((w, du)) = exclusion_check.filter(|_| verbose) {
        println!("  instrument: exclusion's booked work {w:.0} against its energy drop {du:.0}");
    }
    let counted = counted.max(1) as f32;
    let report = Report {
        extent_1s,
        max_ratio,
        end_ratio,
        min_energy,
        end_energy,
        mean_tangential: tangential / counted,
        mean_coherence: coherence / counted,
        mean_mass_radius: mass_radius / counted,
        revolutions,
        exclusion_check,
        max_overlaps,
    };
    if verbose {
        println!(
            "  summary: extent 1 s {:.0}, max x{:.2}, end x{:.2}; KE min {:.0}, end {:.0}; from 6 s tangential {:.2}, coherence {:.2}, mass~radius {:.2}; mean revolutions {:.2}",
            report.extent_1s,
            report.max_ratio,
            report.end_ratio,
            report.min_energy,
            report.end_energy,
            report.mean_tangential,
            report.mean_coherence,
            report.mean_mass_radius,
            report.revolutions,
        );
    }
    report
}

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

/// [`sweep`] at the given dampings.
pub(super) fn sweep_at(label: &str, recipe: &Recipe, dampings: &[f32]) {
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
