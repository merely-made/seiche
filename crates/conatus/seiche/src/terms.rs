// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! Term declarations: what each force is, in the dynamics grammar's words.
//!
//! Every [`Force`](crate::Force) is also [`Declared`]: it lists the terms it
//! applies, each with its topology (who interacts), kernel family, the state
//! it moves, its currency, its class and, for a metric-weighted term, its
//! metric channel, plus the observable that shows it working. A host catalog
//! reads these rather than restating them, and [`crate::instruments`] checks
//! each declared class against the forces the term actually applies.
//!
//! The classes are the dynamics grammar brief's (§3, §6):
//!
//! - **E**: a gradient of a scalar energy, every body at equal inertia;
//! - **Em**: a gradient under a non-identity mass metric, forces `−M⁻¹∇U`;
//! - **H**: conservative, but the minimizer is degenerate (collapse, total
//!   synchrony, a moving target), so the picture is in the trajectory;
//! - **N**: non-conservative (non-reciprocal, velocity-driven or rotational);
//! - **K**: writes state rather than force.
//!
//! An E or Em term exposes its energy, so a later optimizer can minimize it.
//! Contacts, damping and pins live in rapier's step rather than the force
//! list; they are declared here as [`CONTACTS`], [`DAMPING`] and [`PIN`].
//!
//! Plan: `design_docs/mere_docs/implementation_strategy/2026-10-02_dynamics_grammar_plan.md`, G1.

use std::collections::HashMap;

use rapier2d::prelude::*;

use crate::{Force, NodeKey};

/// One term a force applies.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Term {
    /// The term's name within its force ("repulsion", "edge spring").
    pub name: &'static str,
    pub topology: Topology,
    pub kernel: Kernel,
    pub state: State,
    pub currency: Currency,
    pub class: Class,
    /// The mass metric an Em term descends in (and an H term is conservative in).
    pub metric: Option<Metric>,
    pub observable: Observable,
    /// Written once rather than every tick: an initial condition (Orbit's
    /// kick), which does not decide how the law's motion enters the step.
    pub initial: bool,
}

impl Term {
    /// A force-currency term over positions.
    pub const fn force(
        name: &'static str,
        topology: Topology,
        kernel: Kernel,
        class: Class,
        observable: Observable,
    ) -> Self {
        Self {
            name,
            topology,
            kernel,
            state: State::Position,
            currency: Currency::Force,
            class,
            metric: None,
            observable,
            initial: false,
        }
    }

    /// The same term written once, as an initial condition.
    pub const fn once(mut self) -> Self {
        self.initial = true;
        self
    }

    /// The same term weighted by a metric channel.
    pub const fn in_metric(mut self, metric: Metric) -> Self {
        self.metric = Some(metric);
        self
    }

    /// The same term moving other state, in another currency.
    pub const fn moving(mut self, state: State, currency: Currency) -> Self {
        self.state = state;
        self.currency = currency;
        self
    }

    /// Whether an energy is part of the declaration (E and Em).
    pub fn has_energy(&self) -> bool {
        matches!(self.class, Class::E | Class::Em)
    }
}

/// Who interacts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Topology {
    /// Each body alone, against a target or a field.
    Unary,
    /// Along the synced edges.
    Edges,
    /// Given pairs with weights: affinity pairs, a distance table.
    PairList,
    /// Every pair, optionally within a cutoff radius.
    AllPairs { cutoff: Option<f32> },
    /// Members against their group's centroid.
    Groups,
    /// Pairs partitioned by kind, each cell its own coefficient.
    KindMatrix { cutoff: Option<f32> },
    /// A grid field the bodies read and write.
    Medium,
}

impl Topology {
    /// Whether the term acts between bodies, so its forces must balance:
    /// unary terms and media are external.
    pub fn is_internal(self) -> bool {
        !matches!(self, Topology::Unary | Topology::Medium)
    }
}

/// The kernel family: the functional form of the interaction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Kernel {
    /// A central push falling off as `dᵉ`: inverse-square `−2`, charge `−1`.
    Repulsion { exponent: f32 },
    /// A central pull growing as `dᵉ`: ForceAtlas2's `1`, LinLog's `0`,
    /// hub pull's `−1`. The (a, r) notation of Jacomy et al.
    Attraction { exponent: f32 },
    /// Hooke's law toward a rest length, `k(d − L)`.
    Spring,
    /// A pull in proportion to the offset from a target, `s(p − x)`.
    Harmonic,
    /// Plummer-softened gravitation, `G·m·Δ/(d² + ε²)^{3/2}`.
    Plummer,
    /// Particle life's response: a repulsive core, then a tent scaled by the
    /// kind matrix.
    Tent,
    /// A torque turning an edge toward a field direction.
    Needle,
    /// Boids' steering toward mates' heading or centre.
    Steering,
    /// Kuramoto's phase coupling, `(K/deg)·Σ sin(θⱼ − θᵢ)`.
    PhaseCoupling,
    /// A drive along the velocity: a cruise, or damping cancelled.
    Drive,
    /// A response to a numen field.
    Field,
    /// A swirl about a centre with an inward pull.
    Vortex,
    /// Diffusion of a density field, `v = −D∇ρ/ρ`.
    Diffusion,
    /// Non-penetration between bodies.
    Contact,
    /// Linear damping, `−γ·v`.
    Damping,
    /// A velocity written: zeroed, set from a field, scaled, or kicked once.
    VelocityWrite,
    /// A position written: pinned, advected, or proposed and accepted.
    PositionWrite,
}

/// The state a term moves.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum State {
    Position,
    Velocity,
    /// An oscillator phase held inside the force.
    Phase,
    /// A grid field.
    Field,
}

/// How a term's effect enters the step (the currencies Mark ruled
/// 2026-10-02, "Laws declare currency; catalog adapts or refuses").
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Currency {
    /// Adds force; rapier integrates it.
    Force,
    /// Writes velocity or position.
    Kinematic,
    /// Integrates on the GPU.
    Resident,
    /// Realized inside rapier's step, outside the force list (contacts,
    /// damping). *Reading, not ruled*: the brief's "inside the integrator".
    Integrator,
}

/// The term's class (brief §3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Class {
    E,
    Em,
    H,
    N,
    K,
}

/// The metric channel a metric-weighted term is a gradient in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Metric {
    /// The mass source: `ln(deg + 1)`, PageRank, or Orbit's masses.
    Mass,
    /// Node degree: Kuramoto's `1/deg`, Boids' mate averaging.
    Degree,
    /// Density's transport metric (brief §6.1), for its declaration when the
    /// Density law lands.
    Wasserstein,
}

/// The measure that shows a term working (a receipt's signature).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Observable {
    /// Node pairs closer than a body's diameter.
    Overlaps,
    /// RMS distance from the centroid.
    Spread,
    /// The graph-farthest pair's distance over the mean edge.
    Stretch,
    /// Kinetic energy over time: a floor for a living term, a ceiling for a
    /// settling one.
    Energy,
    /// The term's pair lengths against their rest length.
    PairLength,
    /// Distance between groups over the spread within them.
    Separation,
    /// Oscillator phases gathered into clusters.
    PhaseClusters,
    /// Directed edges along a field.
    FieldAlignment,
    /// Offset from the term's targets.
    Residual,
    /// Mass against area, by rank (Density).
    MassAreaRank,
}

/// A layout to read an energy or a metric from: every node body in node-key
/// order, as the laws read them, and the synced edges.
#[derive(Clone, Copy, Debug)]
pub struct Layout<'a> {
    pub nodes: &'a [(NodeKey, Vector)],
    pub edges: &'a [(NodeKey, NodeKey)],
}

impl Layout<'_> {
    /// Node key to its index in [`Self::nodes`].
    pub fn index(&self) -> HashMap<NodeKey, usize> {
        self.nodes
            .iter()
            .enumerate()
            .map(|(i, (key, _))| (*key, i))
            .collect()
    }

    /// The distance between nodes `i` and `j`, in double precision.
    pub fn distance(&self, i: usize, j: usize) -> f64 {
        let (a, b) = (self.nodes[i].1, self.nodes[j].1);
        (f64::from(a.x) - f64::from(b.x)).hypot(f64::from(a.y) - f64::from(b.y))
    }

    /// Node `i`'s position, in double precision.
    pub fn at(&self, i: usize) -> (f64, f64) {
        let p = self.nodes[i].1;
        (f64::from(p.x), f64::from(p.y))
    }

    /// Each edge with both endpoints present and distinct, as index pairs.
    pub fn edge_indices(&self) -> Vec<(usize, usize)> {
        let index = self.index();
        self.edges
            .iter()
            .filter_map(|(a, b)| Some((*index.get(a)?, *index.get(b)?)))
            .filter(|(i, j)| i != j)
            .collect()
    }
}

/// What a force is, term by term. Every [`Force`] declares the terms it
/// applies, in order; a force that bundles several can isolate one, an E or
/// Em term exposes its energy, and a metric-weighted term its weights.
pub trait Declared {
    /// The terms this force applies.
    fn terms(&self) -> Vec<Term>;

    /// Term `term` alone, for a force that applies several; `None` for a
    /// force with one term, which is itself.
    fn isolate(&self, _term: usize) -> Option<Box<dyn Force>> {
        None
    }

    /// Term `term`'s energy at `layout`, for an E or Em term: continuous, and
    /// zero beyond a cutoff. The term's forces are `−M⁻¹∇U`.
    fn energy(&self, _term: usize, _layout: &Layout<'_>) -> Option<f64> {
        None
    }

    /// Term `term`'s metric weights at `layout`, one per node in layout
    /// order, for a term declared with a metric.
    fn metric(&self, _term: usize, _layout: &Layout<'_>) -> Option<Vec<f64>> {
        None
    }

    /// Whether term `term` has a resident kernel or the lagged upload, so a
    /// resident law can take it ([`crate::compose`]).
    fn resident(&self, _term: usize) -> bool {
        false
    }

    /// Term `term` on the common scale ([`crate::scale`], F5): where its
    /// weight is read and the weight, its force there. `None` for a term
    /// whose kernel F5 gives no reference.
    fn scale(&self, _term: usize) -> Option<crate::scale::Scale> {
        None
    }

    /// This force with term `term` at `weight` on the common scale, every
    /// other parameter kept.
    fn reweighted(&self, _term: usize, _weight: f64) -> Option<Box<dyn Force>> {
        None
    }
}

/// Rapier's contact solver between node bodies.
pub const CONTACTS: Term = Term::force(
    "contacts",
    Topology::AllPairs {
        cutoff: Some(2.0 * crate::NODE_BODY_RADIUS),
    },
    Kernel::Contact,
    Class::K,
    Observable::Overlaps,
)
.moving(State::Velocity, Currency::Integrator);

/// Rapier's linear damping on every node body.
pub const DAMPING: Term = Term::force(
    "damping",
    Topology::Unary,
    Kernel::Damping,
    Class::K,
    Observable::Energy,
)
.moving(State::Velocity, Currency::Integrator);

/// A pinned body ([`Simulation::pin`](crate::Simulation::pin)): kinematic,
/// held at its target.
pub const PIN: Term = Term::force(
    "pin",
    Topology::Unary,
    Kernel::PositionWrite,
    Class::K,
    Observable::Residual,
)
.moving(State::Position, Currency::Kinematic);

/// `s/d` past a floor `m`, joined smoothly below it to match a push that
/// grows linearly inside the floor: the energy of a floored inverse-square
/// repulsion.
pub(crate) fn floored_inverse(s: f64, d: f64, m: f64) -> f64 {
    if d >= m {
        s / d
    } else {
        s / m + s * (m * m - d * d) / (2.0 * m * m * m)
    }
}

/// `ln d` past a floor `m`, joined smoothly below it: the energy shape of a
/// floored `1/d` force.
pub(crate) fn floored_log(d: f64, m: f64) -> f64 {
    if d >= m {
        d.ln()
    } else {
        m.ln() - 0.5 + d * d / (2.0 * m * m)
    }
}

/// Hooke's energy, `(k/2)(d − L)²`.
pub(crate) fn spring(k: f64, d: f64, rest: f64) -> f64 {
    0.5 * k * (d - rest) * (d - rest)
}

/// `Σ (s/2)|x − target|²` over the nodes `target` gives a target for.
pub(crate) fn harmonic(
    s: f64,
    layout: &Layout<'_>,
    target: impl Fn(usize) -> Option<(f64, f64)>,
) -> f64 {
    (0..layout.nodes.len())
        .filter_map(|i| {
            let (tx, ty) = target(i)?;
            let (x, y) = layout.at(i);
            Some(0.5 * s * ((x - tx).powi(2) + (y - ty).powi(2)))
        })
        .sum()
}
