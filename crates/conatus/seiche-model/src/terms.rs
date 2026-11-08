// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! Term declarations as data: what each force is, in the dynamics grammar's words.
//!
//! A [`Term`] has a topology (who interacts), kernel family, the state it moves,
//! its currency, its class and, for a metric-weighted term, its metric channel,
//! plus the observable that shows it working. The `Declared` trait that forces
//! implement, and the layouts it reads, stay in `seiche`; these are the plain
//! values a host can name without the engine.
//!
//! Plan: `design_docs/archive_docs/2026-10-28/2026-10-02_dynamics_grammar_plan.md`, G1.

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
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(rename_all = "kebab-case")
)]
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

/// A pinned body (`seiche::Simulation::pin`): kinematic,
/// held at its target.
pub const PIN: Term = Term::force(
    "pin",
    Topology::Unary,
    Kernel::PositionWrite,
    Class::K,
    Observable::Residual,
)
.moving(State::Position, Currency::Kinematic);
