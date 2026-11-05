// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! Raw seiche terms: one kind per force type (F107), with its parameters
//! (F106).
//!
//! Each of the 20 force types a catalog law or overlay builds has a
//! parameter struct here. An omitted parameter takes the force's constructor
//! default, so a parameter struct's `Default` is read off the constructed
//! force rather than restated. A parameter the constructor takes with no
//! default (Stress's unit length, Orbit's counter-damping, Density's
//! resolution, a locus's target) is an `Option` that [`RawTerm::build`]
//! refuses when absent. Graph-derived inputs are named slots holding opaque
//! channel ids, resolved host-side in G4b; the required ones (masses, kinds,
//! groups, depths, pairs) are refused when absent, never filled in.
//!
//! [`RawTerm::build`] builds the force over empty inputs, which is enough to
//! read its declarations, the derived fields. Particle life's seeded matrix
//! is drawn over two kinds there: its kind count comes from the kinds
//! channel, unresolved here, and a seeded matrix over two or more kinds is
//! asymmetric, so it derives as declared (N). *Reading, not ruled.*
//!
//! Plan: `design_docs/archive_docs/2026-10-28/2026-10-02_dynamics_grammar_plan.md`, G4a.

use crate::{
    Anneal, BarnesHutConfig, BarnesHutRepulsion, Boids, Boundary, CounterDamping, DegreeRepulsion,
    Density, DensityStop, DepthGravity, DomainCluster, EdgeSpring, Force, Gravity, GravityLocus,
    GridSnap, Hold, HubGravity, Kuramoto, LinLogForce, MagneticSpring, NodeExclusion, ParticleLife,
    StressSpring,
};

use super::SpecError;

fn wide(x: f32) -> f64 {
    f64::from(x)
}

fn narrow(x: f64) -> f32 {
    x as f32
}

fn wide2((x, y): (f32, f32)) -> (f64, f64) {
    (wide(x), wide(y))
}

fn narrow2((x, y): (f64, f64)) -> (f32, f32) {
    (narrow(x), narrow(y))
}

/// A parameter struct: every field defaulted from the constructed force,
/// unknown fields refused (F98).
macro_rules! params {
    ($(#[$meta:meta])* pub struct $name:ident { $($body:tt)* }) => {
        $(#[$meta])*
        #[derive(Clone, Debug, PartialEq)]
        #[cfg_attr(
            feature = "serde",
            derive(serde::Serialize, serde::Deserialize),
            serde(default, deny_unknown_fields)
        )]
        pub struct $name { $($body)* }
    };
}

params! {
    /// [`NodeExclusion`]: the inverse-square push with a cutoff.
    pub struct NodeExclusionParams {
        pub strength: f64,
        pub cutoff: f64,
        pub min_distance: f64,
    }
}

params! {
    /// [`EdgeSpring`]: Hooke's law along the edges.
    pub struct EdgeSpringParams {
        pub stiffness: f64,
        pub rest_length: f64,
    }
}

params! {
    /// [`Boundary`]: the gentle centring pull.
    pub struct BoundaryParams {
        pub strength: f64,
    }
}

impl BoundaryParams {
    pub fn of(f: &Boundary) -> Self {
        Self {
            strength: wide(f.strength),
        }
    }

    fn build(&self) -> Boundary {
        Boundary {
            strength: narrow(self.strength),
        }
    }
}

impl Default for BoundaryParams {
    fn default() -> Self {
        Self::of(&Boundary::default())
    }
}

params! {
    /// [`BarnesHutRepulsion`]: Charge's `1/d` push over a quadtree.
    pub struct BarnesHutRepulsionParams {
        pub strength: f64,
        pub min_distance: f64,
        pub theta: f64,
        pub min_cell_size: f64,
    }
}

params! {
    /// [`StressSpring`]: springs toward given pair lengths (Stress's graph
    /// distances, Skeleton's spanning tree).
    pub struct StressSpringParams {
        /// The pairs and their lengths, in units of `unit_length`.
        #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
        pub pairs: Option<String>,
        /// The length one unit of pair distance stands for. Required.
        #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
        pub unit_length: Option<f64>,
        pub stiffness: f64,
    }
}

params! {
    /// [`LinLogForce`]: ForceAtlas2's model, LinLog proper at exponent 0.
    pub struct LinlogParams {
        pub attraction: f64,
        pub attraction_exponent: f64,
        pub repulsion: f64,
        pub degree_weighted: bool,
        pub min_distance: f64,
        pub gravity: f64,
    }
}

/// [`CounterDamping`] as a spec writes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(rename_all = "kebab-case")
)]
pub enum CounterDampingParam {
    Off,
    Full,
    Tangential,
}

impl CounterDampingParam {
    pub fn of(c: CounterDamping) -> Self {
        match c {
            CounterDamping::Off => Self::Off,
            CounterDamping::Full => Self::Full,
            CounterDamping::Tangential => Self::Tangential,
        }
    }

    fn build(self) -> CounterDamping {
        match self {
            Self::Off => CounterDamping::Off,
            Self::Full => CounterDamping::Full,
            Self::Tangential => CounterDamping::Tangential,
        }
    }
}

params! {
    /// [`Gravity`]: Orbit's Plummer gravitation and its drives.
    pub struct GravityParams {
        /// The gravitational masses. Required.
        #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
        pub masses: Option<String>,
        /// Which damping the law cancels. Required.
        #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
        pub counter_damping: Option<CounterDampingParam>,
        pub strength: f64,
        pub softening: f64,
        pub orbital_kick: f64,
        pub radial_floor: f64,
    }
}

params! {
    /// [`ParticleLife`]: Kinds' kind matrix. The matrix is drawn from the
    /// spec's seed (F101) unless one is given, row-major and square.
    pub struct ParticleLifeParams {
        /// Each node's kind. Required.
        #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
        pub kinds: Option<String>,
        /// An explicit rule matrix, `k × k` row-major, in place of the seeded one.
        #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
        pub matrix: Option<Vec<f64>>,
        pub radius: f64,
        pub core: f64,
        pub strength: f64,
        pub gravity: f64,
    }
}

/// The kind count a seeded matrix is drawn over when derived (module docs).
const SEEDED_KINDS: usize = 2;

params! {
    /// [`Boids`]: Flock's steering.
    pub struct BoidsParams {
        pub separation: f64,
        pub separation_radius: f64,
        pub alignment: f64,
        pub cohesion: f64,
        pub cruise_speed: f64,
        pub cruise: f64,
        pub gravity: f64,
    }
}

params! {
    /// [`Kuramoto`]: Sync's phase coupling on a ring.
    pub struct KuramotoParams {
        /// Each node's ring radius; `default_radius` for a node it omits.
        #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
        pub radii: Option<String>,
        pub natural_frequency: f64,
        pub coupling: f64,
        pub default_radius: f64,
        pub centre: (f64, f64),
        pub stiffness: f64,
    }
}

params! {
    /// [`MagneticSpring`]: Flow's springs and needle.
    pub struct MagneticSpringParams {
        pub field: (f64, f64),
        pub stiffness: f64,
        pub rest_length: f64,
        pub torque: f64,
        pub repulsion: f64,
        pub min_distance: f64,
    }
}

params! {
    /// [`Anneal`]: Davidson–Harel's walk, seeded by the spec (F101).
    pub struct AnnealParams {
        pub initial_temperature: f64,
        pub cooling: f64,
        pub floor: f64,
        pub step: f64,
        pub ideal_edge: f64,
        pub repulsion: f64,
        pub gravity: f64,
    }
}

params! {
    /// [`Hold`]: Still's zeroed velocities. No parameters.
    #[derive(Default)]
    pub struct HoldParams {}
}

/// [`DensityStop`] as a spec writes it.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(rename_all = "kebab-case")
)]
pub enum DensityStopParam {
    Shift(f64),
    FieldCv(f64),
    Cap,
}

impl DensityStopParam {
    pub fn of(s: DensityStop) -> Self {
        match s {
            DensityStop::Shift(x) => Self::Shift(wide(x)),
            DensityStop::FieldCv(x) => Self::FieldCv(wide(x)),
            DensityStop::Cap => Self::Cap,
        }
    }

    fn build(self) -> DensityStop {
        match self {
            Self::Shift(x) => DensityStop::Shift(narrow(x)),
            Self::FieldCv(x) => DensityStop::FieldCv(narrow(x)),
            Self::Cap => DensityStop::Cap,
        }
    }
}

params! {
    /// [`Density`]: the diffusion cartogram.
    pub struct DensityParams {
        /// The masses areas follow. Required.
        #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
        pub masses: Option<String>,
        /// The grid's cells per side. Required.
        #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
        pub resolution: Option<u32>,
        pub area_per_mass: f64,
        pub centre: (f64, f64),
        pub initial_blur: f64,
        pub seconds: f64,
        pub cfl: f64,
        pub max_substeps: u32,
        pub background: f64,
        pub stop: DensityStopParam,
        pub patience: u32,
        pub max_passes: u32,
        pub min_passes: u32,
        #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
        pub renew: Option<f64>,
        pub converts: bool,
    }
}

params! {
    /// [`DegreeRepulsion`]: Hub room.
    pub struct DegreeRepulsionParams {
        /// The weights; `ln(degree + 1)` from the edges when omitted.
        #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
        pub masses: Option<String>,
        pub strength: f64,
        pub radius: f64,
        pub min_distance: f64,
    }
}

params! {
    /// [`DomainCluster`]: Group pull.
    pub struct DomainClusterParams {
        /// Each node's group. Required.
        #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
        pub groups: Option<String>,
        pub strength: f64,
    }
}

params! {
    /// [`HubGravity`]: Hub pull.
    pub struct HubGravityParams {
        /// The weights; `ln(degree + 1)` from the edges when omitted.
        #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
        pub masses: Option<String>,
        pub strength: f64,
        pub min_distance: f64,
    }
}

params! {
    /// [`DepthGravity`]: Depth's pull toward each node's layer.
    pub struct DepthGravityParams {
        /// Each node's depth. Required.
        #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
        pub depths: Option<String>,
        pub direction: (f64, f64),
        pub spacing: f64,
        pub strength: f64,
    }
}

params! {
    /// [`GridSnap`]: Grid.
    pub struct GridSnapParams {
        pub cell: f64,
        pub strength: f64,
    }
}

params! {
    /// [`GravityLocus`]: Centre, or Tide with an oscillation.
    pub struct GravityLocusParams {
        /// The point pulled toward. Required.
        #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
        pub target: Option<(f64, f64)>,
        pub strength: f64,
        /// `(amplitude, period in seconds)`: a moving target, the tide.
        #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
        pub oscillation: Option<(f64, f64)>,
    }
}

/// A raw seiche term: one kind per force type (F107), serialized by its
/// kind id with its parameters.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(rename_all = "kebab-case")
)]
pub enum RawTerm {
    NodeExclusion(NodeExclusionParams),
    EdgeSpring(EdgeSpringParams),
    Boundary(BoundaryParams),
    BarnesHutRepulsion(BarnesHutRepulsionParams),
    StressSpring(StressSpringParams),
    Linlog(LinlogParams),
    Gravity(GravityParams),
    ParticleLife(ParticleLifeParams),
    Boids(BoidsParams),
    Kuramoto(KuramotoParams),
    MagneticSpring(MagneticSpringParams),
    Anneal(AnnealParams),
    Hold(HoldParams),
    Density(DensityParams),
    DegreeRepulsion(DegreeRepulsionParams),
    DomainCluster(DomainClusterParams),
    HubGravity(HubGravityParams),
    DepthGravity(DepthGravityParams),
    GridSnap(GridSnapParams),
    GravityLocus(GravityLocusParams),
}

/// One graph-derived input of a raw term: its slot, the channel id the spec
/// names (opaque until G4b resolves it), and whether the term needs it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InputSlot<'a> {
    pub slot: &'static str,
    pub channel: Option<&'a str>,
    pub required: bool,
}

impl RawTerm {
    /// Every raw kind id, in declaration order: the 20 force types.
    pub const KINDS: [&'static str; 20] = [
        "node-exclusion",
        "edge-spring",
        "boundary",
        "barnes-hut-repulsion",
        "stress-spring",
        "linlog",
        "gravity",
        "particle-life",
        "boids",
        "kuramoto",
        "magnetic-spring",
        "anneal",
        "hold",
        "density",
        "degree-repulsion",
        "domain-cluster",
        "hub-gravity",
        "depth-gravity",
        "grid-snap",
        "gravity-locus",
    ];

    /// Each kind with its parameters defaulted, in [`Self::KINDS`] order.
    pub fn defaults() -> [RawTerm; 20] {
        [
            RawTerm::NodeExclusion(Default::default()),
            RawTerm::EdgeSpring(Default::default()),
            RawTerm::Boundary(Default::default()),
            RawTerm::BarnesHutRepulsion(Default::default()),
            RawTerm::StressSpring(Default::default()),
            RawTerm::Linlog(Default::default()),
            RawTerm::Gravity(Default::default()),
            RawTerm::ParticleLife(Default::default()),
            RawTerm::Boids(Default::default()),
            RawTerm::Kuramoto(Default::default()),
            RawTerm::MagneticSpring(Default::default()),
            RawTerm::Anneal(Default::default()),
            RawTerm::Hold(Default::default()),
            RawTerm::Density(Default::default()),
            RawTerm::DegreeRepulsion(Default::default()),
            RawTerm::DomainCluster(Default::default()),
            RawTerm::HubGravity(Default::default()),
            RawTerm::DepthGravity(Default::default()),
            RawTerm::GridSnap(Default::default()),
            RawTerm::GravityLocus(Default::default()),
        ]
    }

    /// The kind id, as the spec writes it.
    pub fn kind(&self) -> &'static str {
        let index = match self {
            RawTerm::NodeExclusion(_) => 0,
            RawTerm::EdgeSpring(_) => 1,
            RawTerm::Boundary(_) => 2,
            RawTerm::BarnesHutRepulsion(_) => 3,
            RawTerm::StressSpring(_) => 4,
            RawTerm::Linlog(_) => 5,
            RawTerm::Gravity(_) => 6,
            RawTerm::ParticleLife(_) => 7,
            RawTerm::Boids(_) => 8,
            RawTerm::Kuramoto(_) => 9,
            RawTerm::MagneticSpring(_) => 10,
            RawTerm::Anneal(_) => 11,
            RawTerm::Hold(_) => 12,
            RawTerm::Density(_) => 13,
            RawTerm::DegreeRepulsion(_) => 14,
            RawTerm::DomainCluster(_) => 15,
            RawTerm::HubGravity(_) => 16,
            RawTerm::DepthGravity(_) => 17,
            RawTerm::GridSnap(_) => 18,
            RawTerm::GravityLocus(_) => 19,
        };
        Self::KINDS[index]
    }

    /// The term's graph-derived inputs, which G4b resolves host-side.
    pub fn inputs(&self) -> Vec<InputSlot<'_>> {
        fn slot<'a>(
            slot: &'static str,
            channel: &'a Option<String>,
            required: bool,
        ) -> InputSlot<'a> {
            InputSlot {
                slot,
                channel: channel.as_deref(),
                required,
            }
        }
        match self {
            RawTerm::StressSpring(p) => vec![slot("pairs", &p.pairs, true)],
            RawTerm::Gravity(p) => vec![slot("masses", &p.masses, true)],
            RawTerm::ParticleLife(p) => vec![slot("kinds", &p.kinds, true)],
            RawTerm::Kuramoto(p) => vec![slot("radii", &p.radii, false)],
            RawTerm::Density(p) => vec![slot("masses", &p.masses, true)],
            RawTerm::DegreeRepulsion(p) => vec![slot("masses", &p.masses, false)],
            RawTerm::DomainCluster(p) => vec![slot("groups", &p.groups, true)],
            RawTerm::HubGravity(p) => vec![slot("masses", &p.masses, false)],
            RawTerm::DepthGravity(p) => vec![slot("depths", &p.depths, true)],
            _ => Vec::new(),
        }
    }

    /// The force over empty inputs, enough to read its declarations; a
    /// required parameter or input left out is refused, named at `path`.
    pub fn build(&self, path: &str, seed: u64) -> Result<Box<dyn Force>, SpecError> {
        Ok(match self {
            RawTerm::NodeExclusion(p) => Box::new(p.build()),
            RawTerm::EdgeSpring(p) => Box::new(p.build()),
            RawTerm::Boundary(p) => Box::new(p.build()),
            RawTerm::BarnesHutRepulsion(p) => Box::new(p.build()),
            RawTerm::StressSpring(p) => Box::new(p.build(path)?),
            RawTerm::Linlog(p) => Box::new(p.build()),
            RawTerm::Gravity(p) => Box::new(p.build(path)?),
            RawTerm::ParticleLife(p) => Box::new(p.build(path, seed)?),
            RawTerm::Boids(p) => Box::new(p.build()),
            RawTerm::Kuramoto(p) => Box::new(p.build()),
            RawTerm::MagneticSpring(p) => Box::new(p.build()),
            RawTerm::Anneal(p) => Box::new(p.build(seed)),
            RawTerm::Hold(_) => Box::new(Hold),
            RawTerm::Density(p) => Box::new(p.build(path)?),
            RawTerm::DegreeRepulsion(p) => Box::new(p.build()),
            RawTerm::DomainCluster(p) => Box::new(p.build(path)?),
            RawTerm::HubGravity(p) => Box::new(p.build()),
            RawTerm::DepthGravity(p) => Box::new(p.build(path)?),
            RawTerm::GridSnap(p) => Box::new(p.build()),
            RawTerm::GravityLocus(p) => Box::new(p.build(path)?),
        })
    }
}

/// A required value, or its absence refused at `path`.
fn need<T>(value: Option<T>, path: &str, field: &'static str) -> Result<T, SpecError> {
    value.ok_or_else(|| SpecError::Missing {
        path: path.to_string(),
        field,
    })
}

#[path = "raw/pair_laws.rs"]
mod pair_laws;

#[path = "raw/field_laws.rs"]
mod field_laws;

#[path = "raw/placement_laws.rs"]
mod placement_laws;
