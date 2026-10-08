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
//! Plan: `design_docs/mere_docs/implementation_strategy/2026-10-02_dynamics_grammar_plan.md`, G4a.

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

impl NodeExclusionParams {
    pub fn of(f: &NodeExclusion) -> Self {
        Self {
            strength: wide(f.strength),
            cutoff: wide(f.cutoff),
            min_distance: wide(f.min_distance),
        }
    }

    fn build(&self) -> NodeExclusion {
        NodeExclusion {
            strength: narrow(self.strength),
            cutoff: narrow(self.cutoff),
            min_distance: narrow(self.min_distance),
        }
    }
}

impl Default for NodeExclusionParams {
    fn default() -> Self {
        Self::of(&NodeExclusion::default())
    }
}

params! {
    /// [`EdgeSpring`]: Hooke's law along the edges.
    pub struct EdgeSpringParams {
        pub stiffness: f64,
        pub rest_length: f64,
    }
}

impl EdgeSpringParams {
    pub fn of(f: &EdgeSpring) -> Self {
        Self {
            stiffness: wide(f.stiffness),
            rest_length: wide(f.rest_length),
        }
    }

    fn build(&self) -> EdgeSpring {
        EdgeSpring {
            stiffness: narrow(self.stiffness),
            rest_length: narrow(self.rest_length),
        }
    }
}

impl Default for EdgeSpringParams {
    fn default() -> Self {
        Self::of(&EdgeSpring::default())
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

impl BarnesHutRepulsionParams {
    pub fn of(f: &BarnesHutRepulsion) -> Self {
        Self {
            strength: wide(f.strength),
            min_distance: wide(f.min_distance),
            theta: wide(f.config.theta),
            min_cell_size: wide(f.config.min_cell_size),
        }
    }

    fn build(&self) -> BarnesHutRepulsion {
        BarnesHutRepulsion {
            strength: narrow(self.strength),
            min_distance: narrow(self.min_distance),
            config: BarnesHutConfig {
                theta: narrow(self.theta),
                min_cell_size: narrow(self.min_cell_size),
            },
        }
    }
}

impl Default for BarnesHutRepulsionParams {
    fn default() -> Self {
        Self::of(&BarnesHutRepulsion::default())
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

impl StressSpringParams {
    pub fn of(f: &StressSpring) -> Self {
        Self {
            pairs: None,
            unit_length: Some(wide(f.unit_length)),
            stiffness: wide(f.stiffness),
        }
    }

    fn build(&self, path: &str) -> Result<StressSpring, SpecError> {
        let unit_length = need(self.unit_length, path, "unit_length")?;
        need(self.pairs.as_ref(), path, "pairs")?;
        Ok(
            StressSpring::from_weighted_distances(std::iter::empty(), narrow(unit_length))
                .with_stiffness(narrow(self.stiffness)),
        )
    }
}

impl Default for StressSpringParams {
    fn default() -> Self {
        let mut params = Self::of(&StressSpring::from_weighted_distances(
            std::iter::empty(),
            0.0,
        ));
        params.unit_length = None;
        params
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

impl LinlogParams {
    pub fn of(f: &LinLogForce) -> Self {
        Self {
            attraction: wide(f.attraction),
            attraction_exponent: wide(f.attraction_exponent),
            repulsion: wide(f.repulsion),
            degree_weighted: f.degree_weighted,
            min_distance: wide(f.min_distance),
            gravity: wide(f.gravity),
        }
    }

    fn build(&self) -> LinLogForce {
        LinLogForce {
            attraction: narrow(self.attraction),
            attraction_exponent: narrow(self.attraction_exponent),
            repulsion: narrow(self.repulsion),
            degree_weighted: self.degree_weighted,
            min_distance: narrow(self.min_distance),
            gravity: narrow(self.gravity),
        }
    }
}

impl Default for LinlogParams {
    fn default() -> Self {
        Self::of(&LinLogForce::default())
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

impl GravityParams {
    pub fn of(f: &Gravity) -> Self {
        Self {
            masses: None,
            counter_damping: Some(CounterDampingParam::of(f.counter_damping)),
            strength: wide(f.strength),
            softening: wide(f.softening),
            orbital_kick: wide(f.orbital_kick),
            radial_floor: wide(f.radial_floor),
        }
    }

    fn build(&self, path: &str) -> Result<Gravity, SpecError> {
        need(self.masses.as_ref(), path, "masses")?;
        let counter = need(self.counter_damping, path, "counter_damping")?;
        let mut f = Gravity::new(std::iter::empty(), counter.build());
        f.strength = narrow(self.strength);
        f.softening = narrow(self.softening);
        f.orbital_kick = narrow(self.orbital_kick);
        f.radial_floor = narrow(self.radial_floor);
        Ok(f)
    }
}

impl Default for GravityParams {
    fn default() -> Self {
        let mut params = Self::of(&Gravity::new(std::iter::empty(), CounterDamping::Off));
        params.counter_damping = None;
        params
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

impl ParticleLifeParams {
    pub fn of(f: &ParticleLife) -> Self {
        Self {
            kinds: None,
            matrix: None,
            radius: wide(f.radius),
            core: wide(f.core),
            strength: wide(f.strength),
            gravity: wide(f.gravity),
        }
    }

    fn build(&self, path: &str, seed: u64) -> Result<ParticleLife, SpecError> {
        need(self.kinds.as_ref(), path, "kinds")?;
        let mut f = match &self.matrix {
            None => ParticleLife::seeded(std::iter::empty(), SEEDED_KINDS, seed),
            Some(matrix) => {
                let k = (matrix.len() as f64).sqrt().round() as usize;
                if k == 0 || k * k != matrix.len() || k > usize::from(u8::MAX) + 1 {
                    return Err(SpecError::Invalid {
                        path: path.to_string(),
                        reason: format!(
                            "a kind matrix is square, k × k for 1 to 256 kinds; this one has {} entries",
                            matrix.len()
                        ),
                    });
                }
                ParticleLife::new(
                    std::iter::empty(),
                    k,
                    matrix.iter().map(|x| narrow(*x)).collect(),
                )
            },
        };
        f.radius = narrow(self.radius);
        f.core = narrow(self.core);
        f.strength = narrow(self.strength);
        f.gravity = narrow(self.gravity);
        Ok(f)
    }
}

impl Default for ParticleLifeParams {
    fn default() -> Self {
        Self::of(&ParticleLife::new(std::iter::empty(), 1, Vec::new()))
    }
}

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

impl BoidsParams {
    pub fn of(f: &Boids) -> Self {
        Self {
            separation: wide(f.separation),
            separation_radius: wide(f.separation_radius),
            alignment: wide(f.alignment),
            cohesion: wide(f.cohesion),
            cruise_speed: wide(f.cruise_speed),
            cruise: wide(f.cruise),
            gravity: wide(f.gravity),
        }
    }

    fn build(&self) -> Boids {
        Boids {
            separation: narrow(self.separation),
            separation_radius: narrow(self.separation_radius),
            alignment: narrow(self.alignment),
            cohesion: narrow(self.cohesion),
            cruise_speed: narrow(self.cruise_speed),
            cruise: narrow(self.cruise),
            gravity: narrow(self.gravity),
        }
    }
}

impl Default for BoidsParams {
    fn default() -> Self {
        Self::of(&Boids::default())
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

impl KuramotoParams {
    pub fn of(f: &Kuramoto) -> Self {
        Self {
            radii: None,
            natural_frequency: wide(f.natural_frequency),
            coupling: wide(f.coupling),
            default_radius: wide(f.default_radius),
            centre: wide2(f.centre),
            stiffness: wide(f.stiffness),
        }
    }

    fn build(&self) -> Kuramoto {
        let mut f = Kuramoto::new(std::iter::empty());
        f.natural_frequency = narrow(self.natural_frequency);
        f.coupling = narrow(self.coupling);
        f.default_radius = narrow(self.default_radius);
        f.centre = narrow2(self.centre);
        f.stiffness = narrow(self.stiffness);
        f
    }
}

impl Default for KuramotoParams {
    fn default() -> Self {
        Self::of(&Kuramoto::new(std::iter::empty()))
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

impl MagneticSpringParams {
    pub fn of(f: &MagneticSpring) -> Self {
        Self {
            field: wide2(f.field),
            stiffness: wide(f.stiffness),
            rest_length: wide(f.rest_length),
            torque: wide(f.torque),
            repulsion: wide(f.repulsion),
            min_distance: wide(f.min_distance),
        }
    }

    fn build(&self) -> MagneticSpring {
        MagneticSpring {
            field: narrow2(self.field),
            stiffness: narrow(self.stiffness),
            rest_length: narrow(self.rest_length),
            torque: narrow(self.torque),
            repulsion: narrow(self.repulsion),
            min_distance: narrow(self.min_distance),
        }
    }
}

impl Default for MagneticSpringParams {
    fn default() -> Self {
        Self::of(&MagneticSpring::default())
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

impl AnnealParams {
    pub fn of(f: &Anneal) -> Self {
        Self {
            initial_temperature: wide(f.initial_temperature),
            cooling: wide(f.cooling),
            floor: wide(f.floor),
            step: wide(f.step),
            ideal_edge: wide(f.ideal_edge),
            repulsion: wide(f.repulsion),
            gravity: wide(f.gravity),
        }
    }

    fn build(&self, seed: u64) -> Anneal {
        let mut f = Anneal::seeded(seed);
        f.initial_temperature = narrow(self.initial_temperature);
        f.cooling = narrow(self.cooling);
        f.floor = narrow(self.floor);
        f.step = narrow(self.step);
        f.ideal_edge = narrow(self.ideal_edge);
        f.repulsion = narrow(self.repulsion);
        f.gravity = narrow(self.gravity);
        f
    }
}

impl Default for AnnealParams {
    fn default() -> Self {
        Self::of(&Anneal::seeded(0))
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

impl DensityParams {
    /// The parameters `f` was built with, its resolution aside (the grid
    /// holds it, and a spec states it).
    pub fn of(f: &Density) -> Self {
        Self {
            masses: None,
            resolution: None,
            area_per_mass: wide(f.area_per_mass),
            centre: wide2(f.centre),
            initial_blur: wide(f.initial_blur),
            seconds: wide(f.seconds),
            cfl: wide(f.cfl),
            max_substeps: f.max_substeps,
            background: wide(f.background),
            stop: DensityStopParam::of(f.stop),
            patience: f.patience,
            max_passes: f.max_passes,
            min_passes: f.min_passes,
            renew: f.renew.map(wide),
            converts: f.converts,
        }
    }

    fn build(&self, path: &str) -> Result<Density, SpecError> {
        need(self.masses.as_ref(), path, "masses")?;
        let resolution = need(self.resolution, path, "resolution")?;
        let mut f = Density::new(std::iter::empty(), resolution as usize);
        f.area_per_mass = narrow(self.area_per_mass);
        f.centre = narrow2(self.centre);
        f.initial_blur = narrow(self.initial_blur);
        f.seconds = narrow(self.seconds);
        f.cfl = narrow(self.cfl);
        f.max_substeps = self.max_substeps;
        f.background = narrow(self.background);
        f.stop = self.stop.build();
        f.patience = self.patience;
        f.max_passes = self.max_passes;
        f.min_passes = self.min_passes;
        f.renew = self.renew.map(narrow);
        f.converts = self.converts;
        Ok(f)
    }
}

impl Default for DensityParams {
    fn default() -> Self {
        Self::of(&Density::new(std::iter::empty(), 1))
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

impl DegreeRepulsionParams {
    pub fn of(f: &DegreeRepulsion) -> Self {
        Self {
            masses: None,
            strength: wide(f.strength),
            radius: wide(f.radius),
            min_distance: wide(f.min_distance),
        }
    }

    fn build(&self) -> DegreeRepulsion {
        let mut f = DegreeRepulsion::default();
        f.strength = narrow(self.strength);
        f.radius = narrow(self.radius);
        f.min_distance = narrow(self.min_distance);
        f
    }
}

impl Default for DegreeRepulsionParams {
    fn default() -> Self {
        Self::of(&DegreeRepulsion::default())
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

impl DomainClusterParams {
    pub fn of(f: &DomainCluster) -> Self {
        Self {
            groups: None,
            strength: wide(f.strength),
        }
    }

    fn build(&self, path: &str) -> Result<DomainCluster, SpecError> {
        need(self.groups.as_ref(), path, "groups")?;
        let mut f = DomainCluster::new(std::iter::empty());
        f.strength = narrow(self.strength);
        Ok(f)
    }
}

impl Default for DomainClusterParams {
    fn default() -> Self {
        Self::of(&DomainCluster::new(std::iter::empty()))
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

impl HubGravityParams {
    pub fn of(f: &HubGravity) -> Self {
        Self {
            masses: None,
            strength: wide(f.strength),
            min_distance: wide(f.min_distance),
        }
    }

    fn build(&self) -> HubGravity {
        let mut f = HubGravity::default();
        f.strength = narrow(self.strength);
        f.min_distance = narrow(self.min_distance);
        f
    }
}

impl Default for HubGravityParams {
    fn default() -> Self {
        Self::of(&HubGravity::default())
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

impl DepthGravityParams {
    pub fn of(f: &DepthGravity) -> Self {
        Self {
            depths: None,
            direction: wide2(f.direction),
            spacing: wide(f.spacing),
            strength: wide(f.strength),
        }
    }

    fn build(&self, path: &str) -> Result<DepthGravity, SpecError> {
        need(self.depths.as_ref(), path, "depths")?;
        let mut f = DepthGravity::new(std::iter::empty());
        f.direction = narrow2(self.direction);
        f.spacing = narrow(self.spacing);
        f.strength = narrow(self.strength);
        Ok(f)
    }
}

impl Default for DepthGravityParams {
    fn default() -> Self {
        Self::of(&DepthGravity::new(std::iter::empty()))
    }
}

params! {
    /// [`GridSnap`]: Grid.
    pub struct GridSnapParams {
        pub cell: f64,
        pub strength: f64,
    }
}

impl GridSnapParams {
    pub fn of(f: &GridSnap) -> Self {
        Self {
            cell: wide(f.cell),
            strength: wide(f.strength),
        }
    }

    fn build(&self) -> GridSnap {
        GridSnap {
            cell: narrow(self.cell),
            strength: narrow(self.strength),
        }
    }
}

impl Default for GridSnapParams {
    fn default() -> Self {
        Self::of(&GridSnap::default())
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

impl GravityLocusParams {
    pub fn of(f: &GravityLocus) -> Self {
        Self {
            target: Some(wide2(f.target)),
            strength: wide(f.strength),
            oscillation: f.oscillation.map(wide2),
        }
    }

    fn build(&self, path: &str) -> Result<GravityLocus, SpecError> {
        let target = narrow2(need(self.target, path, "target")?);
        let mut f = match self.oscillation {
            None => GravityLocus::at(target),
            Some((amplitude, period)) => {
                GravityLocus::tidal(target, narrow(amplitude), narrow(period))
            },
        };
        f.strength = narrow(self.strength);
        Ok(f)
    }
}

impl Default for GravityLocusParams {
    fn default() -> Self {
        let mut params = Self::of(&GravityLocus::at((0.0, 0.0)));
        params.target = None;
        params
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
