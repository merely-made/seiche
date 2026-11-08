// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! The physics catalog's vocabulary as data: the laws a graph can move under,
//! the overlays composed over them, and the sources that feed them (kinds,
//! mass, depth), with their ids and labels. The builders that turn these into
//! forces live in `seiche::catalog`; a host names the choice here without the
//! engine.

use crate::Currency;

/// What a law does with one more term.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Admission {
    /// Taken as it is.
    Compose,
    /// Taken converted to the law's currency.
    Convert,
    /// Refused, with the reason a picker shows.
    Refuse(&'static str),
}

impl Admission {
    /// The more restrictive of two admissions: a refusal, then a conversion.
    pub fn and(self, other: Admission) -> Admission {
        match (self, other) {
            (Admission::Refuse(reason), _) | (_, Admission::Refuse(reason)) => {
                Admission::Refuse(reason)
            },
            (Admission::Convert, _) | (_, Admission::Convert) => Admission::Convert,
            _ => Admission::Compose,
        }
    }

    pub fn refusal(self) -> Option<&'static str> {
        match self {
            Admission::Refuse(reason) => Some(reason),
            _ => None,
        }
    }
}

/// The physics law: which dynamics the graph moves under. Ids are technical
/// (`family.method`), labels plain, as the arrangement catalog does it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PhysicsLaw {
    /// Exclusion + edge springs + a centering boundary: the force-directed default.
    Springs,
    /// Barnes–Hut charge repulsion + edge springs: every body repels every other,
    /// distant ones by their cell's centre of mass.
    Charge,
    /// Kamada–Kawai stress: a spring between every connected pair at its
    /// shortest-path distance (relation multiplicity shortens a hop), so the
    /// picture is a metric map of the graph.
    Stress,
    /// ForceAtlas2's force model: attraction linear in distance along edges,
    /// repulsion falling as `1/d`, degree-weighted, so communities separate
    /// and hubs sit central. LinLog proper is its attraction exponent `0`, a
    /// tuning; the id stays `energy.linlog`.
    Energy,
    /// Newtonian gravity with an orbital kick: hubs are suns, leaves circle them,
    /// and it never rests.
    Orbit,
    /// Particle life: nodes carry a kind, and a kind-by-kind rule matrix says who
    /// chases and who flees, and it never rests.
    Kinds,
    /// Boids: separation, alignment, cohesion, and a cruising speed; the graph
    /// moves as a flock.
    Flock,
    /// Kuramoto oscillators: each node has a phase, edges couple phases, and
    /// position follows phase on a ring, so communities become phase clusters.
    Sync,
    /// Magnetic springs: edges align to a field direction, so a directed graph
    /// reads as a flow.
    Flow,
    /// Davidson–Harel simulated annealing: a random walk over a layout energy,
    /// cooling to a minimum.
    Anneal,
    /// No law: bodies hold where the arrangement (or a hand) put them
    /// (velocity zeroed each tick, so contacts can only nudge).
    Still,
    /// Gastner–Newman density equalizing: nodes flow down the gradient of a
    /// diffused mass density until it is even, so room follows mass.
    Density,
}

impl PhysicsLaw {
    pub const ALL: [PhysicsLaw; 12] = [
        PhysicsLaw::Springs,
        PhysicsLaw::Charge,
        PhysicsLaw::Stress,
        PhysicsLaw::Energy,
        PhysicsLaw::Orbit,
        PhysicsLaw::Kinds,
        PhysicsLaw::Flock,
        PhysicsLaw::Sync,
        PhysicsLaw::Flow,
        PhysicsLaw::Anneal,
        PhysicsLaw::Still,
        PhysicsLaw::Density,
    ];

    pub fn id(self) -> &'static str {
        match self {
            PhysicsLaw::Springs => "spring.rapier",
            PhysicsLaw::Charge => "charge.barnes-hut",
            PhysicsLaw::Stress => "stress.kamada-kawai",
            PhysicsLaw::Energy => "energy.linlog",
            PhysicsLaw::Orbit => "orbit.gravity",
            PhysicsLaw::Kinds => "kinds.particle-life",
            PhysicsLaw::Flock => "flock.boids",
            PhysicsLaw::Sync => "sync.kuramoto",
            PhysicsLaw::Flow => "flow.magnetic",
            PhysicsLaw::Anneal => "anneal.davidson-harel",
            PhysicsLaw::Still => "still.default",
            PhysicsLaw::Density => "density.gastner-newman",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            PhysicsLaw::Springs => "Springs",
            PhysicsLaw::Charge => "Charge",
            PhysicsLaw::Stress => "Stress",
            PhysicsLaw::Energy => "Energy",
            PhysicsLaw::Orbit => "Orbit",
            PhysicsLaw::Kinds => "Kinds",
            PhysicsLaw::Flock => "Flock",
            PhysicsLaw::Sync => "Sync",
            PhysicsLaw::Flow => "Flow",
            PhysicsLaw::Anneal => "Anneal",
            PhysicsLaw::Still => "Still",
            PhysicsLaw::Density => "Density",
        }
    }

    pub fn parse(id: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|law| law.id() == id)
    }

    /// Whether the law is a living display that never comes to rest (Orbit,
    /// Kinds, Flock, Sync), so the host keeps ticking rather than settling.
    /// Kinds joined on the P2 fixture's figures: kinetic energy about 18 300
    /// at 6 s and 140 500 at 30 s under continuous ticking, against the floor
    /// of 1 (dynamics grammar plan, G1, F10).
    pub fn never_rests(self) -> bool {
        matches!(
            self,
            PhysicsLaw::Orbit | PhysicsLaw::Kinds | PhysicsLaw::Flock | PhysicsLaw::Sync
        )
    }

    /// Whether the law snapshots graph structure at build (and so is rebuilt on
    /// a topology change): Stress's distances, Orbit's and Density's masses,
    /// Kinds' kinds.
    pub fn graph_bound(self) -> bool {
        matches!(
            self,
            PhysicsLaw::Stress | PhysicsLaw::Orbit | PhysicsLaw::Kinds | PhysicsLaw::Density
        )
    }

    /// Whether the law reads the mass source (and so is rebuilt when it changes).
    pub fn weighted(self) -> bool {
        matches!(self, PhysicsLaw::Orbit | PhysicsLaw::Density)
    }

    /// Why the law refuses `overlay`, if it does: what its currency refuses
    /// ([`Self::admits`]), and on Density every overlay but the two it takes
    /// converted (F73).
    pub fn refuses(self, overlay: PhysicsOverlay) -> Option<&'static str> {
        self.admits(overlay).refusal()
    }
}

/// The overlays Density takes, converted into its flow: the three F73
/// named ("Hub room, Centre, Tide if they hold"), against Density's own bar
/// ("Min 60, bar: all >= 0.7") at all sixteen dealt starts of both bar
/// graphs. Hub room's lowest start reads 0.808 (gen-50) and 0.765 (gen-200),
/// Centre's 0.704 and 0.735, and Tide's 0.699 and 0.736, admitted with them
/// (F81, "Admit Tide too").
pub const DENSITY_ADMITS: &[PhysicsOverlay] = &[
    PhysicsOverlay::DegreeRepulsion,
    PhysicsOverlay::GravityLocus,
    PhysicsOverlay::Tide,
];

/// Why Density refuses the rest.
pub const DENSITY_REFUSAL: &str = "Density takes only Hub room, Centre and Tide: converted \
                                          into its flow, the other overlays keep room from \
                                          following mass.";

/// Overlays a law refused: which, and why. The law itself was applied.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OverlayRefusal {
    pub law: PhysicsLaw,
    pub refused: Vec<PhysicsOverlay>,
    pub reason: &'static str,
}

impl std::fmt::Display for OverlayRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.reason)
    }
}

/// An overlay: one extra force composed onto any law.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PhysicsOverlay {
    /// Hubs push their surroundings apart, by weight.
    DegreeRepulsion,
    /// Nodes drift toward the centroid of their group (by the groups channel,
    /// site by default).
    DomainCluster,
    /// Everything is drawn toward the hubs, by weight.
    HubGravity,
    /// Depth drives the vertical: roots up, leaves down.
    DepthGravity,
    /// A spring to the nearest grid point.
    GridSnap,
    /// A gentle pull toward the canvas centre.
    GravityLocus,
    /// A centre that rides a slow sine, so the graph never fully settles.
    Tide,
    /// The spanning tree's edges held stiff, so the backbone shows under any law.
    Skeleton,
}

impl PhysicsOverlay {
    pub const ALL: [PhysicsOverlay; 8] = [
        PhysicsOverlay::DegreeRepulsion,
        PhysicsOverlay::DomainCluster,
        PhysicsOverlay::HubGravity,
        PhysicsOverlay::DepthGravity,
        PhysicsOverlay::GridSnap,
        PhysicsOverlay::GravityLocus,
        PhysicsOverlay::Tide,
        PhysicsOverlay::Skeleton,
    ];

    pub fn id(self) -> &'static str {
        match self {
            PhysicsOverlay::DegreeRepulsion => "degree-repulsion",
            PhysicsOverlay::DomainCluster => "domain-cluster",
            PhysicsOverlay::HubGravity => "hub-gravity",
            PhysicsOverlay::DepthGravity => "depth-gravity",
            PhysicsOverlay::GridSnap => "grid-snap",
            PhysicsOverlay::GravityLocus => "gravity-locus",
            PhysicsOverlay::Tide => "tide",
            PhysicsOverlay::Skeleton => "skeleton",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            PhysicsOverlay::DegreeRepulsion => "Hub room",
            PhysicsOverlay::DomainCluster => "Group pull",
            PhysicsOverlay::HubGravity => "Hub pull",
            PhysicsOverlay::DepthGravity => "Depth",
            PhysicsOverlay::GridSnap => "Grid",
            PhysicsOverlay::GravityLocus => "Centre",
            PhysicsOverlay::Tide => "Tide",
            PhysicsOverlay::Skeleton => "Skeleton",
        }
    }

    pub fn parse(id: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|overlay| overlay.id() == id)
    }

    /// Whether the overlay snapshots graph structure at build (the grouping,
    /// the depths, the tree), and so is rebuilt on a topology change.
    pub fn graph_bound(self) -> bool {
        matches!(
            self,
            PhysicsOverlay::DomainCluster | PhysicsOverlay::DepthGravity | PhysicsOverlay::Skeleton
        )
    }

    /// Whether the overlay reads the mass source's weights (and so is
    /// graph-bound whenever that source is computed from the topology).
    pub fn weighted(self) -> bool {
        matches!(
            self,
            PhysicsOverlay::DegreeRepulsion | PhysicsOverlay::HubGravity
        )
    }

    /// Whether the overlay keeps the graph moving on its own (the tide).
    pub fn never_rests(self) -> bool {
        matches!(self, PhysicsOverlay::Tide)
    }
}

/// Where the Kinds law reads a node's kind from, and Group pull its groups —
/// the host's choice per scene, one selector each.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PhysicsKindSource {
    /// The URL host: every site a kind.
    Site,
    /// The Louvain community: every cluster a kind.
    Cluster,
    /// A proper colouring (DSATUR): a kind never touches its own kind, so the
    /// rule matrix plays out between neighbours.
    Coloring,
    /// The connected component: every island a kind.
    Component,
    /// Degree bands: isolated, leaf, connected, hub.
    Degree,
    /// What the nodes say: their titles embedded, and the clusters of the
    /// similarity pairs between them (the Meaning channel, dynamics grammar
    /// plan, G2). Site until the first snapshot lands.
    Meaning,
}

impl PhysicsKindSource {
    pub const ALL: [PhysicsKindSource; 6] = [
        PhysicsKindSource::Site,
        PhysicsKindSource::Cluster,
        PhysicsKindSource::Coloring,
        PhysicsKindSource::Component,
        PhysicsKindSource::Degree,
        PhysicsKindSource::Meaning,
    ];

    pub fn id(self) -> &'static str {
        match self {
            PhysicsKindSource::Site => "site",
            PhysicsKindSource::Cluster => "cluster",
            PhysicsKindSource::Coloring => "coloring",
            PhysicsKindSource::Component => "component",
            PhysicsKindSource::Degree => "degree",
            PhysicsKindSource::Meaning => "meaning",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            PhysicsKindSource::Site => "By site",
            PhysicsKindSource::Cluster => "By cluster",
            PhysicsKindSource::Coloring => "By colouring",
            PhysicsKindSource::Component => "By island",
            PhysicsKindSource::Degree => "By degree",
            PhysicsKindSource::Meaning => "By meaning",
        }
    }

    pub fn parse(id: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|source| source.id() == id)
    }
}

/// Where Orbit's masses and the hub overlays' weights come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PhysicsMassSource {
    /// Degree: the most-connected node is the heaviest.
    Degree,
    /// PageRank: the most linked-to node is the heaviest, links from heavy
    /// nodes counting for more.
    PageRank,
}

impl PhysicsMassSource {
    pub const ALL: [PhysicsMassSource; 2] =
        [PhysicsMassSource::Degree, PhysicsMassSource::PageRank];

    pub fn id(self) -> &'static str {
        match self {
            PhysicsMassSource::Degree => "degree",
            PhysicsMassSource::PageRank => "pagerank",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            PhysicsMassSource::Degree => "By degree",
            PhysicsMassSource::PageRank => "By rank",
        }
    }

    pub fn parse(id: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|source| source.id() == id)
    }
}

/// Where the Depth overlay reads a node's depth from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PhysicsDepthSource {
    /// Breadth-first from the nodes nothing points at.
    Roots,
    /// Sugiyama's layer step: cut a feedback arc set, then longest-path layers
    /// over the remaining order — works on cycles.
    Layers,
    /// Dominator-tree depth from the focused node: what you must pass through
    /// to reach a node from the selection. Roots when nothing is focused.
    Focus,
}

impl PhysicsDepthSource {
    pub const ALL: [PhysicsDepthSource; 3] = [
        PhysicsDepthSource::Roots,
        PhysicsDepthSource::Layers,
        PhysicsDepthSource::Focus,
    ];

    pub fn id(self) -> &'static str {
        match self {
            PhysicsDepthSource::Roots => "roots",
            PhysicsDepthSource::Layers => "layers",
            PhysicsDepthSource::Focus => "focus",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            PhysicsDepthSource::Roots => "From roots",
            PhysicsDepthSource::Layers => "By layer",
            PhysicsDepthSource::Focus => "From focus",
        }
    }

    pub fn parse(id: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|source| source.id() == id)
    }
}

impl PhysicsLaw {
    /// How the law's motion enters the step: the first nine move by force,
    /// Anneal, Still and Density write positions. This is a table, not a
    /// computation: `seiche`'s drift test builds each law's forces and fails
    /// if what they declare ever differs.
    pub fn currency(self) -> Currency {
        match self {
            PhysicsLaw::Anneal | PhysicsLaw::Still | PhysicsLaw::Density => Currency::Kinematic,
            _ => Currency::Force,
        }
    }

    /// What the law does with `overlay`: compose it, convert it, or refuse it.
    /// A force law composes every overlay; a kinematic law converts them, and
    /// Density takes only the overlays that hold its bar converted (F73).
    /// Same drift test as [`Self::currency`].
    pub fn admits(self, overlay: PhysicsOverlay) -> Admission {
        if self == PhysicsLaw::Density && !DENSITY_ADMITS.contains(&overlay) {
            return Admission::Refuse(DENSITY_REFUSAL);
        }
        match self.currency() {
            Currency::Kinematic => Admission::Convert,
            _ => Admission::Compose,
        }
    }
}
