// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! The force set as data: what a host sends the simulation instead of boxed
//! forces. A [`ForceSetDescriptor`] is the catalog's own vocabulary made
//! owned values: the law (or composition), the overlays, the sources, and
//! the graph inputs the build snapshots, all resolved on the host's side so
//! nothing in it borrows the graph. `seiche::catalog::build_force_set` turns
//! it into forces on whichever thread the backend runs, by the same builders
//! the in-process path calls, so the forces are the same by construction.
//!
//! The physics channels (degree, PageRank, colouring, islands, depths, the
//! spanning tree, hop distances) are not in the table: they are functions of
//! the nodes and edges, which are, and the builder computes them with the
//! same algorithms a host's cache runs.

use crate::{
    NodeKey, PhysicsComposition, PhysicsDepthSource, PhysicsKindSource, PhysicsLaw,
    PhysicsMassSource, PhysicsOverlay,
};

/// The seed every seeded law (Kinds' rule matrix, Anneal's walk) starts from,
/// so a scene reopens to the same rules.
pub const LAW_SEED: u64 = 0x5EED_CA7A_1064;

/// The sources a build reads attributes through.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LawSources {
    pub kind: PhysicsKindSource,
    /// Group pull's groups: any kind channel, unfolded.
    pub groups: PhysicsKindSource,
    pub mass: PhysicsMassSource,
    pub depth: PhysicsDepthSource,
    /// The focused node, for the Focus depth source.
    pub focus: Option<NodeKey>,
    /// The seed Kinds' matrix and Anneal's walk draw from: the spec's
    /// (F156), [`LAW_SEED`] by default, so no default run moves.
    pub seed: u64,
}

impl LawSources {
    /// The default sources and no focus: enough to build any law over no
    /// graph, to read its declared terms.
    pub fn bare() -> Self {
        Self {
            kind: PhysicsKindSource::Site,
            groups: PhysicsKindSource::Site,
            mass: PhysicsMassSource::Degree,
            depth: PhysicsDepthSource::Roots,
            focus: None,
            seed: LAW_SEED,
        }
    }
}

/// The Meaning snapshot resolved to what a build reads: a group per node it
/// knew, and the extra group a node it does not know yet joins (added since
/// the snapshot was taken) until the next snapshot lands.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MeaningTable {
    pub groups: Vec<(NodeKey, u32)>,
    /// The snapshot's cluster count, the id of the group for unknown nodes.
    pub unknown: u32,
}

/// The graph inputs a build snapshots, resolved: the node set (in key order),
/// the visible spring edges, each node's site, and the cluster and meaning
/// tables when the host has them.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LawInputsData {
    pub nodes: Vec<NodeKey>,
    pub edges: Vec<(NodeKey, NodeKey)>,
    /// `(node, site)` in key order.
    pub sites: Vec<(NodeKey, String)>,
    /// Each node's Louvain community index (`None` without a partition).
    pub clusters: Option<Vec<(NodeKey, u32)>>,
    pub meaning: Option<MeaningTable>,
    /// The host's token for the node and edge lists above: equal tokens mean
    /// the same lists. A backend that keeps the physics channels (PageRank,
    /// colouring, the depths) between builds keys them by it, and still checks
    /// the lists, so a token reused for other lists costs a recomputation and
    /// nothing else. `None` computes each time.
    pub revision: Option<u64>,
}

/// The law slot and the overlays as data: what `PhysicsCommand::SetForceSet`
/// carries in place of `SetForces`' boxed forces.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ForceSetDescriptor {
    /// The picked law (run when there is no composition).
    pub law: PhysicsLaw,
    /// The composition running in the law slot instead of the law, if any.
    pub composition: Option<PhysicsComposition>,
    /// The overlays composed after the law slot, in order.
    pub overlays: Vec<PhysicsOverlay>,
    pub sources: LawSources,
    pub inputs: LawInputsData,
}
