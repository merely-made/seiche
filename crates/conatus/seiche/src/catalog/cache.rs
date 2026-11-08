// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! The physics channels, kept between builds.
//!
//! The inline canvas keeps each physics channel (PageRank, the colouring, the
//! depths, ...) once per view key in its channel registry, so changing the law
//! over an unchanged graph recomputes none of them. A backend that builds from
//! a descriptor sees the nodes and edges and nothing of the host's registry;
//! this is its registry: one slot, for the latest revision a descriptor
//! named, holding each channel the builds asked for. A revision names the
//! lists, and the slot also compares them, so a host that reuses a token for
//! other lists gets a recomputation and never a stale channel.

use std::{cell::RefCell, collections::HashMap, rc::Rc};

use super::{Channels, algo};
use crate::NodeKey;

type Groups = Vec<(NodeKey, u32)>;

/// How many times each channel was computed (not read) in a cache.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ChannelRuns {
    pub mass_degree: u64,
    pub pagerank: u64,
    pub coloring: u64,
    pub components: u64,
    pub degree_bands: u64,
    pub depth_roots: u64,
    pub depth_layers: u64,
    pub depth_focus: u64,
    pub distances: u64,
    pub spanning: u64,
}

impl ChannelRuns {
    /// The total over every channel.
    pub fn total(&self) -> u64 {
        self.mass_degree
            + self.pagerank
            + self.coloring
            + self.components
            + self.degree_bands
            + self.depth_roots
            + self.depth_layers
            + self.depth_focus
            + self.distances
            + self.spanning
    }
}

#[derive(Default)]
struct Slot {
    revision: u64,
    nodes: Vec<NodeKey>,
    edges: Vec<(NodeKey, NodeKey)>,
    mass_degree: Option<Vec<(NodeKey, f32)>>,
    pagerank: Option<Vec<(NodeKey, f32)>>,
    coloring: Option<Groups>,
    components: Option<Groups>,
    degree_bands: Option<Groups>,
    depth_roots: Option<Groups>,
    depth_layers: Option<Groups>,
    depth_focus: HashMap<NodeKey, Groups>,
    distances: Option<Vec<(NodeKey, NodeKey, f32)>>,
    spanning: Option<Vec<(NodeKey, NodeKey)>>,
}

/// The latest revision's channels. See the module doc.
#[derive(Default)]
pub struct ChannelCache {
    slot: Option<Slot>,
    runs: ChannelRuns,
}

impl ChannelCache {
    /// An empty cache, shared by the builds of one backend.
    pub fn shared() -> Rc<RefCell<Self>> {
        Rc::default()
    }

    /// How many times each channel has been computed.
    pub fn runs(&self) -> ChannelRuns {
        self.runs
    }
}

/// The slot for these lists at `revision`, replacing the held one when the
/// revision or the lists differ.
fn slot_for<'a>(
    held: &'a mut Option<Slot>,
    revision: u64,
    nodes: &[NodeKey],
    edges: &[(NodeKey, NodeKey)],
) -> &'a mut Slot {
    let same = held
        .as_ref()
        .is_some_and(|s| s.revision == revision && s.nodes == nodes && s.edges == edges);
    if !same {
        *held = Some(Slot {
            revision,
            nodes: nodes.to_vec(),
            edges: edges.to_vec(),
            ..Slot::default()
        });
    }
    held.as_mut().expect("the slot was just set")
}

/// A [`Channels`] reading one revision of a shared [`ChannelCache`].
pub(super) struct Cached {
    revision: u64,
    cache: Rc<RefCell<ChannelCache>>,
}

impl Cached {
    pub(super) fn new(revision: Option<u64>, cache: Rc<RefCell<ChannelCache>>) -> Self {
        Self {
            revision: revision.expect("a cached read names a revision"),
            cache,
        }
    }
}

macro_rules! cached {
    ($self:ident, $nodes:ident, $edges:ident, $field:ident, $algo:expr) => {{
        let mut cache = $self.cache.borrow_mut();
        let ChannelCache { slot, runs } = &mut *cache;
        let slot = slot_for(slot, $self.revision, $nodes, $edges);
        if slot.$field.is_none() {
            runs.$field += 1;
            slot.$field = Some($algo);
        }
        slot.$field.clone().expect("the channel was just set")
    }};
}

impl Channels for Cached {
    fn mass_degree(&self, nodes: &[NodeKey], edges: &[(NodeKey, NodeKey)]) -> Vec<(NodeKey, f32)> {
        cached!(self, nodes, edges, mass_degree, algo::mass_degree(nodes, edges))
    }
    fn mass_pagerank(
        &self,
        nodes: &[NodeKey],
        edges: &[(NodeKey, NodeKey)],
    ) -> Vec<(NodeKey, f32)> {
        cached!(self, nodes, edges, pagerank, algo::page_rank_weights(nodes, edges))
    }
    fn groups_coloring(&self, nodes: &[NodeKey], edges: &[(NodeKey, NodeKey)]) -> Groups {
        cached!(self, nodes, edges, coloring, algo::coloring_groups(nodes, edges))
    }
    fn groups_component(&self, nodes: &[NodeKey], edges: &[(NodeKey, NodeKey)]) -> Groups {
        cached!(self, nodes, edges, components, algo::component_groups(nodes, edges))
    }
    fn groups_degree(&self, nodes: &[NodeKey], edges: &[(NodeKey, NodeKey)]) -> Groups {
        cached!(self, nodes, edges, degree_bands, algo::degree_bands(nodes, edges))
    }
    fn depth_roots(&self, nodes: &[NodeKey], edges: &[(NodeKey, NodeKey)]) -> Groups {
        cached!(self, nodes, edges, depth_roots, algo::root_depths(nodes, edges))
    }
    fn depth_layers(&self, nodes: &[NodeKey], edges: &[(NodeKey, NodeKey)]) -> Groups {
        cached!(self, nodes, edges, depth_layers, algo::layer_depths(nodes, edges))
    }
    fn depth_focus(
        &self,
        nodes: &[NodeKey],
        edges: &[(NodeKey, NodeKey)],
        focus: NodeKey,
    ) -> Groups {
        let mut cache = self.cache.borrow_mut();
        let ChannelCache { slot, runs } = &mut *cache;
        slot_for(slot, self.revision, nodes, edges)
            .depth_focus
            .entry(focus)
            .or_insert_with(|| {
                runs.depth_focus += 1;
                algo::focus_depths(nodes, edges, focus)
            })
            .clone()
    }
    fn edges_spanning(
        &self,
        nodes: &[NodeKey],
        edges: &[(NodeKey, NodeKey)],
    ) -> Vec<(NodeKey, NodeKey)> {
        cached!(self, nodes, edges, spanning, algo::spanning_edges(nodes, edges))
    }
    fn distances_hops(
        &self,
        nodes: &[NodeKey],
        edges: &[(NodeKey, NodeKey)],
    ) -> Vec<(NodeKey, NodeKey, f32)> {
        cached!(self, nodes, edges, distances, algo::hop_distances(nodes, edges))
    }
}
