use std::borrow::Cow;
use std::collections::HashMap;

use seiche_model::{PhysicsDepthSource, PhysicsKindSource, PhysicsMassSource};

use super::algo;
use crate::NodeKey;

/// A host's cache of the physics view's channels (one computation per view
/// key; F178, F179). The inputs hand it their own node and edge lists, which
/// are the view; each method returns what the matching [`algo`] function
/// would. Without one the inputs compute the channels themselves, uncached.
pub trait Channels {
    /// `mass.degree`.
    fn mass_degree(&self, nodes: &[NodeKey], edges: &[(NodeKey, NodeKey)]) -> Vec<(NodeKey, f32)>;
    /// `mass.pagerank`.
    fn mass_pagerank(&self, nodes: &[NodeKey], edges: &[(NodeKey, NodeKey)])
    -> Vec<(NodeKey, f32)>;
    /// `groups.coloring`.
    fn groups_coloring(
        &self,
        nodes: &[NodeKey],
        edges: &[(NodeKey, NodeKey)],
    ) -> Vec<(NodeKey, u32)>;
    /// `groups.component`.
    fn groups_component(
        &self,
        nodes: &[NodeKey],
        edges: &[(NodeKey, NodeKey)],
    ) -> Vec<(NodeKey, u32)>;
    /// `groups.degree`.
    fn groups_degree(&self, nodes: &[NodeKey], edges: &[(NodeKey, NodeKey)])
    -> Vec<(NodeKey, u32)>;
    /// `depth.roots`.
    fn depth_roots(&self, nodes: &[NodeKey], edges: &[(NodeKey, NodeKey)]) -> Vec<(NodeKey, u32)>;
    /// `depth.layers`.
    fn depth_layers(&self, nodes: &[NodeKey], edges: &[(NodeKey, NodeKey)]) -> Vec<(NodeKey, u32)>;
    /// `depth.focus` from `focus`.
    fn depth_focus(
        &self,
        nodes: &[NodeKey],
        edges: &[(NodeKey, NodeKey)],
        focus: NodeKey,
    ) -> Vec<(NodeKey, u32)>;
    /// `edges.spanning`.
    fn edges_spanning(
        &self,
        nodes: &[NodeKey],
        edges: &[(NodeKey, NodeKey)],
    ) -> Vec<(NodeKey, NodeKey)>;
    /// `distances.hops`.
    fn distances_hops(
        &self,
        nodes: &[NodeKey],
        edges: &[(NodeKey, NodeKey)],
    ) -> Vec<(NodeKey, NodeKey, f32)>;
}

/// The Meaning snapshot resolved to what a build reads: a group per node it
/// knew, and the extra group a node it does not know yet joins (added since
/// the snapshot was taken) until the next snapshot lands.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MeaningGroups<'a> {
    pub groups: Cow<'a, [(NodeKey, u32)]>,
    /// The snapshot's cluster count, the id of the group for unknown nodes.
    pub unknown: u32,
}

/// The graph inputs a law or overlay snapshots at build: the node set, the
/// visible spring edges, and (on demand) the site grouping, the cluster
/// partition, the Meaning snapshot and the physics view's channels. A host
/// resolves its graph into these (Pictograph's constructors over a `Graph`);
/// nothing here borrows the graph.
///
/// The physics channels come from the [`algo`] producers. Over the physics
/// view, handed the host's cache ([`Self::with_channels`]), they are the
/// cache's, one computation per key; over anything else (a grouping's
/// subgraphs, a board's items, the catalog's nominal) they are computed here.
pub struct LawInputs<'a> {
    pub(super) nodes: Vec<NodeKey>,
    pub(super) edges: Vec<(NodeKey, NodeKey)>,
    /// Each node's site (the URL host for a graph node; whatever grouping a
    /// board's host names), the Kinds law's and the group overlay's default.
    pub(super) sites: Cow<'a, HashMap<NodeKey, String>>,
    /// Each node's Louvain community index (`None` without a partition).
    clusters: Option<Cow<'a, [(NodeKey, u32)]>>,
    meaning: Option<MeaningGroups<'a>>,
    /// The cache serving the physics view.
    channels: Option<Box<dyn Channels + 'a>>,
}

impl<'a> LawInputs<'a> {
    /// Graph-free inputs: a node list, an edge list, and a site per node.
    pub fn from_parts(
        mut nodes: Vec<NodeKey>,
        edges: Vec<(NodeKey, NodeKey)>,
        sites: HashMap<NodeKey, String>,
    ) -> Self {
        nodes.sort_by_key(|key| key.index());
        Self {
            nodes,
            edges,
            sites: Cow::Owned(sites),
            clusters: None,
            meaning: None,
            channels: None,
        }
    }

    /// Replace the visible spring edges.
    #[must_use]
    pub fn with_edges(mut self, edges: Vec<(NodeKey, NodeKey)>) -> Self {
        self.edges = edges;
        self
    }

    /// Replace the sites, borrowed or owned.
    #[must_use]
    pub fn with_sites(mut self, sites: Cow<'a, HashMap<NodeKey, String>>) -> Self {
        self.sites = sites;
        self
    }

    /// Read Cluster groups from this partition (site until one exists).
    #[must_use]
    pub fn with_clusters(mut self, clusters: Option<Cow<'a, [(NodeKey, u32)]>>) -> Self {
        self.clusters = clusters;
        self
    }

    /// Read Meaning groups from this resolved snapshot (site until one exists).
    #[must_use]
    pub fn with_meaning(mut self, meaning: Option<MeaningGroups<'a>>) -> Self {
        self.meaning = meaning;
        self
    }

    /// Read the physics channels from `channels`, these inputs being the
    /// physics view it caches for (F178).
    #[must_use]
    pub fn with_channels(mut self, channels: Box<dyn Channels + 'a>) -> Self {
        self.channels = Some(channels);
        self
    }

    /// The node set, in key order.
    pub fn nodes(&self) -> &[NodeKey] {
        &self.nodes
    }

    /// The visible spring edges the forces pull along.
    pub fn edges(&self) -> &[(NodeKey, NodeKey)] {
        &self.edges
    }

    /// Each node's site, for inputs built over a subset of the nodes.
    pub fn sites(&self) -> &HashMap<NodeKey, String> {
        &self.sites
    }

    /// Every node's group by site, as a dense id in first-seen order.
    fn site_groups(&self) -> Vec<(NodeKey, u32)> {
        let mut ids: HashMap<String, u32> = HashMap::new();
        self.nodes
            .iter()
            .map(|&key| {
                let site = self.sites.get(&key).cloned().unwrap_or_default();
                let next = ids.len() as u32;
                (key, *ids.entry(site).or_insert(next))
            })
            .collect()
    }

    /// Every node's Louvain community index (`None` without a partition).
    fn cluster_groups(&self) -> Option<Vec<(NodeKey, u32)>> {
        Some(self.clusters.as_ref()?.to_vec())
    }

    /// Every node's Meaning cluster (`None` without a snapshot). A node the
    /// snapshot does not know yet (added since it was taken) joins one extra
    /// group until the next snapshot lands.
    fn meaning_groups(&self) -> Option<Vec<(NodeKey, u32)>> {
        let meaning = self.meaning.as_ref()?;
        let of: HashMap<NodeKey, u32> = meaning.groups.iter().copied().collect();
        Some(
            self.nodes
                .iter()
                .map(|&key| (key, of.get(&key).copied().unwrap_or(meaning.unknown)))
                .collect(),
        )
    }

    /// Degree bands: isolated, leaf, connected, hub.
    fn degree_groups(&self) -> Vec<(NodeKey, u32)> {
        match &self.channels {
            Some(channels) => channels.groups_degree(&self.nodes, &self.edges),
            None => algo::degree_bands(&self.nodes, &self.edges),
        }
    }

    /// The groups channel: every node's group from any kind channel, as a
    /// dense id, unfolded. Cluster and Meaning read site until their
    /// partition exists.
    pub fn groups(&self, source: PhysicsKindSource) -> Vec<(NodeKey, u32)> {
        match source {
            PhysicsKindSource::Site => self.site_groups(),
            PhysicsKindSource::Cluster => {
                self.cluster_groups().unwrap_or_else(|| self.site_groups())
            },
            PhysicsKindSource::Coloring => self.coloring_groups(),
            PhysicsKindSource::Component => self.component_groups(),
            PhysicsKindSource::Degree => self.degree_groups(),
            PhysicsKindSource::Meaning => {
                self.meaning_groups().unwrap_or_else(|| self.site_groups())
            },
        }
    }

    /// A proper colouring of the visible graph (DSATUR): adjacent nodes never
    /// share a group.
    pub fn coloring_groups(&self) -> Vec<(NodeKey, u32)> {
        match &self.channels {
            Some(channels) => channels.groups_coloring(&self.nodes, &self.edges),
            None => algo::coloring_groups(&self.nodes, &self.edges),
        }
    }

    /// Every node's connected component (island), as a dense id.
    pub fn component_groups(&self) -> Vec<(NodeKey, u32)> {
        match &self.channels {
            Some(channels) => channels.groups_component(&self.nodes, &self.edges),
            None => algo::component_groups(&self.nodes, &self.edges),
        }
    }

    /// PageRank over the directed view, scaled so the mean weight is one
    /// (comparable to log degree on a sparse graph).
    pub fn page_rank_weights(&self) -> Vec<(NodeKey, f32)> {
        match &self.channels {
            Some(channels) => channels.mass_pagerank(&self.nodes, &self.edges),
            None => algo::page_rank_weights(&self.nodes, &self.edges),
        }
    }

    /// The mass channel's values before any reader's transform: degree, or
    /// PageRank with the mean rank one. Orbit reads `1 +` these; the hub
    /// overlays read `ln(1 + degree)` (seiche's default) or the rank itself.
    pub fn mass_values(&self, source: PhysicsMassSource) -> Vec<(NodeKey, f32)> {
        match source {
            PhysicsMassSource::Degree => match &self.channels {
                Some(channels) => channels.mass_degree(&self.nodes, &self.edges),
                None => algo::mass_degree(&self.nodes, &self.edges),
            },
            PhysicsMassSource::PageRank => self.page_rank_weights(),
        }
    }

    /// The depth channel's values.
    pub fn depth_values(
        &self,
        source: PhysicsDepthSource,
        focus: Option<NodeKey>,
    ) -> Vec<(NodeKey, u32)> {
        self.depths(source, focus)
    }

    /// Masses for Orbit and Density: `1 + degree`, or `1 + rank` with the
    /// mean rank one.
    pub fn masses(&self, source: PhysicsMassSource) -> Vec<(NodeKey, f32)> {
        self.mass_values(source)
            .into_iter()
            .map(|(key, value)| (key, 1.0 + value))
            .collect()
    }

    /// A kind per node for the Kinds law, from the chosen source, plus how many
    /// kinds there are.
    pub fn kinds(&self, source: PhysicsKindSource) -> (Vec<(NodeKey, u8)>, usize) {
        let groups = self.groups(source);
        // Particle life reads best with a handful of kinds; fold a long tail of
        // sites into eight, keeping a small catalog its own size.
        let distinct = groups
            .iter()
            .map(|(_, g)| *g)
            .max()
            .map_or(0, |g| g as usize + 1);
        let kind_count = distinct.clamp(1, 8);
        let kinds = groups
            .into_iter()
            .map(|(key, g)| (key, (g as usize % kind_count) as u8))
            .collect();
        (kinds, kind_count)
    }

    /// BFS depth from the roots — the nodes with no incoming visible edge, or
    /// every node when the graph is all cycles.
    pub fn root_depths(&self) -> Vec<(NodeKey, u32)> {
        match &self.channels {
            Some(channels) => channels.depth_roots(&self.nodes, &self.edges),
            None => algo::root_depths(&self.nodes, &self.edges),
        }
    }

    /// Sugiyama's layer step: cut a greedy feedback arc set so the directed
    /// view is acyclic, then longest-path layering in topological order.
    pub fn layer_depths(&self) -> Vec<(NodeKey, u32)> {
        match &self.channels {
            Some(channels) => channels.depth_layers(&self.nodes, &self.edges),
            None => algo::layer_depths(&self.nodes, &self.edges),
        }
    }

    /// Dominator-tree depth from `focus`: the number of nodes every path from
    /// the focus must pass through. Unreachable nodes sit one level below the
    /// deepest reachable one; without a focus, the roots.
    pub fn focus_depths(&self, focus: Option<NodeKey>) -> Vec<(NodeKey, u32)> {
        let Some(focus) = focus else {
            return self.root_depths();
        };
        match &self.channels {
            Some(channels) => channels.depth_focus(&self.nodes, &self.edges, focus),
            None => algo::focus_depths(&self.nodes, &self.edges, focus),
        }
    }

    pub(super) fn depths(
        &self,
        source: PhysicsDepthSource,
        focus: Option<NodeKey>,
    ) -> Vec<(NodeKey, u32)> {
        match source {
            PhysicsDepthSource::Roots => self.root_depths(),
            PhysicsDepthSource::Layers => self.layer_depths(),
            PhysicsDepthSource::Focus => self.focus_depths(focus),
        }
    }

    /// The minimum spanning tree's edges over the undirected view (a pair
    /// with more relations is a shorter edge, so the tree prefers it):
    /// `edges.spanning`, Stress's skeleton.
    pub fn skeleton_edges(&self) -> Vec<(NodeKey, NodeKey)> {
        match &self.channels {
            Some(channels) => channels.edges_spanning(&self.nodes, &self.edges),
            None => algo::spanning_edges(&self.nodes, &self.edges),
        }
    }

    /// Shortest-path distances in hops over the undirected view, each hop
    /// costing `1 / multiplicity`, every connected pair once.
    pub fn weighted_distances(&self) -> Vec<(NodeKey, NodeKey, f32)> {
        match &self.channels {
            Some(channels) => channels.distances_hops(&self.nodes, &self.edges),
            None => algo::hop_distances(&self.nodes, &self.edges),
        }
    }
}
