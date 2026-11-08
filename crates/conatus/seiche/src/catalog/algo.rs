// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! The physics catalog's graph algorithms (dynamics grammar plan, G2c; F21, F149):
//! `mass.degree`, `mass.pagerank`, the colouring, island and degree-band groups,
//! the three depths, `distances.hops` and `edges.spanning`.
//!
//! Each producer reads a node list (in key order) and an edge list, one tuple
//! per visible relation cell. The algorithms run over a petgraph view of the
//! edges: the directed multigraph for PageRank, layering, dominators and
//! components; the undirected simple graph (one edge per pair, cost
//! `1 / multiplicity`) for the colouring, the spanning tree and the shortest
//! paths. They are kernel-free and need only petgraph, so a grouped law can
//! build per-subgroup forces from the subgroup's own inputs on the engine's
//! side. A host that caches them (Pictograph's channel registry) names them
//! through [`super::Channels`].

use std::collections::{HashMap, HashSet, VecDeque};

use petgraph::Direction;
use petgraph::algo::{
    dijkstra, dominators, dsatur_coloring, greedy_feedback_arc_set, min_spanning_tree, page_rank,
    tarjan_scc, toposort,
};
use petgraph::data::Element;
use petgraph::graph::{DiGraph, EdgeIndex, NodeIndex, UnGraph};
use petgraph::visit::EdgeRef;

use crate::NodeKey;

/// PageRank's damping and iteration budget.
const PAGE_RANK_DAMPING: f32 = 0.85;
const PAGE_RANK_ITERATIONS: usize = 50;

/// The petgraph views of a node and edge list.
struct Topology {
    directed: DiGraph<NodeKey, f32>,
    undirected: UnGraph<NodeKey, f32>,
    index_of: HashMap<NodeKey, NodeIndex>,
}

impl Topology {
    fn new(nodes: &[NodeKey], edges: &[(NodeKey, NodeKey)]) -> Self {
        let mut directed = DiGraph::new();
        let mut undirected = UnGraph::new_undirected();
        let mut index_of = HashMap::with_capacity(nodes.len());
        for &key in nodes {
            let d = directed.add_node(key);
            let u = undirected.add_node(key);
            debug_assert_eq!(d, u);
            index_of.insert(key, d);
        }
        let mut multiplicity: HashMap<(NodeIndex, NodeIndex), u32> = HashMap::new();
        for &(a, b) in edges {
            let (Some(&ia), Some(&ib)) = (index_of.get(&a), index_of.get(&b)) else {
                continue;
            };
            if ia == ib {
                continue;
            }
            directed.add_edge(ia, ib, 1.0);
            let pair = if ia < ib { (ia, ib) } else { (ib, ia) };
            *multiplicity.entry(pair).or_default() += 1;
        }
        let mut pairs: Vec<_> = multiplicity.into_iter().collect();
        pairs.sort_by_key(|((a, b), _)| (a.index(), b.index()));
        for ((a, b), m) in pairs {
            undirected.add_edge(a, b, 1.0 / m as f32);
        }
        Self {
            directed,
            undirected,
            index_of,
        }
    }

    fn key(&self, index: NodeIndex) -> NodeKey {
        self.directed[index]
    }
}

fn degrees(edges: &[(NodeKey, NodeKey)]) -> HashMap<NodeKey, u32> {
    let mut degree: HashMap<NodeKey, u32> = HashMap::new();
    for (a, b) in edges {
        *degree.entry(*a).or_default() += 1;
        *degree.entry(*b).or_default() += 1;
    }
    degree
}

/// `mass.degree`: each node's degree over the edges, a pair joined by two
/// relations counting twice (F147).
pub fn mass_degree(nodes: &[NodeKey], edges: &[(NodeKey, NodeKey)]) -> Vec<(NodeKey, f32)> {
    let degree = degrees(edges);
    nodes
        .iter()
        .map(|&key| (key, degree.get(&key).copied().unwrap_or(0) as f32))
        .collect()
}

/// `mass.pagerank`: PageRank over the directed view, scaled so the mean
/// weight is one (comparable to log degree on a sparse graph).
pub fn page_rank_weights(nodes: &[NodeKey], edges: &[(NodeKey, NodeKey)]) -> Vec<(NodeKey, f32)> {
    let view = Topology::new(nodes, edges);
    let n = view.directed.node_count();
    if n == 0 {
        return Vec::new();
    }
    let ranks = page_rank(&view.directed, PAGE_RANK_DAMPING, PAGE_RANK_ITERATIONS);
    nodes
        .iter()
        .map(|&key| {
            let rank = view
                .index_of
                .get(&key)
                .and_then(|index| ranks.get(index.index()))
                .copied()
                .unwrap_or(0.0);
            (key, rank * n as f32)
        })
        .collect()
}

/// `groups.coloring`: a proper colouring (DSATUR), adjacent nodes never
/// sharing a group.
pub fn coloring_groups(nodes: &[NodeKey], edges: &[(NodeKey, NodeKey)]) -> Vec<(NodeKey, u32)> {
    let view = Topology::new(nodes, edges);
    let (colors, _) = dsatur_coloring(&view.undirected);
    nodes
        .iter()
        .map(|&key| {
            let color = view
                .index_of
                .get(&key)
                .and_then(|index| colors.get(index))
                .copied()
                .unwrap_or(0);
            (key, color as u32)
        })
        .collect()
}

/// `groups.component`: every node's connected component (island), as a
/// dense id.
pub fn component_groups(nodes: &[NodeKey], edges: &[(NodeKey, NodeKey)]) -> Vec<(NodeKey, u32)> {
    let view = Topology::new(nodes, edges);
    // The strongly-connected components of an undirected graph are its
    // connected components.
    let mut of: HashMap<NodeKey, u32> = HashMap::new();
    let mut components = tarjan_scc(&view.undirected);
    components.sort_by_key(|members| members.iter().map(|i| i.index()).min());
    for (i, members) in components.iter().enumerate() {
        for &member in members {
            of.insert(view.key(member), i as u32);
        }
    }
    nodes
        .iter()
        .map(|&key| (key, of.get(&key).copied().unwrap_or(0)))
        .collect()
}

/// `groups.degree`: degree bands, isolated, leaf, connected, hub.
pub fn degree_bands(nodes: &[NodeKey], edges: &[(NodeKey, NodeKey)]) -> Vec<(NodeKey, u32)> {
    let degree = degrees(edges);
    nodes
        .iter()
        .map(|&key| {
            let d = degree.get(&key).copied().unwrap_or(0);
            (
                key,
                match d {
                    0 => 0,
                    1 => 1,
                    2..=4 => 2,
                    _ => 3,
                },
            )
        })
        .collect()
}

/// `depth.roots`: BFS depth from the roots, the nodes with no incoming edge,
/// or every node when the graph is all cycles.
pub fn root_depths(nodes: &[NodeKey], edges: &[(NodeKey, NodeKey)]) -> Vec<(NodeKey, u32)> {
    let mut incoming: HashSet<NodeKey> = HashSet::new();
    let mut out: HashMap<NodeKey, Vec<NodeKey>> = HashMap::new();
    for (a, b) in edges {
        incoming.insert(*b);
        out.entry(*a).or_default().push(*b);
    }
    let mut roots: Vec<NodeKey> = nodes
        .iter()
        .copied()
        .filter(|k| !incoming.contains(k))
        .collect();
    if roots.is_empty() {
        roots = nodes.to_vec();
    }
    let mut depth: HashMap<NodeKey, u32> = HashMap::new();
    let mut queue: VecDeque<NodeKey> = VecDeque::new();
    for root in roots {
        depth.insert(root, 0);
        queue.push_back(root);
    }
    while let Some(key) = queue.pop_front() {
        let d = depth[&key];
        if let Some(children) = out.get(&key) {
            for &child in children {
                if !depth.contains_key(&child) {
                    depth.insert(child, d + 1);
                    queue.push_back(child);
                }
            }
        }
    }
    // A node reachable only through a cycle off the roots gets depth one,
    // so nothing is left unplaced.
    nodes
        .iter()
        .map(|&key| (key, depth.get(&key).copied().unwrap_or(1)))
        .collect()
}

/// `depth.layers`: Sugiyama's layer step, a greedy feedback arc set cut so
/// the directed view is acyclic, then longest-path layering in topological
/// order.
pub fn layer_depths(nodes: &[NodeKey], edges: &[(NodeKey, NodeKey)]) -> Vec<(NodeKey, u32)> {
    let view = Topology::new(nodes, edges);
    let mut dag = view.directed.clone();
    let cut: HashSet<EdgeIndex> = greedy_feedback_arc_set(&dag).map(|e| e.id()).collect();
    dag.retain_edges(|_, e| !cut.contains(&e));
    let Ok(order) = toposort(&dag, None) else {
        return root_depths(nodes, edges);
    };
    let mut layer: HashMap<NodeIndex, u32> = HashMap::with_capacity(order.len());
    for v in order {
        let l = dag
            .neighbors_directed(v, Direction::Incoming)
            .filter_map(|u| layer.get(&u).map(|d| d + 1))
            .max()
            .unwrap_or(0);
        layer.insert(v, l);
    }
    nodes
        .iter()
        .map(|&key| {
            let d = view
                .index_of
                .get(&key)
                .and_then(|index| layer.get(index))
                .copied()
                .unwrap_or(0);
            (key, d)
        })
        .collect()
}

/// `depth.focus`: dominator-tree depth from `focus`, the number of nodes
/// every path from the focus must pass through. Unreachable nodes sit one
/// level below the deepest reachable one; a focus outside the view reads
/// the roots' depths.
pub fn focus_depths(
    nodes: &[NodeKey],
    edges: &[(NodeKey, NodeKey)],
    focus: NodeKey,
) -> Vec<(NodeKey, u32)> {
    let view = Topology::new(nodes, edges);
    let Some(&root) = view.index_of.get(&focus) else {
        return root_depths(nodes, edges);
    };
    let dom = dominators::simple_fast(&view.directed, root);
    let mut depth: HashMap<NodeKey, u32> = HashMap::new();
    let mut deepest = 0;
    for &key in nodes {
        let Some(index) = view.index_of.get(&key) else {
            continue;
        };
        if let Some(chain) = dom.dominators(*index) {
            let d = chain.count().saturating_sub(1) as u32;
            deepest = deepest.max(d);
            depth.insert(key, d);
        }
    }
    nodes
        .iter()
        .map(|&key| (key, depth.get(&key).copied().unwrap_or(deepest + 1)))
        .collect()
}

/// `edges.spanning`: the minimum spanning tree's edges over the undirected
/// view (a pair with more relations is a shorter edge, so the tree prefers
/// it), in the order the tree yields them.
pub fn spanning_edges(nodes: &[NodeKey], edges: &[(NodeKey, NodeKey)]) -> Vec<(NodeKey, NodeKey)> {
    let view = Topology::new(nodes, edges);
    min_spanning_tree(&view.undirected)
        .filter_map(|element| match element {
            Element::Edge { source, target, .. } => Some((
                view.key(NodeIndex::new(source)),
                view.key(NodeIndex::new(target)),
            )),
            Element::Node { .. } => None,
        })
        .collect()
}

/// `distances.hops`: shortest-path distances over the undirected view, each
/// hop costing `1 / multiplicity`, every connected pair once.
pub fn hop_distances(
    nodes: &[NodeKey],
    edges: &[(NodeKey, NodeKey)],
) -> Vec<(NodeKey, NodeKey, f32)> {
    let view = Topology::new(nodes, edges);
    let mut out = Vec::new();
    for (i, &a) in nodes.iter().enumerate() {
        let Some(&ia) = view.index_of.get(&a) else {
            continue;
        };
        let reach = dijkstra(&view.undirected, ia, None, |e| *e.weight());
        for &b in &nodes[i + 1..] {
            if let Some(d) = view.index_of.get(&b).and_then(|ib| reach.get(ib)) {
                out.push((a, b, *d));
            }
        }
    }
    out
}

#[cfg(all(test, feature = "engine"))]
mod tests;
