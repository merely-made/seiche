// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! Groups: an outer law between groups and an inner law within each.
//!
//! The grouped combinator (dynamics grammar plan, G3; P7c). The outer law
//! runs over one body per group, at the group's centroid, with the groups'
//! graph (a pair of groups joined when any edge crosses between them) as its
//! edges. Each member takes its weight share of its group's centroid force,
//! `F_g / n_g` (ruled 2026-10-02, F9 "Weight share"): the centroid is the
//! members' mean, so this is the true gradient of the outer energy, and the
//! group answers as one body of all its members' mass. The inner law is its
//! own instance per group, run under a membership mask: it sees only that
//! group's members and the edges inside it, so pairs and edges across groups
//! are the outer law's alone. Unary inner terms act on every member as usual.
//!
//! The first instance is the one Mark named: Charge between meaning
//! clusters, Springs within.
//!
//! *Reading, not ruled:* the outer law's bodies are the centroids, one each,
//! with no mass of their own, so an outer law that reads inertial mass
//! (Orbit) does not belong here.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};

use rapier2d::prelude::*;

use crate::{Declared, Force, ForceContext, Layout, NodeKey, Term, Topology};

/// How a group's centroid force reaches its members.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Spread {
    /// Each member takes `F_g / n_g` (F9, ruled).
    WeightShare,
    /// Each member takes all of `F_g`: F9's rejected option, kept only as the
    /// reciprocity instrument's positive control.
    #[cfg(test)]
    FullForce,
}

/// A partition of the nodes into groups, with the edges inside each group
/// and the groups' own graph. Groups are in id order, members in key order.
#[derive(Clone, Debug, Default)]
pub struct Partition {
    groups: Vec<(u32, Vec<NodeKey>)>,
    inner_edges: Vec<Vec<(NodeKey, NodeKey)>>,
    outer_edges: Vec<(NodeKey, NodeKey)>,
}

impl Partition {
    /// Each node's group, and the topology. A node with no group is left out.
    pub fn new(
        groups: impl IntoIterator<Item = (NodeKey, u32)>,
        edges: &[(NodeKey, NodeKey)],
    ) -> Self {
        let mut by: BTreeMap<u32, Vec<NodeKey>> = BTreeMap::new();
        for (key, group) in groups {
            by.entry(group).or_default().push(key);
        }
        let groups: Vec<(u32, Vec<NodeKey>)> = by
            .into_iter()
            .map(|(id, mut members)| {
                members.sort_by_key(|k| k.index());
                members.dedup();
                (id, members)
            })
            .collect();
        let of: HashMap<NodeKey, usize> = groups
            .iter()
            .enumerate()
            .flat_map(|(g, (_, members))| members.iter().map(move |&k| (k, g)))
            .collect();
        let mut inner_edges = vec![Vec::new(); groups.len()];
        let mut outer = std::collections::BTreeSet::new();
        for &(a, b) in edges {
            let (Some(&ga), Some(&gb)) = (of.get(&a), of.get(&b)) else {
                continue;
            };
            if a == b {
                continue;
            }
            if ga == gb {
                inner_edges[ga].push((a, b));
            } else {
                outer.insert((ga.min(gb), ga.max(gb)));
            }
        }
        let outer_edges = outer
            .into_iter()
            .map(|(a, b)| (NodeKey::new(a), NodeKey::new(b)))
            .collect();
        Self {
            groups,
            inner_edges,
            outer_edges,
        }
    }

    pub fn len(&self) -> usize {
        self.groups.len()
    }

    pub fn is_empty(&self) -> bool {
        self.groups.is_empty()
    }

    /// Group `g`'s members and the edges inside it.
    pub fn group(&self, g: usize) -> (&[NodeKey], &[(NodeKey, NodeKey)]) {
        (&self.groups[g].1, &self.inner_edges[g])
    }

    /// The groups' graph: one key per group (`NodeKey::new(g)`, in group
    /// order) and an edge for each pair of groups an edge crosses between.
    pub fn outer_graph(&self) -> (Vec<NodeKey>, Vec<(NodeKey, NodeKey)>) {
        (
            (0..self.groups.len()).map(NodeKey::new).collect(),
            self.outer_edges.clone(),
        )
    }
}

/// The forces, shared between a composition and its isolated terms.
struct Laws {
    outer: Vec<Box<dyn Force>>,
    inner: Vec<Vec<Box<dyn Force>>>,
}

/// The centroid bodies the outer law runs over.
struct Centroids {
    bodies: RigidBodySet,
    colliders: ColliderSet,
    joints: ImpulseJointSet,
    by_group: HashMap<NodeKey, RigidBodyHandle>,
}

/// Which of the composition's terms act: all of them, or one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Only {
    All,
    Outer(usize, usize),
    Inner(usize, usize),
}

/// An outer law between groups and an inner law within each. See the module docs.
pub struct Grouped {
    partition: Arc<Partition>,
    laws: Arc<Mutex<Laws>>,
    centroids: Arc<Mutex<Centroids>>,
    spread: Spread,
    only: Only,
}

impl Grouped {
    /// `outer` built over [`Partition::outer_graph`]; `inner` one law per
    /// group, in group order, each built over that group's members and edges.
    pub fn new(
        partition: Partition,
        outer: Vec<Box<dyn Force>>,
        inner: Vec<Vec<Box<dyn Force>>>,
    ) -> Self {
        assert_eq!(inner.len(), partition.len(), "one inner law per group");
        let mut bodies = RigidBodySet::new();
        let by_group = (0..partition.len())
            .map(|g| {
                let handle = bodies.insert(RigidBodyBuilder::dynamic().build());
                (NodeKey::new(g), handle)
            })
            .collect();
        Self {
            partition: Arc::new(partition),
            laws: Arc::new(Mutex::new(Laws { outer, inner })),
            centroids: Arc::new(Mutex::new(Centroids {
                bodies,
                colliders: ColliderSet::new(),
                joints: ImpulseJointSet::new(),
                by_group,
            })),
            spread: Spread::WeightShare,
            only: Only::All,
        }
    }

    /// The same composition with another spread rule.
    pub fn with_spread(mut self, spread: Spread) -> Self {
        self.spread = spread;
        self
    }

    pub fn partition(&self) -> &Partition {
        &self.partition
    }

    fn laws(&self) -> std::sync::MutexGuard<'_, Laws> {
        self.laws.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Group `g`'s members present now, keyed to their bodies.
    fn members(
        &self,
        g: usize,
        bodies_by_node: &HashMap<NodeKey, RigidBodyHandle>,
    ) -> HashMap<NodeKey, RigidBodyHandle> {
        self.partition
            .group(g)
            .0
            .iter()
            .filter_map(|k| Some((*k, *bodies_by_node.get(k)?)))
            .collect()
    }

    fn run_inner(
        &self,
        laws: &Laws,
        ctx: &mut ForceContext<'_>,
        dt: f32,
        only: Option<(usize, usize)>,
    ) {
        for g in 0..self.partition.len() {
            let members = self.members(g, ctx.bodies_by_node);
            if members.is_empty() {
                continue;
            }
            let mut masked = ForceContext {
                bodies: &mut *ctx.bodies,
                colliders: ctx.colliders,
                joints: &mut *ctx.joints,
                bodies_by_node: &members,
                edges: self.partition.group(g).1,
                repulsion: None,
                step: ctx.step,
            };
            apply_terms(&laws.inner[g], &mut masked, dt, only);
        }
    }

    fn run_outer(
        &self,
        laws: &Laws,
        ctx: &mut ForceContext<'_>,
        dt: f32,
        only: Option<(usize, usize)>,
    ) {
        let groups: Vec<Vec<(RigidBodyHandle, Vector, Vector)>> = (0..self.partition.len())
            .map(|g| {
                let mut members: Vec<_> = self
                    .members(g, ctx.bodies_by_node)
                    .into_iter()
                    .filter_map(|(k, h)| {
                        let b = ctx.bodies.get(h)?;
                        Some((k.index(), h, b.translation(), b.linvel()))
                    })
                    .collect();
                members.sort_by_key(|m| m.0);
                members.into_iter().map(|(_, h, x, v)| (h, x, v)).collect()
            })
            .collect();
        let mut world = self.centroids.lock().unwrap_or_else(|p| p.into_inner());
        let world = &mut *world;
        for (g, members) in groups.iter().enumerate() {
            let n = members.len().max(1) as f32;
            let (x, v) = members
                .iter()
                .fold((Vector::ZERO, Vector::ZERO), |(x, v), m| (x + m.1, v + m.2));
            if let Some(body) = world
                .by_group
                .get(&NodeKey::new(g))
                .and_then(|h| world.bodies.get_mut(*h))
            {
                body.reset_forces(false);
                body.set_translation(x / n, false);
                body.set_linvel(v / n, false);
            }
        }
        let (_, outer_edges) = self.partition.outer_graph();
        let mut outer_ctx = ForceContext {
            bodies: &mut world.bodies,
            colliders: &world.colliders,
            joints: &mut world.joints,
            bodies_by_node: &world.by_group,
            edges: &outer_edges,
            repulsion: None,
            step: ctx.step,
        };
        apply_terms(&laws.outer, &mut outer_ctx, dt, only);
        for (g, members) in groups.iter().enumerate() {
            let Some(force) = world
                .by_group
                .get(&NodeKey::new(g))
                .and_then(|h| world.bodies.get(*h))
                .map(|b| b.user_force())
            else {
                continue;
            };
            if members.is_empty() {
                continue;
            }
            let share = match self.spread {
                Spread::WeightShare => force / members.len() as f32,
                #[cfg(test)]
                Spread::FullForce => force,
            };
            for (handle, _, _) in members {
                if let Some(body) = ctx.bodies.get_mut(*handle) {
                    body.add_force(share, true);
                }
            }
        }
    }

    /// Outer term `t` as (force, term within it), or inner likewise.
    fn locate(&self, laws: &Laws, term: usize) -> Only {
        let outer = flat_terms(&laws.outer);
        if term < outer.len() {
            let (f, t) = outer[term];
            return Only::Outer(f, t);
        }
        let inner = laws
            .inner
            .first()
            .map(|l| flat_terms(l))
            .unwrap_or_default();
        match inner.get(term - outer.len()) {
            Some(&(f, t)) => Only::Inner(f, t),
            None => Only::All,
        }
    }

    /// Group `g`'s members as a layout, in key order.
    fn group_layout(&self, g: usize, layout: &Layout<'_>) -> Vec<(NodeKey, Vector)> {
        let at: HashMap<NodeKey, Vector> = layout.nodes.iter().copied().collect();
        self.partition
            .group(g)
            .0
            .iter()
            .filter_map(|k| Some((*k, *at.get(k)?)))
            .collect()
    }
}

/// Every term of a force list as (force index, term within it).
fn flat_terms(forces: &[Box<dyn Force>]) -> Vec<(usize, usize)> {
    forces
        .iter()
        .enumerate()
        .flat_map(|(f, force)| (0..force.terms().len()).map(move |t| (f, t)))
        .collect()
}

/// Apply every force, or only term `t` of force `f`.
fn apply_terms(
    forces: &[Box<dyn Force>],
    ctx: &mut ForceContext<'_>,
    dt: f32,
    only: Option<(usize, usize)>,
) {
    match only {
        None => {
            for force in forces {
                force.apply(ctx, dt);
            }
        },
        Some((f, t)) => match forces[f].isolate(t) {
            Some(alone) => alone.apply(ctx, dt),
            None => forces[f].apply(ctx, dt),
        },
    }
}

impl Force for Grouped {
    fn apply(&self, ctx: &mut ForceContext<'_>, dt: f32) {
        let laws = self.laws();
        match self.only {
            Only::All => {
                self.run_inner(&laws, ctx, dt, None);
                self.run_outer(&laws, ctx, dt, None);
            },
            Only::Outer(f, t) => self.run_outer(&laws, ctx, dt, Some((f, t))),
            Only::Inner(f, t) => self.run_inner(&laws, ctx, dt, Some((f, t))),
        }
    }

    fn wants_tick(&self) -> bool {
        let laws = self.laws();
        laws.outer.iter().any(|f| f.wants_tick())
            || laws.inner.iter().flatten().any(|f| f.wants_tick())
    }
}

/// The outer terms over groups, then the inner law's terms (every group's
/// inner law is the same law).
impl Declared for Grouped {
    fn terms(&self) -> Vec<Term> {
        let laws = self.laws();
        // Between groups an outer pair or edge term is groups against groups;
        // an outer unary term still pulls each group alone.
        let outer = laws.outer.iter().flat_map(|f| f.terms()).map(|mut term| {
            if term.topology.is_internal() {
                term.topology = Topology::Groups;
            }
            term
        });
        let inner = laws
            .inner
            .first()
            .map(|l| l.iter().flat_map(|f| f.terms()).collect::<Vec<_>>())
            .unwrap_or_default();
        let all: Vec<Term> = outer.chain(inner).collect();
        match self.only {
            Only::All => all,
            only => (0..all.len())
                .find(|&i| self.locate(&laws, i) == only)
                .map(|i| vec![all[i]])
                .unwrap_or_default(),
        }
    }

    fn isolate(&self, term: usize) -> Option<Box<dyn Force>> {
        if self.only != Only::All {
            return None;
        }
        let only = self.locate(&self.laws(), term);
        (only != Only::All).then(|| {
            Box::new(Grouped {
                partition: self.partition.clone(),
                laws: self.laws.clone(),
                centroids: self.centroids.clone(),
                spread: self.spread,
                only,
            }) as Box<dyn Force>
        })
    }

    /// Outer: the outer term's energy at the centroids. Inner: the sum over
    /// groups of each group's inner term's energy over its members.
    fn energy(&self, term: usize, layout: &Layout<'_>) -> Option<f64> {
        let laws = self.laws();
        let only = match self.only {
            Only::All => self.locate(&laws, term),
            fixed => fixed,
        };
        match only {
            Only::All => None,
            Only::Outer(f, t) => {
                let centroids: Vec<(NodeKey, Vector)> = (0..self.partition.len())
                    .map(|g| {
                        let members = self.group_layout(g, layout);
                        let n = members.len().max(1) as f32;
                        let c = members.iter().fold(Vector::ZERO, |c, m| c + m.1) / n;
                        (NodeKey::new(g), c)
                    })
                    .collect();
                let (_, edges) = self.partition.outer_graph();
                laws.outer[f].energy(
                    t,
                    &Layout {
                        nodes: &centroids,
                        edges: &edges,
                    },
                )
            },
            Only::Inner(f, t) => (0..self.partition.len())
                .map(|g| {
                    let nodes = self.group_layout(g, layout);
                    laws.inner[g][f].energy(
                        t,
                        &Layout {
                            nodes: &nodes,
                            edges: self.partition.group(g).1,
                        },
                    )
                })
                .sum(),
        }
    }
}

#[cfg(test)]
mod tests;
