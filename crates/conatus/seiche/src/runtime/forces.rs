// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

use super::*;

impl Physics {
    /// Reconcile the body set (see [`PhysicsCommand::SyncNodes`]).
    pub fn sync_nodes(&mut self, nodes: Vec<(NodeKey, Point2D<f32>)>) {
        match self {
            Physics::Inline(p) => p.sim.sync_nodes(nodes),
                        Physics::Actor(p) | Physics::Remote(p) => {
                p.link.command(PhysicsCommand::SyncNodes(nodes));
            },
        }
    }

    /// Reconcile after host-observed member exits, with a snapshot barrier on
    /// the actor path so pre-retirement layouts cannot reach replacement slots.
    pub fn sync_nodes_retiring(
        &mut self,
        nodes: Vec<(NodeKey, Point2D<f32>)>,
        retired: Vec<NodeKey>,
    ) {
        if retired.is_empty() {
            self.sync_nodes(nodes);
            return;
        }
        match self {
            Physics::Inline(p) => p.sim.sync_nodes_retiring(nodes, retired),
                        Physics::Actor(p) | Physics::Remote(p) => {
                p.coupling_trace = None;
                p.coupling_membership = None;
                p.link
                    .command(PhysicsCommand::SyncNodesRetiring { nodes, retired });
                p.barrier();
            },
        }
    }

    /// Replace the spring edge topology.
    pub fn sync_edges(&mut self, edges: Vec<(NodeKey, NodeKey)>) {
        self.sync_edge_topology(edges.into_iter().collect());
    }

    /// Replace counted topology on either backend, retaining row order.
    pub fn sync_edge_topology(&mut self, topology: EdgeTopology) {
        match self {
            Physics::Inline(p) => p.sim.sync_edge_topology(topology),
                        Physics::Actor(p) | Physics::Remote(p) => {
                p.link.command(PhysicsCommand::SyncEdgeTopology(topology));
            },
        }
    }

    /// Override existing bodies' positions (a seed / reseed).
    pub fn seed(&mut self, positions: Vec<(NodeKey, Point2D<f32>)>) {
        match self {
            Physics::Inline(p) => {
                p.sim.seed_positions(positions);
                p.pace.forget();
            },
                        Physics::Actor(p) | Physics::Remote(p) => {
                p.link.command(PhysicsCommand::Seed(positions));
                p.barrier();
            },
        }
    }

    /// Replace the live field-coupling forces wholesale — the rebuild a field place /
    /// move / resize / new node triggers (the host re-resolves every coupling against
    /// the current graph). Position-preserving (forces don't touch body state).
    /// (Field regions — rebuild-on-mutation.)
    pub fn set_coupling_forces(&mut self, forces: Vec<CouplingForce>) {
        match self {
            Physics::Inline(p) => p.sim.set_coupling_forces(forces),
                        Physics::Actor(p) | Physics::Remote(p) => {
                p.coupling_trace = None;
                p.coupling_membership = None;
                let bindings = forces
                    .iter()
                    .map(|force| std::sync::Arc::clone(&force.binding))
                    .collect();
                p.link.command(PhysicsCommand::SetCouplingForces(forces));
                p.barrier();
                p.entry.listed(p.command_epoch, bindings);
            },
        }
    }

    /// Enable or disable bounded live coupling observations. Clears old evidence;
    /// actor snapshots produced before this command are rejected by the barrier.
    pub fn set_coupling_trace_limit(&mut self, limit: Option<std::num::NonZeroUsize>) {
        match self {
            Physics::Inline(p) => p.sim.set_coupling_trace_limit(limit),
                        Physics::Actor(p) | Physics::Remote(p) => {
                p.coupling_trace = None;
                p.link
                    .command(PhysicsCommand::SetCouplingTraceLimit(limit));
                p.barrier();
            },
        }
    }

    /// Evidence from the latest executed coupling tick (latest folded snapshot
    /// offloaded). Call `refresh` to receive actor evidence. It is historical,
    /// not an evaluation of today's pose, and excludes post-integration effects.
    pub fn coupling_trace(&self) -> Option<&crate::CouplingTrace> {
        match self {
            Physics::Inline(p) => p.sim.coupling_trace(),
                        Physics::Actor(p) | Physics::Remote(p) => p.coupling_trace.as_ref(),
        }
    }

    /// Install (or clear, with `None`) the pairwise affinity force wholesale — the rebuild a fresh
    /// affinity signal triggers ("cluster by affinity"). Position-preserving (forces never touch
    /// body state). (Graph signals — P4.)
    pub fn set_affinity_force(&mut self, force: Option<AffinitySpring>) {
        match self {
            Physics::Inline(p) => p.sim.set_affinity_force(force),
                        Physics::Actor(p) | Physics::Remote(p) => {
                p.link.command(PhysicsCommand::SetAffinityForce(force));
            },
        }
    }

    /// Install (or clear, with `None`) per-node **anchor** springs toward the
    /// anchored items' arrangement positions (the anchored role, G7).
    /// Position-preserving.
    pub fn set_anchor_force(&mut self, force: Option<crate::AnchorSpring>) {
        match self {
            Physics::Inline(p) => p.sim.set_anchor_force(force),
                        Physics::Actor(p) | Physics::Remote(p) => {
                p.link.command(PhysicsCommand::SetAnchorForce(force));
            },
        }
    }

    /// Replace the law's force set wholesale (see [`PhysicsCommand::SetForces`]):
    /// the physics law plus its overlays. The coupling / affinity / anchor slots
    /// are separate and untouched. Position-preserving. (Physics catalog — P1.)
    pub fn set_forces(&mut self, forces: Vec<Box<dyn Force>>) {
        match self {
            Physics::Inline(p) => p.sim.set_forces(forces),
                        Physics::Actor(p) | Physics::Remote(p) => {
                p.link.command(PhysicsCommand::SetForces(forces));
            },
        }
    }

    /// [`Self::set_forces`] from a descriptor: the forces are built where the
    /// backend runs (`catalog::build_force_set`), here inline and on the actor's
    /// thread offloaded, so a host sends data and not boxed forces.
    pub fn set_force_set(&mut self, descriptor: seiche_model::ForceSetDescriptor) {
        match self {
            Physics::Inline(p) => p
                .sim
                .set_forces(crate::catalog::build_force_set(&descriptor)),
                        Physics::Actor(p) | Physics::Remote(p) => {
                p.link.command(PhysicsCommand::SetForceSet(descriptor));
            },
        }
    }

    /// The node bodies' kinetic energy: read live inline, and from the last
    /// folded snapshot offloaded, so both backends answer the same question.
    /// (Physics catalog — the receipts' energy floor.)
    pub fn kinetic_energy(&self) -> f32 {
        match self {
            Physics::Inline(p) => p.sim.kinetic_energy(),
                        Physics::Actor(p) | Physics::Remote(p) => p.energy,
        }
    }

    /// The number of forces in the law slot (inline backend only). Test introspection.
    pub fn force_count(&self) -> usize {
        match self {
            Physics::Inline(p) => p.sim.force_count(),
                        Physics::Actor(_) | Physics::Remote(_) => 0,
        }
    }

    /// The number of anchored nodes (inline backend only). Test introspection.
    pub fn anchor_count(&self) -> usize {
        match self {
            Physics::Inline(p) => p.sim.anchor_count(),
                        Physics::Actor(_) | Physics::Remote(_) => 0,
        }
    }

    /// The number of affinity pairs in the installed affinity force (inline backend only — the
    /// actor owns its sim on another thread). Test introspection for the P4 wiring. (Graph signals.)
    pub fn affinity_pair_count(&self) -> usize {
        match self {
            Physics::Inline(p) => p.sim.affinity_pair_count(),
                        Physics::Actor(_) | Physics::Remote(_) => 0,
        }
    }

    /// Install (or clear) a lagged evaluator for `NodeExclusion`, inline or
    /// on the actor. See [`Simulation::set_lagged_repulsion`].
    pub fn set_lagged_repulsion(
        &mut self,
        solver: Option<Box<dyn LaggedRepulsion>>,
        threshold: usize,
        max_stale_steps: u32,
    ) {
        match self {
            Physics::Inline(p) => p
                .sim
                .set_lagged_repulsion(solver, threshold, max_stale_steps),
                        Physics::Actor(p) | Physics::Remote(p) => {
                p.link.command(PhysicsCommand::SetLaggedRepulsion {
                    solver,
                    threshold,
                    max_stale_steps,
                });
            },
        }
    }

    /// The lagged lane's counts (inline backend; `None` offloaded, where the
    /// device's own counters are the read).
    pub fn repulsion_stats(&self) -> Option<LaggedStats> {
        match self {
            Physics::Inline(p) => p.sim.repulsion_stats(),
                        Physics::Actor(_) | Physics::Remote(_) => None,
        }
    }
}
