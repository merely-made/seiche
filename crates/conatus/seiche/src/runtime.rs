// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! The runtime: where a [`Simulation`] actually ticks.
//!
//! A [`Simulation`] integrates when someone calls [`Simulation::tick`]; *who*
//! calls it, and on which thread, is a separate decision every host faces.
//! This module is that decision, made once: two shapes behind one [`Physics`]
//! interface, so a host drives physics without owning a frame loop, a thread,
//! or a settle budget of its own. (Lifted out of `mere-canvas` 2026-09-04, so
//! the canvas, the remote board, and any other consumer share one backend.)
//!
//! - [`Physics::Inline`] — the simulation ticks **in the frame loop**, on the
//!   caller's thread. Deterministic (no thread race), so it is what tests
//!   drive; it is also the path for the no-threads `wasm32` browser/PWA
//!   profile, where an OS thread cannot be spawned.
//! - `Physics::Actor` — the simulation runs on an [`armillary`] **actor
//!   thread** (always-offload), behind the default `actor` feature. A heavy
//!   settle (`pipeline.step` over hundreds of bodies) never blocks compositing
//!   or input; the actor emits a [`LayoutSnapshot`] per step and the host folds
//!   it into its [`LayoutView`].
//!
//! Native builds construct [`Physics::Inline`] and immediately
//! `offload` onto an actor (so native always offloads); the host supplies the
//! wake that pokes its event loop when a snapshot lands. Either way the host
//! reads only its [`LayoutView`] — the backend just feeds authoritative
//! positions into it.

#[cfg(feature = "actor")]
use std::sync::mpsc::Receiver;

#[cfg(all(test, feature = "actor"))]
use armillary::Emitter;
#[cfg(feature = "actor")]
use armillary::{ActorHandle, Wake, spawn};
use euclid::default::Point2D;

use crate::{
    AffinitySpring, Basin, CouplingForce, EdgeTopology, FluidParams, LayoutSnapshot, LayoutView,
    NodeCollider, NodeKey, NodeMaterial, SceneEmitter, SceneField, SceneSpec,
};
#[cfg(feature = "engine")]
use crate::{Force, LaggedRepulsion, LaggedStats, Simulation};

pub use seiche_model::pace::TICK_DT;

mod data_command;
mod elapsed;
pub use elapsed::{ElapsedStepConfig, ElapsedStepReport, TICK_DURATION};
#[cfg(feature = "engine")]
mod actor;
mod speed;
#[cfg(feature = "actor")]
#[cfg(feature = "engine")]
use actor::{ActorState, run};
#[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
pub use speed::monotonic_clock;
pub use speed::{
    DEFAULT_BUDGET_SHARE, FALLBACK_DISPLAY_PERIOD, PaceStats, Speed, StepBudget, display_period,
};

mod entry_relay;
mod remote;
pub use remote::Transport;
#[cfg(feature = "engine")]
mod worker;
#[cfg(feature = "engine")]
pub use worker::PhysicsWorker;
#[cfg(feature = "actor")]
mod coupling_trace_delivery;
#[cfg(feature = "actor")]
mod membership_delivery;
#[cfg(feature = "actor")]
pub use membership_delivery::CouplingMembershipDelivery;
#[cfg(all(test, feature = "actor"))]
mod coupling_trace_tests;
mod field_runtime;
#[cfg(all(test, feature = "actor"))]
mod field_runtime_tests;
#[cfg(all(test, feature = "actor"))]
mod entry_relay_tests;
#[cfg(all(test, feature = "engine"))]
mod pause_tests;
#[cfg(all(test, feature = "engine"))]
mod remote_tests;
#[cfg(all(test, feature = "actor"))]
mod retirement_tests;
#[cfg(feature = "actor")]
pub use coupling_trace_delivery::CouplingTraceDelivery;

/// A command the host sends to the physics actor. Mirrors the mutating surface
/// of [`Simulation`] plus the settle/drag drivers; all payloads are `Send`
/// (positions and node keys), so the graph itself never crosses the boundary.
pub enum PhysicsCommand {
    /// Reconcile the body set to exactly these `(node, position)` pairs (position
    /// used only for newly-spawned bodies). See [`Simulation::sync_nodes`].
    SyncNodes(Vec<(NodeKey, Point2D<f32>)>),
    /// Atomically end the retired body lives before reconciling replacements.
    SyncNodesRetiring {
        nodes: Vec<(NodeKey, Point2D<f32>)>,
        retired: Vec<NodeKey>,
    },
    /// Replace the spring edge topology.
    SyncEdges(Vec<(NodeKey, NodeKey)>),
    /// Replace ordered counted topology without expanding the actor payload.
    SyncEdgeTopology(EdgeTopology),
    /// Override the positions of existing bodies (a seed / reseed).
    Seed(Vec<(NodeKey, Point2D<f32>)>),
    ResetFieldRuntime(Vec<(NodeKey, Point2D<f32>)>),
    /// Run an entry handler on the simulation's own thread (the inline path;
    /// a closure, so it cannot cross a message boundary). The data form is
    /// [`Self::SetCouplingEntryRelay`].
    SetCouplingEntryHandler(Option<crate::CouplingEntryHandler>),
    /// Report coupling entries as `CouplingEntryEvent`s in the update stream
    /// instead of running a handler here; the owner's side runs its handler
    /// when the update lands.
    SetCouplingEntryRelay(bool),
    /// Acknowledge all preceding commands in subsequent snapshots. The host
    /// uses this epoch to reject layouts produced before its latest reseed.
    Barrier(u64),
    /// Pin a node to a world position (a drag in progress).
    Pin(NodeKey, Point2D<f32>),
    /// Release a pinned node back to dynamic.
    Unpin(NodeKey),
    /// Extend the settle budget to at least `n` ticks.
    Settle(u32),
    /// Stop all current stepping, including perpetual scenes and dragging.
    /// A later positive Settle resumes; a new drag ticks only while held.
    Halt,
    /// Whether a drag is in progress (keep ticking so neighbors react).
    SetDragging(bool),
    /// Replace the live field-coupling forces wholesale (the rebuild a field place /
    /// move / resize / new node triggers), without a position-losing sim rebuild.
    /// (Field regions — rebuild-on-mutation.)
    SetCouplingForces(Vec<CouplingForce>),
    /// Observe the latest coupling tick with a caller-chosen row cap.
    SetCouplingTraceLimit(Option<std::num::NonZeroUsize>),
    /// Install (or clear) the pairwise affinity force — the rebuild a fresh affinity signal
    /// triggers ("cluster by affinity"). Position-preserving. (Graph signals — P4.)
    SetAffinityForce(Option<AffinitySpring>),
    SetAnchorForce(Option<crate::AnchorSpring>),
    /// Replace the law's force set wholesale — the physics law and its overlays
    /// (the catalog's `laws × overlays`), leaving the coupling / affinity / anchor
    /// slots alone. Position-preserving: no body moves until the next tick.
    /// (Physics catalog — P1.)
#[cfg(feature = "engine")]
    SetForces(Vec<Box<dyn Force>>),
    /// [`Self::SetForces`] from data: the law slot and overlays as a
    /// [`ForceSetDescriptor`](seiche_model::ForceSetDescriptor), built into forces on
    /// the thread the backend runs (`catalog::build_force_set`).
    SetForceSet(seiche_model::ForceSetDescriptor),
    /// Install (or clear) a lagged evaluator for `NodeExclusion` (see
    /// [`Simulation::set_lagged_repulsion`]): how a host's device reaches a
    /// simulation that is already offloaded. (Physics catalog P5b.)
#[cfg(feature = "engine")]
    SetLaggedRepulsion {
        solver: Option<Box<dyn LaggedRepulsion>>,
        threshold: usize,
        max_stale_steps: u32,
    },
    /// Set the linear damping (the "inertia" physics setting) on new + live bodies.
    SetLinearDamping(f32),
    /// Reshape node colliders to per-node face shapes (see [`Simulation::set_node_colliders`]).
    SetNodeColliders(Vec<(NodeKey, NodeCollider)>),
    /// Set per-node physical materials (restitution / friction / density; see
    /// [`Simulation::set_node_materials`]). (Node body & face — material.)
    SetNodeMaterials(Vec<(NodeKey, NodeMaterial)>),
    /// Hold node bodies on axes (see [`Simulation::set_axis_locks`]). (G7.)
    SetAxisLocks(Vec<(NodeKey, crate::Axes)>),
    /// Add a non-graph scene-decoration body (shape, world position, drift velocity).
    /// (Physics scenes P1.)
    AddSceneBody(NodeCollider, Point2D<f32>, (f32, f32)),
    /// Set every node's tangibility (collide with scene bodies, or pass through).
    /// (Physics scenes P2.)
    SetNodesTangible(bool),
    /// Load a declarative scene (clears any prior scene). (Physics scenes P3.)
    LoadScene(SceneSpec),
    /// Remove every scene body. (Physics scenes P3.)
    ClearScene,
    /// Replace static node-contact obstacles independently of scene bodies.
    SetStaticObstacles(Vec<crate::StaticObstacle>),
    /// Load a liquid pool: PBF params, basin, and a `cols × rows` spawn block at `spacing` from
    /// `origin`. (Physics scenes P4c.)
    LoadFluid {
        params: FluidParams,
        basin: Basin,
        origin: Point2D<f32>,
        cols: usize,
        rows: usize,
        spacing: f32,
    },
    /// Remove the liquid pool. (Physics scenes P4c.)
    ClearFluid,
    /// Set (or clear) the scene force-field (whirlpool / well). (Physics scenes P4 fields.)
    SetSceneField(Option<SceneField>),
    /// Add a continuous body emitter (fountain / stream). (Physics scenes — emitters.)
    AddEmitter(SceneEmitter),
    /// Remove every emitter + its bodies. (Physics scenes — emitters.)
    ClearEmitters,
    /// Set the simulation speed (see [`Speed`]).
    SetSpeed(Speed),
    /// Bound the actor's compute per interval above real time; `None` is the
    /// interval itself. The actor reads its own clock.
    SetStepBudget(Option<seiche_model::StepBudgetSpec>),
}

/// One layout the actor produced: the positions plus whether it is still
/// settling (so the host knows to keep requesting frames).
pub struct PhysicsUpdate {
    pub field_clock: crate::FieldClock,
    #[cfg(feature = "actor")]
    pub coupling_membership: Option<CouplingMembershipDelivery>,
    #[cfg(feature = "actor")]
    pub coupling_trace: Option<CouplingTraceDelivery>,
    /// The coupling entries observed since the previous update, when the
    /// relay is on (`SetCouplingEntryRelay`), for the host to run its handler
    /// on. Every one is delivered, including those of an update a later one
    /// supersedes.
    pub coupling_entries: Vec<seiche_model::CouplingEntryEvent>,
    pub snapshot: LayoutSnapshot,
    pub settling: bool,
    pub command_epoch: u64,
    pub pace: PaceStats,
}

#[cfg(feature = "engine")]
/// The in-thread backend: the simulation plus the settle/drag state the frame
/// loop steps it with.
pub struct InlinePhysics {
    sim: Simulation,
    ticks_remaining: u32,
    dragging: bool,
    generation: u64,
    halted: bool,
    pace: speed::Pace,
}

/// How the host reaches the far side of an offloaded backend: the actor thread's
/// handle and update channel, or a [`Transport`] to a Worker.
pub(super) enum Link {
    #[cfg(feature = "actor")]
    Thread {
        handle: ActorHandle<PhysicsCommand>,
        updates: Receiver<PhysicsUpdate>,
    },
    Remote(remote::RemoteLink),
}

impl Link {
    /// Send a command to the far side.
    pub(super) fn command(&mut self, command: PhysicsCommand) {
        match self {
            #[cfg(feature = "actor")]
            Link::Thread { handle, .. } => {
                handle.command(command);
            },
            Link::Remote(link) => link.command(command),
        }
    }

    /// Every update that has arrived, oldest first.
    pub(super) fn drain(&mut self) -> Vec<PhysicsUpdate> {
        match self {
            #[cfg(feature = "actor")]
            Link::Thread { updates, .. } => std::iter::from_fn(|| updates.try_recv().ok()).collect(),
            Link::Remote(link) => link.drain(),
        }
    }
}

/// The offloaded backend ([`Physics::Actor`] on a thread, [`Physics::Remote`]
/// over a transport): the link to the far side, and what the host last saw of it.
pub struct ActorPhysics {
    field_clock: crate::FieldClock,
    coupling_membership: Option<crate::CouplingMembershipFrame>,
    coupling_trace: Option<crate::CouplingTrace>,
    link: Link,
    settling: bool,
    /// The kinetic energy the last folded snapshot carried.
    energy: f32,
    /// The rms speed the last folded snapshot carried.
    speed: f32,
    command_epoch: u64,
    /// The simulation speed set (the dial), not the bodies' rms speed.
    dial: Speed,
    /// The budget the host last set, as it gave it.
    budget: Option<StepBudget>,
    /// The pace the last folded snapshot carried.
    pace: PaceStats,
    /// The owner's entry handler and the coupling lists it resolves events by.
    entry: entry_relay::EntryRelay,
}

impl ActorPhysics {
    fn barrier(&mut self) {
        self.command_epoch = self
            .command_epoch
            .checked_add(1)
            .expect("physics epoch exhausted");
        self.link
            .command(PhysicsCommand::Barrier(self.command_epoch));
    }

    /// The freshest update, and every coupling entry received, in order.
    fn latest(&mut self) -> (Option<PhysicsUpdate>, Vec<seiche_model::CouplingEntryEvent>) {
        let mut latest = None;
        let mut entries = Vec::new();
        for mut update in self.link.drain() {
            entries.append(&mut update.coupling_entries);
            self.entry.acknowledged(update.command_epoch);
            if update.command_epoch >= self.command_epoch {
                latest = Some(update);
            }
        }
        (latest, entries)
    }
}

#[cfg(feature = "engine")]
fn should_tick(sim: &Simulation, ticks: u32, dragging: bool, halted: bool) -> bool {
    dragging || (!halted && (ticks > 0 || sim.wants_continuous_tick()))
}

/// The physics backend a host talks to. Inline by default (tests +
/// wasm); `offload` swaps in the actor thread (feature `actor`).
pub enum Physics {
#[cfg(feature = "engine")]
    Inline(InlinePhysics),
    /// The simulation on an actor thread (only built with the `actor` feature).
    Actor(ActorPhysics),
    /// The simulation on the far side of a [`Transport`]: a Web Worker. Built
    /// from scratch by [`Physics::remote`]; the host sends it data commands and
    /// folds the updates that come back.
    Remote(ActorPhysics),
}

impl Physics {
    /// Set the linear damping (the "inertia" physics setting) on new + live node
    /// bodies. (Physics settings.)
    pub fn set_linear_damping(&mut self, damping: f32) {
        match self {
            #[cfg(feature = "engine")]
            Physics::Inline(p) => p.sim.set_linear_damping(damping),
                        Physics::Actor(p) | Physics::Remote(p) => {
                p.link.command(PhysicsCommand::SetLinearDamping(damping));
            },
        }
    }

    /// Reshape node colliders so each picks/collides at its true face shape + size (the
    /// host maps every node's silhouette / sprite hull to a [`NodeCollider`]). (P0/P5
    /// collider — Decision 5; node-rep — collider matches shape.)
    pub fn set_node_colliders(&mut self, colliders: Vec<(NodeKey, NodeCollider)>) {
        match self {
            #[cfg(feature = "engine")]
            Physics::Inline(p) => p.sim.set_node_colliders(colliders),
                        Physics::Actor(p) | Physics::Remote(p) => {
                p.link
                    .command(PhysicsCommand::SetNodeColliders(colliders));
            },
        }
    }

    /// Apply per-node physical materials (restitution / friction / density) to the live bodies
    /// (the host maps each node's [`NodeMaterial`] override). (Node body & face — material.)
    pub fn set_node_materials(&mut self, materials: Vec<(NodeKey, NodeMaterial)>) {
        match self {
            #[cfg(feature = "engine")]
            Physics::Inline(p) => p.sim.set_node_materials(materials),
                        Physics::Actor(p) | Physics::Remote(p) => {
                p.link
                    .command(PhysicsCommand::SetNodeMaterials(materials));
            },
        }
    }

    /// Add a non-graph scene-decoration body to the world — a drifting backdrop / scene
    /// element, intangible to the graph by default. Inline (pre-offload) adds it
    /// synchronously; once offloaded it rides a command. (Physics scenes P1.)
    pub fn add_scene_body(
        &mut self,
        collider: NodeCollider,
        position: Point2D<f32>,
        velocity: (f32, f32),
    ) {
        match self {
            #[cfg(feature = "engine")]
            Physics::Inline(p) => {
                p.sim.add_scene_body(collider, position, velocity);
            },
                        Physics::Actor(p) | Physics::Remote(p) => {
                p.link
                    .command(PhysicsCommand::AddSceneBody(collider, position, velocity));
            },
        }
    }

    /// Set every node's tangibility to the scene (the scene-wide lever). (Physics scenes P2.)
    pub fn set_nodes_tangible(&mut self, tangible: bool) {
        match self {
            #[cfg(feature = "engine")]
            Physics::Inline(p) => p.sim.set_nodes_tangible(tangible),
                        Physics::Actor(p) | Physics::Remote(p) => {
                p.link.command(PhysicsCommand::SetNodesTangible(tangible));
            },
        }
    }

    /// Load a declarative scene into the world (clears any prior scene). (Physics scenes P3.)
    pub fn load_scene(&mut self, spec: SceneSpec) {
        match self {
            #[cfg(feature = "engine")]
            Physics::Inline(p) => p.sim.load_scene(&spec),
                        Physics::Actor(p) | Physics::Remote(p) => {
                p.link.command(PhysicsCommand::LoadScene(spec));
            },
        }
    }

    /// Replace static node-contact obstacles (inline or on the physics actor).
    pub fn set_static_obstacles(&mut self, obstacles: Vec<crate::StaticObstacle>) {
        match self {
            #[cfg(feature = "engine")]
            Physics::Inline(p) => p.sim.set_static_obstacles(obstacles),
                        Physics::Actor(p) | Physics::Remote(p) => {
                p.link
                    .command(PhysicsCommand::SetStaticObstacles(obstacles));
            },
        }
    }

    /// Remove every scene body. (Physics scenes P3.)
    pub fn clear_scene(&mut self) {
        match self {
            #[cfg(feature = "engine")]
            Physics::Inline(p) => p.sim.clear_scene(),
                        Physics::Actor(p) | Physics::Remote(p) => {
                p.link.command(PhysicsCommand::ClearScene);
            },
        }
    }

    /// Load a liquid pool (replacing any prior fluid). (Physics scenes P4c.)
    pub fn load_fluid(
        &mut self,
        params: FluidParams,
        basin: Basin,
        origin: Point2D<f32>,
        cols: usize,
        rows: usize,
        spacing: f32,
    ) {
        match self {
            #[cfg(feature = "engine")]
            Physics::Inline(p) => p.sim.load_fluid(params, basin, origin, cols, rows, spacing),
                        Physics::Actor(p) | Physics::Remote(p) => {
                p.link.command(PhysicsCommand::LoadFluid {
                    params,
                    basin,
                    origin,
                    cols,
                    rows,
                    spacing,
                });
            },
        }
    }

    /// Remove the liquid pool. (Physics scenes P4c.)
    pub fn clear_fluid(&mut self) {
        match self {
            #[cfg(feature = "engine")]
            Physics::Inline(p) => p.sim.clear_fluid(),
                        Physics::Actor(p) | Physics::Remote(p) => {
                p.link.command(PhysicsCommand::ClearFluid);
            },
        }
    }

    /// Set (or clear) the scene force-field (whirlpool / well). (Physics scenes P4 fields.)
    pub fn set_scene_field(&mut self, field: Option<SceneField>) {
        match self {
            #[cfg(feature = "engine")]
            Physics::Inline(p) => p.sim.set_scene_field(field),
                        Physics::Actor(p) | Physics::Remote(p) => {
                p.link.command(PhysicsCommand::SetSceneField(field));
            },
        }
    }

    /// Add a continuous body emitter (fountain / stream). (Physics scenes — emitters.)
    pub fn add_emitter(&mut self, spec: SceneEmitter) {
        match self {
            #[cfg(feature = "engine")]
            Physics::Inline(p) => p.sim.add_emitter(spec),
                        Physics::Actor(p) | Physics::Remote(p) => {
                p.link.command(PhysicsCommand::AddEmitter(spec));
            },
        }
    }

    /// Remove every emitter + its bodies. (Physics scenes — emitters.)
    pub fn clear_emitters(&mut self) {
        match self {
            #[cfg(feature = "engine")]
            Physics::Inline(p) => p.sim.clear_emitters(),
                        Physics::Actor(p) | Physics::Remote(p) => {
                p.link.command(PhysicsCommand::ClearEmitters);
            },
        }
    }

    /// Pin a node to a world position (a drag in progress).
    pub fn pin(&mut self, node: NodeKey, position: Point2D<f32>) {
        match self {
            #[cfg(feature = "engine")]
            Physics::Inline(p) => p.sim.pin(node, position),
                        Physics::Actor(p) | Physics::Remote(p) => {
                p.link.command(PhysicsCommand::Pin(node, position));
            },
        }
    }

    /// Release a pinned node back to dynamic.
    pub fn unpin(&mut self, node: NodeKey) {
        match self {
            #[cfg(feature = "engine")]
            Physics::Inline(p) => p.sim.unpin(node),
                        Physics::Actor(p) | Physics::Remote(p) => {
                p.link.command(PhysicsCommand::Unpin(node));
            },
        }
    }

    /// Hold node bodies on the given axes (an encoded axis, F28); see
    /// [`Simulation::set_axis_locks`]. (Dynamics grammar plan, G7.)
    pub fn set_axis_locks(&mut self, locks: Vec<(NodeKey, crate::Axes)>) {
        match self {
            #[cfg(feature = "engine")]
            Physics::Inline(p) => p.sim.set_axis_locks(locks),
                        Physics::Actor(p) | Physics::Remote(p) => {
                p.link.command(PhysicsCommand::SetAxisLocks(locks));
            },
        }
    }
}

impl ActorPhysics {
    fn fold_latest(&mut self, view: &mut LayoutView) {
        let (latest, entries) = self.latest();
        if let Some(update) = latest {
            view.apply_snapshot(&update.snapshot);
            self.settling = update.settling;
            self.energy = update.snapshot.energy;
            self.speed = update.snapshot.speed;
            self.pace = update.pace;
            self.field_clock = update.field_clock;
            // A remote backend has no diagnostic frames: they stay on the far side.
            #[cfg(feature = "actor")]
            {
                self.coupling_membership = update
                    .coupling_membership
                    .and_then(|delivery| delivery.read());
                self.coupling_trace = update.coupling_trace.and_then(|delivery| delivery.read());
            }
        }
        if !entries.is_empty() {
            self.entry.dispatch(entries, self.coupling_membership.as_mut());
        }
    }
}

#[path = "runtime/scheduling.rs"]
mod scheduling;

#[path = "runtime/forces.rs"]
mod forces;
