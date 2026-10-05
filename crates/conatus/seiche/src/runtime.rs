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
    AffinitySpring, Basin, CouplingForce, FluidParams, Force, LaggedRepulsion, LaggedStats,
    LayoutSnapshot, LayoutView, NodeCollider, NodeKey, NodeMaterial, SceneEmitter, SceneField,
    SceneSpec, Simulation,
};

/// Per-tick timestep handed to the simulation: one 60fps frame. The whole
/// stack integrates at this step, inline or on the actor, so a settle budget
/// in ticks reads directly as seconds.
pub const TICK_DT: f32 = 1.0 / 60.0;

mod elapsed;
pub use elapsed::{ElapsedStepConfig, ElapsedStepReport, TICK_DURATION};
#[cfg(feature = "actor")]
mod actor;
mod speed;
#[cfg(feature = "actor")]
use actor::{ActorState, run};
#[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
pub use speed::monotonic_clock;
pub use speed::{
    DEFAULT_BUDGET_SHARE, FALLBACK_DISPLAY_PERIOD, PaceStats, Speed, StepBudget, display_period,
};

#[cfg(test)]
mod pause_tests;

/// A command the host sends to the physics actor. Mirrors the mutating surface
/// of [`Simulation`] plus the settle/drag drivers; all payloads are `Send`
/// (positions and node keys), so the graph itself never crosses the boundary.
pub enum PhysicsCommand {
    /// Reconcile the body set to exactly these `(node, position)` pairs (position
    /// used only for newly-spawned bodies). See [`Simulation::sync_nodes`].
    SyncNodes(Vec<(NodeKey, Point2D<f32>)>),
    /// Replace the spring edge topology.
    SyncEdges(Vec<(NodeKey, NodeKey)>),
    /// Override the positions of existing bodies (a seed / reseed).
    Seed(Vec<(NodeKey, Point2D<f32>)>),
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
    /// Install (or clear) the pairwise affinity force — the rebuild a fresh affinity signal
    /// triggers ("cluster by affinity"). Position-preserving. (Graph signals — P4.)
    SetAffinityForce(Option<AffinitySpring>),
    SetAnchorForce(Option<crate::AnchorSpring>),
    /// Replace the law's force set wholesale — the physics law and its overlays
    /// (the catalog's `laws × overlays`), leaving the coupling / affinity / anchor
    /// slots alone. Position-preserving: no body moves until the next tick.
    /// (Physics catalog — P1.)
    SetForces(Vec<Box<dyn Force>>),
    /// Install (or clear) a lagged evaluator for `NodeExclusion` (see
    /// [`Simulation::set_lagged_repulsion`]): how a host's device reaches a
    /// simulation that is already offloaded. (Physics catalog P5b.)
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
    /// interval itself.
    SetStepBudget(Option<std::time::Duration>),
}

/// One layout the actor produced: the positions plus whether it is still
/// settling (so the host knows to keep requesting frames).
pub struct PhysicsUpdate {
    pub snapshot: LayoutSnapshot,
    pub settling: bool,
    pub command_epoch: u64,
    pub pace: PaceStats,
}

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

/// The off-thread backend: the actor handle, its update channel, and the last
/// settling flag the host saw.
#[cfg(feature = "actor")]
pub struct ActorPhysics {
    handle: ActorHandle<PhysicsCommand>,
    updates: Receiver<PhysicsUpdate>,
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
}

#[cfg(feature = "actor")]
impl ActorPhysics {
    fn barrier(&mut self) {
        self.command_epoch = self
            .command_epoch
            .checked_add(1)
            .expect("physics epoch exhausted");
        self.handle
            .command(PhysicsCommand::Barrier(self.command_epoch));
    }

    fn latest(&mut self) -> Option<PhysicsUpdate> {
        let mut latest = None;
        while let Ok(update) = self.updates.try_recv() {
            if update.command_epoch >= self.command_epoch {
                latest = Some(update);
            }
        }
        latest
    }
}

fn should_tick(sim: &Simulation, ticks: u32, dragging: bool, halted: bool) -> bool {
    dragging || (!halted && (ticks > 0 || sim.wants_continuous_tick()))
}

/// The physics backend a host talks to. Inline by default (tests +
/// wasm); `offload` swaps in the actor thread (feature `actor`).
pub enum Physics {
    Inline(InlinePhysics),
    #[cfg(feature = "actor")]
    Actor(ActorPhysics),
}

impl Physics {
    /// A new in-thread backend over `sim`, with an initial settle budget.
    pub fn inline(sim: Simulation, initial_settle: u32) -> Self {
        Physics::Inline(InlinePhysics {
            sim,
            ticks_remaining: initial_settle,
            dragging: false,
            generation: 0,
            halted: false,
            pace: speed::Pace::default(),
        })
    }

    /// Move the simulation onto an [`armillary`] actor thread (always-offload).
    /// A no-op if already offloaded. `wake` pokes the host's event loop when a
    /// snapshot is ready. The actor inherits the current settle budget so an
    /// in-flight first settle continues uninterrupted across the move.
    #[cfg(feature = "actor")]
    pub fn offload(&mut self, wake: Wake) {
        let Physics::Inline(inline) = self else {
            return;
        };
        // Move the real simulation out (swap in a throwaway empty one) so it can
        // be built-into the actor thread; `Simulation: Send` makes this sound.
        let sim = std::mem::replace(&mut inline.sim, Simulation::new());
        let initial_settle = inline.ticks_remaining;
        let dragging = inline.dragging;
        let halted = inline.halted;
        let speed = inline.pace.speed;
        let pace = inline.pace.stats();
        let settling = should_tick(&sim, initial_settle, dragging, halted);
        let state = ActorState {
            ticks_remaining: initial_settle,
            dragging,
            halted,
            command_epoch: 0,
            pace: speed::Pace::for_actor(
                speed,
                pace.ticks,
                inline.pace.budget.map(|budget| budget.per_frame),
            ),
        };
        let (handle, updates) = spawn(wake, move |commands, out| {
            run(sim, state, commands, out);
        });
        *self = Physics::Actor(ActorPhysics {
            handle,
            updates,
            settling,
            energy: 0.0,
            speed: 0.0,
            command_epoch: 0,
            dial: speed,
            budget: inline.pace.budget,
            pace,
        });
    }

    /// Set the simulation speed: ticks per frame at the fixed [`TICK_DT`].
    /// The trajectory does not depend on it.
    pub fn set_speed(&mut self, speed: Speed) {
        match self {
            Physics::Inline(p) => {
                p.pace.speed = speed;
                if speed >= Speed::REAL_TIME {
                    p.pace.previous = None;
                }
            },
            #[cfg(feature = "actor")]
            Physics::Actor(p) => {
                p.dial = speed;
                p.handle.command(PhysicsCommand::SetSpeed(speed));
            },
        }
    }

    pub fn speed(&self) -> Speed {
        match self {
            Physics::Inline(p) => p.pace.speed,
            #[cfg(feature = "actor")]
            Physics::Actor(p) => p.dial,
        }
    }

    /// Bound a frame's ticks above real time (see [`StepBudget`]). Offloaded,
    /// the actor measures on its own clock and the host's is ignored.
    pub fn set_step_budget(&mut self, budget: Option<StepBudget>) {
        match self {
            Physics::Inline(p) => p.pace.budget = budget,
            #[cfg(feature = "actor")]
            Physics::Actor(p) => {
                p.budget = budget;
                p.handle.command(PhysicsCommand::SetStepBudget(
                    budget.map(|budget| budget.per_frame),
                ));
            },
        }
    }

    /// The step budget last set, as the host gave it.
    pub fn step_budget(&self) -> Option<StepBudget> {
        match self {
            Physics::Inline(p) => p.pace.budget,
            #[cfg(feature = "actor")]
            Physics::Actor(p) => p.budget,
        }
    }

    /// Ticks run, the effective speed, and whether the budget bound.
    pub fn pace(&self) -> PaceStats {
        match self {
            Physics::Inline(p) => p.pace.stats(),
            #[cfg(feature = "actor")]
            Physics::Actor(p) => p.pace,
        }
    }

    /// Reconcile the body set (see [`PhysicsCommand::SyncNodes`]).
    pub fn sync_nodes(&mut self, nodes: Vec<(NodeKey, Point2D<f32>)>) {
        match self {
            Physics::Inline(p) => p.sim.sync_nodes(nodes),
            #[cfg(feature = "actor")]
            Physics::Actor(p) => {
                p.handle.command(PhysicsCommand::SyncNodes(nodes));
            },
        }
    }

    /// Replace the spring edge topology.
    pub fn sync_edges(&mut self, edges: Vec<(NodeKey, NodeKey)>) {
        match self {
            Physics::Inline(p) => p.sim.sync_edges(edges),
            #[cfg(feature = "actor")]
            Physics::Actor(p) => {
                p.handle.command(PhysicsCommand::SyncEdges(edges));
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
            #[cfg(feature = "actor")]
            Physics::Actor(p) => {
                p.handle.command(PhysicsCommand::Seed(positions));
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
            #[cfg(feature = "actor")]
            Physics::Actor(p) => {
                p.handle.command(PhysicsCommand::SetCouplingForces(forces));
            },
        }
    }

    /// Install (or clear, with `None`) the pairwise affinity force wholesale — the rebuild a fresh
    /// affinity signal triggers ("cluster by affinity"). Position-preserving (forces never touch
    /// body state). (Graph signals — P4.)
    pub fn set_affinity_force(&mut self, force: Option<AffinitySpring>) {
        match self {
            Physics::Inline(p) => p.sim.set_affinity_force(force),
            #[cfg(feature = "actor")]
            Physics::Actor(p) => {
                p.handle.command(PhysicsCommand::SetAffinityForce(force));
            },
        }
    }

    /// Install (or clear, with `None`) per-node **anchor** springs toward the
    /// anchored items' arrangement positions (the anchored role, G7).
    /// Position-preserving.
    pub fn set_anchor_force(&mut self, force: Option<crate::AnchorSpring>) {
        match self {
            Physics::Inline(p) => p.sim.set_anchor_force(force),
            #[cfg(feature = "actor")]
            Physics::Actor(p) => {
                p.handle.command(PhysicsCommand::SetAnchorForce(force));
            },
        }
    }

    /// Replace the law's force set wholesale (see [`PhysicsCommand::SetForces`]):
    /// the physics law plus its overlays. The coupling / affinity / anchor slots
    /// are separate and untouched. Position-preserving. (Physics catalog — P1.)
    pub fn set_forces(&mut self, forces: Vec<Box<dyn Force>>) {
        match self {
            Physics::Inline(p) => p.sim.set_forces(forces),
            #[cfg(feature = "actor")]
            Physics::Actor(p) => {
                p.handle.command(PhysicsCommand::SetForces(forces));
            },
        }
    }

    /// The node bodies' kinetic energy: read live inline, and from the last
    /// folded snapshot offloaded, so both backends answer the same question.
    /// (Physics catalog — the receipts' energy floor.)
    pub fn kinetic_energy(&self) -> f32 {
        match self {
            Physics::Inline(p) => p.sim.kinetic_energy(),
            #[cfg(feature = "actor")]
            Physics::Actor(p) => p.energy,
        }
    }

    /// The number of forces in the law slot (inline backend only). Test introspection.
    pub fn force_count(&self) -> usize {
        match self {
            Physics::Inline(p) => p.sim.force_count(),
            #[cfg(feature = "actor")]
            Physics::Actor(_) => 0,
        }
    }

    /// The number of anchored nodes (inline backend only). Test introspection.
    pub fn anchor_count(&self) -> usize {
        match self {
            Physics::Inline(p) => p.sim.anchor_count(),
            #[cfg(feature = "actor")]
            Physics::Actor(_) => 0,
        }
    }

    /// The number of affinity pairs in the installed affinity force (inline backend only — the
    /// actor owns its sim on another thread). Test introspection for the P4 wiring. (Graph signals.)
    pub fn affinity_pair_count(&self) -> usize {
        match self {
            Physics::Inline(p) => p.sim.affinity_pair_count(),
            #[cfg(feature = "actor")]
            Physics::Actor(_) => 0,
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
            #[cfg(feature = "actor")]
            Physics::Actor(p) => {
                p.handle.command(PhysicsCommand::SetLaggedRepulsion {
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
            #[cfg(feature = "actor")]
            Physics::Actor(_) => None,
        }
    }

    /// Set the linear damping (the "inertia" physics setting) on new + live node
    /// bodies. (Physics settings.)
    pub fn set_linear_damping(&mut self, damping: f32) {
        match self {
            Physics::Inline(p) => p.sim.set_linear_damping(damping),
            #[cfg(feature = "actor")]
            Physics::Actor(p) => {
                p.handle.command(PhysicsCommand::SetLinearDamping(damping));
            },
        }
    }

    /// Reshape node colliders so each picks/collides at its true face shape + size (the
    /// host maps every node's silhouette / sprite hull to a [`NodeCollider`]). (P0/P5
    /// collider — Decision 5; node-rep — collider matches shape.)
    pub fn set_node_colliders(&mut self, colliders: Vec<(NodeKey, NodeCollider)>) {
        match self {
            Physics::Inline(p) => p.sim.set_node_colliders(colliders),
            #[cfg(feature = "actor")]
            Physics::Actor(p) => {
                p.handle
                    .command(PhysicsCommand::SetNodeColliders(colliders));
            },
        }
    }

    /// Apply per-node physical materials (restitution / friction / density) to the live bodies
    /// (the host maps each node's [`NodeMaterial`] override). (Node body & face — material.)
    pub fn set_node_materials(&mut self, materials: Vec<(NodeKey, NodeMaterial)>) {
        match self {
            Physics::Inline(p) => p.sim.set_node_materials(materials),
            #[cfg(feature = "actor")]
            Physics::Actor(p) => {
                p.handle
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
            Physics::Inline(p) => {
                p.sim.add_scene_body(collider, position, velocity);
            },
            #[cfg(feature = "actor")]
            Physics::Actor(p) => {
                p.handle
                    .command(PhysicsCommand::AddSceneBody(collider, position, velocity));
            },
        }
    }

    /// Set every node's tangibility to the scene (the scene-wide lever). (Physics scenes P2.)
    pub fn set_nodes_tangible(&mut self, tangible: bool) {
        match self {
            Physics::Inline(p) => p.sim.set_nodes_tangible(tangible),
            #[cfg(feature = "actor")]
            Physics::Actor(p) => {
                p.handle.command(PhysicsCommand::SetNodesTangible(tangible));
            },
        }
    }

    /// Load a declarative scene into the world (clears any prior scene). (Physics scenes P3.)
    pub fn load_scene(&mut self, spec: SceneSpec) {
        match self {
            Physics::Inline(p) => p.sim.load_scene(&spec),
            #[cfg(feature = "actor")]
            Physics::Actor(p) => {
                p.handle.command(PhysicsCommand::LoadScene(spec));
            },
        }
    }

    /// Remove every scene body. (Physics scenes P3.)
    pub fn clear_scene(&mut self) {
        match self {
            Physics::Inline(p) => p.sim.clear_scene(),
            #[cfg(feature = "actor")]
            Physics::Actor(p) => {
                p.handle.command(PhysicsCommand::ClearScene);
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
            Physics::Inline(p) => p.sim.load_fluid(params, basin, origin, cols, rows, spacing),
            #[cfg(feature = "actor")]
            Physics::Actor(p) => {
                p.handle.command(PhysicsCommand::LoadFluid {
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
            Physics::Inline(p) => p.sim.clear_fluid(),
            #[cfg(feature = "actor")]
            Physics::Actor(p) => {
                p.handle.command(PhysicsCommand::ClearFluid);
            },
        }
    }

    /// Set (or clear) the scene force-field (whirlpool / well). (Physics scenes P4 fields.)
    pub fn set_scene_field(&mut self, field: Option<SceneField>) {
        match self {
            Physics::Inline(p) => p.sim.set_scene_field(field),
            #[cfg(feature = "actor")]
            Physics::Actor(p) => {
                p.handle.command(PhysicsCommand::SetSceneField(field));
            },
        }
    }

    /// Add a continuous body emitter (fountain / stream). (Physics scenes — emitters.)
    pub fn add_emitter(&mut self, spec: SceneEmitter) {
        match self {
            Physics::Inline(p) => p.sim.add_emitter(spec),
            #[cfg(feature = "actor")]
            Physics::Actor(p) => {
                p.handle.command(PhysicsCommand::AddEmitter(spec));
            },
        }
    }

    /// Remove every emitter + its bodies. (Physics scenes — emitters.)
    pub fn clear_emitters(&mut self) {
        match self {
            Physics::Inline(p) => p.sim.clear_emitters(),
            #[cfg(feature = "actor")]
            Physics::Actor(p) => {
                p.handle.command(PhysicsCommand::ClearEmitters);
            },
        }
    }

    /// Pin a node to a world position (a drag in progress).
    pub fn pin(&mut self, node: NodeKey, position: Point2D<f32>) {
        match self {
            Physics::Inline(p) => p.sim.pin(node, position),
            #[cfg(feature = "actor")]
            Physics::Actor(p) => {
                p.handle.command(PhysicsCommand::Pin(node, position));
            },
        }
    }

    /// Release a pinned node back to dynamic.
    pub fn unpin(&mut self, node: NodeKey) {
        match self {
            Physics::Inline(p) => p.sim.unpin(node),
            #[cfg(feature = "actor")]
            Physics::Actor(p) => {
                p.handle.command(PhysicsCommand::Unpin(node));
            },
        }
    }

    /// Extend the settle budget to at least `ticks` (start / prolong a settle).
    pub fn settle(&mut self, ticks: u32) {
        match self {
            Physics::Inline(p) => {
                p.ticks_remaining = p.ticks_remaining.max(ticks);
                if ticks > 0 {
                    p.halted = false;
                }
            },
            #[cfg(feature = "actor")]
            Physics::Actor(p) => {
                p.handle.command(PhysicsCommand::Settle(ticks));
                if ticks > 0 {
                    p.settling = true;
                }
            },
        }
    }

    /// Stop current stepping, including perpetual scenes. A later positive
    /// settle resumes; a new drag can temporarily step while held.
    pub fn halt(&mut self) {
        match self {
            Physics::Inline(p) => {
                p.ticks_remaining = 0;
                p.dragging = false;
                p.halted = true;
                p.pace.forget();
                p.pace.meter.clear();
            },
            #[cfg(feature = "actor")]
            Physics::Actor(p) => {
                p.handle.command(PhysicsCommand::Halt);
                p.barrier();
                p.settling = false;
            },
        }
    }

    /// Tell the backend whether a drag is in progress (keep ticking so the
    /// pinned node's neighbors react through the springs).
    pub fn set_dragging(&mut self, dragging: bool) {
        match self {
            Physics::Inline(p) => p.dragging = dragging,
            #[cfg(feature = "actor")]
            Physics::Actor(p) => {
                p.handle.command(PhysicsCommand::SetDragging(dragging));
            },
        }
    }

    /// Whether the layout is still moving (settle in progress or a node dragged).
    pub fn is_settling(&self) -> bool {
        match self {
            Physics::Inline(p) => should_tick(&p.sim, p.ticks_remaining, p.dragging, p.halted),
            #[cfg(feature = "actor")]
            Physics::Actor(p) => p.settling,
        }
    }

    /// Fold the layout into `view` **without stepping** — for a host that
    /// synced bodies and must read or draw them before the next frame.
    /// Offloaded, this drains only what the actor has already emitted.
    pub fn refresh(&mut self, view: &mut LayoutView) {
        match self {
            Physics::Inline(p) => {
                let settling = should_tick(&p.sim, p.ticks_remaining, p.dragging, p.halted);
                speed::fold(p, view, settling);
            },
            #[cfg(feature = "actor")]
            Physics::Actor(p) => p.fold_latest(view),
        }
    }

    /// Advance one frame, folding the latest positions into `view`, and return
    /// whether the layout is still settling.
    ///
    /// - Inline: owe one frame-equivalent ([`TICK_DURATION`]) at the speed, so
    ///   one tick a call at real time, and snapshot. Deterministic callers own
    ///   their tick count: a fraction from elapsed-time driving does not carry.
    /// - Actor: drain the update channel, applying the most recent snapshot
    ///   (older queued ones are superseded — only the freshest layout matters).
    pub fn advance_frame(&mut self, view: &mut LayoutView) -> bool {
        match self {
            Physics::Inline(p) => speed::advance_frame(p, view),
            #[cfg(feature = "actor")]
            Physics::Actor(p) => {
                p.fold_latest(view);
                p.settling
            },
        }
    }

    /// The node bodies' rms speed (see [`Simulation::rms_speed`]): live
    /// inline, the last folded snapshot's offloaded. (G7, F46.)
    pub fn rms_speed(&self) -> f32 {
        match self {
            Physics::Inline(p) => p.sim.rms_speed(),
            #[cfg(feature = "actor")]
            Physics::Actor(p) => p.speed,
        }
    }

    /// Hold node bodies on the given axes (an encoded axis, F28); see
    /// [`Simulation::set_axis_locks`]. (Dynamics grammar plan, G7.)
    pub fn set_axis_locks(&mut self, locks: Vec<(NodeKey, crate::Axes)>) {
        match self {
            Physics::Inline(p) => p.sim.set_axis_locks(locks),
            #[cfg(feature = "actor")]
            Physics::Actor(p) => {
                p.handle.command(PhysicsCommand::SetAxisLocks(locks));
            },
        }
    }

    /// Why the layout keeps moving, for a host's diagnostics: whether the
    /// world asks for ticks of its own (a scene, or a force such as a flow
    /// that has not converged), and the settle budget left. The actor reports
    /// neither and answers `(false, 0)`.
    pub fn tick_demand(&self) -> (bool, u32) {
        match self {
            Physics::Inline(p) => (p.sim.wants_continuous_tick(), p.ticks_remaining),
            #[cfg(feature = "actor")]
            Physics::Actor(_) => (false, 0),
        }
    }
}

#[cfg(feature = "actor")]
impl ActorPhysics {
    fn fold_latest(&mut self, view: &mut LayoutView) {
        if let Some(update) = self.latest() {
            view.apply_snapshot(&update.snapshot);
            self.settling = update.settling;
            self.energy = update.snapshot.energy;
            self.speed = update.snapshot.speed;
            self.pace = update.pace;
        }
    }
}
