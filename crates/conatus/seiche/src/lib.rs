// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! # seiche
//!
//! Rapier-backed force integration for Mere: the crate that realizes the graph's
//! bodies as motion. A *seiche* is the standing-wave oscillation of a lake or bay,
//! the whole body of water rocking rhythmically back and forth; it moves the *mere*
//! (a mere is a lake) the way this crate moves the graph's bodies. Sibling to the
//! `numen` field-algebra crate: numen defines and evaluates fields and couplings;
//! `seiche` applies and integrates the force laws (rapier bodies, collision,
//! stepping) into positions.
//!
//! ## Place in the architecture
//!
//! The host's graph is the source of truth for node identity, topology,
//! and the *committed* position used for persistence. seiche holds the rapier
//! world that drives the *projected* position — bodies bound to nodes, forces
//! from pluggable [`Force`] implementors, and a tick that mirrors body
//! translations back to the graph each step.
//!
//! Petgraph (inside kernel) owns topology; seiche owns physical state. Both sit at
//! the substrate tier, consumed by everything above.
//!
//! ## Forces
//!
//! - [`Simulation`] — owns the rapier world, a [`NodeKey`] ↔
//!   [`RigidBodyHandle`] bimap, position-based hit-test / cull, and the
//!   registered forces.
//! - [`Simulation::sync_nodes`] / [`Simulation::sync_edges`] — keep bodies
//!   and edge topology in step with the host's node/edge lists. Idempotent.
//! - [`Simulation::tick`] — apply each registered [`Force`], then step the world.
//! - Built-in forces ([`NodeExclusion`], [`EdgeSpring`], [`Boundary`]) are the
//!   fast Rust default-path; numen's couplings are the general, scriptable
//!   path that compiles to the same [`Force`] contract.
//!
//! Until a [`Force`] is registered, the world ticks empty — bodies settle to
//! rest under damping alone.
//!
//! ## Decomposition
//!
//! The `Simulation` impl is split across files by concern so no one stays over
//! the per-file size ceiling: the struct + `new` + force registration + read
//! model + the [`tick`](Simulation::tick) heartbeat live here; the node-body
//! shape/material axis in [`node_body`], body lifecycle in [`sync`], the rigid
//! scene tier in [`scene_sim`], the fluid seam in [`fluid_coupling`], the
//! force-field tier in [`field`], and emitters in [`emitter`]. (Rust lets an
//! `impl` span modules; private fields stay reachable from these child modules.)

#![doc(html_root_url = "https://docs.rs/seiche/0.0.1")]

use std::collections::HashMap;

use euclid::default::{Box2D, Point2D};
use rapier2d::prelude::*;

/// The opaque identifier seiche binds a body to: petgraph's stable node index, the
/// same type mere's kernel and chartulary use as `NodeKey`. A consumer supplies its
/// own keys (from any graph, or minted directly); seiche never inspects them.
pub type NodeKey = petgraph::stable_graph::NodeIndex;

mod edge_topology;
pub use edge_topology::{EdgeIter, EdgeRun, EdgeTopology};

/// Where a [`Simulation`] ticks: the inline / actor backend a host drives
/// instead of owning a frame loop and a settle budget of its own. (Lifted out
/// of `mere-canvas` 2026-09-04.)
pub mod runtime;
#[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
pub use runtime::monotonic_clock;
pub use runtime::{
    DEFAULT_BUDGET_SHARE, ElapsedStepConfig, ElapsedStepReport, FALLBACK_DISPLAY_PERIOD, PaceStats,
    Physics, PhysicsCommand, PhysicsUpdate, Speed, StepBudget, TICK_DT, TICK_DURATION,
    display_period,
};

/// What each force is, term by term (topology, kernel, state, currency,
/// class, metric, observable), and the instruments that check it.
pub mod instruments;
pub mod terms;
pub use terms::{
    Class, Currency, Declared, Kernel, Layout, Metric, Observable, State, Term, Topology,
};

/// The combinators (dynamics grammar plan, G3): composition by currency, the
/// weighted sum, groups, and the observables their receipts read.
pub mod compose;
pub mod grouped;
pub mod observe;
pub mod scale;
/// The dynamics spec: a live layout's drive as portable data (G4).
pub mod spec;
pub mod weighted;
pub use compose::Admission;
pub use grouped::{Grouped, Partition, Spread};
pub use spec::{DynamicsSpec, SpecCatalog, SpecError};
pub use weighted::Weighted;

/// Built-in force forces for the force-directed orrery layout.
pub mod forces;
pub use forces::{Boundary, EdgeSpring, NodeExclusion};

/// The physics catalog's **laws**: distinct dynamics over the node bodies,
/// each a [`Force`] a host installs through [`Simulation::set_forces`].
/// Springs (the trio above), Charge ([`BarnesHutRepulsion`] + springs) and
/// Still (no forces) need nothing here; the rest are their own files.
pub mod laws;
pub mod overlays;
pub use laws::{
    Anneal, Boids, CounterDamping, Density, DensityGrid, DensityMedium, DensityStop, Gravity, Hold,
    Kuramoto, LinLogForce, MagneticSpring, ParticleLife, StressSpring, graph_distances,
};
pub use overlays::{
    DegreeRepulsion, DepthGravity, DomainCluster, GravityLocus, GridSnap, HubGravity,
};

/// Tensorized N-body force laws. Burn stays opt-in behind `tensor-burn` so
/// ordinary layout consumers do not acquire the GPU/runtime stack.
pub mod tensor_forces;
pub use tensor_forces::{
    NodeExclusionParams, RepulsionInputError, RepulsionParams, node_exclusion_reference,
    repulsion_reference,
};
#[cfg(feature = "tensor-burn")]
pub use tensor_forces::{node_exclusion, repulsion};
#[cfg(feature = "tensor-burn-wgpu")]
pub use tensor_forces::{node_exclusion_wgpu_roundtrip, repulsion_wgpu_roundtrip};

/// The lagged seam: a pairwise law evaluated elsewhere (the host's GPU) and
/// applied a step or more later, the CPU law covering the steps it cannot.
pub mod lagged;
pub use lagged::{
    DEFAULT_MAX_STALE_STEPS, LaggedLane, LaggedRepulsion, LaggedStats, Repulsion, RepulsionRoute,
};

/// [`LaggedRepulsion`] on the host's device through `conatus::resident`.
#[cfg(feature = "gpu")]
pub mod gpu;

/// Barnes–Hut quadtree for O(n log n) approximate n-body repulsion — harvested
/// from the retired `graph-layout`, for big-graph charge repulsion. The
/// [`barnes_hut::repulsion_forces`] primitive is ready; wiring it as a live
/// [`Force`] is a tuning step (it would supplement [`NodeExclusion`]'s local
/// separation with global spreading).
pub mod barnes_hut;
pub use barnes_hut::{BarnesHutConfig, BarnesHutRepulsion, repulsion_forces};

/// The aether→seiche seam: a [`numen::Coupling`] compiled to a
/// [`Force`] (the general, scriptable path the built-in forces specialize).
pub mod coupling_force;
pub mod coupling_membership;
pub mod coupling_trace;
pub mod field_clock;
pub use coupling_force::{CouplingEffect, CouplingEvaluation, CouplingForce, CouplingInertReason};
pub use coupling_membership::{
    CouplingEligibility, CouplingEntryHandler, CouplingEntryOutcome, CouplingMembershipCause,
    CouplingMembershipChange, CouplingMembershipFrame, CouplingMembershipReport,
};
pub use coupling_trace::{
    CouplingAppliedEffect, CouplingBinding, CouplingBodyState, CouplingSource,
    CouplingTargetIdentity, CouplingTrace, CouplingTraceRecord, CouplingTraceSession,
};
pub use field_clock::FieldClock;

/// The pairwise **affinity** force: a weighted, attract-only spring over a
/// `(a, b, weight)` signal that clusters structurally-similar nodes ("cluster by
/// affinity" on force-directed). (Graph signals — P4.)
pub mod affinity_force;
pub use affinity_force::{
    AffinitySpring, DEFAULT_AFFINITY_REST_LENGTH, DEFAULT_AFFINITY_STIFFNESS,
};

pub mod anchor_force;
pub use anchor_force::{AnchorSpring, DEFAULT_ANCHOR_SLACK, DEFAULT_ANCHOR_STIFFNESS};

/// Arrangement roles: seeded, anchored or pinned, at recipe, group and item
/// scope, and the axis locks an encoded axis takes. (Dynamics grammar plan, G7.)
pub mod roles;
pub use roles::{Axes, Role, RoleTable};
pub mod settle;
pub use settle::{DynamicsRunner, SettleEnd, SettleError, SettleReport, settle};

/// Position-Based Fluids (PBF): our own small SPH liquid for the orrery (salva lags rapier badly,
/// so we roll our own). The solver lives here; its seam onto the rigid world (loading + the two-way
/// coupling) is [`Simulation`]'s fluid tier in [`fluid_coupling`]. (Physics scenes P4c.)
pub mod fluid;
pub use fluid::{Basin, ContactShape, Fluid, FluidContact, FluidParams};

/// The declarative scene library: ready-made [`SceneSpec`]s the orrery loads behind the
/// graph (drop-bowl, pyramid, dominoes, Galton board, funnel, drift, chain, whirlpool, fountain).
/// Data, not engine code; the growing catalog lives here to keep `lib.rs` under the size ceiling.
pub mod scenes;
pub use scenes::{
    ball_and_chain_scene, bridge_scene, chain_scene, cradle_scene, domino_scene, drift_scene,
    drop_bowl_scene, fountain_scene, funnel_scene, galton_scene, mixer_scene, pyramid_scene,
    whirlpool_scene,
};

/// The declarative scene **format**: the data types a [`SceneSpec`] is built from (bodies,
/// joints, world settings). Split from `lib.rs` because the format grows independently of the
/// simulation core. (Physics scenes P4b.)
mod scene_spec;
pub use scene_spec::{
    JointMotorSpec, SceneBodyId, SceneBodySpec, SceneBodyType, SceneEmitter, SceneField,
    SceneJoint, SceneJointSpec, SceneSpec,
};

/// The node-body axis: a node's collider **shape** ([`NodeCollider`]) and physical **material**
/// ([`NodeMaterial`]), and the [`Simulation`] methods that re-apply each to live bodies. (Node-rep.)
mod node_body;
mod obstacles;
pub use node_body::{NodeCollider, NodeMaterial};
pub use obstacles::StaticObstacle;

/// The **sieve**: collision predicates as data — nodes carry [`Kinds`], a sieve scene body
/// blocks the kinds it names and passes the rest, computed by rapier's interaction groups
/// with no per-frame work. (Tactile tier T2.)
mod sift;
pub use sift::Kinds;

/// Node-body lifecycle on [`Simulation`]: reconciling the body set with the graph (`sync_*`),
/// the spring-edge topology, and the body accessors the read model is built from.
mod sync;

/// Read-only geometric **proposals** ("physics proposes, the record disposes"): containments
/// and supports, for the host to promote to facts at explicit commitments. (P1 + tactile T3.)
mod propose;
pub use propose::Support;

/// The rigid **scene** tier on [`Simulation`]: scene-decoration bodies, declarative [`SceneSpec`]
/// loading + joints, world gravity, and the per-node tangibility lever.
mod scene_sim;

/// The **fluid** tier on [`Simulation`]: loading the PBF pool and the two-way coupling between
/// pool and rapier bodies (the solver itself is [`fluid`]).
mod fluid_coupling;

/// The force-**field** tier on [`Simulation`] (a whirlpool / well) plus the actor's keep-ticking
/// predicate [`Simulation::wants_continuous_tick`].
mod field;

/// Continuous body **emitters** (fountains / streams) on [`Simulation`].
mod emitter;
use emitter::LiveEmitter;

/// Scene-geometry queries for the orrery canvas: edge geometry, edge picking,
/// and marquee rect-select (node point-pick + cull live on `Simulation`
/// directly). Split out to keep `lib.rs` under the per-file size ceiling.
mod query;

/// The rapier-free read model — a position-only [`LayoutView`] the host renders
/// and hit-tests from, plus the [`LayoutSnapshot`] a physics actor emits to
/// refresh it. The seam that lets the simulation run off the UI thread.
mod view;
pub use view::{LayoutSnapshot, LayoutView, RectSelection, SceneBodyView};

/// Physical radius used for every node body's collider.
///
/// Mirrors `platen::CanvasSceneOptions::default()`'s
/// `default_node_radius` so simulation-driven separation matches
/// the renderer's drawn radius. When per-node radii become a real
/// thing, this constant will get replaced by a per-node lookup.
pub const NODE_BODY_RADIUS: f32 = 18.0;

/// Density for node-body colliders, chosen so each node's mass is ~= 1.0
/// (mass = density * pi * r^2 ~= 0.001 * pi * 18^2 ~= 1.0). The layout forces in
/// [`forces`] are then intuitive accelerations rather than being swamped by the
/// large mass a default density would give an 18px ball.
pub const NODE_BODY_DENSITY: f32 = 0.001;

/// Linear damping applied to every node body. Tuned for a slippery,
/// inertial feel: a released node coasts a moment before settling, rather
/// than stopping dead. Low enough to glide, high enough that the layout
/// still comes to rest without continuous input.
const DEFAULT_LINEAR_DAMPING: f32 = 2.5;

/// Angular damping. Bodies don't visibly rotate in the orrery
/// renderer (circles are isotropic), but rapier still tracks angular
/// velocity; high damping prevents the integrator from accumulating
/// spurious spin.
const DEFAULT_ANGULAR_DAMPING: f32 = 4.0;

/// Upper bound on bodies a single [`SceneSpec`] loads (and a live emitter spawns), so a runaway
/// scene can't swamp the physics actor's per-tick budget. (Physics scenes P3.)
const SCENE_BODY_CAP: usize = 200;

/// Collision groups that keep the graph and the scene from hard-colliding by
/// default. Node colliders are in [`NODE_GROUP`] and collide only with other nodes
/// (intangible to the scene); scene colliders are in [`SCENE_GROUP`] and collide
/// with each other while *admitting* nodes — so a node opts into touching the scene
/// by adding [`SCENE_GROUP`] to its own filter (the tangibility lever, P2). rapier's
/// pairwise test needs both sides to admit the other, so the node's filter alone
/// gates node-scene contact. (Physics scenes P1.)
const NODE_GROUP: Group = Group::GROUP_1;
const SCENE_GROUP: Group = Group::GROUP_2;

/// Interaction groups for a scene-decoration collider: member of [`SCENE_GROUP`],
/// colliding with the scene and admitting nodes (a node still passes through unless
/// its own filter opts in).
fn scene_groups() -> InteractionGroups {
    InteractionGroups::new(
        SCENE_GROUP,
        SCENE_GROUP | NODE_GROUP,
        InteractionTestMode::And,
    )
}

/// A pluggable force-applier. Forces read the body store and apply
/// forces / impulses; seiche's tick walks every registered force
/// before stepping the world.
///
/// Implementors should be cheap — the tick budget is ~16ms total —
/// and write through `bodies.get_mut(...).add_force(...)` or
/// equivalent. Forces are queried in registration order each tick.
///
/// Every force is [`Declared`]: it says what terms it applies, so a catalog
/// reads its class and currency rather than restating them.
pub trait Force: Declared + Send {
    fn apply(&self, ctx: &mut ForceContext<'_>, dt: f32);

    /// Whether the force has motion of its own still to run, so the host
    /// keeps ticking past its settle budget (a flow that has not converged).
    fn wants_tick(&self) -> bool {
        false
    }
}

/// The complete law a [`RepulsionSolver`] must preserve.
///
/// A solver receives this rather than loose constants so it cannot quietly omit
/// `NodeExclusion`'s cutoff or substitute smooth softening for its hard floor.
#[derive(Clone, Copy, Debug)]
pub struct RepulsionRequest {
    pub strength: f32,
    pub cutoff: f32,
    pub min_distance: f32,
}

/// A validated force vector returned by a [`RepulsionSolver`].
#[derive(Clone, Debug)]
pub struct RepulsionForces {
    x: Vec<f32>,
    y: Vec<f32>,
}

impl RepulsionForces {
    /// Construct output for exactly `expected` bodies, rejecting a bad or
    /// non-finite result before it can enter Rapier.
    pub fn new(expected: usize, x: Vec<f32>, y: Vec<f32>) -> Result<Self, RepulsionSolverError> {
        if x.len() != expected || y.len() != expected {
            return Err(RepulsionSolverError::OutputLength {
                expected,
                x: x.len(),
                y: y.len(),
            });
        }
        for (index, value) in x.iter().enumerate() {
            if !value.is_finite() {
                return Err(RepulsionSolverError::NonFinite {
                    index,
                    component: RepulsionComponent::X,
                });
            }
        }
        for (index, value) in y.iter().enumerate() {
            if !value.is_finite() {
                return Err(RepulsionSolverError::NonFinite {
                    index,
                    component: RepulsionComponent::Y,
                });
            }
        }
        Ok(Self { x, y })
    }

    pub fn components(&self) -> (&[f32], &[f32]) {
        (&self.x, &self.y)
    }
}

/// One component of an invalid solver result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RepulsionComponent {
    X,
    Y,
}

/// A solver result that was unsafe to apply to the Rapier world.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RepulsionSolverError {
    OutputLength {
        expected: usize,
        x: usize,
        y: usize,
    },
    NonFinite {
        index: usize,
        component: RepulsionComponent,
    },
    Backend(String),
}

impl std::fmt::Display for RepulsionSolverError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OutputLength { expected, x, y } => write!(
                formatter,
                "repulsion solver returned {x} x-forces and {y} y-forces for {expected} bodies"
            ),
            Self::NonFinite { index, component } => {
                write!(
                    formatter,
                    "repulsion solver returned non-finite {component:?} force at {index}"
                )
            },
            Self::Backend(error) => write!(formatter, "repulsion solver failed: {error}"),
        }
    }
}

impl std::error::Error for RepulsionSolverError {}

/// A host-injected, synchronous pairwise-repulsion solver: the route tests
/// and benches keep beside the lagged one ([`LaggedRepulsion`]).
///
/// Positions are supplied as host slices and validated forces return to
/// Rapier in the same step. [`NodeExclusion`] routes to it above
/// [`Simulation::set_repulsion_solver`]'s threshold, falling back to its CPU
/// law if it fails.
pub type RepulsionSolver = std::sync::Arc<
    dyn Fn(&[f32], &[f32], RepulsionRequest) -> Result<RepulsionForces, RepulsionSolverError>
        + Send
        + Sync,
>;

/// Default node count above which [`NodeExclusion`] uses an installed
/// [`RepulsionSolver`]. This is only a host-configurable staging guard, not a
/// residency or Barnes–Hut crossover claim; hosts set it from their own receipt.
pub const DEFAULT_GPU_REPULSION_THRESHOLD: usize = 1_000;

/// Per-tick view a [`Force`] sees: mutable access to the rapier body
/// store plus the NodeKey ↔ handle bimap so forces can reason about
/// pairs / topology when they need to.
pub struct ForceContext<'a> {
    pub bodies: &'a mut RigidBodySet,
    pub colliders: &'a ColliderSet,
    pub joints: &'a mut ImpulseJointSet,
    pub bodies_by_node: &'a HashMap<NodeKey, RigidBodyHandle>,
    /// Topology the layout forces pull along (e.g. [`EdgeSpring`]). Node-key
    /// pairs, set via [`Simulation::sync_edges`]; seiche stays relation-taxonomy
    /// agnostic, so the caller decides which edge families feed the layout.
    pub edges: &'a [(NodeKey, NodeKey)],
    /// The staged evaluator (synchronous or lagged) and the node count at or
    /// above which [`NodeExclusion`] uses it, threaded from the
    /// [`Simulation`]. `None` = always the native CPU scan.
    pub repulsion: Option<&'a mut Repulsion>,
    /// How many ticks the simulation has taken before this one: the clock a
    /// lagged answer's age is measured on.
    pub step: u64,
    /// Accepted field time before this tick; independent of lagged result age.
    pub field_clock: FieldClock,
}

/// One rapier world + bookkeeping. The host owns one of these per
/// graph; seiche stays pure (no global state, no static handles).
pub struct Simulation {
    pipeline: PhysicsPipeline,
    parameters: IntegrationParameters,
    gravity: Vector,
    islands: IslandManager,
    broad_phase: DefaultBroadPhase,
    narrow_phase: NarrowPhase,
    bodies: RigidBodySet,
    colliders: ColliderSet,
    impulse_joints: ImpulseJointSet,
    multibody_joints: MultibodyJointSet,
    // Empty: seiche makes no soft bodies, but rapier 0.36 threads the set through.
    soft_bodies: SoftBodySet,
    ccd_solver: CCDSolver,
    bodies_by_node: HashMap<NodeKey, RigidBodyHandle>,
    static_obstacles: Vec<RigidBodyHandle>,
    edges: EdgeTopology,
    forces: Vec<Box<dyn Force>>,
    /// Field couplings as a separately-replaceable force list (built-in layout
    /// forces stay in `forces`). A `CouplingForce` snapshots its target node set at
    /// build time, so a field move / resize / new node needs the whole set
    /// re-resolved — the host rebuilds these wholesale via
    /// [`set_coupling_forces`](Self::set_coupling_forces). (Field regions —
    /// rebuild-on-mutation.)
    coupling_forces: Vec<CouplingForce>,
    coupling_trace_session: CouplingTraceSession,
    coupling_trace_limit: Option<std::num::NonZeroUsize>,
    coupling_trace: Option<CouplingTrace>,
    coupling_binding_generation: u64,
    field_clock: FieldClock,
    coupling_membership: coupling_membership::MembershipRuntime,
    coupling_membership_frame: Option<CouplingMembershipFrame>,
    coupling_entry_handler: Option<CouplingEntryHandler>,
    /// The pairwise **affinity** force, if installed: a weighted attract-only spring that clusters
    /// structurally-similar nodes ("cluster by affinity"). Like the couplings it is a snapshot —
    /// the host rebuilds it wholesale via [`set_affinity_force`](Self::set_affinity_force) when the
    /// affinity signal recomputes — and it applies in the same reset window as the built-ins.
    /// `None` = off (the default). (Graph signals — P4.)
    affinity_force: Option<AffinitySpring>,
    /// Per-node springs toward the anchored items' arrangement positions (the
    /// anchored role, [`Role::Anchored`]). Rebuilt wholesale via
    /// [`set_anchor_force`](Self::set_anchor_force).
    anchor_force: Option<AnchorSpring>,
    /// Optional host-injected staged evaluator (synchronous or lagged) + the
    /// node count at or above which [`NodeExclusion`] routes to it instead of
    /// its native CPU scan. `None` = always native (the default). Rapier keeps
    /// every role either way.
    repulsion: Option<Repulsion>,
    /// Ticks taken so far; a lagged answer's age is counted on it.
    steps: u64,
    /// Linear damping applied to every node body — runtime-tunable (the "inertia"
    /// the physics settings expose): lower keeps more drift after a settle, higher
    /// brings nodes to rest sooner. New bodies take this; [`set_linear_damping`]
    /// also re-applies it to the live ones. (Physics settings.)
    linear_damping: f32,
    /// Non-graph scene-decoration bodies sharing this world (the "living backdrop" /
    /// interactive scene), each stored as `(handle, collider shape)` keyed by a
    /// [`SceneBodyId`]. The shape rides along so the host can paint the true face (P4b).
    /// They tick and collide like any body but carry no [`NodeKey`], so the layout forces
    /// (which key off `bodies_by_node`) ignore them. (Physics scenes P1.)
    scene_bodies: HashMap<SceneBodyId, (RigidBodyHandle, NodeCollider)>,
    /// Sparse sprite handles for scene bodies that wear one (a textured crate / barrel), keyed by
    /// [`SceneBodyId`]. An opaque host key seiche never interprets — it rides through to the body's
    /// [`SceneBodyView`] for the paint, like the collider shape. Most props have none. (Scene-prop
    /// sprites.)
    scene_sprites: HashMap<SceneBodyId, String>,
    /// Monotonic id source for [`scene_bodies`](Self::scene_bodies).
    next_scene_id: u64,
    /// Whether the loaded scene wants to keep moving forever (a perpetual backdrop) rather
    /// than settle — the physics actor reads this via [`Self::scene_perpetual`] to keep
    /// ticking instead of parking at rest. Reset by [`Self::clear_scene`]. (Physics scenes P4a.)
    scene_perpetual: bool,
    /// The liquid pool sharing this world, if any ([`fluid`] PBF). `None` until one is loaded;
    /// stepped each [`Self::tick`] and emitted in the snapshot for rendering. (Physics scenes P4c.)
    fluid: Option<Fluid>,
    /// Scene-wide tangibility lever state (mirrors [`Self::set_nodes_tangible`]): when `true`, node
    /// bodies couple to the fluid (the graph stirs the pool); when `false` they pass through. (P4c.)
    nodes_tangible: bool,
    /// The sieve state (tactile T2): each node's declared [`Kinds`] and each sieve scene
    /// body's blocked set, remembered so body re-syncs and tangibility remasks compose.
    sift: sift::Sift,
    /// Per-node material overrides (tactile T1), remembered so a re-synced body comes back
    /// with the material it was tuned to, not the spawn defaults.
    node_materials: HashMap<NodeKey, NodeMaterial>,
    /// A continuous force-field over the scene's dynamic bodies (a whirlpool / well), if any. Keeps
    /// the actor ticking while set; cleared by [`Self::clear_scene`]. (Physics scenes P4 fields.)
    scene_field: Option<SceneField>,
    /// Continuous body emitters (fountains / streams). Stepped each tick: spawn by rate, reap by
    /// age. Keep the actor ticking while present. (Physics scenes — emitters.)
    emitters: Vec<LiveEmitter>,
}

impl Default for Simulation {
    fn default() -> Self {
        Self::new()
    }
}

// The simulation must be `Send` so a host can build it on, and run its tick on,
// a dedicated physics actor thread (P6 always-offload). It is intentionally not
// required to be `Sync` — only one thread (the actor) ever touches it.
const _: fn() = || {
    fn assert_send<T: Send>() {}
    assert_send::<Simulation>();
    assert_send::<LayoutSnapshot>();
};

// The simulation + built-in-force test suite, now kernel-free (drives the
// physics through `sync_nodes` / `sync_edges`, not a mere `Graph`).
#[cfg(test)]
mod tests;

/// The lagged seam's receipts on evaluators whose timing the test controls.
#[cfg(test)]
mod lagged_tests;

/// Integration tests for the scene / fluid / field / emitter tiers (split alongside them).
#[cfg(test)]
mod scene_tests;

#[cfg(test)]
mod field_runtime_tests;

#[path = "simulation.rs"]
mod simulation;
