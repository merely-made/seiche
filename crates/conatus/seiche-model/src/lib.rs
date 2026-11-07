// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! The rapier-free model of `seiche`: what a host reads, names or sends without
//! running the physics engine.
//!
//! `seiche` integrates force laws over a rapier world. Most of what a canvas
//! says *about* that world is plain data: node keys, collider shapes and
//! materials, scene specs, arrangement roles, the field clock, the edge
//! topology and the position-only [`LayoutView`] a host renders from. Those live
//! here, with no dependency on rapier, parry or nalgebra, so a build that only
//! names or displays a layout can link this crate and leave the engine out (the
//! lazy physics Worker plan, Graphshell host plan T7). `seiche` re-exports every
//! item under its old path, so existing consumers compile unchanged.
//!
//! What is not here, and why: the [`Force`](https://docs.rs/seiche) trait and
//! its context (they hand out rapier bodies), `Simulation`, the force laws, the
//! dynamics spec (it builds forces), and the `Physics` runtime with its command
//! set (it owns a `Simulation` and carries boxed forces). Those move, or become
//! descriptors, in the next step.

/// The opaque identifier a body is bound to: petgraph's stable node index, the
/// same type mere's kernel and chartulary use as `NodeKey`. A consumer supplies
/// its own keys (from any graph, or minted directly); seiche never inspects them.
pub type NodeKey = petgraph::stable_graph::NodeIndex;

/// Radius of every node body, in world units (1 unit is about 1 px).
pub const NODE_BODY_RADIUS: f32 = 18.0;

/// Density of a node body, so its mass is area times this.
pub const NODE_BODY_DENSITY: f32 = 0.001;

pub mod edge_topology;
pub mod field_clock;
pub mod fluid;
pub mod node_body;
pub mod observe;
pub mod roles;
pub mod scene_spec;
pub mod view;

pub use edge_topology::{EdgeIter, EdgeRun, EdgeTopology};
pub use field_clock::FieldClock;
pub use fluid::{Basin, ContactShape, Fluid, FluidContact, FluidParams};
pub use node_body::{NodeCollider, NodeMaterial};
pub use roles::{Axes, Role, RoleTable};
pub use scene_spec::{
    JointMotorSpec, SceneBodyId, SceneBodySpec, SceneBodyType, SceneEmitter, SceneField,
    SceneJoint, SceneJointSpec, SceneSpec,
};
pub use view::{LayoutSnapshot, LayoutView, RectSelection, SceneBodyView};
