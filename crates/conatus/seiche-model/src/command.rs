// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! The data half of the physics command set: every command a host sends the
//! simulation whose payload is plain values.
//!
//! `seiche`'s `PhysicsCommand` mixes these with commands that carry boxed
//! forces, closures and function pointers. Those cannot cross a thread-less
//! message boundary; these can. With the `serde` feature each [`DataCommand`]
//! has a wire form, and `seiche` converts to and from `PhysicsCommand`
//! without loss (a round-trip test per variant).
//!
//! The springs travel as specs ([`AnchorSpec`], [`AffinitySpec`]): the engine
//! builds its spring from them, so the spring types keep their private state.

use std::num::NonZeroUsize;

use euclid::default::Point2D;

use crate::{
    Axes, Basin, EdgeTopology, FluidParams, ForceSetDescriptor, NodeCollider, NodeKey,
    NodeMaterial, SceneEmitter, SceneField, SceneSpec, Speed, StaticObstacle,
};

/// An anchor spring as data: each node's anchor, the stiffness and the slack.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AnchorSpec {
    /// `(node, (x, y))` anchors, in a fixed order.
    pub anchors: Vec<(NodeKey, (f32, f32))>,
    /// Force per unit of offset from the anchor.
    pub stiffness: f32,
    /// Offset below which the anchor exerts nothing.
    pub slack: f32,
}

/// An affinity spring as data: weighted pairs, the stiffness and the rest length.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AffinitySpec {
    /// `(a, b, weight)` with the weight in `0..=1`.
    pub pairs: Vec<(NodeKey, NodeKey, f32)>,
    /// Attraction strength at weight 1.
    pub stiffness: f32,
    /// The separation a weight-1 pair settles toward.
    pub rest_length: f32,
}

/// A command whose payload is plain data. Mirrors the matching variants of
/// `seiche::runtime::PhysicsCommand`.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DataCommand {
    /// Reconcile the body set to exactly these `(node, position)` pairs.
    SyncNodes(Vec<(NodeKey, Point2D<f32>)>),
    /// End the retired body lives before reconciling replacements.
    SyncNodesRetiring {
        nodes: Vec<(NodeKey, Point2D<f32>)>,
        retired: Vec<NodeKey>,
    },
    /// Replace the spring edge topology.
    SyncEdges(Vec<(NodeKey, NodeKey)>),
    /// Replace the ordered counted topology.
    SyncEdgeTopology(EdgeTopology),
    /// Override the positions of existing bodies.
    Seed(Vec<(NodeKey, Point2D<f32>)>),
    /// Reset the field runtime to these positions.
    ResetFieldRuntime(Vec<(NodeKey, Point2D<f32>)>),
    /// Acknowledge all preceding commands in subsequent updates.
    Barrier(u64),
    /// Pin a node to a world position.
    Pin(NodeKey, Point2D<f32>),
    /// Release a pinned node.
    Unpin(NodeKey),
    /// Extend the settle budget to at least this many ticks.
    Settle(u32),
    /// Stop all current stepping.
    Halt,
    /// Whether a drag is in progress.
    SetDragging(bool),
    /// Observe the latest coupling tick with this row cap.
    SetCouplingTraceLimit(Option<NonZeroUsize>),
    /// Install or clear the affinity spring.
    SetAffinityForce(Option<AffinitySpec>),
    /// Install or clear the anchor spring.
    SetAnchorForce(Option<AnchorSpec>),
    /// Set the linear damping on new and live bodies.
    SetLinearDamping(f32),
    /// Reshape node colliders.
    SetNodeColliders(Vec<(NodeKey, NodeCollider)>),
    /// Set per-node physical materials.
    SetNodeMaterials(Vec<(NodeKey, NodeMaterial)>),
    /// Hold node bodies on axes.
    SetAxisLocks(Vec<(NodeKey, Axes)>),
    /// Add a scene-decoration body: shape, position, drift velocity.
    AddSceneBody(NodeCollider, Point2D<f32>, (f32, f32)),
    /// Set every node's tangibility to the scene.
    SetNodesTangible(bool),
    /// Load a declarative scene.
    LoadScene(SceneSpec),
    /// Remove every scene body.
    ClearScene,
    /// Replace the static node-contact obstacles.
    SetStaticObstacles(Vec<StaticObstacle>),
    /// Load a liquid pool.
    LoadFluid {
        params: FluidParams,
        basin: Basin,
        origin: Point2D<f32>,
        cols: usize,
        rows: usize,
        spacing: f32,
    },
    /// Remove the liquid pool.
    ClearFluid,
    /// Set or clear the scene force-field.
    SetSceneField(Option<SceneField>),
    /// Add a continuous body emitter.
    AddEmitter(SceneEmitter),
    /// Remove every emitter and its bodies.
    ClearEmitters,
    /// Set the simulation speed.
    SetSpeed(Speed),
    /// Replace the law's force set wholesale, from its descriptor: the engine
    /// builds the forces (`seiche::catalog::build_force_set`).
    SetForceSet(ForceSetDescriptor),
}
