// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! The node-body axis: a node's collider **shape** ([`NodeCollider`] — the geometry the body
//! collides + hit-tests at) and its physical **material** ([`NodeMaterial`] — restitution /
//! friction / density), plus the [`Simulation`] methods that re-apply each to live bodies. Split
//! from `lib.rs` to keep the simulation core under the per-file size ceiling. (Node-rep.)

use crate::NODE_BODY_DENSITY;

/// The collider shape the host wants for a node's physics body, so the hard-collision /
/// hit geometry matches the node's *visible* face rather than a uniform ball. The host maps
/// its own form vocabulary (orrery `NodeShape`, a custom sprite hull) onto this; seiche lowers
/// each to a parry shape in [`Simulation::set_node_colliders`]. Sizes/points are in world
/// units (the same space as positions). (Node-rep — collider matches shape.)
#[derive(Clone, Debug, PartialEq)]
pub enum NodeCollider {
    /// A circle of `radius` — the default, and the circle content silhouette.
    Ball { radius: f32 },
    /// An axis-aligned square with half-extent `half` (a document face).
    Square { half: f32 },
    /// A square with rounded corners: total half-extent `half`, corners rounded by `border`
    /// (a menu / rounded face).
    RoundedSquare { half: f32, border: f32 },
    /// A custom convex hull in body-local world units — the sprite's traced outline or a
    /// hand-edited polygon. Falls back to a ball of `fallback` if the hull is degenerate
    /// (fewer than 3 points / collinear). (Node-rep — sprite hull / shape editor.)
    Hull {
        points: Vec<(f32, f32)>,
        fallback: f32,
    },
}

/// A node's physical **material** on the Body axis: how its rapier body responds to contact
/// and weight. The defaults match the spawn constants, so an unconfigured node is unchanged;
/// the orrery pushes per-node overrides via [`Simulation::set_node_materials`] (the same
/// re-apply-to-live-bodies shape as [`Simulation::set_linear_damping`]). Independent of the
/// node's shape (the collider geometry) and its face (the texture). (Node body & face — material.)
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NodeMaterial {
    /// Bounciness on contact: 0 (dead) to 1 (perfectly elastic).
    pub restitution: f32,
    /// Surface grip: 0 (frictionless) upward.
    pub friction: f32,
    /// Mass density (mass = density * area); higher is heavier and harder to push.
    pub density: f32,
    /// How much world gravity acts on this node: `0.0` (the default — layout nodes never
    /// fall) to `1.0` (full weight). Weight is opt-in per node, which is what lets a mere
    /// say "old documents are heavy" without every node dropping off the canvas.
    /// (Tactile tier T1.)
    pub gravity_scale: f32,
}

impl Default for NodeMaterial {
    fn default() -> Self {
        Self {
            restitution: 0.0,
            friction: 0.0,
            density: NODE_BODY_DENSITY,
            gravity_scale: 0.0,
        }
    }
}
