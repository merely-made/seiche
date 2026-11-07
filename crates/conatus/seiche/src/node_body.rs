// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! The node-body axis: a node's collider **shape** ([`NodeCollider`] — the geometry the body
//! collides + hit-tests at) and its physical **material** ([`NodeMaterial`] — restitution /
//! friction / density), plus the [`Simulation`] methods that re-apply each to live bodies. Split
//! from `lib.rs` to keep the simulation core under the per-file size ceiling. (Node-rep.)

use crate::NodeKey;
use rapier2d::prelude::*;

use crate::{NodeCollider, NodeMaterial, Simulation};

/// Lower to the parry shape rapier collides + queries with. Extents are clamped to a
/// positive minimum so a zero-size node still has a pickable body.
/// Lower a [`NodeCollider`] to its parry shape.
pub(crate) fn shared_shape(collider: &NodeCollider) -> SharedShape {
    match collider {
        NodeCollider::Ball { radius } => SharedShape::ball(radius.max(1.0)),
        NodeCollider::Square { half } => {
            let h = half.max(1.0);
            SharedShape::cuboid(h, h)
        },
        NodeCollider::RoundedSquare { half, border } => {
            let b = border.clamp(0.1, half - 0.1).max(0.1);
            let h = (half - b).max(1.0);
            SharedShape::round_cuboid(h, h, b)
        },
        NodeCollider::Hull { points, fallback } => {
            // parry's `convex_hull` *asserts* `len >= 2` (it panics, not returns `None`) and
            // still yields nothing useful for a collinear set, so guard the degenerate cases
            // to the ball fallback; a real polygon needs at least 3 points.
            if points.len() < 3 {
                SharedShape::ball(fallback.max(1.0))
            } else {
                let pts: Vec<Vector> = points.iter().map(|&(x, y)| Vector::new(x, y)).collect();
                SharedShape::convex_hull(&pts)
                    .unwrap_or_else(|| SharedShape::ball(fallback.max(1.0)))
            }
        },
    }
}

impl Simulation {
    /// Resize **and reshape** each listed node's collider (hit-target + hard-collision
    /// geometry) to match its visible face — a ball, square, rounded square, or custom hull —
    /// so a node collides at its true face shape and size, not the uniform ball
    /// (Decision 5: the face IS the collider, so physics and picture stay in sync). Bodies
    /// keep position and velocity; only the shape changes. Mass is left at the spawn value — it is
    /// the face geometry, not the inertia, that tracks size. Nodes without a body are skipped.
    /// (P0/P5 collider; node-rep — collider matches shape.)
    pub fn set_node_colliders(
        &mut self,
        colliders: impl IntoIterator<Item = (NodeKey, NodeCollider)>,
    ) {
        for (node, collider) in colliders {
            let Some(&body_handle) = self.bodies_by_node.get(&node) else {
                continue;
            };
            // Copy the handles out so the immutable body borrow ends before the
            // mutable collider borrow (distinct fields, but the borrow checker needs
            // the split made explicit).
            let collider_handles: Vec<ColliderHandle> = self
                .bodies
                .get(body_handle)
                .map(|body| body.colliders().to_vec())
                .unwrap_or_default();
            let shape = crate::node_body::shared_shape(&collider);
            for handle in collider_handles {
                if let Some(c) = self.colliders.get_mut(handle) {
                    c.set_shape(shape.clone());
                }
            }
        }
    }

    /// Apply each listed node's physical **material** (restitution / friction / density /
    /// gravity scale) to its live body, re-applying immediately like
    /// [`set_linear_damping`](Self::set_linear_damping), and remember it so a re-synced body
    /// comes back tuned the same way. A density change re-derives the body's mass from its
    /// colliders, so a heavier node resists pushing. The defaults match the spawn values, so
    /// an unconfigured node is a no-op. Nodes without a body still have their material
    /// remembered for when one appears. (Node body & face — material; tactile T1.)
    pub fn set_node_materials(
        &mut self,
        materials: impl IntoIterator<Item = (NodeKey, NodeMaterial)>,
    ) {
        for (node, material) in materials {
            self.node_materials.insert(node, material);
            let Some(&body_handle) = self.bodies_by_node.get(&node) else {
                continue;
            };
            // Split the borrows (immutable body read for the handles, then mutable collider
            // writes) the way `set_node_colliders` does.
            let collider_handles: Vec<ColliderHandle> = self
                .bodies
                .get(body_handle)
                .map(|body| body.colliders().to_vec())
                .unwrap_or_default();
            for handle in collider_handles {
                if let Some(c) = self.colliders.get_mut(handle) {
                    c.set_restitution(material.restitution);
                    c.set_friction(material.friction);
                    c.set_density(material.density);
                }
            }
            // Density feeds mass, which rapier caches on the body — recompute it so the new
            // weight takes effect (distinct fields: `bodies` mut, `colliders` read).
            if let Some(body) = self.bodies.get_mut(body_handle) {
                body.recompute_mass_properties_from_colliders(&self.colliders);
                body.set_gravity_scale(material.gravity_scale, true);
            }
        }
    }

    /// The material a node's **live** body actually carries right now, read back off rapier
    /// state (collider restitution / friction / density, body gravity scale) rather than the
    /// remembered override — so a receipt can assert what was set is what runs. `None` for a
    /// node with no body. (Tactile T1.)
    pub fn node_material(&self, node: NodeKey) -> Option<NodeMaterial> {
        let &handle = self.bodies_by_node.get(&node)?;
        let body = self.bodies.get(handle)?;
        let collider = self.colliders.get(*body.colliders().first()?)?;
        Some(NodeMaterial {
            restitution: collider.restitution(),
            friction: collider.friction(),
            density: collider.density(),
            gravity_scale: body.gravity_scale(),
        })
    }
}
