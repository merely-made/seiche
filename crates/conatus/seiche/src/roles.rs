// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! The `Simulation` half of the roles: holding node bodies on axes. The data
//! (`Role`, `Axes`, `RoleTable`) lives in `seiche-model`.

use rapier2d::prelude::*;

pub use seiche_model::roles::{Axes, Role, RoleTable};

use crate::{NodeKey, Simulation};

impl Simulation {
    /// Hold node bodies on the given axes: a held axis cannot translate, so
    /// the body keeps its coordinate there exactly while forces and contacts
    /// move it on the others. [`Axes::NONE`] frees a body again. Applied to
    /// the live bodies only: a host re-applies it after a sync spawns new ones.
    pub fn set_axis_locks(&mut self, locks: impl IntoIterator<Item = (NodeKey, Axes)>) {
        for (node, axes) in locks {
            let Some(&handle) = self.bodies_by_node.get(&node) else {
                continue;
            };
            if let Some(body) = self.bodies.get_mut(handle) {
                body.set_enabled_translations(!axes.x, !axes.y, true);
            }
        }
    }

    /// Which axes a node body has held.
    pub fn axis_locks(&self, node: NodeKey) -> Option<Axes> {
        let body = self.bodies.get(*self.bodies_by_node.get(&node)?)?;
        let locked = body.locked_axes();
        Some(Axes {
            x: locked.contains(LockedAxes::TRANSLATION_LOCKED_X),
            y: locked.contains(LockedAxes::TRANSLATION_LOCKED_Y),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Boundary, EdgeSpring, NodeExclusion};
    use euclid::default::Point2D;
    use std::collections::HashMap;

    /// Two bodies on one x, overlapping: with x held on both, contacts and
    /// repulsion separate them on y alone and x stays exact. The same pair
    /// with nothing held moves on x too (the control).
    #[test]
    fn a_held_axis_stays_exact_while_the_free_axis_separates() {
        let run = |locks: Axes| {
            let mut sim = Simulation::new();
            let (a, b) = (NodeKey::new(0), NodeKey::new(1));
            sim.sync_nodes([(a, Point2D::new(40.0, 0.0)), (b, Point2D::new(43.0, 6.0))]);
            sim.add_force(NodeExclusion::default());
            sim.add_force(EdgeSpring::default());
            sim.add_force(Boundary::default());
            sim.set_axis_locks([(a, locks), (b, locks)]);
            assert_eq!(sim.axis_locks(a), Some(locks));
            for _ in 0..360 {
                sim.tick(1.0 / 60.0);
            }
            let at: HashMap<_, _> = sim.positions().collect();
            (at[&a], at[&b])
        };
        let x_held = Axes { x: true, y: false };
        let (a, b) = run(x_held);
        assert_eq!((a.x, b.x), (40.0, 43.0), "x is held exactly");
        assert!((a.y - b.y).abs() > 36.0, "y separated: {a:?} {b:?}");
        let (a, b) = run(Axes::NONE);
        assert!(
            (a.x - 40.0).abs() > 1.0 || (b.x - 43.0).abs() > 1.0,
            "free, x moves: {a:?} {b:?}"
        );
    }

    /// A drag pins a body kinematically and its release frees it; the held
    /// axis survives the round trip.
    #[test]
    fn a_held_axis_survives_a_pin_and_release() {
        let mut sim = Simulation::new();
        let a = NodeKey::new(0);
        sim.sync_nodes([(a, Point2D::new(10.0, 0.0))]);
        let held = Axes { x: false, y: true };
        sim.set_axis_locks([(a, held)]);
        sim.pin(a, Point2D::new(30.0, 30.0));
        sim.tick(1.0 / 60.0);
        sim.unpin(a);
        assert_eq!(sim.axis_locks(a), Some(held));
    }
}
