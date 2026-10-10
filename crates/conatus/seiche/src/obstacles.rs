// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! Static placement obstacles, independent of living scene decorations.

use crate::{NODE_GROUP, NodeCollider, Simulation};
use rapier2d::prelude::*;

/// A fixed obstacle in world units. Hosts validate their geometry before binding.
#[derive(Clone, Debug)]
pub struct StaticObstacle {
    pub collider: NodeCollider,
    pub position: (f32, f32),
    pub rotation: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Kinds, NodeKey};
    use euclid::default::Point2D;

    #[test]
    fn a_fixed_obstacle_excludes_a_node_under_the_running_law() {
        let mut sim = Simulation::new();
        let node = NodeKey::new(0);
        sim.sync_nodes([(node, Point2D::new(35.0, 0.0))]);
        sim.set_node_colliders([(node, NodeCollider::Ball { radius: 5.0 })]);
        sim.add_force(crate::Boundary::default());
        sim.set_static_obstacles(vec![StaticObstacle {
            collider: NodeCollider::Square { half: 40.0 },
            position: (0.0, 0.0),
            rotation: 0.0,
        }]);
        for _ in 0..360 {
            sim.tick(1.0 / 60.0);
        }
        assert!(sim.position_of(node).unwrap().x >= 44.5);
    }

    #[test]
    fn replacing_obstacles_preserves_nodes_and_living_scene_state() {
        let mut sim = Simulation::new();
        let node = NodeKey::new(0);
        sim.sync_nodes([(node, Point2D::new(100.0, 100.0))]);
        sim.set_node_kinds(node, Kinds(3));
        sim.set_nodes_tangible(false);
        sim.set_gravity((0.0, 120.0));
        sim.add_scene_body(
            NodeCollider::Ball { radius: 6.0 },
            Point2D::new(-200.0, 0.0),
            (20.0, 0.0),
        );
        let body = sim.body_for(node).unwrap();
        sim.set_static_obstacles(vec![StaticObstacle {
            collider: NodeCollider::Square { half: 40.0 },
            position: (0.0, 0.0),
            rotation: 0.0,
        }]);
        assert_eq!(sim.body_for(node), Some(body));
        assert_eq!(sim.scene_body_count(), 1);
        assert_eq!(sim.node_kinds(node), Kinds(3));
        assert!(!sim.node_is_tangible(node));
        assert_eq!(sim.gravity, Vector::new(0.0, 120.0));
        sim.clear_scene();
        assert_eq!(
            sim.static_obstacles.len(),
            1,
            "clearing living scenery leaves placement geometry"
        );
        sim.set_static_obstacles(Vec::new());
        assert!(sim.static_obstacles.is_empty());
        assert_eq!(sim.body_for(node), Some(body));
        assert_eq!(sim.position_of(node), Some(Point2D::new(100.0, 100.0)));
    }
}

impl Simulation {
    /// Replace the placement obstacles without changing nodes, gravity, scene
    /// bodies, kinds, tangibility, forces or velocities. Obstacles participate
    /// in node contact but have no node identity and are absent from scene paint.
    pub fn set_static_obstacles(&mut self, obstacles: Vec<StaticObstacle>) {
        for handle in self.static_obstacles.drain(..) {
            self.bodies.remove(
                handle,
                &mut self.islands,
                &mut self.colliders,
                &mut self.impulse_joints,
                &mut self.multibody_joints,
                &mut self.soft_bodies,
                true,
            );
        }
        for obstacle in obstacles {
            let body = RigidBodyBuilder::fixed()
                .translation(Vector::new(obstacle.position.0, obstacle.position.1))
                .rotation(obstacle.rotation)
                .build();
            let handle = self.bodies.insert(body);
            let collider = ColliderBuilder::new(obstacle.collider.to_shared_shape())
                .collision_groups(InteractionGroups::new(
                    NODE_GROUP,
                    NODE_GROUP,
                    InteractionTestMode::And,
                ))
                .restitution(0.0)
                .friction(0.0)
                .build();
            self.colliders
                .insert_with_parent(collider, handle, &mut self.bodies);
            self.static_obstacles.push(handle);
        }
    }
}
