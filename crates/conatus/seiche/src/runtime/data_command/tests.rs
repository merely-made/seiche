// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

use std::num::NonZeroUsize;

use euclid::default::Point2D;
use seiche_model::{
    AffinitySpec, AnchorSpec, Axes, Basin, DataCommand, EdgeTopology, FluidParams, NodeCollider,
    NodeKey, NodeMaterial, SceneEmitter, SceneField, Speed, StaticObstacle,
};

use super::*;

fn k(i: usize) -> NodeKey {
    NodeKey::new(i)
}

/// One of every data command, with non-default payloads.
fn samples() -> Vec<DataCommand> {
    let p = |x, y| Point2D::new(x, y);
    let ball = NodeCollider::Ball { radius: 9.0 };
    vec![
        DataCommand::SyncNodes(vec![(k(0), p(1.0, 2.0)), (k(3), p(-4.0, 5.5))]),
        DataCommand::SyncNodesRetiring {
            nodes: vec![(k(1), p(0.0, 1.0))],
            retired: vec![k(2), k(7)],
        },
        DataCommand::SyncEdges(vec![(k(0), k(1)), (k(1), k(0))]),
        DataCommand::SyncEdgeTopology(
            [(k(0), k(1)), (k(0), k(1)), (k(1), k(2))]
                .into_iter()
                .collect::<EdgeTopology>(),
        ),
        DataCommand::Seed(vec![(k(4), p(3.0, 3.0))]),
        DataCommand::ResetFieldRuntime(vec![(k(5), p(6.0, 7.0))]),
        DataCommand::Barrier(41),
        DataCommand::Pin(k(2), p(10.0, -10.0)),
        DataCommand::Unpin(k(2)),
        DataCommand::Settle(120),
        DataCommand::Halt,
        DataCommand::SetDragging(true),
        DataCommand::SetCouplingTraceLimit(NonZeroUsize::new(8)),
        DataCommand::SetAffinityForce(Some(AffinitySpec {
            pairs: vec![(k(0), k(1), 0.5), (k(1), k(2), 1.0)],
            stiffness: 3.0,
            rest_length: 70.0,
        })),
        DataCommand::SetAnchorForce(Some(AnchorSpec {
            anchors: vec![(k(0), (1.0, 2.0)), (k(2), (3.0, 4.0))],
            stiffness: 9.0,
            slack: 2.0,
        })),
        DataCommand::SetLinearDamping(1.25),
        DataCommand::SetNodeColliders(vec![
            (k(0), ball.clone()),
            (
                k(1),
                NodeCollider::Hull {
                    points: vec![(0.0, 0.0), (4.0, 0.0), (0.0, 4.0)],
                    fallback: 3.0,
                },
            ),
        ]),
        DataCommand::SetNodeMaterials(vec![(k(0), NodeMaterial::default())]),
        DataCommand::SetAxisLocks(vec![(k(0), Axes { x: true, y: false })]),
        DataCommand::AddSceneBody(ball.clone(), p(5.0, 5.0), (1.0, -1.0)),
        DataCommand::SetNodesTangible(true),
        DataCommand::LoadScene(crate::domino_scene()),
        DataCommand::ClearScene,
        DataCommand::SetStaticObstacles(vec![StaticObstacle {
            collider: ball.clone(),
            position: (1.0, 2.0),
            rotation: 0.25,
        }]),
        DataCommand::LoadFluid {
            params: FluidParams::default(),
            basin: Basin {
                min_x: -50.0,
                max_x: 50.0,
                floor_y: 80.0,
            },
            origin: p(0.0, 0.0),
            cols: 4,
            rows: 3,
            spacing: 6.0,
        },
        DataCommand::ClearFluid,
        DataCommand::SetSceneField(Some(SceneField::Vortex {
            center: (1.0, 1.0),
            strength: 2.0,
            inward: 0.5,
        })),
        DataCommand::AddEmitter(SceneEmitter {
            collider: ball,
            position: (0.0, 0.0),
            position_jitter: (1.0, 1.0),
            velocity: (0.0, 1.0),
            velocity_jitter: (0.5, 0.5),
            rate_per_sec: 3.0,
            lifetime_secs: 4.0,
            max_alive: 20,
        }),
        DataCommand::ClearEmitters,
        DataCommand::SetSpeed(Speed::from_factor(2.0)),
    ]
}

#[test]
fn every_data_command_round_trips_through_physics_command() {
    let samples = samples();
    assert_eq!(samples.len(), 30, "one sample per data command");
    for command in samples {
        let before = format!("{command:?}");
        let physics: PhysicsCommand = command.into();
        let after =
            DataCommand::try_from(physics).unwrap_or_else(|_| panic!("{before} must convert back"));
        assert_eq!(before, format!("{after:?}"));
    }
}

#[test]
fn the_non_data_commands_come_back_unchanged() {
    for command in [
        PhysicsCommand::SetForces(Vec::new()),
        PhysicsCommand::SetCouplingForces(Vec::new()),
        PhysicsCommand::SetCouplingEntryHandler(None),
        PhysicsCommand::SetStepBudget(None),
    ] {
        assert!(DataCommand::try_from(command).is_err());
    }
}

#[cfg(feature = "serde")]
#[test]
fn every_data_command_survives_its_wire_form() {
    for command in samples() {
        let before = format!("{command:?}");
        let wire = serde_json::to_string(&command).expect("serialize");
        let back: DataCommand = serde_json::from_str(&wire).expect("deserialize");
        assert_eq!(before, format!("{back:?}"), "{wire}");
    }
}
