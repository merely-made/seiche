// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

use std::num::NonZeroUsize;

use euclid::default::Point2D;
use seiche_model::{
    AffinitySpec, AnchorSpec, Axes, Basin, CouplingDescriptor, CouplingSource,
    CouplingTargetIdentity, DataCommand, EdgeTopology, FluidParams,
    ForceSetDescriptor, LawInputsData, LawSources, NodeCollider, NodeKey, NodeMaterial, PhysicsLaw,
    PhysicsOverlay, SceneEmitter, SceneField, Speed, StaticObstacle, StepBudgetSpec,
};

use super::*;
use crate::StepBudget;

fn k(i: usize) -> NodeKey {
    NodeKey::new(i)
}

/// A descriptor with every field set to something other than its default.
pub(crate) fn coupling_descriptor() -> CouplingDescriptor {
    use numen::{
        CouplingId, CouplingResponse, FieldDefinition, FieldExtent, FieldId, FieldRegistry,
        NodeSelector, ScalarField,
    };
    let mut registry = FieldRegistry::new();
    let sampled = registry.insert_scalar("well", ScalarField::Const(2.5));
    CouplingDescriptor {
        response: CouplingResponse::open("mere://test/descriptor"),
        strength: 0.75,
        targets: vec![k(0), k(2), k(2)],
        field: FieldDefinition::Scalar(ScalarField::Add(
            Box::new(ScalarField::Sample(sampled)),
            Box::new(ScalarField::Time),
        )),
        registry,
        extent: FieldExtent::Region {
            min_x: -10.0,
            min_y: -20.0,
            max_x: 30.0,
            max_y: 40.0,
        },
        selector: Some(NodeSelector::Tagged("hot".into())),
        source: Some(CouplingSource {
            field: FieldId::from_uuid(uuid::Uuid::from_u128(11)),
            coupling: CouplingId::from_uuid(uuid::Uuid::from_u128(12)),
            revision: Some(7),
        }),
        target_identities: vec![
            Some(CouplingTargetIdentity {
                member: 21,
                occurrence: None,
            }),
            None,
            Some(CouplingTargetIdentity {
                member: 22,
                occurrence: Some(23),
            }),
        ],
    }
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
        DataCommand::SetCouplingForces(vec![coupling_descriptor()]),
        DataCommand::SetCouplingEntryRelay(true),
        DataCommand::SetStepBudget(Some(StepBudgetSpec {
            per_frame: std::time::Duration::from_micros(8_333),
            margin: std::time::Duration::from_micros(500),
        })),
        DataCommand::SetForceSet(ForceSetDescriptor {
            law: PhysicsLaw::Charge,
            composition: None,
            overlays: vec![PhysicsOverlay::Skeleton],
            sources: LawSources::bare(),
            inputs: LawInputsData {
                nodes: vec![NodeKey::new(0), NodeKey::new(1)],
                edges: vec![(NodeKey::new(0), NodeKey::new(1))],
                sites: vec![(NodeKey::new(0), "a".into()), (NodeKey::new(1), "b".into())],
                clusters: None,
                meaning: None,
                revision: Some(7),
            },
            contact_cutoff: Some(36.0),
        }),
    ]
}

#[test]
fn every_data_command_round_trips_through_physics_command() {
    let samples = samples();
    assert_eq!(samples.len(), 34, "one sample per data command");
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
        PhysicsCommand::SetCouplingEntryHandler(None),
        PhysicsCommand::SetLaggedRepulsion {
            solver: None,
            threshold: 1,
            max_stale_steps: 1,
        },
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

/// A descriptor builds a force whose binding is the one the builder chain
/// makes, and describing the force gives the descriptor back.
#[test]
fn a_coupling_descriptor_builds_the_same_force_as_the_builder_chain() {
    let d = coupling_descriptor();
    let from_data = CouplingForce::from_descriptor(&d);
    let by_hand = CouplingForce::new(
        d.response.clone(),
        d.strength,
        d.targets.iter().copied(),
        d.field.clone(),
    )
    .with_registry(d.registry.clone())
    .with_extent(d.extent.clone())
    .with_selector(d.selector.clone().unwrap())
    .with_provenance(d.source.unwrap(), |slot, _| d.target_identities[slot]);
    assert_eq!(from_data.binding(), by_hand.binding());
    assert!(from_data.binding().animated, "the Time term is seen through the registry");
    assert_eq!(from_data.describe(), d);
}

/// A descriptor whose identities do not match its targets still builds a
/// binding with one slot per target (the membership walk indexes them).
#[test]
fn a_coupling_descriptor_with_mismatched_identities_is_normalised() {
    let mut d = coupling_descriptor();
    d.target_identities.truncate(1);
    assert_eq!(
        CouplingForce::from_descriptor(&d).binding().target_identities.len(),
        d.targets.len()
    );
    d.target_identities = vec![None; 9];
    assert_eq!(
        CouplingForce::from_descriptor(&d).binding().target_identities.len(),
        d.targets.len()
    );
}

/// The budget's data form keeps the allowance and the margin; the clock is
/// whoever runs the step's.
#[test]
fn a_step_budget_spec_carries_the_allowance_and_margin_and_takes_any_clock() {
    use std::time::Duration;
    fn clock() -> Duration {
        Duration::from_millis(5)
    }
    let budget = StepBudget {
        per_frame: Duration::from_micros(8_333),
        clock,
        margin: Duration::from_micros(250),
    };
    let spec = budget.spec();
    assert_eq!(spec.per_frame, budget.per_frame);
    assert_eq!(spec.margin, budget.margin);
    let back = StepBudget::from_spec(spec, clock);
    assert_eq!((back.per_frame, back.margin, (back.clock)()), (budget.per_frame, budget.margin, clock()));
}

/// The same samples through the Worker's codec (postcard writes no field names
/// and refuses formats serde-json forgives, so JSON passing proves less).
#[test]
fn every_data_command_survives_the_postcard_wire() {
    use seiche_model::wire::{ToWorker, decode, encode};
    for command in samples() {
        let before = format!("{command:?}");
        let bytes = encode(&ToWorker::Command(command));
        let Ok(ToWorker::Command(back)) = decode::<ToWorker>(&bytes) else {
            panic!("{before} did not decode");
        };
        assert_eq!(before, format!("{back:?}"));
    }
}
