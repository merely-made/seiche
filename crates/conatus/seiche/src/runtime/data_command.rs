// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! `PhysicsCommand` <-> `DataCommand`: the commands whose payloads are plain
//! data (`SetForceSet` among them) convert without loss; the rest (`SetForces`,
//! `SetCouplingForces`, `SetCouplingEntryHandler`, `SetLaggedRepulsion`,
//! `SetStepBudget`) are returned unchanged in `Err`.

use seiche_model::DataCommand;

use super::PhysicsCommand;
use crate::{AffinitySpring, AnchorSpring};

impl From<DataCommand> for PhysicsCommand {
    fn from(command: DataCommand) -> Self {
        use DataCommand as D;
        match command {
            D::SyncNodes(n) => PhysicsCommand::SyncNodes(n),
            D::SyncNodesRetiring { nodes, retired } => {
                PhysicsCommand::SyncNodesRetiring { nodes, retired }
            },
            D::SyncEdges(e) => PhysicsCommand::SyncEdges(e),
            D::SyncEdgeTopology(t) => PhysicsCommand::SyncEdgeTopology(t),
            D::Seed(p) => PhysicsCommand::Seed(p),
            D::ResetFieldRuntime(p) => PhysicsCommand::ResetFieldRuntime(p),
            D::Barrier(e) => PhysicsCommand::Barrier(e),
            D::Pin(n, p) => PhysicsCommand::Pin(n, p),
            D::Unpin(n) => PhysicsCommand::Unpin(n),
            D::Settle(n) => PhysicsCommand::Settle(n),
            D::Halt => PhysicsCommand::Halt,
            D::SetDragging(d) => PhysicsCommand::SetDragging(d),
            D::SetCouplingTraceLimit(l) => PhysicsCommand::SetCouplingTraceLimit(l),
            D::SetAffinityForce(s) => {
                PhysicsCommand::SetAffinityForce(s.as_ref().map(AffinitySpring::from_spec))
            },
            D::SetAnchorForce(s) => {
                PhysicsCommand::SetAnchorForce(s.as_ref().map(AnchorSpring::from_spec))
            },
            D::SetLinearDamping(d) => PhysicsCommand::SetLinearDamping(d),
            D::SetNodeColliders(c) => PhysicsCommand::SetNodeColliders(c),
            D::SetNodeMaterials(m) => PhysicsCommand::SetNodeMaterials(m),
            D::SetAxisLocks(a) => PhysicsCommand::SetAxisLocks(a),
            D::AddSceneBody(c, p, v) => PhysicsCommand::AddSceneBody(c, p, v),
            D::SetNodesTangible(t) => PhysicsCommand::SetNodesTangible(t),
            D::LoadScene(s) => PhysicsCommand::LoadScene(s),
            D::ClearScene => PhysicsCommand::ClearScene,
            D::SetStaticObstacles(o) => PhysicsCommand::SetStaticObstacles(o),
            D::LoadFluid {
                params,
                basin,
                origin,
                cols,
                rows,
                spacing,
            } => PhysicsCommand::LoadFluid {
                params,
                basin,
                origin,
                cols,
                rows,
                spacing,
            },
            D::ClearFluid => PhysicsCommand::ClearFluid,
            D::SetSceneField(f) => PhysicsCommand::SetSceneField(f),
            D::AddEmitter(e) => PhysicsCommand::AddEmitter(e),
            D::ClearEmitters => PhysicsCommand::ClearEmitters,
            D::SetSpeed(s) => PhysicsCommand::SetSpeed(s),
            D::SetForceSet(d) => PhysicsCommand::SetForceSet(d),
        }
    }
}

impl TryFrom<PhysicsCommand> for DataCommand {
    type Error = PhysicsCommand;

    fn try_from(command: PhysicsCommand) -> Result<Self, PhysicsCommand> {
        use PhysicsCommand as P;
        Ok(match command {
            P::SyncNodes(n) => DataCommand::SyncNodes(n),
            P::SyncNodesRetiring { nodes, retired } => {
                DataCommand::SyncNodesRetiring { nodes, retired }
            },
            P::SyncEdges(e) => DataCommand::SyncEdges(e),
            P::SyncEdgeTopology(t) => DataCommand::SyncEdgeTopology(t),
            P::Seed(p) => DataCommand::Seed(p),
            P::ResetFieldRuntime(p) => DataCommand::ResetFieldRuntime(p),
            P::Barrier(e) => DataCommand::Barrier(e),
            P::Pin(n, p) => DataCommand::Pin(n, p),
            P::Unpin(n) => DataCommand::Unpin(n),
            P::Settle(n) => DataCommand::Settle(n),
            P::Halt => DataCommand::Halt,
            P::SetDragging(d) => DataCommand::SetDragging(d),
            P::SetCouplingTraceLimit(l) => DataCommand::SetCouplingTraceLimit(l),
            P::SetAffinityForce(s) => {
                DataCommand::SetAffinityForce(s.as_ref().map(AffinitySpring::to_spec))
            },
            P::SetAnchorForce(s) => {
                DataCommand::SetAnchorForce(s.as_ref().map(AnchorSpring::to_spec))
            },
            P::SetLinearDamping(d) => DataCommand::SetLinearDamping(d),
            P::SetNodeColliders(c) => DataCommand::SetNodeColliders(c),
            P::SetNodeMaterials(m) => DataCommand::SetNodeMaterials(m),
            P::SetAxisLocks(a) => DataCommand::SetAxisLocks(a),
            P::AddSceneBody(c, p, v) => DataCommand::AddSceneBody(c, p, v),
            P::SetNodesTangible(t) => DataCommand::SetNodesTangible(t),
            P::LoadScene(s) => DataCommand::LoadScene(s),
            P::ClearScene => DataCommand::ClearScene,
            P::SetStaticObstacles(o) => DataCommand::SetStaticObstacles(o),
            P::LoadFluid {
                params,
                basin,
                origin,
                cols,
                rows,
                spacing,
            } => DataCommand::LoadFluid {
                params,
                basin,
                origin,
                cols,
                rows,
                spacing,
            },
            P::ClearFluid => DataCommand::ClearFluid,
            P::SetSceneField(f) => DataCommand::SetSceneField(f),
            P::AddEmitter(e) => DataCommand::AddEmitter(e),
            P::ClearEmitters => DataCommand::ClearEmitters,
            P::SetSpeed(s) => DataCommand::SetSpeed(s),
            P::SetForceSet(d) => DataCommand::SetForceSet(d),
            other @ (P::SetCouplingEntryHandler(_)
            | P::SetCouplingForces(_)
            | P::SetForces(_)
            | P::SetLaggedRepulsion { .. }
            | P::SetStepBudget(_)) => return Err(other),
        })
    }
}

#[cfg(test)]
mod tests;
