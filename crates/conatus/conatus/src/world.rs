// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
};

use crate::{
    BodyDesc, BodyId, BodyKind, BodyState, CharacterConfig, CharacterMove, ColliderDesc,
    ColliderId, ColliderShape, SpatialFilter, Transform, Velocity, VoxelChange, VoxelEdit,
};

mod queries;
mod rapier;

pub use queries::{ShapeContact, VoxelBox, VoxelCells};
use rapier::RapierBodyBackend;

#[derive(Clone, Debug, PartialEq)]
pub enum BodyError {
    UnknownBody(BodyId),
    UnknownCollider(ColliderId),
    InvalidBody(&'static str),
    InvalidCollider {
        part: u32,
        reason: &'static str,
    },
    InvalidQuery(&'static str),
    InvalidOperation {
        body: BodyId,
        operation: &'static str,
    },
    NotVoxelCollider(ColliderId),
}

impl fmt::Display for BodyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownBody(id) => write!(f, "unknown or stale body id {}", id.raw()),
            Self::UnknownCollider(id) => write!(
                f,
                "unknown collider part {} on body {}",
                id.part(),
                id.body().raw()
            ),
            Self::InvalidBody(reason) => write!(f, "invalid body: {reason}"),
            Self::InvalidCollider { part, reason } => {
                write!(f, "invalid collider part {part}: {reason}")
            },
            Self::InvalidQuery(reason) => write!(f, "invalid spatial query: {reason}"),
            Self::InvalidOperation { body, operation } => {
                write!(f, "body {} does not support {operation}", body.raw())
            },
            Self::NotVoxelCollider(id) => write!(
                f,
                "collider part {} on body {} is not a voxel grid",
                id.part(),
                id.body().raw()
            ),
        }
    }
}

impl Error for BodyError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Interaction {
    Contact,
    Sensor,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InteractionState {
    Started,
    Stopped,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InteractionEvent {
    pub tick: u64,
    pub state: InteractionState,
    pub interaction: Interaction,
    pub a: ColliderId,
    pub b: ColliderId,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayHit {
    pub collider: ColliderId,
    pub distance: f32,
    pub point: [f32; 3],
    pub normal: [f32; 3],
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct StepUpdate {
    pub tick: u64,
    pub revision: u64,
    pub changed: Vec<BodyState>,
    pub removed: Vec<BodyId>,
    pub voxel_changes: Vec<VoxelChange>,
    pub interactions: Vec<InteractionEvent>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VoxelEditSummary {
    pub previous_revision: u64,
    pub revision: u64,
    pub requested: usize,
    pub changed: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct InteractionKey {
    interaction: Interaction,
    a: ColliderId,
    b: ColliderId,
}

impl InteractionKey {
    fn new(interaction: Interaction, a: ColliderId, b: ColliderId) -> Option<Self> {
        if a.body() == b.body() {
            return None;
        }
        let (a, b) = if a <= b { (a, b) } else { (b, a) };
        Some(Self { interaction, a, b })
    }

    fn event(self, tick: u64, state: InteractionState) -> InteractionEvent {
        InteractionEvent {
            tick,
            state,
            interaction: self.interaction,
            a: self.a,
            b: self.b,
        }
    }

    fn contains(self, body: BodyId) -> bool {
        self.a.body() == body || self.b.body() == body
    }
}

struct Slot {
    generation: u32,
    occupied: bool,
    collider_revisions: Vec<u64>,
}

impl Slot {
    fn vacant(generation: u32) -> Self {
        Self {
            generation,
            occupied: false,
            collider_revisions: Vec::new(),
        }
    }
}

/// A host-neutral 3D tactile world.
///
/// The backend is deliberately private. Callers retain Conatus ids, arrays,
/// descriptors, and frame changes instead of backend handles.
pub struct BodyWorld {
    backend: RapierBodyBackend,
    slots: Vec<Slot>,
    free: Vec<u32>,
    tick: u64,
    revision: u64,
    dirty_bodies: BTreeSet<BodyId>,
    pending_removed: Vec<BodyId>,
    pending_voxel_changes: Vec<VoxelChange>,
    active_interactions: BTreeSet<InteractionKey>,
    pending_events: Vec<InteractionEvent>,
    /// Bodies whose colliders queries may not see yet (ruling 352).
    query_stale: BTreeSet<BodyId>,
}

impl Default for BodyWorld {
    fn default() -> Self {
        Self::new([0.0, -9.81, 0.0])
    }
}

fn validate_body(desc: &BodyDesc) -> Result<(), BodyError> {
    validate_transform(desc.transform).map_err(BodyError::InvalidBody)?;
    if !finite3(desc.velocity.linear) || !finite3(desc.velocity.angular) {
        return Err(BodyError::InvalidBody("velocity must be finite"));
    }
    for (value, name) in [
        (desc.linear_damping, "linear damping"),
        (desc.angular_damping, "angular damping"),
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(BodyError::InvalidBody(match name {
                "linear damping" => "linear damping must be finite and non-negative",
                _ => "angular damping must be finite and non-negative",
            }));
        }
    }
    if !desc.gravity_scale.is_finite() {
        return Err(BodyError::InvalidBody("gravity scale must be finite"));
    }
    for (part, collider) in desc.colliders.iter().enumerate() {
        validate_collider(collider).map_err(|reason| BodyError::InvalidCollider {
            part: part as u32,
            reason,
        })?;
    }
    Ok(())
}

fn validate_character(config: CharacterConfig) -> Result<(), BodyError> {
    if !finite3(config.up) || squared_length(config.up) <= f32::EPSILON {
        return Err(BodyError::InvalidBody(
            "character up direction must be finite and non-zero",
        ));
    }
    if !config.offset.is_finite() || config.offset <= 0.0 {
        return Err(BodyError::InvalidBody(
            "character offset must be finite and positive",
        ));
    }
    for angle in [config.max_slope_climb_angle, config.min_slope_slide_angle] {
        if !angle.is_finite() || !(0.0..=std::f32::consts::PI).contains(&angle) {
            return Err(BodyError::InvalidBody(
                "character slope angles must be finite and between zero and pi",
            ));
        }
    }
    if config
        .snap_to_ground
        .is_some_and(|distance| !distance.is_finite() || distance < 0.0)
    {
        return Err(BodyError::InvalidBody(
            "character ground snap must be finite and non-negative",
        ));
    }
    if config.autostep.is_some_and(|step| {
        !step.max_height.is_finite()
            || step.max_height <= 0.0
            || !step.min_width.is_finite()
            || step.min_width <= 0.0
    }) {
        return Err(BodyError::InvalidBody(
            "character autostep dimensions must be finite and positive",
        ));
    }
    Ok(())
}

fn validate_collider(desc: &ColliderDesc) -> Result<(), &'static str> {
    validate_transform(desc.local_transform)?;
    validate_shape(&desc.shape)?;
    if !desc.material.density.is_finite() || desc.material.density < 0.0 {
        return Err("density must be finite and non-negative");
    }
    if !desc.material.friction.is_finite() || desc.material.friction < 0.0 {
        return Err("friction must be finite and non-negative");
    }
    if !desc.material.restitution.is_finite() || !(0.0..=1.0).contains(&desc.material.restitution) {
        return Err("restitution must be finite and between zero and one");
    }
    Ok(())
}

fn validate_shape(shape: &ColliderShape) -> Result<(), &'static str> {
    match shape {
        ColliderShape::Sphere { radius } => positive(*radius, "sphere radius"),
        ColliderShape::Box { half_extents } => positive3(*half_extents, "box half-extents"),
        ColliderShape::CapsuleY {
            half_height,
            radius,
        } => {
            positive(*half_height, "capsule half-height")?;
            positive(*radius, "capsule radius")
        },
        ColliderShape::CylinderY {
            half_height,
            radius,
        } => {
            positive(*half_height, "cylinder half-height")?;
            positive(*radius, "cylinder radius")
        },
        ColliderShape::VoxelGrid { cell_size, .. } => positive3(*cell_size, "voxel cell size"),
    }
}

fn validate_transform(transform: Transform) -> Result<(), &'static str> {
    if !finite3(transform.translation) || !transform.rotation.iter().all(|v| v.is_finite()) {
        return Err("transform values must be finite");
    }
    let norm = transform.rotation.iter().map(|v| v * v).sum::<f32>();
    if norm <= f32::EPSILON {
        return Err("rotation quaternion must be non-zero");
    }
    Ok(())
}

fn positive(value: f32, name: &'static str) -> Result<(), &'static str> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(match name {
            "sphere radius" => "sphere radius must be finite and positive",
            "capsule half-height" => "capsule half-height must be finite and positive",
            "capsule radius" => "capsule radius must be finite and positive",
            "cylinder half-height" => "cylinder half-height must be finite and positive",
            "cylinder radius" => "cylinder radius must be finite and positive",
            _ => "shape dimension must be finite and positive",
        })
    }
}

fn positive3(values: [f32; 3], name: &'static str) -> Result<(), &'static str> {
    if values.iter().all(|value| value.is_finite() && *value > 0.0) {
        Ok(())
    } else {
        Err(match name {
            "box half-extents" => "box half-extents must be finite and positive",
            "voxel cell size" => "voxel cell size must be finite and positive",
            _ => "shape dimensions must be finite and positive",
        })
    }
}

fn finite3(value: [f32; 3]) -> bool {
    value.iter().all(|value| value.is_finite())
}

fn squared_length(value: [f32; 3]) -> f32 {
    value.iter().map(|v| v * v).sum()
}

#[cfg(test)]
#[path = "world/tests.rs"]
mod tests;

#[path = "world/bodies.rs"]
mod bodies;
