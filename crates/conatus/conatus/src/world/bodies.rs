// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

use super::*;

impl BodyWorld {
    pub fn new(gravity: [f32; 3]) -> Self {
        Self::try_new(gravity).expect("BodyWorld gravity must be finite")
    }

    pub fn try_new(gravity: [f32; 3]) -> Result<Self, BodyError> {
        if !finite3(gravity) {
            return Err(BodyError::InvalidBody("gravity must be finite"));
        }
        Ok(Self {
            backend: RapierBodyBackend::new(gravity),
            slots: Vec::new(),
            free: Vec::new(),
            tick: 0,
            revision: 0,
            dirty_bodies: BTreeSet::new(),
            pending_removed: Vec::new(),
            pending_voxel_changes: Vec::new(),
            active_interactions: BTreeSet::new(),
            pending_events: Vec::new(),
            query_stale: BTreeSet::new(),
        })
    }

    pub const fn tick(&self) -> u64 {
        self.tick
    }

    pub const fn revision(&self) -> u64 {
        self.revision
    }

    pub fn len(&self) -> usize {
        self.slots.iter().filter(|slot| slot.occupied).count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn contains(&self, id: BodyId) -> bool {
        self.slot(id).is_some()
    }

    pub fn gravity(&self) -> [f32; 3] {
        self.backend.gravity()
    }

    pub fn set_gravity(&mut self, gravity: [f32; 3]) -> Result<(), BodyError> {
        if !finite3(gravity) {
            return Err(BodyError::InvalidBody("gravity must be finite"));
        }
        self.backend.set_gravity(gravity);
        self.bump_revision();
        Ok(())
    }

    pub fn spawn(&mut self, desc: BodyDesc) -> Result<BodyId, BodyError> {
        validate_body(&desc)?;
        let id = self.reserve_id();

        self.backend.insert(id, desc);

        let revision = self.bump_revision();
        let slot = &mut self.slots[id.slot() as usize];
        slot.occupied = true;
        slot.collider_revisions = vec![revision; self.backend.collider_count(id)];
        self.dirty_bodies.insert(id);
        self.query_stale.insert(id);
        Ok(id)
    }

    pub fn despawn(&mut self, id: BodyId) -> Result<BodyState, BodyError> {
        let state = self.state(id).ok_or(BodyError::UnknownBody(id))?;
        let ended: Vec<_> = self
            .active_interactions
            .iter()
            .copied()
            .filter(|interaction| interaction.contains(id))
            .collect();
        for interaction in ended {
            self.active_interactions.remove(&interaction);
            self.pending_events
                .push(interaction.event(self.tick, InteractionState::Stopped));
        }

        self.backend.remove(id)?;
        let slot = &mut self.slots[id.slot() as usize];
        slot.occupied = false;
        slot.collider_revisions.clear();
        slot.generation = slot.generation.wrapping_add(1).max(1);
        self.free.push(id.slot());
        self.dirty_bodies.remove(&id);
        self.query_stale.remove(&id);
        self.pending_removed.push(id);
        self.bump_revision();
        Ok(state)
    }

    pub fn state(&self, id: BodyId) -> Option<BodyState> {
        self.slot(id)?;
        self.backend.state(id)
    }

    /// A stable-slot-ordered state snapshot for save preparation, renderer
    /// bootstrap, or a newly attached consumer.
    pub fn states(&self) -> Vec<BodyState> {
        self.ids().filter_map(|id| self.state(id)).collect()
    }

    pub fn ids(&self) -> impl Iterator<Item = BodyId> + '_ {
        self.slots
            .iter()
            .enumerate()
            .filter(|(_, entry)| entry.occupied)
            .map(|(slot, entry)| BodyId::from_parts(slot as u32, entry.generation))
    }

    pub fn collider_ids(&self, body: BodyId) -> Result<Vec<ColliderId>, BodyError> {
        let slot = self.slot(body).ok_or(BodyError::UnknownBody(body))?;
        Ok((0..slot.collider_revisions.len())
            .map(|part| ColliderId::new(body, part as u32))
            .collect())
    }

    pub fn collider_revision(&self, collider: ColliderId) -> Result<u64, BodyError> {
        let (slot, index) = self.collider_slot(collider)?;
        Ok(slot.collider_revisions[index])
    }

    pub fn set_transform(
        &mut self,
        id: BodyId,
        transform: Transform,
        wake: bool,
    ) -> Result<(), BodyError> {
        validate_transform(transform).map_err(BodyError::InvalidBody)?;
        self.ensure_body(id)?;
        self.backend.set_transform(id, transform, wake)?;
        self.dirty_bodies.insert(id);
        self.query_stale.insert(id);
        self.bump_revision();
        Ok(())
    }

    pub fn set_next_kinematic_transform(
        &mut self,
        id: BodyId,
        transform: Transform,
    ) -> Result<(), BodyError> {
        validate_transform(transform).map_err(BodyError::InvalidBody)?;
        self.ensure_body(id)?;
        self.backend.set_next_kinematic_transform(id, transform)?;
        self.bump_revision();
        Ok(())
    }

    pub fn set_velocity(
        &mut self,
        id: BodyId,
        velocity: Velocity,
        wake: bool,
    ) -> Result<(), BodyError> {
        if !finite3(velocity.linear) || !finite3(velocity.angular) {
            return Err(BodyError::InvalidBody("velocity must be finite"));
        }
        self.ensure_body(id)?;
        self.backend.set_velocity(id, velocity, wake)?;
        self.dirty_bodies.insert(id);
        self.bump_revision();
        Ok(())
    }

    pub fn set_kind(&mut self, id: BodyId, kind: BodyKind) -> Result<(), BodyError> {
        self.ensure_body(id)?;
        self.backend.set_kind(id, kind)?;
        self.dirty_bodies.insert(id);
        self.query_stale.insert(id);
        self.bump_revision();
        Ok(())
    }

    pub fn apply_force(&mut self, id: BodyId, force: [f32; 3]) -> Result<(), BodyError> {
        if !finite3(force) {
            return Err(BodyError::InvalidBody("force must be finite"));
        }
        self.ensure_body(id)?;
        self.backend.apply_force(id, force)?;
        self.bump_revision();
        Ok(())
    }

    pub fn apply_torque(&mut self, id: BodyId, torque: [f32; 3]) -> Result<(), BodyError> {
        if !finite3(torque) {
            return Err(BodyError::InvalidBody("torque must be finite"));
        }
        self.ensure_body(id)?;
        self.backend.apply_torque(id, torque)?;
        self.bump_revision();
        Ok(())
    }

    pub fn apply_impulse(&mut self, id: BodyId, impulse: [f32; 3]) -> Result<(), BodyError> {
        if !finite3(impulse) {
            return Err(BodyError::InvalidBody("impulse must be finite"));
        }
        self.ensure_body(id)?;
        self.backend.apply_impulse(id, impulse)?;
        self.bump_revision();
        Ok(())
    }

    pub fn apply_torque_impulse(&mut self, id: BodyId, impulse: [f32; 3]) -> Result<(), BodyError> {
        if !finite3(impulse) {
            return Err(BodyError::InvalidBody("torque impulse must be finite"));
        }
        self.ensure_body(id)?;
        self.backend.apply_torque_impulse(id, impulse)?;
        self.bump_revision();
        Ok(())
    }

    /// Constrain a requested translation against the world and queue the
    /// resulting target on a position-kinematic body. The selected collider is
    /// the character shape; its sibling colliders move with the same body.
    pub fn move_character(
        &mut self,
        collider: ColliderId,
        requested: [f32; 3],
        dt: f32,
        config: CharacterConfig,
    ) -> Result<CharacterMove, BodyError> {
        validate_character(config)?;
        if !finite3(requested) || !dt.is_finite() || dt <= 0.0 {
            return Err(BodyError::InvalidOperation {
                body: collider.body(),
                operation: "finite character movement with a positive step",
            });
        }
        self.collider_slot(collider)?;
        let movement = self
            .backend
            .move_character(collider, requested, dt, config)?;
        self.bump_revision();
        Ok(movement)
    }

    /// Apply sparse edits directly to a voxel collider's acceleration
    /// structure. The collider retains its identity and material.
    pub fn edit_voxels(
        &mut self,
        collider: ColliderId,
        edits: impl IntoIterator<Item = VoxelEdit>,
    ) -> Result<VoxelEditSummary, BodyError> {
        let edits: Vec<_> = edits.into_iter().collect();
        let previous_revision = self.collider_revision(collider)?;
        self.collider_slot(collider)?;
        let effective = self.backend.edit_voxels(collider, &edits)?;

        if effective.is_empty() {
            return Ok(VoxelEditSummary {
                previous_revision,
                revision: previous_revision,
                requested: edits.len(),
                changed: 0,
            });
        }

        let revision = self.bump_revision();
        let (slot, index) = self.collider_slot_mut(collider)?;
        slot.collider_revisions[index] = revision;
        self.dirty_bodies.insert(collider.body());
        self.query_stale.insert(collider.body());
        let changed = effective.len();
        self.pending_voxel_changes.push(VoxelChange {
            collider,
            previous_revision,
            revision,
            edits: effective,
        });
        Ok(VoxelEditSummary {
            previous_revision,
            revision,
            requested: edits.len(),
            changed,
        })
    }

    pub fn raycast(
        &self,
        origin: [f32; 3],
        direction: [f32; 3],
        max_distance: f32,
        solid: bool,
        filter: SpatialFilter,
    ) -> Result<Option<RayHit>, BodyError> {
        if !finite3(origin) || !finite3(direction) || !max_distance.is_finite() {
            return Err(BodyError::InvalidQuery("ray values must be finite"));
        }
        if max_distance < 0.0 {
            return Err(BodyError::InvalidQuery(
                "ray maximum distance must not be negative",
            ));
        }
        if !filter.include_sensors && !filter.include_solids {
            return Ok(None);
        }
        let direction_length = squared_length(direction).sqrt();
        if direction_length <= f32::EPSILON {
            return Err(BodyError::InvalidQuery("ray direction must be non-zero"));
        }
        self.backend.raycast(
            origin,
            direction,
            direction_length,
            max_distance,
            solid,
            filter,
        )
    }

    pub fn overlaps(
        &self,
        transform: Transform,
        shape: &ColliderShape,
        filter: SpatialFilter,
    ) -> Result<Vec<ColliderId>, BodyError> {
        validate_transform(transform).map_err(BodyError::InvalidQuery)?;
        validate_shape(shape).map_err(BodyError::InvalidQuery)?;
        if !filter.include_sensors && !filter.include_solids {
            return Ok(Vec::new());
        }
        self.backend.overlaps(transform, shape, filter)
    }

    pub fn step(&mut self, dt: f32) -> Result<StepUpdate, BodyError> {
        if !dt.is_finite() || dt <= 0.0 {
            return Err(BodyError::InvalidBody(
                "simulation step must be finite and greater than zero",
            ));
        }

        let before: BTreeMap<_, _> = self
            .ids()
            .filter_map(|id| self.state(id).map(|state| (id, state.transform)))
            .collect();
        self.backend.step(dt);
        self.query_stale.clear();
        self.tick = self.tick.saturating_add(1);
        let revision = self.bump_revision();

        let current_interactions = self.backend.collect_interactions();
        let mut interactions = self.drain_events();
        let removed = self.drain_removed();
        let voxel_changes = self.drain_voxel_changes();
        interactions.extend(
            current_interactions
                .difference(&self.active_interactions)
                .copied()
                .map(|key| key.event(self.tick, InteractionState::Started)),
        );
        interactions.extend(
            self.active_interactions
                .difference(&current_interactions)
                .copied()
                .map(|key| key.event(self.tick, InteractionState::Stopped)),
        );
        self.active_interactions = current_interactions;

        let dirty = std::mem::take(&mut self.dirty_bodies);
        let changed = self
            .ids()
            .filter(|id| {
                dirty.contains(id)
                    || self
                        .state(*id)
                        .is_some_and(|state| before.get(id) != Some(&state.transform))
            })
            .filter_map(|id| self.state(id))
            .collect();

        Ok(StepUpdate {
            tick: self.tick,
            revision,
            changed,
            removed,
            voxel_changes,
            interactions,
        })
    }

    /// Interaction stops caused by a despawn are available before another
    /// physics step. Engine frames drain these even when elapsed time produces
    /// zero fixed steps.
    pub fn drain_events(&mut self) -> Vec<InteractionEvent> {
        std::mem::take(&mut self.pending_events)
    }

    /// Drain bodies changed by commands since the last step. A renderer can
    /// use this on a zero-step frame so spawns, despawns of its own handles,
    /// teleports, and voxel edits are not held until simulation advances.
    pub fn drain_changes(&mut self) -> Vec<BodyState> {
        let dirty = std::mem::take(&mut self.dirty_bodies);
        dirty.into_iter().filter_map(|id| self.state(id)).collect()
    }

    /// Drain stable ids removed since the previous step or drain. Renderers
    /// and other projections use this to retire their own instances.
    pub fn drain_removed(&mut self) -> Vec<BodyId> {
        std::mem::take(&mut self.pending_removed)
    }

    pub fn drain_voxel_changes(&mut self) -> Vec<VoxelChange> {
        std::mem::take(&mut self.pending_voxel_changes)
    }

    pub(super) fn reserve_id(&mut self) -> BodyId {
        if let Some(slot) = self.free.pop() {
            let generation = self.slots[slot as usize].generation;
            BodyId::from_parts(slot, generation)
        } else {
            let slot = self.slots.len() as u32;
            self.slots.push(Slot::vacant(1));
            BodyId::from_parts(slot, 1)
        }
    }

    pub(super) fn slot(&self, id: BodyId) -> Option<&Slot> {
        let slot = self.slots.get(id.slot() as usize)?;
        (slot.generation == id.generation() && slot.occupied).then_some(slot)
    }

    pub(super) fn ensure_body(&self, id: BodyId) -> Result<(), BodyError> {
        self.slot(id).map(|_| ()).ok_or(BodyError::UnknownBody(id))
    }

    pub(super) fn collider_slot(&self, id: ColliderId) -> Result<(&Slot, usize), BodyError> {
        let slot = self
            .slot(id.body())
            .ok_or(BodyError::UnknownBody(id.body()))?;
        let index = id.part() as usize;
        if index >= slot.collider_revisions.len() {
            return Err(BodyError::UnknownCollider(id));
        }
        Ok((slot, index))
    }

    pub(super) fn collider_slot_mut(
        &mut self,
        id: ColliderId,
    ) -> Result<(&mut Slot, usize), BodyError> {
        let index = id.part() as usize;
        let slot = self
            .slots
            .get_mut(id.body().slot() as usize)
            .filter(|slot| slot.generation == id.body().generation() && slot.occupied)
            .ok_or(BodyError::UnknownBody(id.body()))?;
        if index >= slot.collider_revisions.len() {
            return Err(BodyError::UnknownCollider(id));
        }
        Ok((slot, index))
    }

    pub(super) fn bump_revision(&mut self) -> u64 {
        self.revision = self.revision.saturating_add(1);
        self.revision
    }
}
