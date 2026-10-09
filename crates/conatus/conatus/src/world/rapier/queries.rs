// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! The Rapier side of `BodyWorld`'s refresh, point, voxel and contact queries.

use rapier3d::geometry::ContactManifold;
use rapier3d::parry::bounding_volume::BoundingVolume;
use rapier3d::parry::shape::{VoxelData, Voxels};

use super::{RapierBodyBackend, array, ivector, pose, shared_shape, vector};
use crate::world::{BodyError, ShapeContact, VoxelBox};
use crate::{BodyId, ColliderId, ColliderShape, SpatialFilter, Transform};

/// Edge of the blocks `VoxelCellWalk` orders by. It must be a multiple of
/// parry's 3D voxel chunk edge (8, private to parry), so each block lies in
/// one chunk and parry yields its cells in x, y, z order; 8 is also T2's
/// brick grain (ruling 331). The canonical-order tests catch a parry change.
const BLOCK: i32 = 8;

/// Added to twice the prediction to form the margin parry is asked for.
/// Parry's voxel candidate search loosens each shape's box by half the
/// margin over half-open cell ranges, so it finds only gaps below half the
/// margin, and at margin 0 it drops exact touching, convex pairs included.
/// Measured over all 25 shape pairs at gaps on either side of the
/// prediction (2026-10-09, `b01` log): 28 of 350 cases missed at margin =
/// prediction, 12 at twice it, none at twice plus 1/1024 or plus 1/64. The
/// slack is 1/64 for headroom; points beyond the prediction are dropped.
const CONTACT_SLACK: f32 = 1.0 / 64.0;

fn contact_margin(prediction: f32) -> f32 {
    2.0 * prediction + CONTACT_SLACK
}

impl RapierBodyBackend {
    /// Sync one body's colliders into the query structures. Rapier's own
    /// `detect_collisions` would do this but drops the island bookkeeping of
    /// bodies inserted since the last step, which then never simulate
    /// (measured 2026-10-09), so only the cached pose and the broad-phase
    /// entry are written; the next step still sees every change it owes.
    pub(in crate::world) fn refresh_body(&mut self, id: BodyId) {
        let Some(entry) = self.bodies.get(&id) else {
            return;
        };
        let params = self.physics.integration_parameters;
        let body_pose = *self.physics.bodies[entry.body].position();
        for &handle in &entry.colliders {
            let collider = &self.physics.colliders[handle];
            let current = collider
                .position_wrt_parent()
                .map_or(*collider.position(), |local| body_pose * *local);
            if *collider.position() != current {
                self.physics.colliders[handle].set_position(current);
            }
            let aabb = self.physics.colliders[handle]
                .compute_broad_phase_aabb(&params, &self.physics.bodies);
            self.physics.broad_phase.set_aabb(&params, handle, aabb);
        }
    }

    pub(in crate::world) fn colliders_at_point(
        &self,
        point: [f32; 3],
        filter: SpatialFilter,
    ) -> Result<Vec<ColliderId>, BodyError> {
        let filter = self.query_filter(filter)?;
        let mut hits: Vec<_> = self
            .physics
            .intersect_point(vector(point), filter)
            .filter_map(|(handle, _)| self.collider_id(handle))
            .collect();
        hits.sort_unstable();
        hits.dedup();
        Ok(hits)
    }

    pub(in crate::world) fn voxel_filled(
        &self,
        collider: ColliderId,
        cell: [i32; 3],
    ) -> Result<bool, BodyError> {
        Ok(self
            .voxels(collider)?
            .voxel_state(ivector(cell))
            .is_some_and(|state| !state.is_empty()))
    }

    pub(in crate::world) fn voxel_cells(
        &self,
        collider: ColliderId,
        within: Option<VoxelBox>,
    ) -> Result<VoxelCellWalk<'_>, BodyError> {
        let voxels = self.voxels(collider)?;
        let [mins, maxs] = voxels.domain();
        let mut min = [mins.x, mins.y, mins.z];
        let mut max = [maxs.x, maxs.y, maxs.z];
        if let Some(bounds) = within {
            for axis in 0..3 {
                min[axis] = min[axis].max(bounds.min[axis]);
                max[axis] = max[axis].min(bounds.max[axis]);
            }
        }
        let pending = if (0..3).all(|axis| min[axis] < max[axis]) {
            vec![(min, max)]
        } else {
            Vec::new()
        };
        Ok(VoxelCellWalk {
            voxels,
            pending,
            current: None,
        })
    }

    pub(in crate::world) fn contacts(
        &self,
        transform: Transform,
        shape: &ColliderShape,
        prediction: f32,
        filter: SpatialFilter,
    ) -> Result<Vec<ShapeContact>, BodyError> {
        let query = shared_shape(shape);
        let query_pose = pose(transform);
        let margin = contact_margin(prediction);
        let search = query.compute_aabb(&query_pose).loosened(margin);
        let dispatcher = self.physics.narrow_phase.query_dispatcher();
        let mut manifolds: Vec<ContactManifold> = Vec::new();
        let mut contacts = Vec::new();
        for (handle, world_collider) in self
            .physics
            .intersect_aabb_conservative(search, self.query_filter(filter)?)
        {
            let Some(collider) = self.collider_id(handle) else {
                continue;
            };
            // The world collider is shape 1, so points and normals are on
            // its side whatever order parry runs the pair in internally.
            let collider_pose = *world_collider.position();
            let pos12 = collider_pose.inv_mul(&query_pose);
            let mut workspace = None;
            manifolds.clear();
            dispatcher
                .contact_manifolds(
                    &pos12,
                    world_collider.shape(),
                    query.as_ref(),
                    margin,
                    &mut manifolds,
                    &mut workspace,
                )
                .map_err(|_| BodyError::InvalidQuery("the backend cannot test this shape pair"))?;
            for manifold in &manifolds {
                // Composite shapes (voxel grids) give points in the touched
                // sub-shape's frame.
                let side = manifold
                    .subshape_pos1()
                    .map_or(collider_pose, |sub| collider_pose * *sub);
                let normal = array(side.rotation * manifold.local_n1);
                contacts.extend(
                    manifold
                        .points
                        .iter()
                        .filter(|point| point.dist <= prediction)
                        .map(|point| ShapeContact {
                            collider,
                            point: array(side * point.local_p1),
                            normal,
                            distance: point.dist,
                        }),
                );
            }
        }
        contacts.sort_by(|a, b| {
            a.collider
                .cmp(&b.collider)
                .then(a.distance.total_cmp(&b.distance))
                .then_with(|| total_order(a.point, b.point))
                .then_with(|| total_order(a.normal, b.normal))
        });
        Ok(contacts)
    }

    fn voxels(&self, collider: ColliderId) -> Result<&Voxels, BodyError> {
        let handle = self.collider_handle(collider)?;
        self.physics.colliders[handle]
            .shape()
            .as_voxels()
            .ok_or(BodyError::NotVoxelCollider(collider))
    }
}

/// Walks a cell box by halving it along x, then y, then z down to single
/// blocks, lower half first, skipping halves that hold no voxel. Blocks thus
/// come in ascending (x, y, z) order without sorting or copying any cells.
pub(in crate::world) struct VoxelCellWalk<'a> {
    voxels: &'a Voxels,
    pending: Vec<([i32; 3], [i32; 3])>,
    current: Option<Box<dyn Iterator<Item = VoxelData> + 'a>>,
}

impl Iterator for VoxelCellWalk<'_> {
    type Item = [i32; 3];

    fn next(&mut self) -> Option<[i32; 3]> {
        loop {
            if let Some(cells) = &mut self.current {
                if let Some(voxel) = cells.next() {
                    let cell = voxel.grid_coords;
                    return Some([cell.x, cell.y, cell.z]);
                }
                self.current = None;
            }
            let (min, max) = self.pending.pop()?;
            let mut range = self.voxels.voxels_in_range(ivector(min), ivector(max));
            let Some((axis, middle)) = block_split(min, max) else {
                self.current = Some(Box::new(range));
                continue;
            };
            if range.next().is_none() {
                continue;
            }
            let mut upper_min = min;
            upper_min[axis] = middle;
            let mut lower_max = max;
            lower_max[axis] = middle;
            self.pending.push((upper_min, max));
            self.pending.push((min, lower_max));
        }
    }
}

/// The first axis on which the box spans more than one block, and the block
/// boundary splitting it in half; `None` for a box inside one block.
fn block_split(min: [i32; 3], max: [i32; 3]) -> Option<(usize, i32)> {
    (0..3).find_map(|axis| {
        let first = min[axis].div_euclid(BLOCK);
        let last = (max[axis] - 1).div_euclid(BLOCK);
        (last > first).then(|| (axis, (first + (last - first + 1) / 2) * BLOCK))
    })
}

fn total_order(a: [f32; 3], b: [f32; 3]) -> std::cmp::Ordering {
    (0..3)
        .map(|axis| a[axis].total_cmp(&b[axis]))
        .find(|order| order.is_ne())
        .unwrap_or(std::cmp::Ordering::Equal)
}

#[cfg(test)]
mod tests {
    use rapier3d::parry::query::{
        ContactManifold, DefaultQueryDispatcher, PersistentQueryDispatcher,
    };
    use rapier3d::parry::shape::{Ball, Voxels};
    use rapier3d::prelude::{IVector, Pose, Vector};

    use super::contact_margin;

    /// Points parry reports within `prediction` for a ball over a voxel top
    /// face at `gap`, asking parry for `margin`.
    fn ball_over_voxels(gap: f32, margin: f32, prediction: f32) -> usize {
        let voxels = Voxels::new(Vector::splat(1.0), &[IVector::new(0, 0, 0)]);
        let ball = Ball::new(0.25);
        let pos12 = Pose::translation(0.5, 1.0 + gap + 0.25, 0.5);
        let mut manifolds: Vec<ContactManifold<(), ()>> = Vec::new();
        DefaultQueryDispatcher
            .contact_manifolds(&pos12, &voxels, &ball, margin, &mut manifolds, &mut None)
            .unwrap();
        manifolds
            .iter()
            .flat_map(|manifold| &manifold.points)
            .filter(|point| point.dist <= prediction)
            .count()
    }

    /// The upstream quirk the margin compensates (upstream candidates
    /// ledger, item 12). If this starts failing, parry changed: re-measure
    /// `CONTACT_SLACK` and update the ledger.
    /// Dyadic values, so the boundary is not left to f32 rounding.
    #[test]
    fn parry_still_needs_the_compensated_margin() {
        let (gap, prediction) = (0.0625, 0.125);
        assert_eq!(
            ball_over_voxels(gap, prediction, prediction),
            0,
            "uncompensated gap"
        );
        assert_eq!(ball_over_voxels(0.0, 0.0, 0.0), 0, "uncompensated touching");
        assert_eq!(
            ball_over_voxels(gap, contact_margin(prediction), prediction),
            1
        );
        assert_eq!(ball_over_voxels(0.0, contact_margin(0.0), 0.0), 1);
    }
}
