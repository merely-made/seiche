// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! One layout step: clear settle, bin, exclusion, weighted springs, then
//! integration with pins applied.

use cubecl::prelude::*;

use super::{CELL_SLACK, ResidentGraph, groups, kernels};
use crate::resident::{binning, kernels as law};

impl ResidentGraph {
    /// Queue the force dispatches without integrating: settle cleared, cells
    /// binned, exclusion, springs. [`Self::step`] is this plus integration;
    /// a receipt reads [`Self::read_forces`] after it.
    pub fn compute_forces(&mut self) {
        let n = self.used;
        if n == 0 {
            return;
        }
        let dim = CubeDim::new_1d(law::CUBE_DIM);
        let capacity = self.capacity;
        let table = self.table;
        let exclusion = self.params.exclusion;
        let inv_cell = 1.0 / (exclusion.cutoff.max(1e-6) * CELL_SLACK);
        let mask = table as u32 - 1;
        let b = &self.bufs;
        let client = &self.client;
        unsafe {
            law::clear_settle::launch_unchecked(
                client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                BufferArg::from_raw_parts(b.settle.clone(), 1),
            );
            binning::clear::launch_unchecked(
                client,
                groups(table as u32 + 1),
                dim,
                BufferArg::from_raw_parts(b.counts.clone(), table + 1),
                table as u32 + 1,
            );
            kernels::count_buckets::launch_unchecked(
                client,
                groups(n),
                dim,
                BufferArg::from_raw_parts(b.positions.clone(), capacity * 4),
                BufferArg::from_raw_parts(b.flags.clone(), capacity),
                BufferArg::from_raw_parts(b.counts.clone(), table + 1),
                n,
                inv_cell,
                mask,
            );
        }
        binning::exclusive_scan(client, &b.counts, table + 1);
        unsafe {
            binning::cursors::launch_unchecked(
                client,
                groups(table as u32),
                dim,
                BufferArg::from_raw_parts(b.counts.clone(), table + 1),
                BufferArg::from_raw_parts(b.cursors.clone(), table),
                table as u32,
            );
            kernels::scatter_buckets::launch_unchecked(
                client,
                groups(n),
                dim,
                BufferArg::from_raw_parts(b.positions.clone(), capacity * 4),
                BufferArg::from_raw_parts(b.flags.clone(), capacity),
                BufferArg::from_raw_parts(b.cursors.clone(), table),
                BufferArg::from_raw_parts(b.sorted.clone(), capacity * 4),
                n,
                inv_cell,
                mask,
            );
            kernels::exclude_buckets::launch_unchecked(
                client,
                groups(n),
                dim,
                BufferArg::from_raw_parts(b.positions.clone(), capacity * 4),
                BufferArg::from_raw_parts(b.sorted.clone(), capacity * 4),
                BufferArg::from_raw_parts(b.counts.clone(), table + 1),
                BufferArg::from_raw_parts(b.flags.clone(), capacity),
                BufferArg::from_raw_parts(b.forces.clone(), capacity * 4),
                n,
                inv_cell,
                mask,
                exclusion.strength,
                exclusion.cutoff * exclusion.cutoff,
                exclusion.min_distance,
            );
            if self.springs.edges > 0 {
                kernels::springs_weighted::launch_unchecked(
                    client,
                    groups(n),
                    dim,
                    BufferArg::from_raw_parts(b.positions.clone(), capacity * 4),
                    BufferArg::from_raw_parts(b.forces.clone(), capacity * 4),
                    BufferArg::from_raw_parts(b.flags.clone(), capacity),
                    BufferArg::from_raw_parts(b.generations.clone(), capacity),
                    BufferArg::from_raw_parts(b.offsets.clone(), capacity + 1),
                    BufferArg::from_raw_parts(b.edge_ids.clone(), self.springs.edges * 3),
                    BufferArg::from_raw_parts(b.edge_weights.clone(), self.springs.edges),
                    n,
                    self.params.spring_k,
                    self.params.rest_length,
                );
            }
        }
    }

    /// One layout step: forces, then integration with pins applied. Positions
    /// update in place and the revision advances. Queues only; see the module
    /// docs for the flush contract.
    pub fn step(&mut self) {
        let n = self.used;
        if n == 0 {
            return;
        }
        self.compute_forces();
        let capacity = self.capacity;
        let b = &self.bufs;
        unsafe {
            kernels::integrate_pinned::launch_unchecked(
                &self.client,
                groups(n),
                CubeDim::new_1d(law::CUBE_DIM),
                BufferArg::from_raw_parts(b.positions.clone(), capacity * 4),
                BufferArg::from_raw_parts(b.velocities.clone(), capacity * 4),
                BufferArg::from_raw_parts(b.forces.clone(), capacity * 4),
                BufferArg::from_raw_parts(b.flags.clone(), capacity),
                BufferArg::from_raw_parts(b.pin_targets.clone(), capacity * 4),
                BufferArg::from_raw_parts(b.settle.clone(), 1),
                n,
                self.params.dt,
                self.params.damping,
                self.params.centering,
            );
        }
        self.revision += 1;
    }
}
