// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! The weighted spring set: structural edges and Meaning affinity pairs,
//! built as CSR on the host and uploaded once per [`ResidentGraph::set_springs`].

use super::{Buf, GraphError, ResidentGraph, SlotId};

/// The host's copy of the CSR, kept so growth can upload it again.
pub(super) struct Springs {
    /// One row per slot of the capacity, plus the end.
    offsets: Vec<u32>,
    /// Directed edges, three words each: target index, target generation,
    /// source generation.
    ids: Vec<u32>,
    weights: Vec<f32>,
    pub(super) edges: usize,
    pub(super) revision: u64,
}

impl Springs {
    pub(super) fn new(capacity: usize) -> Self {
        Self {
            offsets: vec![0; capacity + 1],
            ids: Vec::new(),
            weights: Vec::new(),
            edges: 0,
            revision: 0,
        }
    }

    /// New slots have empty rows.
    pub(super) fn resize(&mut self, capacity: usize) {
        let last = *self.offsets.last().unwrap_or(&0);
        self.offsets.resize(capacity + 1, last);
    }
}

impl ResidentGraph {
    /// The revision the caller stamped on the current spring set: which
    /// Meaning revision the springs reflect.
    pub fn spring_revision(&self) -> u64 {
        self.springs.revision
    }

    /// Replace the spring set with `pairs`, each undirected with a weight that
    /// scales `spring_k`, and stamp it `revision`. The whole call is refused,
    /// naming the first offender, if any pair names a dead or stale slot or a
    /// non-finite weight; nothing is applied then. A pair of one slot with
    /// itself is dropped.
    ///
    /// Springs follow their slots: removing a node silences its springs on the
    /// device, and a reused index never inherits them. The set is not edited
    /// for removals, so a host refreshes it when its pairs change.
    pub fn set_springs(
        &mut self,
        revision: u64,
        pairs: &[(SlotId, SlotId, f32)],
    ) -> Result<(), GraphError> {
        for (pair, &(a, b, weight)) in pairs.iter().enumerate() {
            for slot in [a, b] {
                if self.check(slot).is_err() {
                    return Err(GraphError::SpringPair { pair, slot });
                }
            }
            if !weight.is_finite() {
                return Err(GraphError::NotFinite);
            }
        }
        let rows = self.capacity;
        let mut offsets = vec![0u32; rows + 1];
        let kept = || pairs.iter().filter(|(a, b, _)| a.index != b.index);
        for (a, b, _) in kept() {
            offsets[a.index as usize + 1] += 1;
            offsets[b.index as usize + 1] += 1;
        }
        for row in 0..rows {
            offsets[row + 1] += offsets[row];
        }
        let edges = offsets[rows] as usize;
        let mut cursor = offsets.clone();
        let mut ids = vec![0u32; edges * 3];
        let mut weights = vec![0f32; edges];
        for (a, b, weight) in kept() {
            for (from, to) in [(a, b), (b, a)] {
                let e = cursor[from.index as usize] as usize;
                cursor[from.index as usize] += 1;
                ids[e * 3..e * 3 + 3].copy_from_slice(&[to.index, to.generation, from.generation]);
                weights[e] = *weight;
            }
        }
        self.springs = Springs {
            offsets,
            ids,
            weights,
            edges,
            revision,
        };
        self.upload_springs();
        Ok(())
    }

    /// Send the host's CSR to the device: the offsets in place, the edges as
    /// fresh allocations sized to them.
    pub(super) fn upload_springs(&mut self) {
        let offsets = self.springs.offsets.clone();
        self.put(Buf::Offsets, 0, bytemuck::cast_slice(&offsets));
        // An empty set still needs a bound allocation; one zeroed edge is
        // never read because every row is empty.
        let ids: &[u32] = if self.springs.edges == 0 {
            &[0; 3]
        } else {
            &self.springs.ids
        };
        let weights: &[f32] = if self.springs.edges == 0 {
            &[0.0]
        } else {
            &self.springs.weights
        };
        self.bufs.edge_ids = self.client.create_from_slice(bytemuck::cast_slice(ids));
        self.bufs.edge_weights = self.client.create_from_slice(bytemuck::cast_slice(weights));
        self.stats.uploaded_bytes += (ids.len() * 4 + weights.len() * 4) as u64;
    }
}
