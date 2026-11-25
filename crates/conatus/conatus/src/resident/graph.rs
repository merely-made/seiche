// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! The resident graph layout: one law integrated on the host's device, with
//! positions that stay there from step through drawing.
//!
//! The law is `seiche::NodeExclusion` (short range, with a cutoff) plus
//! weighted springs and a weak centering, damped. A host adds and removes
//! nodes, pins dragged ones, refreshes the spring pairs, and draws, picks and
//! labels from the published buffer; nothing reads the whole field back.
//!
//! # Slots
//!
//! Buffers are allocated at a capacity and indexed by slot. A [`SlotId`] is an
//! index plus a generation: removing a slot clears its alive bit and bumps the
//! generation, and the index is reused later. Every host write that names a
//! dead or stale slot is rejected, so a pointer drag that outlives its node
//! cannot move whoever took the index.
//!
//! # Submission contract
//!
//! [`ResidentGraph::step`] only queues work on the host's CubeCL client, which
//! owns the host's wgpu queue. A host that draws from the published buffer in
//! the same frame calls [`ResidentGraph::flush`] after `step`, submits its own
//! draw to the same queue, and issues picks and reads after that: queue order
//! then makes the draw, the pick and the labels all see the revision the
//! step produced. [`ResidentGraph::flush`] is CubeCL's own client flush, which
//! submits the pending command buffer to the queue.
//!
//! # Reallocation
//!
//! Growth doubles the capacity and so replaces the buffers. The leases carry
//! an `allocation_epoch` that changes only then, so a renderer rebuilds its
//! bind group on an epoch change and otherwise only re-reads the revision.

use cubecl::bytes::Bytes;
use cubecl::client::Client;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::exclusion::ExclusionParams;
use super::{
    ChunkStamp, PlaneElementType, ReadEpoch, ResidentClient, SpatialLease, binning, kernels as law,
};

mod kernels;
mod reads;
mod springs;

pub use reads::Pending;

/// Slot flag: the slot holds a node.
pub const SLOT_ALIVE: u32 = 1;
/// Slot flag: the slot is held at its pin target.
pub const SLOT_PINNED: u32 = 2;

/// Smallest capacity: a handful of nodes still gets a usable hash table.
const MIN_CAPACITY: usize = 64;
/// Cells are this much wider than the cutoff, so float rounding in a cell
/// index cannot put two bodies inside the cutoff two cells apart even at
/// large coordinates (1e-4 would fail past a few hundred cells from the origin).
const CELL_SLACK: f32 = 1.03;

/// A node's place in the buffers, and which occupant of it this names.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SlotId {
    pub index: u32,
    pub generation: u32,
}

/// The step's constants.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GraphParams {
    pub dt: f32,
    /// Velocity retained per step. Below 1 the system settles.
    pub damping: f32,
    /// Base spring constant, scaled by each pair's weight.
    pub spring_k: f32,
    pub rest_length: f32,
    /// A weak pull toward the origin.
    pub centering: f32,
    /// The exclusion law. `cutoff` must be a positive finite length.
    pub exclusion: ExclusionParams,
}

impl Default for GraphParams {
    fn default() -> Self {
        Self {
            dt: 1.0 / 60.0,
            damping: 0.85,
            spring_k: 0.08,
            rest_length: 30.0,
            centering: 0.05,
            // `seiche::NodeExclusion::default()`, which this crate does not depend on.
            exclusion: ExclusionParams {
                strength: 220_000.0,
                cutoff: 1_000.0,
                min_distance: 8.0,
            },
        }
    }
}

/// Why a graph call was refused or a read failed.
#[derive(Clone, Debug, PartialEq)]
pub enum GraphError {
    /// The slot is dead, or its generation is not the live one.
    StaleSlot(SlotId),
    /// `set_springs` refused the whole call: pair `pair` names a dead slot.
    SpringPair { pair: usize, slot: SlotId },
    /// A position, target or weight was not finite (`pair` for a weight).
    NotFinite,
    /// The device read failed (a lost device, a failed submit).
    Device(String),
    /// The readback held fewer bytes than asked.
    Length { expected: usize, actual: usize },
}

impl std::fmt::Display for GraphError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::StaleSlot(slot) => write!(
                formatter,
                "slot {} generation {} is not live",
                slot.index, slot.generation
            ),
            Self::SpringPair { pair, slot } => write!(
                formatter,
                "spring pair {pair} names dead slot {} generation {}",
                slot.index, slot.generation
            ),
            Self::NotFinite => write!(formatter, "a position, target or weight was not finite"),
            Self::Device(error) => write!(formatter, "graph device read failed: {error}"),
            Self::Length { expected, actual } => {
                write!(formatter, "graph readback held {actual} bytes for {expected}")
            },
        }
    }
}

impl std::error::Error for GraphError {}

/// Host-side traffic since construction, for a receipt to read.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GraphStats {
    pub uploaded_bytes: u64,
    pub read_bytes: u64,
}

/// A published range with the epoch of the allocation behind it.
#[derive(Debug)]
pub struct GraphLease<'a> {
    /// Range, shape (`[slots, width, 1]`) and the revision, as every lease.
    pub lease: SpatialLease<'a>,
    /// Changes only when growth reallocates: rebuild bind groups then.
    pub allocation_epoch: u64,
    /// Slots the buffers have ever used. Draw this many; flags say which live.
    pub slots: u32,
}

/// The resolved wgpu range behind one handle.
type Resolved = (wgpu::Buffer, u64, u64);

/// Every device allocation of one epoch.
struct Buffers {
    positions: Handle,
    velocities: Handle,
    forces: Handle,
    flags: Handle,
    generations: Handle,
    pin_targets: Handle,
    offsets: Handle,
    edge_ids: Handle,
    edge_weights: Handle,
    counts: Handle,
    cursors: Handle,
    sorted: Handle,
    settle: Handle,
    pick_scratch: Handle,
    pick_distances: Handle,
    pick_out: Handle,
    positions_range: Resolved,
    flags_range: Resolved,
}

#[derive(Clone, Copy)]
enum Buf {
    Positions,
    Velocities,
    Flags,
    Generations,
    PinTargets,
    Offsets,
}

/// The resident graph layout. See the module docs.
pub struct ResidentGraph {
    resident: ResidentClient,
    client: Client,
    params: GraphParams,
    capacity: usize,
    table: usize,
    /// Slots ever handed out: kernels run over `0..used`.
    used: u32,
    live: u32,
    free: Vec<u32>,
    /// Host mirrors of the device's generations and flags.
    generations: Vec<u32>,
    flags: Vec<u32>,
    pins: Vec<[f32; 4]>,
    bufs: Buffers,
    epoch: u64,
    revision: u64,
    springs: springs::Springs,
    stats: GraphStats,
}

fn zeros(bytes: usize) -> Vec<u8> {
    vec![0u8; bytes]
}

fn resolve(client: &Client, handle: &Handle) -> Resolved {
    let managed = client
        .get_resource::<cubecl::wgpu::WgpuServer<cubecl::wgpu::AutoCompiler>>(handle.clone())
        .expect("resident allocation");
    let resource = managed.resource();
    (resource.buffer.clone(), resource.offset, resource.size)
}

fn groups(count: u32) -> CubeCount {
    CubeCount::Static(count.div_ceil(law::CUBE_DIM).max(1), 1, 1)
}

fn table_for(capacity: usize) -> usize {
    (2 * capacity).next_power_of_two().max(256)
}

impl Buffers {
    /// Fresh zeroed state at `capacity`; the CSR starts empty.
    fn new(client: &Client, capacity: usize) -> Self {
        let table = table_for(capacity);
        let state = |bytes: usize| client.create_from_slice(&zeros(bytes));
        let positions = state(capacity * 16);
        let flags = state(capacity * 4);
        Self {
            positions_range: resolve(client, &positions),
            flags_range: resolve(client, &flags),
            positions,
            velocities: state(capacity * 16),
            forces: state(capacity * 16),
            flags,
            generations: state(capacity * 4),
            pin_targets: state(capacity * 16),
            offsets: state((capacity + 1) * 4),
            edge_ids: state(12),
            edge_weights: state(4),
            counts: client.empty((table + 1) * 4),
            cursors: client.empty(table * 4),
            sorted: client.empty(capacity * 16),
            settle: state(4),
            pick_scratch: state(8),
            pick_distances: client.empty(capacity * 4),
            pick_out: state(8),
        }
    }
}

impl std::fmt::Debug for ResidentGraph {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ResidentGraph")
            .field("capacity", &self.capacity)
            .field("slots", &self.used)
            .field("live", &self.live)
            .field("epoch", &self.epoch)
            .field("revision", &self.revision)
            .finish_non_exhaustive()
    }
}

impl ResidentGraph {
    /// A graph on the host's client with room for `capacity` nodes; it grows
    /// by doubling when full.
    pub fn new(client: &ResidentClient, capacity: usize, params: GraphParams) -> Self {
        let capacity = capacity.max(MIN_CAPACITY);
        let compute = client.compute_client().clone();
        Self {
            resident: client.clone(),
            bufs: Buffers::new(&compute, capacity),
            client: compute,
            params,
            capacity,
            table: table_for(capacity),
            used: 0,
            live: 0,
            free: Vec::new(),
            generations: vec![0; capacity],
            flags: vec![0; capacity],
            pins: vec![[0.0; 4]; capacity],
            epoch: 0,
            revision: 0,
            springs: springs::Springs::new(capacity),
            stats: GraphStats::default(),
        }
    }

    pub fn params(&self) -> GraphParams {
        self.params
    }

    /// Change the step's constants; nothing is uploaded, they ride as kernel
    /// arguments.
    pub fn set_params(&mut self, params: GraphParams) {
        self.params = params;
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Live nodes.
    pub fn len(&self) -> usize {
        self.live as usize
    }

    pub fn is_empty(&self) -> bool {
        self.live == 0
    }

    /// Slots ever used (live, or dead and awaiting reuse): the draw range.
    pub fn slots(&self) -> u32 {
        self.used
    }

    /// Changes only when growth reallocates the buffers.
    pub fn allocation_epoch(&self) -> u64 {
        self.epoch
    }

    /// Counts every change to what the leases publish: each step, insert,
    /// remove, pin and unpin.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn stats(&self) -> GraphStats {
        self.stats
    }

    /// Whether `slot` names a live node.
    pub fn is_live(&self, slot: SlotId) -> bool {
        let i = slot.index as usize;
        i < self.used as usize
            && self.generations[i] == slot.generation
            && self.flags[i] & SLOT_ALIVE != 0
    }

    fn check(&self, slot: SlotId) -> Result<(), GraphError> {
        if self.is_live(slot) {
            Ok(())
        } else {
            Err(GraphError::StaleSlot(slot))
        }
    }

    /// The live occupant of `index`, if any.
    pub fn slot_at(&self, index: u32) -> Option<SlotId> {
        let i = index as usize;
        (i < self.used as usize && self.flags[i] & SLOT_ALIVE != 0).then(|| SlotId {
            index,
            generation: self.generations[i],
        })
    }

    /// A targeted upload of `data` at element `first` of `buf`, ordered with
    /// the kernels around it on the client's stream.
    fn put(&mut self, buf: Buf, first: usize, data: &[u8]) {
        let (handle, unit, total) = match buf {
            Buf::Positions => (&self.bufs.positions, 16, self.capacity * 16),
            Buf::Velocities => (&self.bufs.velocities, 16, self.capacity * 16),
            Buf::Flags => (&self.bufs.flags, 4, self.capacity * 4),
            Buf::Generations => (&self.bufs.generations, 4, self.capacity * 4),
            Buf::PinTargets => (&self.bufs.pin_targets, 16, self.capacity * 16),
            Buf::Offsets => (&self.bufs.offsets, 4, (self.capacity + 1) * 4),
        };
        let start = first * unit;
        let slice = handle
            .clone()
            .offset_start(start as u64)
            .offset_end((total - start - data.len()) as u64);
        self.stats.uploaded_bytes += data.len() as u64;
        self.client
            .write(&slice, Bytes::from_bytes_vec(data.to_vec()));
    }

    /// Add one node. See [`Self::insert_many`].
    pub fn insert(&mut self, position: [f32; 4]) -> Result<SlotId, GraphError> {
        Ok(self.insert_many(&[position])?.remove(0))
    }

    /// Add nodes at `positions` (padded 3D, `w` is kept for the host). Free
    /// indices are reused first, each with the generation its removal left;
    /// the rest extend the high-water mark, growing the buffers if needed.
    /// Consecutive indices go up in one upload per buffer.
    pub fn insert_many(&mut self, positions: &[[f32; 4]]) -> Result<Vec<SlotId>, GraphError> {
        if positions.iter().flatten().any(|v| !v.is_finite()) {
            return Err(GraphError::NotFinite);
        }
        let fresh = positions.len().saturating_sub(self.free.len());
        let wanted = self.used as usize + fresh;
        if wanted > self.capacity {
            self.grow((self.capacity * 2).max(wanted.next_power_of_two()));
        }
        let mut slots = Vec::with_capacity(positions.len());
        for _ in positions {
            let index = self.free.pop().unwrap_or_else(|| {
                self.used += 1;
                self.used - 1
            });
            self.flags[index as usize] = SLOT_ALIVE;
            slots.push(SlotId {
                index,
                generation: self.generations[index as usize],
            });
        }
        self.live += slots.len() as u32;
        let mut from = 0;
        while from < slots.len() {
            let mut to = from + 1;
            while to < slots.len() && slots[to].index == slots[to - 1].index + 1 {
                to += 1;
            }
            let first = slots[from].index as usize;
            let run = to - from;
            self.put(Buf::Positions, first, bytemuck::cast_slice(&positions[from..to]));
            self.put(Buf::Velocities, first, &zeros(run * 16));
            let flags = self.flags[first..first + run].to_vec();
            self.put(Buf::Flags, first, bytemuck::cast_slice(&flags));
            let generations = self.generations[first..first + run].to_vec();
            self.put(Buf::Generations, first, bytemuck::cast_slice(&generations));
            from = to;
        }
        self.revision += 1;
        Ok(slots)
    }

    /// Remove a node: its alive bit clears and its generation bumps, so its
    /// springs fall silent on the device and every handle to it goes stale.
    pub fn remove(&mut self, slot: SlotId) -> Result<(), GraphError> {
        self.check(slot)?;
        let i = slot.index as usize;
        self.flags[i] = 0;
        self.generations[i] = self.generations[i].wrapping_add(1);
        self.free.push(slot.index);
        self.live -= 1;
        let flags = self.flags[i];
        self.put(Buf::Flags, i, bytemuck::bytes_of(&flags));
        let generation = self.generations[i];
        self.put(Buf::Generations, i, bytemuck::bytes_of(&generation));
        self.revision += 1;
        Ok(())
    }

    /// Hold `slot` at `target` from the next step on. Repeating the call
    /// moves the target with one sixteen-byte write, which is a pointer drag.
    pub fn pin(&mut self, slot: SlotId, target: [f32; 3]) -> Result<(), GraphError> {
        self.check(slot)?;
        if !target.iter().all(|v| v.is_finite()) {
            return Err(GraphError::NotFinite);
        }
        let i = slot.index as usize;
        let held = [target[0], target[1], target[2], 1.0];
        self.pins[i] = held;
        self.put(Buf::PinTargets, i, bytemuck::bytes_of(&held));
        if self.flags[i] & SLOT_PINNED == 0 {
            self.flags[i] |= SLOT_PINNED;
            let flags = self.flags[i];
            self.put(Buf::Flags, i, bytemuck::bytes_of(&flags));
        }
        self.revision += 1;
        Ok(())
    }

    /// Release a pinned slot; it keeps the position it was held at, at rest.
    pub fn unpin(&mut self, slot: SlotId) -> Result<(), GraphError> {
        self.check(slot)?;
        let i = slot.index as usize;
        if self.flags[i] & SLOT_PINNED != 0 {
            self.flags[i] &= !SLOT_PINNED;
            let flags = self.flags[i];
            self.put(Buf::Flags, i, bytemuck::bytes_of(&flags));
            self.revision += 1;
        }
        Ok(())
    }

    /// Reallocate at `capacity`, carrying positions and velocities across on
    /// the device. Everything the host mirrors is uploaded again.
    fn grow(&mut self, capacity: usize) {
        let old = std::mem::replace(&mut self.bufs, Buffers::new(&self.client, capacity));
        let used = self.used;
        let copy = |from: &Handle, to: &Handle| unsafe {
            kernels::copy_f32::launch_unchecked(
                &self.client,
                groups(used * 4),
                CubeDim::new_1d(law::CUBE_DIM),
                BufferArg::from_raw_parts(from.clone(), used as usize * 4),
                BufferArg::from_raw_parts(to.clone(), used as usize * 4),
                used * 4,
            );
        };
        if used > 0 {
            copy(&old.positions, &self.bufs.positions);
            copy(&old.velocities, &self.bufs.velocities);
        }
        self.capacity = capacity;
        self.table = table_for(capacity);
        self.generations.resize(capacity, 0);
        self.flags.resize(capacity, 0);
        self.epoch += 1;
        self.bufs.positions_range = resolve(&self.client, &self.bufs.positions);
        self.bufs.flags_range = resolve(&self.client, &self.bufs.flags);
        let flags = self.flags.clone();
        let generations = self.generations.clone();
        self.put(Buf::Flags, 0, bytemuck::cast_slice(&flags));
        self.put(Buf::Generations, 0, bytemuck::cast_slice(&generations));
        self.pins.resize(capacity, [0.0; 4]);
        let pins = self.pins.clone();
        self.put(Buf::PinTargets, 0, bytemuck::cast_slice(&pins));
        self.springs.resize(capacity);
        self.upload_springs();
    }

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

    /// Submit everything queued to the host's wgpu queue (CubeCL's client
    /// flush). Call it after [`Self::step`] and before the host submits a draw
    /// that reads the leases, so both land on the queue in that order.
    pub fn flush(&self) -> Result<(), GraphError> {
        self.client
            .flush()
            .map_err(|error| GraphError::Device(error.to_string()))
    }

    /// The positions, `[slots, 4, 1]` of `f32`, published for drawing.
    pub fn positions_lease(&self) -> GraphLease<'_> {
        self.lease(&self.bufs.positions_range, 4, PlaneElementType::F32)
    }

    /// Per-slot flags ([`SLOT_ALIVE`], [`SLOT_PINNED`]), `[slots, 1, 1]` of
    /// `u32`, so a renderer can skip dead slots and mark pinned ones.
    pub fn flags_lease(&self) -> GraphLease<'_> {
        self.lease(&self.bufs.flags_range, 1, PlaneElementType::U32)
    }

    fn lease<'a>(&'a self, range: &'a Resolved, width: usize, element: PlaneElementType) -> GraphLease<'a> {
        GraphLease {
            lease: SpatialLease {
                buffer: &range.0,
                offset: range.1,
                size: range.2,
                shape: [self.used as usize, width, 1],
                element_type: element,
                stamp: ChunkStamp {
                    revision: self.revision,
                    valid_read_epoch: ReadEpoch::new(self.revision),
                },
            },
            allocation_epoch: self.epoch,
            slots: self.used,
        }
    }
}
