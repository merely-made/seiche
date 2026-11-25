// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! Bounded reads of the graph: the settle word, a pick, and a gather of named
//! slots. Each is queued after the step it follows and read back without
//! blocking, in [`super::super::PendingExclusion`]'s style: `try_take` polls,
//! `wait` blocks on native only. Whole-field reads exist for receipts and are
//! named so.

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll, Waker};

use cubecl::prelude::*;
use cubecl::server::{Handle, ServerError};

use super::{GraphError, ResidentGraph, SlotId, groups, kernels};
use crate::resident::{ResidentClient, kernels as law};

type Readback =
    Pin<Box<dyn Future<Output = Result<Vec<cubecl::bytes::Bytes>, ServerError>> + Send>>;
type Decode<T> = Box<dyn FnOnce(&[u8]) -> Result<T, GraphError> + Send>;

enum State<T> {
    Reading(Readback, Decode<T>),
    Ready(Result<T, GraphError>),
    Taken,
}

/// A queued read whose bytes are on their way back.
pub struct Pending<T> {
    device: ResidentClient,
    state: State<T>,
}

impl<T> Pending<T> {
    /// Poll the read once, decoding it if it has arrived.
    fn advance(&mut self) {
        let State::Reading(read, _) = &mut self.state else {
            return;
        };
        let mut context = Context::from_waker(Waker::noop());
        let Poll::Ready(result) = read.as_mut().poll(&mut context) else {
            return;
        };
        let State::Reading(_, decode) = std::mem::replace(&mut self.state, State::Taken) else {
            unreachable!("state was just matched")
        };
        self.state = State::Ready(match result {
            Err(error) => Err(GraphError::Device(error.to_string())),
            Ok(mut bytes) => decode(&bytes.remove(0)),
        });
    }

    /// The result if it has arrived, without waiting. `None` while the read
    /// is in flight, and after the result has been taken.
    pub fn try_take(&mut self) -> Option<Result<T, GraphError>> {
        self.device.poll_device();
        self.advance();
        match std::mem::replace(&mut self.state, State::Taken) {
            State::Ready(result) => Some(result),
            other => {
                self.state = other;
                None
            },
        }
    }

    /// Wait for the result (native only: a browser's event loop is what
    /// finishes the read there).
    #[cfg(not(target_family = "wasm"))]
    pub fn wait(mut self) -> Result<T, GraphError> {
        loop {
            if let Some(result) = self.try_take() {
                return result;
            }
            std::thread::yield_now();
        }
    }
}

impl<T> Drop for Pending<T> {
    fn drop(&mut self) {
        if let State::Reading(..) = &self.state
            && let State::Reading(read, _) = std::mem::replace(&mut self.state, State::Taken)
        {
            self.device.adopt(Box::pin(async move {
                let _ = read.await;
            }));
        }
    }
}

fn sized(bytes: &[u8], expected: usize) -> Result<&[u8], GraphError> {
    bytes.get(..expected).ok_or(GraphError::Length {
        expected,
        actual: bytes.len(),
    })
}

impl ResidentGraph {
    /// Start reading `bytes` of `handle`. The first poll queues the copy, so
    /// it is made here, and any readback a dropped reader left is finished
    /// first.
    fn read<T>(&mut self, handle: Handle, bytes: usize, decode: Decode<T>) -> Pending<T> {
        self.resident.poll_device();
        self.stats.read_bytes += bytes as u64;
        let client = self.client.clone();
        let mut pending = Pending {
            device: self.resident.clone(),
            state: State::Reading(
                Box::pin(async move { client.read_async(vec![handle]).await }),
                decode,
            ),
        };
        pending.advance();
        pending
    }

    /// The fastest body's speed in the last step: the settle word, four
    /// bytes, the only per-step read a host needs.
    pub fn request_settle(&mut self) -> Pending<f32> {
        let handle = self.bufs.settle.clone();
        self.read(
            handle,
            4,
            Box::new(|bytes| {
                let word = sized(bytes, 4)?;
                Ok(f32::from_bits(u32::from_le_bytes([
                    word[0], word[1], word[2], word[3],
                ])))
            }),
        )
    }

    /// [`Self::request_settle`], waited for.
    #[cfg(not(target_family = "wasm"))]
    pub fn max_speed(&mut self) -> Result<f32, GraphError> {
        self.request_settle().wait()
    }

    /// Queue a pick: the nearest live slot within `radius` of `point`
    /// (lowest index on a tie), eight bytes back. Queued after any step
    /// already queued, so it sees that step's positions.
    pub fn request_pick(&mut self, point: [f32; 3], radius: f32) -> Pending<Option<SlotId>> {
        let n = self.used;
        let capacity = self.capacity;
        let b = &self.bufs;
        let dim = CubeDim::new_1d(law::CUBE_DIM);
        unsafe {
            kernels::pick_init::launch_unchecked(
                &self.client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(2),
                BufferArg::from_raw_parts(b.pick_scratch.clone(), 2),
            );
            if n > 0 {
                kernels::pick_distance::launch_unchecked(
                    &self.client,
                    groups(n),
                    dim,
                    BufferArg::from_raw_parts(b.positions.clone(), capacity * 4),
                    BufferArg::from_raw_parts(b.flags.clone(), capacity),
                    BufferArg::from_raw_parts(b.pick_distances.clone(), capacity),
                    BufferArg::from_raw_parts(b.pick_scratch.clone(), 2),
                    n,
                    point[0],
                    point[1],
                    point[2],
                    radius * radius,
                );
                kernels::pick_lowest::launch_unchecked(
                    &self.client,
                    groups(n),
                    dim,
                    BufferArg::from_raw_parts(b.pick_distances.clone(), capacity),
                    BufferArg::from_raw_parts(b.pick_scratch.clone(), 2),
                    n,
                );
            }
            kernels::pick_finish::launch_unchecked(
                &self.client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                BufferArg::from_raw_parts(b.pick_scratch.clone(), 2),
                BufferArg::from_raw_parts(b.generations.clone(), capacity),
                BufferArg::from_raw_parts(b.pick_out.clone(), 2),
            );
        }
        let out = b.pick_out.clone();
        self.read(
            out,
            8,
            Box::new(|bytes| {
                let words: &[u32] = bytemuck::cast_slice(sized(bytes, 8)?);
                Ok((words[0] != u32::MAX).then_some(SlotId {
                    index: words[0],
                    generation: words[1],
                }))
            }),
        )
    }

    /// [`Self::request_pick`], waited for.
    #[cfg(not(target_family = "wasm"))]
    pub fn pick(&mut self, point: [f32; 3], radius: f32) -> Result<Option<SlotId>, GraphError> {
        self.request_pick(point, radius).wait()
    }

    /// Queue a gather of the named slots' positions into a small buffer and
    /// read only that back: how labels for visible nodes are placed. Any
    /// dead or stale slot refuses the call. The result is in request order.
    pub fn read_slots(&mut self, slots: &[SlotId]) -> Result<Pending<Vec<[f32; 4]>>, GraphError> {
        for &slot in slots {
            self.check(slot)?;
        }
        let count = slots.len();
        let indices: Vec<u32> = slots.iter().map(|slot| slot.index).collect();
        let out = self.client.empty(count.max(1) * 16);
        if count > 0 {
            let uploaded = self
                .client
                .create_from_slice(bytemuck::cast_slice(&indices));
            self.stats.uploaded_bytes += count as u64 * 4;
            unsafe {
                kernels::gather_positions::launch_unchecked(
                    &self.client,
                    groups(count as u32),
                    CubeDim::new_1d(law::CUBE_DIM),
                    BufferArg::from_raw_parts(self.bufs.positions.clone(), self.capacity * 4),
                    BufferArg::from_raw_parts(uploaded, count),
                    BufferArg::from_raw_parts(out.clone(), count * 4),
                    count as u32,
                );
            }
        }
        Ok(self.read(
            out,
            count * 16,
            Box::new(move |bytes| Ok(bytemuck::cast_slice(sized(bytes, count * 16)?).to_vec())),
        ))
    }

    /// Whole-field reads, for receipts and tests. Not the frame path.
    fn read_field(&mut self, handle: Handle, bytes: usize) -> Result<Vec<u8>, GraphError> {
        self.stats.read_bytes += bytes as u64;
        let read = self
            .client
            .read_one(handle)
            .map_err(|error| GraphError::Device(error.to_string()))?;
        Ok(sized(&read, bytes)?.to_vec())
    }

    /// Positions of slots `0..slots()`, dead ones included.
    pub fn read_positions(&mut self) -> Result<Vec<[f32; 4]>, GraphError> {
        let (handle, bytes) = (self.bufs.positions.clone(), self.used as usize * 16);
        Ok(bytemuck::cast_slice(&self.read_field(handle, bytes)?).to_vec())
    }

    pub fn read_velocities(&mut self) -> Result<Vec<[f32; 4]>, GraphError> {
        let (handle, bytes) = (self.bufs.velocities.clone(), self.used as usize * 16);
        Ok(bytemuck::cast_slice(&self.read_field(handle, bytes)?).to_vec())
    }

    /// The forces the last [`Self::compute_forces`] or step left.
    pub fn read_forces(&mut self) -> Result<Vec<[f32; 4]>, GraphError> {
        let (handle, bytes) = (self.bufs.forces.clone(), self.used as usize * 16);
        Ok(bytemuck::cast_slice(&self.read_field(handle, bytes)?).to_vec())
    }

    /// The device's copy of the flags, which must equal the host's.
    pub fn read_flags(&mut self) -> Result<Vec<u32>, GraphError> {
        let (handle, bytes) = (self.bufs.flags.clone(), self.used as usize * 4);
        Ok(bytemuck::cast_slice(&self.read_field(handle, bytes)?).to_vec())
    }
}
