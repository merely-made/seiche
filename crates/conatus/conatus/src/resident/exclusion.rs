// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! The exclusion lane: `seiche::NodeExclusion`'s law on the host's device,
//! for a host whose integrator stays rapier.
//!
//! Positions go up, [`kernels::exclude`] runs, and the forces come back
//! through a readback that is started at [`Exclusion::submit`] and finished
//! whenever the host next asks, never by blocking. On native CubeCL's poll
//! thread drives the map; in a browser the event loop does, so a result
//! submitted in one frame is ready no earlier than the next turn.
//! (Physics catalog plan, P5a.)

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll, Waker};

use cubecl::client::ComputeClient;
use cubecl::prelude::*;
use cubecl::server::ServerError;
use cubecl::wgpu::WgpuRuntime;

use super::{ResidentClient, kernels};

/// The law's constants, as `seiche::NodeExclusion` holds them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExclusionParams {
    pub strength: f32,
    pub cutoff: f32,
    pub min_distance: f32,
}

/// Why a staged exclusion produced no forces.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExclusionError {
    /// The device read failed (a lost device, a failed submit).
    Readback(String),
    /// The readback held fewer bytes than the bodies submitted.
    Length { expected: usize, actual: usize },
}

impl std::fmt::Display for ExclusionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Readback(error) => write!(formatter, "exclusion readback failed: {error}"),
            Self::Length { expected, actual } => write!(
                formatter,
                "exclusion readback held {actual} bytes for {expected} expected"
            ),
        }
    }
}

impl std::error::Error for ExclusionError {}

type Readback =
    Pin<Box<dyn Future<Output = Result<Vec<cubecl::bytes::Bytes>, ServerError>> + Send>>;

/// The exclusion dispatch on a host's CubeCL client.
pub struct Exclusion {
    client: ComputeClient<WgpuRuntime>,
    dispatches: u64,
}

impl std::fmt::Debug for Exclusion {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Exclusion")
            .field("dispatches", &self.dispatches)
            .finish_non_exhaustive()
    }
}

impl Exclusion {
    pub fn new(client: &ResidentClient) -> Self {
        Self {
            client: client.compute_client().clone(),
            dispatches: 0,
        }
    }

    /// How many kernels this lane has launched: the count a receipt reads to
    /// prove the device ran rather than a fallback.
    pub fn dispatches(&self) -> u64 {
        self.dispatches
    }

    /// Upload `positions` (padded 3D, `w` ignored), launch the kernel and
    /// start reading the forces back. Returns at once.
    pub fn submit(&mut self, positions: &[[f32; 4]], params: ExclusionParams) -> PendingExclusion {
        let n = positions.len();
        if n == 0 {
            return PendingExclusion {
                n,
                state: State::Ready(Vec::new()),
            };
        }
        let bytes = n * 16;
        let input = self
            .client
            .create_from_slice(bytemuck::cast_slice(positions));
        let output = self.client.empty(bytes);
        let cubes = (n as u32).div_ceil(kernels::CUBE_DIM);
        unsafe {
            kernels::exclude::launch_unchecked::<WgpuRuntime>(
                &self.client,
                CubeCount::Static(cubes, 1, 1),
                CubeDim::new_1d(kernels::CUBE_DIM),
                BufferArg::from_raw_parts(input, n * 4),
                BufferArg::from_raw_parts(output.clone(), n * 4),
                n as u32,
                params.strength,
                params.cutoff * params.cutoff,
                params.min_distance,
                (kernels::CUBE_DIM * kernels::STRIDE) as usize,
            );
        }
        self.dispatches += 1;
        // `read_async` borrows its client, so the future owns a clone. The
        // first poll is what queues the copy and the map, so poll once now.
        let client = self.client.clone();
        let mut pending = PendingExclusion {
            n,
            state: State::Reading(Box::pin(
                async move { client.read_async(vec![output]).await },
            )),
        };
        if let Some(result) = pending.try_take() {
            pending.state = match result {
                Ok(forces) => State::Ready(forces),
                Err(error) => State::Failed(error),
            };
        }
        pending
    }
}

enum State {
    Reading(Readback),
    Ready(Vec<[f32; 4]>),
    Failed(ExclusionError),
    Taken,
}

/// One submitted exclusion whose forces are on their way back.
pub struct PendingExclusion {
    n: usize,
    state: State,
}

impl PendingExclusion {
    /// The body count this result answers for.
    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// The forces if they have arrived, without waiting. `None` while the
    /// read is in flight, and after the result has been taken.
    pub fn try_take(&mut self) -> Option<Result<Vec<[f32; 4]>, ExclusionError>> {
        if let State::Reading(read) = &mut self.state {
            let mut context = Context::from_waker(Waker::noop());
            match read.as_mut().poll(&mut context) {
                Poll::Pending => return None,
                Poll::Ready(Err(error)) => {
                    self.state = State::Taken;
                    return Some(Err(ExclusionError::Readback(error.to_string())));
                },
                Poll::Ready(Ok(mut bytes)) => {
                    let bytes = bytes.remove(0);
                    let expected = self.n * 16;
                    if bytes.len() < expected {
                        self.state = State::Taken;
                        return Some(Err(ExclusionError::Length {
                            expected,
                            actual: bytes.len(),
                        }));
                    }
                    self.state = State::Ready(bytemuck::cast_slice(&bytes[..expected]).to_vec());
                },
            }
        }
        match std::mem::replace(&mut self.state, State::Taken) {
            State::Ready(forces) => Some(Ok(forces)),
            State::Failed(error) => Some(Err(error)),
            other => {
                self.state = other;
                None
            },
        }
    }

    /// Wait for the forces (native only: a browser cannot block its own
    /// event loop, which is what finishes the read there). Tests and
    /// benches use this; the frame path uses [`Self::try_take`].
    #[cfg(not(target_family = "wasm"))]
    pub fn wait(mut self) -> Result<Vec<[f32; 4]>, ExclusionError> {
        loop {
            if let Some(result) = self.try_take() {
                return result;
            }
            std::thread::yield_now();
        }
    }
}
