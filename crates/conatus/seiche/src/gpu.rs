// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! [`LaggedRepulsion`] on the host's own device (feature `gpu`): the exclusion
//! lane of `conatus::resident`, every pair below its cell threshold and the
//! cell list above it. (Physics catalog plan, P5b.)
//!
//! A host builds one [`PhysicsDevice`] from its renderer's wgpu handles and
//! clones it to every simulation that should use it; each simulation gets a
//! [`DeviceRepulsion`] of its own from [`PhysicsDevice::repulsion`].

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

pub use conatus::resident::{DEFAULT_CELL_THRESHOLD, ResidentClient};
use conatus::resident::{Exclusion, ExclusionParams, PendingExclusion};

use crate::{
    DEFAULT_GPU_REPULSION_THRESHOLD, DEFAULT_MAX_STALE_STEPS, LaggedRepulsion, RepulsionForces,
    RepulsionRequest, RepulsionSolverError, Simulation,
};

/// The host's device, shared: one CubeCL client per device, cloned cheaply
/// into every canvas and board. Its counters add up every lane made from it,
/// so a host whose physics runs on an actor thread can still read them.
#[derive(Clone, Debug)]
pub struct PhysicsDevice {
    client: ResidentClient,
    /// What the host's adapter offers and its device holds, when the device
    /// came from the host's handles.
    features: Option<DeviceFeatures>,
    cell_threshold: usize,
    threshold: usize,
    max_stale_steps: u32,
    counts: Arc<DeviceCounts>,
}

/// The wgpu features of a host's adapter and of the device it booted, for a
/// tenant that must know what the device lacks (the Meaning model's CubeCL
/// times kernels on the device whenever the adapter has timestamp queries;
/// dynamics grammar plan, G2, F31).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceFeatures {
    pub adapter: wgpu::Features,
    pub device: wgpu::Features,
}

impl DeviceFeatures {
    /// Whether the adapter offers timestamp queries the device was booted
    /// without: the case in which CubeCL, which decides device timing from
    /// the adapter, fails its first timed launch.
    pub fn lacks_adapter_timestamps(&self) -> bool {
        let timestamps = wgpu::Features::TIMESTAMP_QUERY;
        self.adapter.contains(timestamps) && !self.device.contains(timestamps)
    }
}

#[derive(Debug, Default)]
struct DeviceCounts {
    submissions: AtomicU64,
    answers: AtomicU64,
    cell_submissions: AtomicU64,
}

impl PhysicsDevice {
    pub fn new(client: ResidentClient) -> Self {
        Self {
            client,
            features: None,
            cell_threshold: DEFAULT_CELL_THRESHOLD,
            threshold: DEFAULT_GPU_REPULSION_THRESHOLD,
            max_stale_steps: DEFAULT_MAX_STALE_STEPS,
            counts: Arc::default(),
        }
    }

    /// From the host's four wgpu handles, never a device of its own.
    pub fn from_wgpu(
        instance: wgpu::Instance,
        adapter: wgpu::Adapter,
        device: wgpu::Device,
        queue: wgpu::Queue,
    ) -> Self {
        let features = DeviceFeatures {
            adapter: adapter.features(),
            device: device.features(),
        };
        Self {
            features: Some(features),
            ..Self::new(ResidentClient::from_wgpu(instance, adapter, device, queue))
        }
    }

    /// The adapter's and the device's features, when this device came from
    /// the host's handles ([`Self::from_wgpu`]).
    pub fn features(&self) -> Option<DeviceFeatures> {
        self.features
    }

    /// Node count at or above which a lane walks the cell list rather than
    /// every pair.
    pub fn with_cell_threshold(mut self, threshold: usize) -> Self {
        self.cell_threshold = threshold;
        self
    }

    /// Node count at or above which a simulation uses the device at all
    /// (0 behaves as 1: always). Each host sets its own measured crossover.
    pub fn with_threshold(mut self, threshold: usize) -> Self {
        self.threshold = threshold;
        self
    }

    /// How many steps old an answer may be and still apply (1 on a host
    /// that steps once a frame; the web tree's three-step frames want 3).
    pub fn with_max_stale_steps(mut self, steps: u32) -> Self {
        self.max_stale_steps = steps.max(1);
        self
    }

    pub fn threshold(&self) -> usize {
        self.threshold
    }

    pub fn max_stale_steps(&self) -> u32 {
        self.max_stale_steps
    }

    pub fn client(&self) -> &ResidentClient {
        &self.client
    }

    /// A lagged evaluator for one simulation.
    pub fn repulsion(&self) -> DeviceRepulsion {
        let mut lane = Exclusion::new(&self.client);
        lane.set_cell_threshold(self.cell_threshold);
        DeviceRepulsion {
            lane,
            pending: VecDeque::new(),
            counts: self.counts.clone(),
        }
    }

    /// Install a lane from this device on `sim`, with this device's
    /// threshold and staleness limit.
    pub fn install(&self, sim: &mut Simulation) {
        sim.set_lagged_repulsion(
            Some(Box::new(self.repulsion())),
            self.threshold,
            self.max_stale_steps,
        );
    }

    /// Submissions every lane from this device has made.
    pub fn submissions(&self) -> u64 {
        self.counts.submissions.load(Ordering::Relaxed)
    }

    /// Answers every lane from this device has received.
    pub fn answers(&self) -> u64 {
        self.counts.answers.load(Ordering::Relaxed)
    }

    /// Submissions that walked the cell list.
    pub fn cell_submissions(&self) -> u64 {
        self.counts.cell_submissions.load(Ordering::Relaxed)
    }
}

/// `NodeExclusion`'s law on a [`PhysicsDevice`]; submissions are answered in
/// order, as the device's one queue runs them.
pub struct DeviceRepulsion {
    lane: Exclusion,
    pending: VecDeque<PendingExclusion>,
    counts: Arc<DeviceCounts>,
}

impl LaggedRepulsion for DeviceRepulsion {
    fn submit(
        &mut self,
        xs: &[f32],
        ys: &[f32],
        request: RepulsionRequest,
    ) -> Result<(), RepulsionSolverError> {
        let positions: Vec<[f32; 4]> = xs.iter().zip(ys).map(|(&x, &y)| [x, y, 0.0, 0.0]).collect();
        let pending = self.lane.submit(
            &positions,
            ExclusionParams {
                strength: request.strength,
                cutoff: request.cutoff,
                min_distance: request.min_distance,
            },
        );
        self.counts.submissions.fetch_add(1, Ordering::Relaxed);
        if pending.used_cells() {
            self.counts.cell_submissions.fetch_add(1, Ordering::Relaxed);
        }
        self.pending.push_back(pending);
        Ok(())
    }

    fn poll(&mut self) -> Option<Result<RepulsionForces, RepulsionSolverError>> {
        let pending = self.pending.front_mut()?;
        let n = pending.len();
        let result = pending.try_take()?;
        self.pending.pop_front();
        self.counts.answers.fetch_add(1, Ordering::Relaxed);
        Some(
            result
                .map_err(|error| RepulsionSolverError::Backend(error.to_string()))
                .and_then(|forces| {
                    RepulsionForces::new(
                        n,
                        forces.iter().map(|f| f[0]).collect(),
                        forces.iter().map(|f| f[1]).collect(),
                    )
                }),
        )
    }

    fn in_flight(&self) -> usize {
        self.pending.len()
    }
}
