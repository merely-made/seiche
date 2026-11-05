// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

use super::*;

impl ResidentClient {
    pub fn init(setup: WgpuSetup) -> Self {
        let wgpu_device = setup.device.clone();
        let device = init_device(setup, RuntimeOptions::default());
        Self {
            wgpu_device: Some(wgpu_device),
            ..Self::from_registered_device(device)
        }
    }

    /// [`Self::init`] from a host's four wgpu handles (a renderer's
    /// `WgpuHandles`, say), reading the backend off the adapter. Each call
    /// registers one more CubeCL device: a host makes one client per device
    /// and clones it.
    pub fn from_wgpu(
        instance: wgpu::Instance,
        adapter: wgpu::Adapter,
        device: wgpu::Device,
        queue: wgpu::Queue,
    ) -> Self {
        let backend = adapter.get_info().backend;
        Self::init(WgpuSetup {
            instance,
            adapter,
            device,
            queue,
            backend,
        })
    }

    pub fn from_registered_device(device: WgpuDevice) -> Self {
        let compute = cubecl::Device::from(device.clone()).client();
        Self {
            compute,
            device,
            wgpu_device: None,
            orphans: Arc::default(),
        }
    }

    /// The CubeCL client every resident allocation is made through.
    ///
    /// Public because the field lane allocates its own buffers on the
    /// same client the chunk bundles use: one allocator is what lets a
    /// kernel pass and a tensor pass meet without a bridge.
    pub fn compute_client(&self) -> &Client {
        &self.compute
    }

    /// Bytes in use on this client's device, every stream of it: what a
    /// model loaded onto the host's device adds (the Meaning receipt's
    /// single-device check). CubeCL 0.11 reads memory through its report,
    /// where pre.4 had `memory_usage`.
    pub fn bytes_in_use(&self) -> u64 {
        self.compute
            .memory_report(cubecl::MemoryScope::Device)
            .usage()
            .bytes_in_use
    }

    pub fn device(&self) -> &WgpuDevice {
        &self.device
    }

    /// Deliver any finished map callbacks without waiting (a no-op in a
    /// browser, whose event loop delivers them). CubeCL's own poll thread
    /// delivers them too, but measured late: with a caller that sleeps
    /// between polls a readback took 14 ms at the median against 1 ms when
    /// the caller spun, and 2 ms with this poll first (2026-10-02).
    pub fn poll_device(&self) {
        if let Some(device) = &self.wgpu_device {
            let _ = device.poll(wgpu::PollType::Poll);
        }
        self.reap();
    }

    /// Keep a readback whose reader is gone until it completes. Dropping an
    /// unfinished CubeCL read releases its staging buffer while the buffer is
    /// mapped or about to be, and the next submit that reuses it fails wgpu
    /// validation ("still mapped"); finishing it lets CubeCL unmap and
    /// release the buffer itself. Never blocks, so it serves a browser too.
    pub fn adopt(&self, readback: Orphan) {
        if let Ok(mut orphans) = self.orphans.lock() {
            orphans.push(readback);
        }
    }

    /// Poll every adopted readback once and drop the finished ones.
    pub fn reap(&self) {
        let Ok(mut orphans) = self.orphans.lock() else {
            return;
        };
        let mut context = Context::from_waker(Waker::noop());
        orphans.retain_mut(|readback| readback.as_mut().poll(&mut context).is_pending());
    }

    /// Readbacks adopted and not yet finished.
    pub fn orphans(&self) -> usize {
        self.orphans.lock().map_or(0, |orphans| orphans.len())
    }
}
