// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! The shared device: what the tenant asks of it, and the handles it borrows.

/// What the tenant asks of a device a host boots for several tenants.
///
/// Every capability beyond these degrades at run time: clustered lights
/// need compute and three fragment storage buffers, skinning and morphs
/// five vertex storage buffers, and without them the tenant falls back to
/// eight fixed lights and rest poses.
#[derive(Clone, Debug, Default)]
pub struct DeviceNeeds {
    /// Features the tenant cannot work without.
    pub required_features: wgpu::Features,
    /// Features the tenant uses when the adapter has them.
    pub optional_features: wgpu::Features,
    /// Limits the tenant needs; `None` is wgpu's defaults.
    pub limits: Option<wgpu::Limits>,
}

impl DeviceNeeds {
    /// The tenant's needs: nothing beyond wgpu's defaults.
    pub fn tenant() -> Self {
        Self::default()
    }
}

/// The host's device handles. Clones of wgpu handles share the device.
#[derive(Clone, Debug)]
pub struct HostDevice {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}
