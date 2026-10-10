// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! The lit body tenant.
//!
//! Bodies (rigid palette meshes and glTF) drawn on the host's device into
//! the host's target and encoder, lit by the stack's [`LightBlock`], and
//! depth-joined with a tracer through a depth pre-pass. kiss3d sits
//! beneath and never shows: the API is plain arrays and wgpu.

mod body;
mod camera;
mod device;
mod frame;
mod light;

pub use body::{BodyError, BodyId, Palette, PaletteMesh, Pose};
pub use camera::Camera;
pub use device::{DeviceNeeds, HostDevice};
pub use frame::{Exports, FrameError, FrameReport, Tenant, Tonemap};
pub use light::{Ambient, LightBlock, PointLight, Sun};
