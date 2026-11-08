// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! The physics catalog's engine half: `laws × overlays` built into forces.
//!
//! The vocabulary (laws, overlays, sources, compositions) is data in
//! `seiche-model`'s `catalog`. This module turns a choice of it into the
//! [`Force`](crate::Force)s a simulation runs, against the graph inputs the
//! build snapshots ([`LawInputs`]): the node set, the visible spring edges
//! and, on demand, the site, cluster and meaning groups and the physics
//! view's channels. A host that caches the channels (Pictograph's registry)
//! names them through [`Channels`]; without one they are computed here by the
//! [`algo`] functions. The same builders serve every backend, so the forces
//! are identical by construction wherever the simulation runs.
//!
//! A **law** is which dynamics the graph moves under. An **overlay** is one
//! more pull or push composed onto any law. A **source** is where a law or
//! overlay reads a node attribute from. Plan:
//! `design_docs/archive_docs/2026-10-28/2026-09-02_physics_catalog_plan.md`.

pub mod algo;
mod build;
mod cache;
mod composition;
mod inputs;

pub use build::density_law;
pub use cache::{ChannelCache, ChannelRuns};
pub use composition::{build_force_set, build_force_set_cached};
pub use inputs::{Channels, LawInputs, MeaningGroups};
pub use seiche_model::{LAW_SEED, LawSources};

/// The Skeleton overlay's tree-edge stiffness, against `EdgeSpring`'s 10.
pub const SKELETON_STIFFNESS: f32 = 60.0;
/// Charge's repulsion, calibrated so that at contact (a node diameter, 36)
/// the `1/d` push matches `NodeExclusion`'s inverse-square one
/// (`220_000 / 36² ≈ 170`): the seiche default of 2 400 left bodies
/// touching under the edge springs. (Physics catalog — the Charge receipt.)
pub const CHARGE_STRENGTH: f32 = 6_000.0;
/// Orbit's exclusion reaches two node diameters: a guard at contact, where
/// Springs' reach of 1,000 outweighed gravitation at every range inside it
/// and threw the graph apart. Orbit's centring is a weak well that holds what
/// the orbits spread. (Ruled 2026-10-04, "Frictionless orbits + centring".)
pub const ORBIT_EXCLUSION_REACH: f32 = 4.0 * crate::NODE_BODY_RADIUS;
pub const ORBIT_CENTRING: f32 = 0.02;
/// Density's grid resolution on the CPU tier (cells per side): 64², ruled
/// 2026-10-02 (the same ranks as 128² at about a sixteenth of the cost).
pub const DENSITY_RESOLUTION: usize = 64;
/// Density's convergence test (ruled 2026-10-02): passes end once the mean
/// node shift stays under a twentieth of a spacing for three passes in a
/// row; the pass cap ends them if the test never does.
pub const DENSITY_STOP: crate::DensityStop = crate::DensityStop::Shift(0.05);
pub const DENSITY_PATIENCE: u32 = 3;
pub const DENSITY_MAX_PASSES: u32 = 120;
/// The fewest passes before the stop test may end them (ruled 2026-10-03,
/// "Min 60, bar: all >= 0.7").
pub const DENSITY_MIN_PASSES: u32 = 60;

#[cfg(test)]
mod descriptor_tests;
