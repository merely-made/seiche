// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! The common scale (ruled 2026-10-02, F5 "Reference-configuration
//! normalization"): a term's weight-1 strength is its force at a declared
//! reference, so weight 1 is unit force there and a weight means the same
//! push whichever term carries it.
//!
//! - A repulsion's reference is two bodies at contact, a node diameter
//!   ([`CONTACT`], 36), as Charge's calibration matched `NodeExclusion`.
//! - A spring's is a pair one rest length past its rest length.
//! - A unary pull's is one body [`UNIT_LENGTH`] from its target.
//!
//! A term reports its [`Scale`] through [`Declared::scale`](crate::Declared)
//! and is rebuilt at another weight through `reweighted`. Today's
//! calibrated strengths are their weights, so the default compositions are
//! unchanged. F5 names no reference for attractions, gravitation or the
//! dynamics kernels (tent, steering, drive, needle, phase coupling); those
//! terms report none.
//!
//! Plan: `design_docs/mere_docs/implementation_strategy/2026-10-02_dynamics_grammar_plan.md`, G3.

use crate::{Kernel, NODE_BODY_RADIUS, Term, Topology};

/// Two bodies touching: a node diameter.
pub const CONTACT: f32 = 2.0 * NODE_BODY_RADIUS;

/// One rest length of offset for a unary pull: `EdgeSpring`'s rest length.
/// The plan's open question 5 names it the obvious candidate and a per-term
/// length the other; it is built here, in this one place, at the first.
pub const UNIT_LENGTH: f32 = 170.0;

/// Where a term's weight is read.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Reference {
    /// Two bodies [`CONTACT`] apart.
    Contact,
    /// Two joined bodies `2 · rest` apart.
    Stretch { rest: f32 },
    /// One body [`UNIT_LENGTH`] from its target.
    Offset,
}

/// Which reference F5 gives a declaration: a unary or group pull at an
/// offset, a pair repulsion at contact, a pair spring at a stretch. `None`
/// for the kernels F5 does not name.
pub fn family(term: &Term) -> Option<Family> {
    match (term.topology, term.kernel) {
        (Topology::Unary | Topology::Groups, Kernel::Harmonic | Kernel::Spring) => {
            Some(Family::Offset)
        },
        (Topology::Unary | Topology::Medium, _) => None,
        (_, Kernel::Repulsion { .. }) => Some(Family::Contact),
        (_, Kernel::Spring) => Some(Family::Stretch),
        _ => None,
    }
}

/// A reference without its rest length.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    Contact,
    Stretch,
    Offset,
}

impl Reference {
    pub fn family(self) -> Family {
        match self {
            Reference::Contact => Family::Contact,
            Reference::Stretch { .. } => Family::Stretch,
            Reference::Offset => Family::Offset,
        }
    }
}

/// A term on the common scale: where its weight is read, and the weight.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Scale {
    pub reference: Reference,
    /// The term's force at its reference.
    pub weight: f64,
}

/// A push `strength · d^exponent` at contact.
pub fn at_contact(strength: f32, exponent: f32) -> f64 {
    f64::from(strength) * f64::from(CONTACT).powf(f64::from(exponent))
}

/// The strength whose push at contact is `weight`.
pub fn strength_at_contact(weight: f64, exponent: f32) -> f32 {
    (weight / f64::from(CONTACT).powf(f64::from(exponent))) as f32
}

/// A pull `strength · offset` at the unit length.
pub fn at_offset(strength: f32) -> f64 {
    f64::from(strength) * f64::from(UNIT_LENGTH)
}

/// The strength whose pull at the unit length is `weight`.
pub fn strength_at_offset(weight: f64) -> f32 {
    (weight / f64::from(UNIT_LENGTH)) as f32
}

/// A spring's `k · rest` at one rest length of stretch.
pub fn at_stretch(stiffness: f32, rest: f32) -> f64 {
    f64::from(stiffness) * f64::from(rest)
}

/// The stiffness whose pull at one rest length of stretch is `weight`.
pub fn stiffness_at_stretch(weight: f64, rest: f32) -> f32 {
    (weight / f64::from(rest)) as f32
}

#[cfg(test)]
mod tests;
