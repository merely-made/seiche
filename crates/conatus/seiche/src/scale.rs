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
//! - A unary pull's is one body [`UNIT_LENGTH`] from its target (F79:
//!   "170, Grid at half cell"), except Grid's, whose cell of 120 cannot hold
//!   that offset: it is read half a cell from its grid point.
//! - Where distance matters (F80): the attractions (LinLog's, Hub pull),
//!   gravitation (Plummer, read on a standard node body at unit
//!   gravitational mass) and Boids' cohesion, which pulls in proportion to
//!   its offset from the mates' centre, are read at contact too.
//!
//! A term reports its [`Scale`] through [`Declared::scale`](crate::Declared)
//! and is rebuilt at another weight through `reweighted`. Today's
//! calibrated strengths are their weights, so the default compositions are
//! unchanged. Alignment, phase coupling, the needle, the two drives, the
//! writes and Density's diffusion depend on no distance and report none,
//! weight 1 meaning as calibrated. Particle life's tent reports none: at
//! contact it reads its repulsive core whatever the kind rule, so it stays
//! off the scale (F90, "Off the scale").
//!
//! Plan: `design_docs/mere_docs/implementation_strategy/2026-10-02_dynamics_grammar_plan.md`, G3.

use crate::{Kernel, NODE_BODY_DENSITY, NODE_BODY_RADIUS, Observable, Term, Topology};

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
    /// One body half a `cell` from its nearest grid point (F79).
    HalfCell { cell: f32 },
}

/// Which reference a declaration takes: a unary or group pull at an offset,
/// a pair repulsion, attraction or gravitation at contact, a pair spring at
/// a stretch, and a steering toward the mates' centre (it moves pair
/// lengths) at contact (F5, F80). `None` for kernels that depend on no
/// distance, and for the tent while its reference is open.
pub fn family(term: &Term) -> Option<Family> {
    match (term.topology, term.kernel) {
        (Topology::Unary | Topology::Groups, Kernel::Harmonic | Kernel::Spring) => {
            Some(Family::Offset)
        },
        (Topology::Unary | Topology::Medium, _) => None,
        (_, Kernel::Repulsion { .. } | Kernel::Attraction { .. } | Kernel::Plummer) => {
            Some(Family::Contact)
        },
        (_, Kernel::Steering) if term.observable == Observable::PairLength => Some(Family::Contact),
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
            Reference::Offset | Reference::HalfCell { .. } => Family::Offset,
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

/// Plummer gravitation's pull per unit `G` at contact, softened by
/// `softening`, on a standard node body (density times disc area) at unit
/// gravitational mass: the law applies acceleration times inertial mass.
pub fn plummer_at_contact(softening: f32) -> f64 {
    let (c, e) = (f64::from(CONTACT), f64::from(softening));
    let r = f64::from(NODE_BODY_RADIUS);
    let body = f64::from(NODE_BODY_DENSITY) * std::f64::consts::PI * r * r;
    c / (c * c + e * e).powf(1.5) * body
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
