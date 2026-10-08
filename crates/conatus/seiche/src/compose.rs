// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! Composition by currency: which terms a law takes as they are, which it
//! takes converted, and which it refuses.
//!
//! Ruled 2026-10-02, "Laws declare currency; catalog adapts or refuses":
//! forces compose freely; a kinematic law takes forces converted to its
//! currency ("overdamped, v = F/γ, the rule Hold already follows"); a
//! resident law takes only forces with resident kernels or the lagged
//! upload; the pickers grey out the rest with the reason.
//!
//! A law's currency comes from its declared terms ([`law_currency`]). The
//! conversion is the kinematic law's own write: each tick it leaves its
//! moving bodies at rest, so the forces other terms add are integrated for
//! one step from rest, `v = F·dt / (m·(1 + c·dt))`, which is `F/γ` with
//! `γ = m·(1/dt + c)` (`c` the linear damping). Hold has always done this,
//! Anneal now does, and Density does once it takes a force
//! ([`Density::converts`](crate::Density::converts)), off while overlays on
//! Density stay refused, since on, a crowded start spreads less.
//!
//! Plan: `design_docs/mere_docs/implementation_strategy/2026-10-02_dynamics_grammar_plan.md`, G3.

use crate::{Currency, Force, Term};

/// Why a kinematic law refuses a term that also writes state.
pub const TWO_WRITERS: &str = "this law writes the bodies' state, and so would this term: \
                               two writes do not compose";
/// Why a resident law refuses a force with no resident kernel.
pub const NOT_RESIDENT: &str = "this law integrates on the GPU and takes only forces with a \
                                resident kernel or the lagged upload";
/// Why a resident law refuses a term that writes state on the CPU.
pub const RESIDENT_WRITE: &str = "this law integrates on the GPU and takes no term that writes \
                                  the bodies' state on the CPU";
/// Why a weighted mix refuses a law that does not add force.
pub const UNWEIGHTED: &str = "a weight scales a law's forces, and this law writes the bodies' \
                              state instead, so a mix takes force laws only";

/// How a law's motion enters the step: kinematic if a term writes state
/// every tick, resident if a term integrates on the GPU, force otherwise.
/// Initial conditions ([`Term::once`]) and the integrator's own terms
/// (contacts, damping) do not decide it.
pub fn law_currency(terms: &[Term]) -> Currency {
    let mut currency = Currency::Force;
    for term in terms
        .iter()
        .filter(|t| !t.initial && t.currency != Currency::Integrator)
    {
        match term.currency {
            Currency::Resident => return Currency::Resident,
            Currency::Kinematic => currency = Currency::Kinematic,
            Currency::Force | Currency::Integrator => {},
        }
    }
    currency
}

/// The currency of a whole force list (a law's forces, in order).
pub fn currency_of(forces: &[Box<dyn Force>]) -> Currency {
    let terms: Vec<Term> = forces.iter().flat_map(|force| force.terms()).collect();
    law_currency(&terms)
}

/// What a law does with one more term.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Admission {
    /// Taken as it is.
    Compose,
    /// Taken converted to the law's currency.
    Convert,
    /// Refused, with the reason a picker shows.
    Refuse(&'static str),
}

impl Admission {
    /// The more restrictive of two admissions: a refusal, then a conversion.
    pub fn and(self, other: Admission) -> Admission {
        match (self, other) {
            (Admission::Refuse(reason), _) | (_, Admission::Refuse(reason)) => {
                Admission::Refuse(reason)
            },
            (Admission::Convert, _) | (_, Admission::Convert) => Admission::Convert,
            _ => Admission::Compose,
        }
    }

    pub fn refusal(self) -> Option<&'static str> {
        match self {
            Admission::Refuse(reason) => Some(reason),
            _ => None,
        }
    }
}

/// What a law of `law` currency does with `term`; `resident` is whether the
/// term has a resident kernel or the lagged upload.
pub fn admit(law: Currency, term: &Term, resident: bool) -> Admission {
    if term.currency == Currency::Integrator || term.initial {
        return Admission::Compose;
    }
    match (law, term.currency) {
        (Currency::Force | Currency::Integrator, _) => Admission::Compose,
        (Currency::Kinematic, Currency::Force) => Admission::Convert,
        (Currency::Kinematic, _) => Admission::Refuse(TWO_WRITERS),
        (Currency::Resident, Currency::Force) if resident => Admission::Compose,
        (Currency::Resident, Currency::Force) => Admission::Refuse(NOT_RESIDENT),
        (Currency::Resident, Currency::Resident) => Admission::Compose,
        (Currency::Resident, _) => Admission::Refuse(RESIDENT_WRITE),
    }
}

/// What a law of `law` currency does with every term of `force`.
pub fn admit_force(law: Currency, force: &dyn Force) -> Admission {
    force
        .terms()
        .iter()
        .enumerate()
        .map(|(i, term)| admit(law, term, force.resident(i)))
        .fold(Admission::Compose, Admission::and)
}

/// Why a weighted mix of laws of these currencies is refused, if it is: a
/// weight scales forces, so a mix holds force laws only.
pub fn mix_refusal(laws: &[Currency]) -> Option<&'static str> {
    laws.iter()
        .any(|c| *c != Currency::Force)
        .then_some(UNWEIGHTED)
}

#[cfg(test)]
mod tests;
