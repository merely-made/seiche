// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! A weighted force: another force's push, scaled.
//!
//! The weighted sum (dynamics grammar plan, G3; "Weighted multi-law lists"):
//! a law in a mix contributes its forces times its weight, so a mix of two
//! laws at weights 1 and 0 is the first law and at 0.5 each is half of both.
//! Only the wrapped force's own share is scaled; what earlier forces added to
//! a body this tick is left as it was. A weight of 0 leaves the force out,
//! its state unadvanced, since a law's state (a phase, a clock) cannot be
//! scaled. A weight scales force, so a force that writes state is refused
//! ([`crate::compose::UNWEIGHTED`]).

use rapier2d::prelude::*;

use crate::{Currency, Declared, Force, ForceContext, Layout, Term, compose};

pub struct Weighted {
    inner: Box<dyn Force>,
    factor: f32,
}

impl Weighted {
    /// `inner`'s forces times `factor`; refused for a force that writes state.
    pub fn new(inner: Box<dyn Force>, factor: f32) -> Result<Self, &'static str> {
        if compose::law_currency(&inner.terms()) != Currency::Force {
            return Err(compose::UNWEIGHTED);
        }
        Ok(Self {
            inner,
            factor: factor.max(0.0),
        })
    }

    pub fn factor(&self) -> f32 {
        self.factor
    }
}

impl Force for Weighted {
    fn apply(&self, ctx: &mut ForceContext<'_>, dt: f32) {
        if self.factor == 0.0 {
            return;
        }
        let before: Vec<(RigidBodyHandle, Vector)> = ctx
            .bodies_by_node
            .values()
            .map(|&h| {
                (
                    h,
                    ctx.bodies.get(h).map_or(Vector::ZERO, |b| b.user_force()),
                )
            })
            .collect();
        self.inner.apply(ctx, dt);
        for (handle, earlier) in before {
            if let Some(body) = ctx.bodies.get_mut(handle) {
                let own = body.user_force() - earlier;
                body.reset_forces(false);
                body.add_force(earlier + own * self.factor, true);
            }
        }
    }

    fn wants_tick(&self) -> bool {
        self.factor != 0.0 && self.inner.wants_tick()
    }
}

impl Declared for Weighted {
    fn terms(&self) -> Vec<Term> {
        self.inner.terms()
    }

    fn isolate(&self, term: usize) -> Option<Box<dyn Force>> {
        let only = self.inner.isolate(term)?;
        Some(Box::new(Self {
            inner: only,
            factor: self.factor,
        }))
    }

    fn energy(&self, term: usize, layout: &Layout<'_>) -> Option<f64> {
        Some(self.inner.energy(term, layout)? * f64::from(self.factor))
    }

    fn metric(&self, term: usize, layout: &Layout<'_>) -> Option<Vec<f64>> {
        self.inner.metric(term, layout)
    }

    fn resident(&self, term: usize) -> bool {
        self.inner.resident(term)
    }
}
