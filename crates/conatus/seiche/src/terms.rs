// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! Term declarations: what each force is, in the dynamics grammar's words.
//!
//! Every [`Force`](crate::Force) is also [`Declared`]: it lists the terms it
//! applies, each with its topology (who interacts), kernel family, the state
//! it moves, its currency, its class and, for a metric-weighted term, its
//! metric channel, plus the observable that shows it working. A host catalog
//! reads these rather than restating them, and [`crate::instruments`] checks
//! each declared class against the forces the term actually applies.
//!
//! The classes are the dynamics grammar brief's (§3, §6):
//!
//! - **E**: a gradient of a scalar energy, every body at equal inertia;
//! - **Em**: a gradient under a non-identity mass metric, forces `−M⁻¹∇U`;
//! - **H**: conservative, but the minimizer is degenerate (collapse, total
//!   synchrony, a moving target), so the picture is in the trajectory;
//! - **N**: non-conservative (non-reciprocal, velocity-driven or rotational);
//! - **K**: writes state rather than force.
//!
//! An E or Em term exposes its energy, so a later optimizer can minimize it.
//! Contacts, damping and pins live in rapier's step rather than the force
//! list; they are declared here as [`CONTACTS`], [`DAMPING`] and [`PIN`].
//!
//! Plan: `design_docs/archive_docs/2026-10-28/2026-10-02_dynamics_grammar_plan.md`, G1.

use std::collections::HashMap;

use rapier2d::prelude::*;

pub use seiche_model::terms::{
    CONTACTS, Class, Currency, DAMPING, Kernel, Metric, Observable, PIN, State, Term, Topology,
};

use crate::{Force, NodeKey};

/// A layout to read an energy or a metric from: every node body in node-key
/// order, as the laws read them, and the synced edges.
#[derive(Clone, Copy, Debug)]
pub struct Layout<'a> {
    pub nodes: &'a [(NodeKey, Vector)],
    pub edges: &'a [(NodeKey, NodeKey)],
}

impl Layout<'_> {
    /// Node key to its index in [`Self::nodes`].
    pub fn index(&self) -> HashMap<NodeKey, usize> {
        self.nodes
            .iter()
            .enumerate()
            .map(|(i, (key, _))| (*key, i))
            .collect()
    }

    /// The distance between nodes `i` and `j`, in double precision.
    pub fn distance(&self, i: usize, j: usize) -> f64 {
        let (a, b) = (self.nodes[i].1, self.nodes[j].1);
        (f64::from(a.x) - f64::from(b.x)).hypot(f64::from(a.y) - f64::from(b.y))
    }

    /// Node `i`'s position, in double precision.
    pub fn at(&self, i: usize) -> (f64, f64) {
        let p = self.nodes[i].1;
        (f64::from(p.x), f64::from(p.y))
    }

    /// Each edge with both endpoints present and distinct, as index pairs.
    pub fn edge_indices(&self) -> Vec<(usize, usize)> {
        let index = self.index();
        self.edges
            .iter()
            .filter_map(|(a, b)| Some((*index.get(a)?, *index.get(b)?)))
            .filter(|(i, j)| i != j)
            .collect()
    }
}

/// What a force is, term by term. Every [`Force`] declares the terms it
/// applies, in order; a force that bundles several can isolate one, an E or
/// Em term exposes its energy, and a metric-weighted term its weights.
pub trait Declared {
    /// The terms this force applies.
    fn terms(&self) -> Vec<Term>;

    /// Term `term` alone, for a force that applies several; `None` for a
    /// force with one term, which is itself.
    fn isolate(&self, _term: usize) -> Option<Box<dyn Force>> {
        None
    }

    /// Term `term`'s energy at `layout`, for an E or Em term: continuous, and
    /// zero beyond a cutoff. The term's forces are `−M⁻¹∇U`.
    fn energy(&self, _term: usize, _layout: &Layout<'_>) -> Option<f64> {
        None
    }

    /// Term `term`'s metric weights at `layout`, one per node in layout
    /// order, for a term declared with a metric.
    fn metric(&self, _term: usize, _layout: &Layout<'_>) -> Option<Vec<f64>> {
        None
    }

    /// Whether term `term` has a resident kernel or the lagged upload, so a
    /// resident law can take it ([`crate::compose`]).
    fn resident(&self, _term: usize) -> bool {
        false
    }

    /// Term `term` on the common scale ([`crate::scale`], F5): where its
    /// weight is read and the weight, its force there. `None` for a term
    /// whose kernel F5 gives no reference.
    fn scale(&self, _term: usize) -> Option<crate::scale::Scale> {
        None
    }

    /// This force with term `term` at `weight` on the common scale, every
    /// other parameter kept.
    fn reweighted(&self, _term: usize, _weight: f64) -> Option<Box<dyn Force>> {
        None
    }
}

/// `s/d` past a floor `m`, joined smoothly below it to match a push that
/// grows linearly inside the floor: the energy of a floored inverse-square
/// repulsion.
pub(crate) fn floored_inverse(s: f64, d: f64, m: f64) -> f64 {
    if d >= m {
        s / d
    } else {
        s / m + s * (m * m - d * d) / (2.0 * m * m * m)
    }
}

/// `ln d` past a floor `m`, joined smoothly below it: the energy shape of a
/// floored `1/d` force.
pub(crate) fn floored_log(d: f64, m: f64) -> f64 {
    if d >= m {
        d.ln()
    } else {
        m.ln() - 0.5 + d * d / (2.0 * m * m)
    }
}

/// Hooke's energy, `(k/2)(d − L)²`.
pub(crate) fn spring(k: f64, d: f64, rest: f64) -> f64 {
    0.5 * k * (d - rest) * (d - rest)
}

/// `Σ (s/2)|x − target|²` over the nodes `target` gives a target for.
pub(crate) fn harmonic(
    s: f64,
    layout: &Layout<'_>,
    target: impl Fn(usize) -> Option<(f64, f64)>,
) -> f64 {
    (0..layout.nodes.len())
        .filter_map(|i| {
            let (tx, ty) = target(i)?;
            let (x, y) = layout.at(i);
            Some(0.5 * s * ((x - tx).powi(2) + (y - ty).powi(2)))
        })
        .sum()
}
