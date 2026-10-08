// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! Grid snap: a spring to the nearest grid point.
//!
//! Every node is pulled toward the closest intersection of a square grid,
//! so the picture settles onto cells while the law still decides which
//! cell. The donor's `GridSnap` extra (its crystal preset), over any law.

use rapier2d::prelude::*;

use crate::laws::node_positions;
use crate::terms::harmonic;
use crate::{Class, Declared, Force, ForceContext, Kernel, Layout, Observable, Term, Topology};

#[derive(Clone, Copy, Debug)]
pub struct GridSnap {
    /// Grid pitch, world units.
    pub cell: f32,
    /// Pull per unit of offset from the nearest grid point.
    pub strength: f32,
}

impl Default for GridSnap {
    fn default() -> Self {
        Self {
            cell: 120.0,
            strength: 10.0,
        }
    }
}

impl Force for GridSnap {
    fn apply(&self, ctx: &mut ForceContext<'_>, _dt: f32) {
        let cell = self.cell.max(1.0);
        for (_, handle, position) in node_positions(ctx) {
            let target = Vector::new(
                (position.x / cell).round() * cell,
                (position.y / cell).round() * cell,
            );
            if let Some(body) = ctx.bodies.get_mut(handle) {
                body.add_force((target - position) * self.strength, true);
            }
        }
    }
}

impl Declared for GridSnap {
    /// The pull half a cell from the grid point (F79, "170, Grid at half
    /// cell"): the unit length does not fit inside a cell.
    fn scale(&self, _term: usize) -> Option<crate::scale::Scale> {
        let cell = self.cell.max(1.0);
        Some(crate::scale::Scale {
            reference: crate::scale::Reference::HalfCell { cell },
            weight: f64::from(self.strength) * f64::from(cell / 2.0),
        })
    }

    fn reweighted(&self, _term: usize, weight: f64) -> Option<Box<dyn Force>> {
        let mut force = *self;
        force.strength = (weight / f64::from(self.cell.max(1.0) / 2.0)) as f32;
        Some(Box::new(force))
    }

    fn terms(&self) -> Vec<Term> {
        vec![Term::force(
            "grid",
            Topology::Unary,
            Kernel::Harmonic,
            Class::E,
            Observable::Residual,
        )]
    }

    /// `(s/2)|x − nearest grid point|²`, continuous across cell boundaries.
    fn energy(&self, _term: usize, layout: &Layout<'_>) -> Option<f64> {
        let cell = f64::from(self.cell.max(1.0));
        Some(harmonic(f64::from(self.strength), layout, |i| {
            let (x, y) = layout.at(i);
            Some(((x / cell).round() * cell, (y / cell).round() * cell))
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NodeKey, Simulation};
    use euclid::default::Point2D;

    /// Nodes end within a few units of grid points.
    #[test]
    fn bodies_land_on_the_grid() {
        let keys: Vec<NodeKey> = (0..5).map(NodeKey::new).collect();
        let mut sim = Simulation::new();
        sim.set_linear_damping(3.0);
        sim.sync_nodes(keys.iter().enumerate().map(|(i, &k)| {
            (
                k,
                Point2D::new(i as f32 * 133.0 + 17.0, i as f32 * 41.0 + 9.0),
            )
        }));
        sim.sync_edges(Vec::<(NodeKey, NodeKey)>::new());
        sim.set_forces(vec![Box::new(GridSnap::default())]);
        for _ in 0..600 {
            sim.tick(1.0 / 60.0);
        }
        for &k in &keys {
            let p = sim.position_of(k).unwrap();
            let off = |v: f32| (v - (v / 120.0).round() * 120.0).abs();
            assert!(off(p.x) < 4.0 && off(p.y) < 4.0, "{p:?} is off the grid");
        }
    }
}
