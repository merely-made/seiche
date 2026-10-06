// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! Degree repulsion: hubs push their surroundings apart.
//!
//! Extra separation from every node scaled by its weight — `ln(degree + 1)`
//! from the edges unless the host supplies its own — within a radius, so a
//! well-connected node clears room for the spokes it will draw in. The
//! donor's `DegreeRepulsion` extra, over any law. The falloff is `1/d`, not
//! `1/d²`: it has to still be felt at spring length (the springs hold spokes
//! at ~170 under [`EdgeSpring`](crate::EdgeSpring)'s default), where an
//! inverse-square push sized safely for close range is nothing.

use std::collections::HashMap;

use rapier2d::prelude::*;

use crate::laws::{degrees, node_positions};
use crate::terms::floored_log;
use crate::{
    Class, Declared, Force, ForceContext, Kernel, Layout, Metric, NodeKey, Observable, Term,
    Topology,
};

#[derive(Clone, Debug)]
pub struct DegreeRepulsion {
    /// Push at unit distance per unit of weight.
    pub strength: f32,
    /// Beyond this a hub pushes nothing.
    pub radius: f32,
    pub min_distance: f32,
    /// The host's per-node weights; `None` reads `ln(degree + 1)` from the
    /// edges each tick.
    weights: Option<HashMap<NodeKey, f32>>,
}

impl Default for DegreeRepulsion {
    fn default() -> Self {
        Self {
            strength: 40_000.0,
            radius: 400.0,
            min_distance: 8.0,
            weights: None,
        }
    }
}

impl DegreeRepulsion {
    /// Use these weights instead of log degree — a node absent here pushes
    /// nothing.
    pub fn with_weights(mut self, weights: impl IntoIterator<Item = (NodeKey, f32)>) -> Self {
        self.weights = Some(weights.into_iter().collect());
        self
    }
}

impl Force for DegreeRepulsion {
    fn apply(&self, ctx: &mut ForceContext<'_>, _dt: f32) {
        let nodes = node_positions(ctx);
        let weights: Vec<f32> = match &self.weights {
            Some(map) => nodes
                .iter()
                .map(|(key, _, _)| map.get(key).copied().unwrap_or(0.0).max(0.0))
                .collect(),
            None => {
                let degree = degrees(ctx.edges);
                nodes
                    .iter()
                    .map(|(key, _, _)| ((degree.get(key).copied().unwrap_or(0) + 1) as f32).ln())
                    .collect()
            },
        };
        let radius2 = self.radius * self.radius;
        let mut forces = vec![Vector::ZERO; nodes.len()];
        for i in 0..nodes.len() {
            let weight = weights[i];
            if weight <= 0.0 {
                continue;
            }
            for j in 0..nodes.len() {
                if i == j {
                    continue;
                }
                let delta = nodes[j].2 - nodes[i].2;
                if delta.length_squared() > radius2 {
                    continue;
                }
                let dist = delta.length().max(self.min_distance);
                forces[j] += delta / dist * (self.strength * weight / dist);
            }
        }
        for (i, (_, handle, _)) in nodes.iter().enumerate() {
            if let Some(body) = ctx.bodies.get_mut(*handle) {
                body.add_force(forces[i], true);
            }
        }
    }
}

impl DegreeRepulsion {
    /// Each node's weight: the host's, or `ln(degree + 1)`.
    fn weights_at(&self, layout: &Layout<'_>) -> Vec<f64> {
        let degree = degrees(layout.edges);
        layout
            .nodes
            .iter()
            .map(|(key, _)| match &self.weights {
                Some(map) => f64::from(map.get(key).copied().unwrap_or(0.0).max(0.0)),
                None => f64::from(((degree.get(key).copied().unwrap_or(0) + 1) as f32).ln()),
            })
            .collect()
    }
}

/// Each side is pushed by the other's weight alone, so the pair does not
/// balance; it is the gradient of `−s·wᵢwⱼ·ln d` in the metric of the
/// weights (class Em; the brief's finding F-c, declared as it is, F8).
impl Declared for DegreeRepulsion {
    fn scale(&self, _term: usize) -> Option<crate::scale::Scale> {
        Some(crate::scale::Scale {
            reference: crate::scale::Reference::Contact,
            weight: crate::scale::at_contact(self.strength, -1.0),
        })
    }

    fn reweighted(&self, _term: usize, weight: f64) -> Option<Box<dyn Force>> {
        let mut force = self.clone();
        force.strength = crate::scale::strength_at_contact(weight, -1.0);
        Some(Box::new(force))
    }

    fn terms(&self) -> Vec<Term> {
        vec![
            Term::force(
                "hub room",
                Topology::AllPairs {
                    cutoff: Some(self.radius),
                },
                Kernel::Repulsion { exponent: -1.0 },
                Class::Em,
                Observable::Spread,
            )
            .in_metric(Metric::Mass),
        ]
    }

    /// `−s·wᵢwⱼ·ln(d/R)` within the radius, floored.
    fn energy(&self, _term: usize, layout: &Layout<'_>) -> Option<f64> {
        let w = self.weights_at(layout);
        let (s, r, m) = (
            f64::from(self.strength),
            f64::from(self.radius),
            f64::from(self.min_distance),
        );
        let mut energy = 0.0;
        for i in 0..w.len() {
            for j in (i + 1)..w.len() {
                let d = layout.distance(i, j);
                if d <= r {
                    energy -= s * w[i] * w[j] * (floored_log(d, m) - r.ln());
                }
            }
        }
        Some(energy)
    }

    fn metric(&self, _term: usize, layout: &Layout<'_>) -> Option<Vec<f64>> {
        Some(self.weights_at(layout))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Boundary, EdgeSpring, NodeExclusion, Simulation};
    use euclid::default::Point2D;

    /// A hub's spokes end further from it with the overlay than without.
    #[test]
    fn a_hub_clears_room() {
        let run = |overlay: bool| {
            let keys: Vec<NodeKey> = (0..6).map(NodeKey::new).collect();
            let mut sim = Simulation::new();
            sim.sync_nodes(keys.iter().enumerate().map(|(i, &k)| {
                let a = i as f32 * 1.2;
                let r = if i > 0 { 60.0 } else { 0.0 };
                (k, Point2D::new(r * a.cos(), r * a.sin()))
            }));
            sim.sync_edges(
                keys[1..]
                    .iter()
                    .map(|&leaf| (keys[0], leaf))
                    .collect::<Vec<_>>(),
            );
            let mut forces: Vec<Box<dyn Force>> = vec![
                Box::new(NodeExclusion::default()),
                Box::new(EdgeSpring::default()),
                Box::new(Boundary::default()),
            ];
            if overlay {
                forces.push(Box::new(DegreeRepulsion::default()));
            }
            sim.set_forces(forces);
            for _ in 0..600 {
                sim.tick(1.0 / 60.0);
            }
            let hub = sim.position_of(keys[0]).unwrap();
            keys[1..]
                .iter()
                .map(|&k| (sim.position_of(k).unwrap() - hub).length())
                .sum::<f32>()
                / 5.0
        };
        let with = run(true);
        let without = run(false);
        assert!(
            with > without * 1.15,
            "spokes with {with:.0} should exceed without {without:.0}"
        );
    }
}
