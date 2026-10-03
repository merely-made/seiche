// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! Energy: ForceAtlas2's force model, with LinLog as a tuning.
//!
//! Attraction along edges grows as `a·dᵉ` and repulsion between every pair
//! falls as `r/d`, weighted by `(deg_i + 1)(deg_j + 1)` so hubs push
//! everything away and sit central. In the (a, r) notation of Jacomy et al.
//! (ForceAtlas2, *PLoS ONE* 9(6), 2014) the default is ForceAtlas2's
//! (1, −1): attraction linear in distance, which they place between
//! Fruchterman–Reingold's (2, −1) and Noack's LinLog. LinLog is (0, −1): its
//! *energy* is linear in distance, so its attractive force is constant, and
//! its optimal layouts relax modularity clusterings (Noack, *Phys. Rev. E*
//! 79, 2009). [`LinLogForce::attraction_exponent`] `0.0` is that tuning.
//! Either way the settled distance between two groups grows as fewer edges
//! join them, so communities come out as separated islands where a
//! spring-electrical layout smears them.
//!
//! The law id stays `energy.linlog` for saved scenes (ruled 2026-10-02, F7:
//! "Keep (1, −1), relabel ForceAtlas2").

use rapier2d::prelude::*;

use crate::terms::floored_log;
use crate::{Class, Declared, Force, ForceContext, Kernel, Layout, Observable, Term, Topology};

use super::{degrees, node_positions};

/// ForceAtlas2's attraction and repulsion, degree-weighted by default, with
/// the attraction exponent a parameter (LinLog at `0.0`).
#[derive(Clone, Copy, Debug)]
pub struct LinLogForce {
    /// Attraction along an edge at unit distance: the pull is
    /// `attraction · d^attraction_exponent`.
    pub attraction: f32,
    /// How the pull grows with distance: `1.0` is ForceAtlas2's linear
    /// attraction (the default), `0.0` LinLog's constant one.
    pub attraction_exponent: f32,
    /// Repulsion at unit distance between any two bodies.
    pub repulsion: f32,
    /// Weight repulsion and centring by `deg + 1` per side — ForceAtlas2's
    /// hub term. Off, every pair repels alike.
    pub degree_weighted: bool,
    /// Distance floor for the repulsion.
    pub min_distance: f32,
    /// Weak centering so disconnected pieces stay bounded.
    pub gravity: f32,
}

impl Default for LinLogForce {
    fn default() -> Self {
        Self {
            attraction: 4.0,
            attraction_exponent: 1.0,
            repulsion: 60_000.0,
            degree_weighted: true,
            min_distance: 10.0,
            gravity: 0.02,
        }
    }
}

impl Force for LinLogForce {
    fn apply(&self, ctx: &mut ForceContext<'_>, _dt: f32) {
        let nodes = node_positions(ctx);
        let degree = degrees(ctx.edges);
        let mut forces = vec![Vector::ZERO; nodes.len()];
        let index: std::collections::HashMap<_, _> = nodes
            .iter()
            .enumerate()
            .map(|(i, (key, _, _))| (*key, i))
            .collect();
        // Repulsion between all pairs: r · w_i · w_j / d, along the separation.
        for i in 0..nodes.len() {
            let w_i = if self.degree_weighted {
                (degree.get(&nodes[i].0).copied().unwrap_or(0) + 1) as f32
            } else {
                1.0
            };
            for j in (i + 1)..nodes.len() {
                let w_j = if self.degree_weighted {
                    (degree.get(&nodes[j].0).copied().unwrap_or(0) + 1) as f32
                } else {
                    1.0
                };
                let delta = nodes[i].2 - nodes[j].2;
                let dist = delta.length().max(self.min_distance);
                let push = delta / dist * (self.repulsion * w_i * w_j / dist);
                forces[i] += push;
                forces[j] -= push;
            }
            // Gravity toward the origin, proportional to distance and degree.
            forces[i] -= nodes[i].2 * (self.gravity * w_i);
        }
        // Attraction along edges, `a·dᵉ`: at the default exponent 1 the
        // factor `d^0` is exactly one, so the pull is `delta · a` as before.
        for &(a, b) in ctx.edges {
            let (Some(&i), Some(&j)) = (index.get(&a), index.get(&b)) else {
                continue;
            };
            if i == j {
                continue;
            }
            let delta = nodes[j].2 - nodes[i].2;
            let dist = delta.length();
            if self.attraction_exponent < 1.0 && dist < 1e-3 {
                continue;
            }
            let pull = delta * (self.attraction * dist.powf(self.attraction_exponent - 1.0));
            forces[i] += pull;
            forces[j] -= pull;
        }
        for (i, (_, handle, _)) in nodes.iter().enumerate() {
            if let Some(body) = ctx.bodies.get_mut(*handle) {
                body.add_force(forces[i], true);
            }
        }
    }
}

impl LinLogForce {
    /// Each node's weight, `deg + 1` when degree-weighted, else one.
    fn weights(&self, layout: &Layout<'_>) -> Vec<f64> {
        let degree = degrees(layout.edges);
        layout
            .nodes
            .iter()
            .map(|(key, _)| {
                if self.degree_weighted {
                    f64::from(degree.get(key).copied().unwrap_or(0) + 1)
                } else {
                    1.0
                }
            })
            .collect()
    }
}

impl Declared for LinLogForce {
    fn terms(&self) -> Vec<Term> {
        vec![
            Term::force(
                "repulsion",
                Topology::AllPairs { cutoff: None },
                Kernel::Repulsion { exponent: -1.0 },
                Class::E,
                Observable::Separation,
            ),
            Term::force(
                "attraction",
                Topology::Edges,
                Kernel::Attraction {
                    exponent: self.attraction_exponent,
                },
                Class::E,
                Observable::Separation,
            ),
            Term::force(
                "centring",
                Topology::Unary,
                Kernel::Harmonic,
                Class::E,
                Observable::Spread,
            ),
        ]
    }

    fn isolate(&self, term: usize) -> Option<Box<dyn Force>> {
        let only = match term {
            0 => Self {
                attraction: 0.0,
                gravity: 0.0,
                ..*self
            },
            1 => Self {
                repulsion: 0.0,
                gravity: 0.0,
                ..*self
            },
            2 => Self {
                repulsion: 0.0,
                attraction: 0.0,
                ..*self
            },
            _ => return None,
        };
        Some(Box::new(only))
    }

    /// Repulsion `−r·wᵢwⱼ·ln d` (floored), attraction `a·d^(e+1)/(e+1)` per
    /// edge, centring `(g·wᵢ/2)|x|²`.
    fn energy(&self, term: usize, layout: &Layout<'_>) -> Option<f64> {
        let w = self.weights(layout);
        match term {
            0 => {
                let (r, m) = (f64::from(self.repulsion), f64::from(self.min_distance));
                let mut energy = 0.0;
                for i in 0..w.len() {
                    for j in (i + 1)..w.len() {
                        energy -= r * w[i] * w[j] * floored_log(layout.distance(i, j), m);
                    }
                }
                Some(energy)
            },
            1 => {
                let (a, e) = (
                    f64::from(self.attraction),
                    f64::from(self.attraction_exponent),
                );
                Some(
                    layout
                        .edge_indices()
                        .into_iter()
                        .map(|(i, j)| a * layout.distance(i, j).powf(e + 1.0) / (e + 1.0))
                        .sum(),
                )
            },
            2 => {
                let g = f64::from(self.gravity);
                Some(
                    (0..w.len())
                        .map(|i| {
                            let (x, y) = layout.at(i);
                            0.5 * g * w[i] * (x * x + y * y)
                        })
                        .sum(),
                )
            },
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Boundary, EdgeSpring, NodeExclusion, NodeKey, Simulation};
    use euclid::default::Point2D;

    /// Two triangles joined by a single edge, seeded interleaved so the law
    /// has to separate them.
    fn two_cliques() -> (Vec<NodeKey>, Vec<(NodeKey, NodeKey)>) {
        let keys: Vec<NodeKey> = (0..6).map(NodeKey::new).collect();
        let mut edges = vec![];
        for group in [&keys[0..3], &keys[3..6]] {
            edges.push((group[0], group[1]));
            edges.push((group[1], group[2]));
            edges.push((group[2], group[0]));
        }
        edges.push((keys[2], keys[3]));
        (keys, edges)
    }

    fn centroid(sim: &Simulation, keys: &[NodeKey]) -> Vector {
        let sum = keys
            .iter()
            .filter_map(|k| sim.position_of(*k))
            .fold(Vector::ZERO, |acc, p| acc + Vector::new(p.x, p.y));
        sum / keys.len() as f32
    }

    fn settle(forces: Vec<Box<dyn Force>>) -> f32 {
        let (keys, edges) = two_cliques();
        let mut sim = Simulation::new();
        sim.sync_nodes(keys.iter().enumerate().map(|(i, &k)| {
            (
                k,
                Point2D::new((i % 2) as f32 * 40.0, (i / 2) as f32 * 40.0),
            )
        }));
        sim.sync_edges(edges);
        sim.set_forces(forces);
        for _ in 0..900 {
            sim.tick(1.0 / 60.0);
        }
        let gap = centroid(&sim, &keys[0..3]) - centroid(&sim, &keys[3..6]);
        let spread = keys[0..3]
            .iter()
            .filter_map(|k| sim.position_of(*k))
            .map(|p| (Vector::new(p.x, p.y) - centroid(&sim, &keys[0..3])).length())
            .fold(0.0, f32::max);
        gap.length() / spread.max(1.0)
    }

    /// The law's claim: two cliques joined by one edge separate further,
    /// relative to their own size, under Energy than under Springs.
    #[test]
    fn communities_separate_further_than_under_springs() {
        let energy = settle(vec![Box::new(LinLogForce::default())]);
        let springs = settle(vec![
            Box::new(NodeExclusion::default()),
            Box::new(EdgeSpring::default()),
            Box::new(Boundary::default()),
        ]);
        assert!(
            energy > springs * 1.3,
            "island separation ratio under Energy {energy:.2} should exceed Springs {springs:.2}"
        );
    }

    /// LinLog proper, the attraction exponent at `0`, is a tuning of the same
    /// law: it settles the same fixture to a finite picture.
    #[test]
    fn linlog_proper_runs_as_a_tuning() {
        let linlog = settle(vec![Box::new(LinLogForce {
            attraction_exponent: 0.0,
            ..LinLogForce::default()
        })]);
        let forceatlas2 = settle(vec![Box::new(LinLogForce::default())]);
        println!("island separation: LinLog {linlog:.2}, ForceAtlas2 {forceatlas2:.2}");
        assert!(
            linlog.is_finite() && linlog > 0.0,
            "LinLog settles: {linlog}"
        );
    }
}
