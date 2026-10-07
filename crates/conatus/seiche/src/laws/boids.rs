// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! Flock: Reynolds' boids over the graph.
//!
//! Three steering forces — *separation* from anything too close, *alignment*
//! with flockmates' headings, *cohesion* toward flockmates' centre — with the
//! graph's edges deciding who is a flockmate. A connected group moves as a
//! constellation, keeping its shape while it drifts; unconnected groups pass
//! each other. Nothing settles: like [`Gravity`](super::Gravity) this is a
//! law of motion, and a receipt reads its energy, not its rest.

use std::collections::HashMap;

use rapier2d::prelude::*;

use crate::terms::harmonic;
use crate::{
    Class, Declared, Force, ForceContext, Kernel, Layout, Metric, NodeKey, Observable, Term,
    Topology,
};

use super::node_positions;

/// Boids with edge-neighbours as flockmates.
#[derive(Clone, Copy, Debug)]
pub struct Boids {
    /// Push away from any body closer than `separation_radius`.
    pub separation: f32,
    pub separation_radius: f32,
    /// Steer toward flockmates' mean heading.
    pub alignment: f32,
    /// Steer toward flockmates' centre.
    pub cohesion: f32,
    /// The speed every body is nudged toward, so the flock keeps moving.
    pub cruise_speed: f32,
    /// Steering toward the cruising speed, per unit of speed short of it.
    pub cruise: f32,
    /// Weak centering so the flock stays on the canvas.
    pub gravity: f32,
}

impl Default for Boids {
    fn default() -> Self {
        Self {
            separation: 4_000.0,
            separation_radius: 60.0,
            alignment: 3.0,
            cohesion: 0.6,
            cruise_speed: 40.0,
            cruise: 0.5,
            gravity: 0.03,
        }
    }
}

impl Force for Boids {
    fn apply(&self, ctx: &mut ForceContext<'_>, _dt: f32) {
        let nodes = node_positions(ctx);
        let velocities: Vec<Vector> = nodes
            .iter()
            .map(|(_, handle, _)| {
                ctx.bodies
                    .get(*handle)
                    .map(|b| b.linvel())
                    .unwrap_or(Vector::ZERO)
            })
            .collect();
        let index: HashMap<NodeKey, usize> = nodes
            .iter()
            .enumerate()
            .map(|(i, (key, _, _))| (*key, i))
            .collect();
        let mut mates: Vec<Vec<usize>> = vec![Vec::new(); nodes.len()];
        for &(a, b) in ctx.edges {
            if let (Some(&i), Some(&j)) = (index.get(&a), index.get(&b)) {
                if i != j {
                    mates[i].push(j);
                    mates[j].push(i);
                }
            }
        }
        let mut forces = vec![Vector::ZERO; nodes.len()];
        let sep2 = self.separation_radius * self.separation_radius;
        for i in 0..nodes.len() {
            let mut force = Vector::ZERO;
            // Separation: from everyone near, flockmate or not.
            for j in 0..nodes.len() {
                if i == j {
                    continue;
                }
                let delta = nodes[i].2 - nodes[j].2;
                let d2 = delta.length_squared();
                if d2 < sep2 && d2 > 1e-6 {
                    let d = d2.sqrt();
                    force += delta / d
                        * (self.separation * (1.0 - d / self.separation_radius) / d.max(1.0));
                }
            }
            if !mates[i].is_empty() {
                let n = mates[i].len() as f32;
                let mean_velocity = mates[i]
                    .iter()
                    .fold(Vector::ZERO, |acc, &j| acc + velocities[j])
                    / n;
                let centre = mates[i]
                    .iter()
                    .fold(Vector::ZERO, |acc, &j| acc + nodes[j].2)
                    / n;
                force += (mean_velocity - velocities[i]) * self.alignment;
                force += (centre - nodes[i].2) * self.cohesion;
            }
            // Cruise: keep a heading, so a still flock starts and a fast one slows.
            let speed = velocities[i].length();
            if speed > 1e-3 {
                force += velocities[i] / speed * ((self.cruise_speed - speed) * self.cruise);
            } else {
                // A body with no heading takes one from its index, deterministically.
                let angle = i as f32 * 2.399_963;
                force += Vector::new(angle.cos(), angle.sin()) * (self.cruise_speed * self.cruise);
            }
            force -= nodes[i].2 * self.gravity;
            forces[i] = force;
        }
        for (i, (_, handle, _)) in nodes.iter().enumerate() {
            if let Some(body) = ctx.bodies.get_mut(*handle) {
                body.add_force(forces[i], true);
            }
        }
    }
}

/// Separation is a pair potential (E). Cohesion toward each node's own
/// mates is a gradient in the metric of mate counts (Em); alignment and the
/// cruise read velocity (N).
impl Declared for Boids {
    /// Separation at contact, its taper `1 − d/R` included, and the
    /// centring at the unit length; F5 names no reference for steering or
    /// the cruise.
    fn scale(&self, term: usize) -> Option<crate::scale::Scale> {
        let taper = f64::from(1.0 - crate::scale::CONTACT / self.separation_radius).max(0.0);
        match term {
            0 => Some(crate::scale::Scale {
                reference: crate::scale::Reference::Contact,
                weight: crate::scale::at_contact(self.separation, -1.0) * taper,
            }),
            2 => Some(crate::scale::Scale {
                reference: crate::scale::Reference::Contact,
                weight: crate::scale::at_contact(self.cohesion, 1.0),
            }),
            4 => Some(crate::scale::Scale {
                reference: crate::scale::Reference::Offset,
                weight: crate::scale::at_offset(self.gravity),
            }),
            _ => None,
        }
    }

    fn reweighted(&self, term: usize, weight: f64) -> Option<Box<dyn Force>> {
        let taper = f64::from(1.0 - crate::scale::CONTACT / self.separation_radius).max(0.0);
        let mut force = *self;
        match term {
            0 if taper > 0.0 => {
                force.separation = crate::scale::strength_at_contact(weight / taper, -1.0)
            },
            2 => force.cohesion = crate::scale::strength_at_contact(weight, 1.0),
            4 => force.gravity = crate::scale::strength_at_offset(weight),
            _ => return None,
        }
        Some(Box::new(force))
    }

    fn terms(&self) -> Vec<Term> {
        vec![
            Term::force(
                "separation",
                Topology::AllPairs {
                    cutoff: Some(self.separation_radius),
                },
                Kernel::Repulsion { exponent: -1.0 },
                Class::E,
                Observable::Overlaps,
            ),
            Term::force(
                "alignment",
                Topology::Edges,
                Kernel::Steering,
                Class::N,
                Observable::Energy,
            )
            .in_metric(Metric::Degree),
            Term::force(
                "cohesion",
                Topology::Edges,
                Kernel::Steering,
                Class::Em,
                Observable::PairLength,
            )
            .in_metric(Metric::Degree),
            Term::force(
                "cruise",
                Topology::Unary,
                Kernel::Drive,
                Class::N,
                Observable::Energy,
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
        let none = Self {
            separation: 0.0,
            alignment: 0.0,
            cohesion: 0.0,
            cruise: 0.0,
            gravity: 0.0,
            ..*self
        };
        let only = match term {
            0 => Self {
                separation: self.separation,
                ..none
            },
            1 => Self {
                alignment: self.alignment,
                ..none
            },
            2 => Self {
                cohesion: self.cohesion,
                ..none
            },
            3 => Self {
                cruise: self.cruise,
                ..none
            },
            4 => Self {
                gravity: self.gravity,
                ..none
            },
            _ => return None,
        };
        Some(Box::new(only))
    }

    /// Separation `s(ln(R/d) + d/R − 1)` within the radius (joined smoothly
    /// below one unit), cohesion `(c/2)Σ|xᵢ − xⱼ|²` over the mates, centring
    /// `(g/2)|x|²`.
    fn energy(&self, term: usize, layout: &Layout<'_>) -> Option<f64> {
        match term {
            0 => {
                let (s, r) = (
                    f64::from(self.separation),
                    f64::from(self.separation_radius),
                );
                let mut energy = 0.0;
                for i in 0..layout.nodes.len() {
                    for j in (i + 1)..layout.nodes.len() {
                        let d = layout.distance(i, j);
                        if d >= r || d * d <= 1e-6 {
                            continue;
                        }
                        energy += if d >= 1.0 {
                            s * ((r / d).ln() + d / r - 1.0)
                        } else {
                            s * (r.ln() + 1.0 / (2.0 * r)) - s * (d - d * d / (2.0 * r))
                        };
                    }
                }
                Some(energy)
            },
            2 => {
                let c = f64::from(self.cohesion);
                Some(
                    layout
                        .edge_indices()
                        .into_iter()
                        .map(|(i, j)| 0.5 * c * layout.distance(i, j).powi(2))
                        .sum(),
                )
            },
            4 => Some(harmonic(f64::from(self.gravity), layout, |_| {
                Some((0.0, 0.0))
            })),
            _ => None,
        }
    }

    /// Each node's mate count: the averaging that makes cohesion and
    /// alignment a gradient only in this metric.
    fn metric(&self, term: usize, layout: &Layout<'_>) -> Option<Vec<f64>> {
        if term != 1 && term != 2 {
            return None;
        }
        let mut count = vec![0.0; layout.nodes.len()];
        for (i, j) in layout.edge_indices() {
            count[i] += 1.0;
            count[j] += 1.0;
        }
        Some(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Simulation;
    use euclid::default::Point2D;

    /// The law's claim: a connected flock keeps moving, and its members move
    /// the same way — headings aligned — after 600 ticks.
    #[test]
    fn a_flock_moves_together() {
        let keys: Vec<NodeKey> = (0..6).map(NodeKey::new).collect();
        let mut sim = Simulation::new();
        sim.set_linear_damping(0.2);
        sim.sync_nodes(keys.iter().enumerate().map(|(i, &k)| {
            (
                k,
                Point2D::new((i % 3) as f32 * 70.0, (i / 3) as f32 * 70.0),
            )
        }));
        // A path through all six: every node has flockmates.
        sim.sync_edges(keys.windows(2).map(|w| (w[0], w[1])).collect::<Vec<_>>());
        sim.set_forces(vec![Box::new(Boids::default())]);
        for _ in 0..600 {
            sim.tick(1.0 / 60.0);
        }
        assert!(
            sim.kinetic_energy() > 1.0,
            "the flock should still be moving"
        );
        let headings: Vec<Vector> = keys
            .iter()
            .filter_map(|&k| sim.velocity_of(k))
            .filter(|v| v.length() > 1e-3)
            .map(|v| v / v.length())
            .collect();
        assert!(headings.len() >= 5, "every member should have a heading");
        let mean = headings.iter().fold(Vector::ZERO, |acc, h| acc + *h) / headings.len() as f32;
        assert!(
            mean.length() > 0.6,
            "headings should be aligned (mean resultant {:.2})",
            mean.length()
        );
    }
}
