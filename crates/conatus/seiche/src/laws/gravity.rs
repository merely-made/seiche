// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! Orbit: n-body gravitation, the graph as a solar system.
//!
//! Every body attracts every other with `G · m_i · m_j / (d² + ε²)`, mass by
//! degree (a hub is a sun), and on the first tick each body receives a
//! tangential kick about the mass centre so it orbits rather than falls in.
//! The orbits never settle: the law's whole point is motion. Under
//! [`CounterDamping::Tangential`] only the orbital motion is frictionless, so
//! energy another force adds (exclusion's push, a drag) leaves as radial motion
//! the host's damping settles, and the system stays bound. What it reveals is
//! hierarchy as gravity — hubs sink inside, leaves circle them.

use std::collections::HashMap;
use std::sync::Mutex;

use rapier2d::prelude::*;

use crate::{
    Class, Currency, Declared, Force, ForceContext, Kernel, Layout, Metric, NodeKey, Observable,
    State, Term, Topology,
};

use super::node_positions;

/// How [`Gravity`] meets the host's linear damping (the "inertia" setting).
/// Gravity conserves energy only in a frictionless world; damped, every leaf
/// spirals into its hub within seconds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CounterDamping {
    /// No drive: the host's damping acts on every body.
    Off,
    /// Each body's damping cancelled along its whole velocity, a frictionless
    /// world: energy any other force adds stays, and an unbound body coasts.
    Full,
    /// Only each body's tangential motion about the gravitational mass centre,
    /// moving with that centre, is frictionless; radial motion and the
    /// system's drift settle under the host's damping, so orbits circularize
    /// and keep going. Orbit's (ruled 2026-10-04, "Frictionless orbits +
    /// centring").
    Tangential,
}

/// N-body attraction with an orbital kick.
#[derive(Debug)]
pub struct Gravity {
    masses: HashMap<NodeKey, f32>,
    /// The gravitational constant, in force per unit mass² at unit distance.
    pub strength: f32,
    /// Softening length: no singularity at close approach.
    pub softening: f32,
    /// Tangential speed given on the first tick, per unit distance from the
    /// mass centre; `0.0` lets the graph simply fall together.
    pub orbital_kick: f32,
    /// Which part of each body's motion the host's damping is cancelled on.
    pub counter_damping: CounterDamping,
    kicked: Mutex<bool>,
}

impl Gravity {
    /// Masses per node; a node absent here weighs `1.0`. A host typically
    /// passes `degree + 1`. The counter-damping is the caller's choice: there
    /// is no default, so no caller's meaning changes silently (seiche 0.0.6).
    pub fn new(
        masses: impl IntoIterator<Item = (NodeKey, f32)>,
        counter_damping: CounterDamping,
    ) -> Self {
        Self {
            masses: masses.into_iter().collect(),
            strength: 9_000.0,
            softening: 24.0,
            orbital_kick: 1.0,
            counter_damping,
            kicked: Mutex::new(false),
        }
    }

    fn mass(&self, key: &NodeKey) -> f32 {
        self.masses.get(key).copied().unwrap_or(1.0).max(0.1)
    }
}

impl Force for Gravity {
    fn apply(&self, ctx: &mut ForceContext<'_>, _dt: f32) {
        let nodes = node_positions(ctx);
        if nodes.is_empty() {
            return;
        }
        let masses: Vec<f32> = nodes.iter().map(|(key, _, _)| self.mass(key)).collect();
        let total: f32 = masses.iter().sum();
        let centre = nodes
            .iter()
            .zip(&masses)
            .fold(Vector::ZERO, |acc, ((_, _, p), m)| acc + *p * *m)
            / total;

        // The kick, once: perpendicular to the radius vector at the circular
        // orbital speed for the mass inside that radius, `√(G M / r)`, scaled
        // by `orbital_kick` (1.0 is a circle, less an ellipse that falls in).
        let mut kicked = self
            .kicked
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if !*kicked && self.orbital_kick > 0.0 {
            // The mass that pulls a body inward is the mass inside its radius
            // (the shell theorem, near enough for a graph): order by radius
            // and accumulate, so an outer leaf is not kicked for a mass that
            // is beside it rather than beneath it.
            let mut by_radius: Vec<(usize, f32)> = nodes
                .iter()
                .enumerate()
                .map(|(i, (_, _, p))| (i, (*p - centre).length()))
                .collect();
            by_radius.sort_by(|a, b| a.1.total_cmp(&b.1));
            let mut enclosed = 0.0;
            for (i, r) in by_radius {
                let (_, handle, position) = &nodes[i];
                if r >= 1e-3 && enclosed > 0.0 {
                    let speed = (self.strength * enclosed / r).sqrt() * self.orbital_kick;
                    let radius = *position - centre;
                    let tangent = Vector::new(-radius.y, radius.x) / r * speed;
                    if let Some(body) = ctx.bodies.get_mut(*handle) {
                        body.set_linvel(tangent, true);
                    }
                }
                enclosed += masses[i];
            }
            *kicked = true;
        }
        drop(kicked);

        // Gravitational mass is the host's map; inertial mass is the body's
        // own, so the acceleration a body feels is `G · m_other / d²` whatever
        // rapier weighed it at — the law, not the collider density, decides
        // who circles whom.
        let soft2 = self.softening * self.softening;
        let mut accelerations = vec![Vector::ZERO; nodes.len()];
        for i in 0..nodes.len() {
            for j in (i + 1)..nodes.len() {
                let delta = nodes[j].2 - nodes[i].2;
                let dist2 = delta.length_squared() + soft2;
                let direction = delta / dist2.sqrt();
                accelerations[i] += direction * (self.strength * masses[j] / dist2);
                accelerations[j] -= direction * (self.strength * masses[i] / dist2);
            }
        }
        // The system's drift, mass-weighted as the centre is: Tangential
        // measures each body's orbital motion against it.
        let drift = match self.counter_damping {
            CounterDamping::Tangential => {
                nodes
                    .iter()
                    .zip(&masses)
                    .fold(Vector::ZERO, |acc, ((_, handle, _), m)| {
                        acc + ctx.bodies.get(*handle).map_or(Vector::ZERO, |b| b.linvel()) * *m
                    })
                    / total
            },
            CounterDamping::Off | CounterDamping::Full => Vector::ZERO,
        };
        for (i, (_, handle, position)) in nodes.iter().enumerate() {
            if let Some(body) = ctx.bodies.get_mut(*handle) {
                let inertial = body.mass().max(1e-3);
                // rapier applies `v /= 1 + dt·d` each step; a force of `m·d·v`
                // puts back what that takes, to first order.
                let restored = match self.counter_damping {
                    CounterDamping::Off => Vector::ZERO,
                    CounterDamping::Full => body.linvel(),
                    CounterDamping::Tangential => {
                        let radius = *position - centre;
                        let r = radius.length();
                        if r < 1e-3 {
                            Vector::ZERO
                        } else {
                            let tangent = Vector::new(-radius.y, radius.x) / r;
                            tangent * (body.linvel() - drift).dot(tangent)
                        }
                    },
                };
                let force =
                    accelerations[i] * inertial + restored * (body.linear_damping() * inertial);
                body.add_force(force, true);
            }
        }
    }
}

/// Gravitation is conservative in the metric of the gravitational masses,
/// but its minimizer is collapse (class H); the drive that cancels damping is
/// velocity-driven (N), and the kick is a one-time velocity write (K).
impl Declared for Gravity {
    fn terms(&self) -> Vec<Term> {
        vec![
            Term::force(
                "gravitation",
                Topology::AllPairs { cutoff: None },
                Kernel::Plummer,
                Class::H,
                Observable::Energy,
            )
            .in_metric(Metric::Mass),
            Term::force(
                "counter-damping",
                Topology::Unary,
                Kernel::Drive,
                Class::N,
                Observable::Energy,
            ),
            Term::force(
                "orbital kick",
                Topology::Unary,
                Kernel::VelocityWrite,
                Class::K,
                Observable::Energy,
            )
            .moving(State::Velocity, Currency::Kinematic),
        ]
    }

    fn isolate(&self, term: usize) -> Option<Box<dyn Force>> {
        let off = CounterDamping::Off;
        let (strength, counter_damping, orbital_kick) = match term {
            0 => (self.strength, off, 0.0),
            1 => (0.0, self.counter_damping, 0.0),
            2 => (0.0, off, self.orbital_kick),
            _ => return None,
        };
        let kicked = *self
            .kicked
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        Some(Box::new(Self {
            masses: self.masses.clone(),
            strength,
            softening: self.softening,
            orbital_kick,
            counter_damping,
            kicked: Mutex::new(kicked),
        }))
    }

    fn metric(&self, term: usize, layout: &Layout<'_>) -> Option<Vec<f64>> {
        (term == 0).then(|| {
            layout
                .nodes
                .iter()
                .map(|(key, _)| f64::from(self.mass(key)))
                .collect()
        })
    }
}

#[cfg(test)]
mod diag;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Simulation;
    use euclid::default::Point2D;

    /// Tangential's claim, on the P2 fixture from the boot Spiral's shape under
    /// the tree page's damping, composed as the catalog composes Orbit
    /// (exclusion to two node diameters, centring 0.02): the blow-out leaves
    /// as radial motion, so the extent stays within 3x its first second's and
    /// the motion stays an orbit. The control is the same set under `Full`,
    /// which keeps the blow-out and breathes instead of orbiting.
    #[test]
    fn tangential_counter_damping_keeps_the_orbits_bound() {
        use super::diag::{Counter, Fixture, Recipe, run_quietly};
        let fixture = Fixture::p2();
        let ruled = Recipe {
            counter: Counter::Law,
            centring: Some(0.02),
            ..Recipe::catalog().reach(2.0)
        };
        let control = Recipe {
            counter: Counter::Full,
            ..ruled
        };
        let read = |recipe: &Recipe| {
            run_quietly("", &fixture, 19.0, 2.5, recipe.parts(&fixture), 30, false)
        };
        let orbit = read(&ruled);
        assert!(
            orbit.max_ratio < 3.0 && orbit.min_energy > 1.0,
            "bound and moving: x{:.2}, energy at least {:.0}",
            orbit.max_ratio,
            orbit.min_energy
        );
        assert!(
            orbit.mean_tangential > 0.8 && orbit.mean_coherence > 0.8 && orbit.revolutions > 1.0,
            "an orbit: tangential {:.2}, coherence {:.2}, revolutions {:.2}",
            orbit.mean_tangential,
            orbit.mean_coherence,
            orbit.revolutions
        );
        let breathing = read(&control);
        assert!(
            breathing.mean_tangential < 0.8,
            "the control keeps the blow-out: tangential {:.2}",
            breathing.mean_tangential
        );
    }

    /// The law's claim: a hub with leaves keeps moving — kinetic energy stays
    /// above a floor after 600 ticks — where Springs would have come to rest.
    #[test]
    fn a_solar_system_never_rests() {
        let keys: Vec<NodeKey> = (0..7).map(NodeKey::new).collect();
        let mut sim = Simulation::new();
        sim.set_linear_damping(0.0);
        sim.sync_nodes(keys.iter().enumerate().map(|(i, &k)| {
            let angle = i as f32 * 0.9;
            let r = if i == 0 { 0.0 } else { 120.0 + 20.0 * i as f32 };
            (k, Point2D::new(r * angle.cos(), r * angle.sin()))
        }));
        sim.sync_edges(
            keys[1..]
                .iter()
                .map(|&leaf| (keys[0], leaf))
                .collect::<Vec<_>>(),
        );
        let masses = keys
            .iter()
            .enumerate()
            .map(|(i, &k)| (k, if i == 0 { 12.0 } else { 1.0 }));
        sim.set_forces(vec![Box::new(Gravity::new(masses, CounterDamping::Full))]);
        let mut energies = Vec::new();
        for tick in 0..600 {
            sim.tick(1.0 / 60.0);
            if tick % 100 == 99 {
                energies.push(sim.kinetic_energy());
            }
        }
        assert!(
            energies.iter().all(|&e| e > 1.0),
            "kinetic energy should stay above the floor throughout: {energies:?}"
        );
        // And nothing flew off: every leaf is still bound — within a few
        // starting radii of the hub after ten seconds of orbit.
        let hub = sim.position_of(keys[0]).unwrap();
        for &leaf in &keys[1..] {
            let r = (sim.position_of(leaf).unwrap() - hub).length();
            assert!(r < 800.0, "a leaf escaped to {r:.0}");
        }
    }
}
