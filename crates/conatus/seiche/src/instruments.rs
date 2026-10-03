// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! Instruments: does a term do what its declaration says?
//!
//! A [`Probe`] holds a fixture's bodies and edges in a world of their own and
//! reads one force at a time, every clock held. On it:
//!
//! - [`descent`]: under overdamped flow an E or Em term's energy never rises;
//! - [`gradient_error`]: its forces are its energy's gradient, in its metric;
//! - [`balance`]: an internal term's metric-weighted forces sum to zero;
//! - for a term with no energy, three candidate tests of conservativeness,
//!   to be chosen between by the positive control (Kinds, seeded against
//!   symmetrized): [`loop_work`] around a closed loop in joint configuration
//!   space, [`jacobian_asymmetry`], and [`persistent_motion`] under the
//!   integrator;
//! - [`velocity_read`]: whether the force at fixed positions changes with the
//!   bodies' velocities, which no position-only test can see.
//!
//! Plan: `design_docs/mere_docs/implementation_strategy/2026-10-02_dynamics_grammar_plan.md`, G1.

use euclid::default::Point2D;
use rapier2d::prelude::*;

use crate::laws::Rng;
use crate::{Force, ForceContext, Layout, NodeKey, Simulation};

/// The position and force type the instruments take: rapier's.
pub use rapier2d::prelude::Vector;

/// A fixture's bodies and edges with no other force, read one force at a time.
pub struct Probe {
    sim: Simulation,
    keys: Vec<NodeKey>,
    velocities: Vec<Vector>,
}

impl Probe {
    pub fn new(nodes: &[NodeKey], edges: &[(NodeKey, NodeKey)]) -> Self {
        let mut keys = nodes.to_vec();
        keys.sort_by_key(|key| key.index());
        keys.dedup();
        let mut sim = Simulation::new();
        sim.sync_nodes(keys.iter().map(|&key| (key, Point2D::origin())));
        sim.sync_edges(edges.iter().copied());
        let velocities = vec![Vector::ZERO; keys.len()];
        Self {
            sim,
            keys,
            velocities,
        }
    }

    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Seeded positions, uniform in a disc of `radius` about the origin.
    pub fn scatter(&self, seed: u64, radius: f32) -> Vec<Vector> {
        let mut rng = Rng::new(seed);
        (0..self.len())
            .map(|_| {
                let r = radius * rng.unit().sqrt();
                let a = rng.unit() * std::f32::consts::TAU;
                Vector::new(r * a.cos(), r * a.sin())
            })
            .collect()
    }

    fn handle(&self, i: usize) -> Option<RigidBodyHandle> {
        self.sim.bodies_by_node.get(&self.keys[i]).copied()
    }

    /// Put every body at `positions` (in node-key order), at rest.
    pub fn place(&mut self, positions: &[Vector]) {
        self.velocities = vec![Vector::ZERO; self.len()];
        for (i, &p) in positions.iter().enumerate().take(self.len()) {
            if let Some(body) = self.handle(i).and_then(|h| self.sim.bodies.get_mut(h)) {
                body.set_translation(p, false);
                body.set_linvel(Vector::ZERO, false);
                body.set_angvel(0.0, false);
            }
        }
    }

    pub fn positions(&self) -> Vec<Vector> {
        (0..self.len())
            .map(|i| {
                self.handle(i)
                    .and_then(|h| self.sim.bodies.get(h))
                    .map_or(Vector::ZERO, |b| b.translation())
            })
            .collect()
    }

    fn nodes(&self) -> Vec<(NodeKey, Vector)> {
        self.keys.iter().copied().zip(self.positions()).collect()
    }

    /// `force`'s push on each body at the current positions, read with
    /// `dt = 0` so no clock or phase advances, the bodies at the probe's
    /// velocities (rest unless [`velocity_read`] set them).
    pub fn forces(&mut self, force: &dyn Force) -> Vec<Vector> {
        for i in 0..self.len() {
            let v = self.velocities[i];
            if let Some(body) = self.handle(i).and_then(|h| self.sim.bodies.get_mut(h)) {
                body.reset_forces(false);
                body.set_linvel(v, false);
            }
        }
        let sim = &mut self.sim;
        let mut ctx = ForceContext {
            bodies: &mut sim.bodies,
            colliders: &sim.colliders,
            joints: &mut sim.impulse_joints,
            bodies_by_node: &sim.bodies_by_node,
            edges: &sim.edges,
            repulsion: None,
            step: sim.steps,
        };
        force.apply(&mut ctx, 0.0);
        (0..self.len())
            .map(|i| {
                self.handle(i)
                    .and_then(|h| self.sim.bodies.get(h))
                    .map_or(Vector::ZERO, |b| b.user_force())
            })
            .collect()
    }

    /// Term `term`'s energy at the current positions.
    pub fn energy(&self, force: &dyn Force, term: usize) -> Option<f64> {
        let nodes = self.nodes();
        force.energy(
            term,
            &Layout {
                nodes: &nodes,
                edges: &self.sim.edges,
            },
        )
    }

    /// Term `term`'s metric weights, or ones for a term with no metric.
    pub fn weights(&self, force: &dyn Force, term: usize) -> Vec<f64> {
        let nodes = self.nodes();
        force
            .metric(
                term,
                &Layout {
                    nodes: &nodes,
                    edges: &self.sim.edges,
                },
            )
            .unwrap_or_else(|| vec![1.0; self.len()])
    }
}

/// A term alone: the force itself when it has one term, else its isolation.
pub fn isolated(force: Box<dyn Force>, term: usize) -> Box<dyn Force> {
    match force.isolate(term) {
        Some(only) => only,
        None => force,
    }
}

/// The energy along an overdamped flow.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Descent {
    pub start: f64,
    pub end: f64,
    /// The largest rise in one step.
    pub max_rise: f64,
}

impl Descent {
    /// The largest single rise as a fraction of the whole fall.
    pub fn relative_rise(&self) -> f64 {
        self.max_rise / (self.start - self.end).abs().max(f64::MIN_POSITIVE)
    }
}

/// Run `x ← x + h·F(x)` from `start` for `steps`, `h` chosen so no body
/// moves more than `step` world units, and record term `term`'s energy.
/// `None` for a term with no energy.
pub fn descent(
    probe: &mut Probe,
    force: &dyn Force,
    term: usize,
    start: &[Vector],
    steps: usize,
    step: f32,
) -> Option<Descent> {
    probe.place(start);
    let first = probe.energy(force, term)?;
    let (mut last, mut max_rise) = (first, 0.0f64);
    for _ in 0..steps {
        let f = probe.forces(force);
        let largest = f.iter().map(|v| v.length()).fold(0.0f32, f32::max);
        if largest <= 0.0 {
            break;
        }
        let h = step / largest;
        let next: Vec<Vector> = probe
            .positions()
            .iter()
            .zip(&f)
            .map(|(x, v)| *x + *v * h)
            .collect();
        probe.place(&next);
        let now = probe.energy(force, term)?;
        max_rise = max_rise.max(now - last);
        last = now;
    }
    Some(Descent {
        start: first,
        end: last,
        max_rise,
    })
}

/// `‖M·F + ∇U‖ / ‖∇U‖` at `at`, the gradient by central differences of `h`
/// and `M` the term's metric. `None` for a term with no energy.
pub fn gradient_error(
    probe: &mut Probe,
    force: &dyn Force,
    term: usize,
    at: &[Vector],
    h: f32,
) -> Option<f64> {
    probe.place(at);
    let weights = probe.weights(force, term);
    let f = probe.forces(force);
    let (mut error, mut norm) = (0.0, 0.0);
    for i in 0..probe.len() {
        for axis in 0..2 {
            let mut shifted = at.to_vec();
            let base = component(at[i], axis);
            set_component(&mut shifted[i], axis, base + h);
            let plus_at = component(shifted[i], axis);
            probe.place(&shifted);
            let plus = probe.energy(force, term)?;
            set_component(&mut shifted[i], axis, base - h);
            let minus_at = component(shifted[i], axis);
            probe.place(&shifted);
            let minus = probe.energy(force, term)?;
            let grad = (plus - minus) / f64::from(plus_at - minus_at);
            let weighted = weights[i] * f64::from(component(f[i], axis));
            error += (weighted + grad).powi(2);
            norm += grad * grad;
        }
    }
    probe.place(at);
    Some((error / norm.max(f64::MIN_POSITIVE)).sqrt())
}

/// The metric-weighted force sum of an internal term.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Balance {
    /// `|Σ wᵢFᵢ|`.
    pub residual: f64,
    /// `Σ wᵢ|Fᵢ|`.
    pub scale: f64,
}

impl Balance {
    pub fn relative(&self) -> f64 {
        self.residual / self.scale.max(f64::MIN_POSITIVE)
    }
}

/// `Σ wᵢFᵢ` at `at`: zero for a pair term reciprocal in the metric `w`.
pub fn balance(probe: &mut Probe, force: &dyn Force, weights: &[f64], at: &[Vector]) -> Balance {
    probe.place(at);
    let f = probe.forces(force);
    let (mut sx, mut sy, mut scale) = (0.0, 0.0, 0.0);
    for (v, w) in f.iter().zip(weights) {
        sx += w * f64::from(v.x);
        sy += w * f64::from(v.y);
        scale += w.abs() * f64::from(v.length());
    }
    Balance {
        residual: sx.hypot(sy),
        scale,
    }
}

/// The work `∮ Σ wᵢFᵢ·ẋᵢ dt` around `x(t) = x₀ + a(A cos t + B sin t)`,
/// with `A` and `B` seeded displacements of every body, as a fraction of the
/// absolute work `∮ Σ |wᵢFᵢ·ẋᵢ| dt`. A gradient in the metric `w` does no
/// net work around any closed loop, whatever kinks its energy has.
pub fn loop_work(
    probe: &mut Probe,
    force: &dyn Force,
    weights: &[f64],
    at: &[Vector],
    amplitude: f32,
    seed: u64,
    samples: usize,
) -> f64 {
    let mut rng = Rng::new(seed);
    let mut dir = || {
        (0..probe.len())
            .map(|_| Vector::new(rng.signed(), rng.signed()))
            .collect::<Vec<_>>()
    };
    let (a, b) = (dir(), dir());
    let (mut work, mut absolute) = (0.0, 0.0);
    for k in 0..samples {
        let t = std::f32::consts::TAU * k as f32 / samples as f32;
        let (c, s) = (t.cos(), t.sin());
        let x: Vec<Vector> = (0..probe.len())
            .map(|i| at[i] + (a[i] * c + b[i] * s) * amplitude)
            .collect();
        probe.place(&x);
        let f = probe.forces(force);
        for i in 0..probe.len() {
            let velocity = (b[i] * c - a[i] * s) * amplitude;
            let power = weights[i] * f64::from(f[i].dot(velocity));
            work += power;
            absolute += power.abs();
        }
    }
    probe.place(at);
    work.abs() / absolute.max(f64::MIN_POSITIVE)
}

/// `‖A‖ / ‖J̃‖` at `at`, for `J̃ = M·∂F/∂x` by central differences of `h`
/// and `A` its antisymmetric part: zero for a gradient in the metric.
pub fn jacobian_asymmetry(
    probe: &mut Probe,
    force: &dyn Force,
    weights: &[f64],
    at: &[Vector],
    h: f32,
) -> f64 {
    let n = 2 * probe.len();
    let mut jacobian = vec![0.0f64; n * n];
    for col in 0..n {
        let (i, axis) = (col / 2, col % 2);
        let mut shifted = at.to_vec();
        let base = component(at[i], axis);
        set_component(&mut shifted[i], axis, base + h);
        let plus_at = component(shifted[i], axis);
        probe.place(&shifted);
        let plus = probe.forces(force);
        set_component(&mut shifted[i], axis, base - h);
        let minus_at = component(shifted[i], axis);
        probe.place(&shifted);
        let minus = probe.forces(force);
        let width = f64::from(plus_at - minus_at);
        for row in 0..n {
            let (j, axis) = (row / 2, row % 2);
            let delta = f64::from(component(plus[j], axis) - component(minus[j], axis));
            jacobian[row * n + col] = weights[j] * delta / width;
        }
    }
    probe.place(at);
    let (mut anti, mut whole) = (0.0, 0.0);
    for row in 0..n {
        for col in 0..n {
            let (a, b) = (jacobian[row * n + col], jacobian[col * n + row]);
            anti += (0.5 * (a - b)).powi(2);
            whole += a * a;
        }
    }
    (anti / whole.max(f64::MIN_POSITIVE)).sqrt()
}

/// Kinetic energy under the integrator: the peak over the first window and
/// the mean over the last.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Motion {
    pub early: f32,
    pub late: f32,
}

impl Motion {
    pub fn ratio(&self) -> f32 {
        self.late / self.early.max(f32::MIN_POSITIVE)
    }
}

/// Tick the probe's world under `force` alone (rapier, its damping and
/// contacts) for `ticks` at 60 Hz from `start`: a conservative term with
/// damping comes to rest, a driven or non-conservative one may not.
pub fn persistent_motion(
    probe: &mut Probe,
    force: Box<dyn Force>,
    start: &[Vector],
    ticks: usize,
    window: usize,
) -> Motion {
    probe.place(start);
    probe.sim.set_forces(vec![force]);
    let mut energies = Vec::with_capacity(ticks);
    for _ in 0..ticks {
        probe.sim.tick(1.0 / 60.0);
        energies.push(probe.sim.kinetic_energy());
    }
    probe.sim.set_forces(Vec::new());
    probe.place(start);
    let window = window.clamp(1, ticks.max(1));
    let early = energies.iter().take(window).copied().fold(0.0, f32::max);
    let tail = &energies[energies.len().saturating_sub(window)..];
    Motion {
        early,
        late: tail.iter().sum::<f32>() / tail.len().max(1) as f32,
    }
}

/// `‖F(x, v) − F(x, 0)‖ / (‖F(x, 0)‖ + ‖F(x, v)‖)` with `v` seeded at
/// `speed`: zero for a force that reads positions only.
pub fn velocity_read(
    probe: &mut Probe,
    force: &dyn Force,
    at: &[Vector],
    speed: f32,
    seed: u64,
) -> f64 {
    probe.place(at);
    let still = probe.forces(force);
    let mut rng = Rng::new(seed);
    probe.velocities = (0..probe.len())
        .map(|_| {
            let a = rng.unit() * std::f32::consts::TAU;
            Vector::new(a.cos(), a.sin()) * speed
        })
        .collect();
    let moving = probe.forces(force);
    probe.place(at);
    let norm = |f: &[Vector]| {
        f.iter()
            .map(|v| f64::from(v.length_squared()))
            .sum::<f64>()
            .sqrt()
    };
    let diff: Vec<Vector> = moving.iter().zip(&still).map(|(a, b)| *a - *b).collect();
    norm(&diff) / (norm(&still) + norm(&moving)).max(f64::MIN_POSITIVE)
}

fn component(v: Vector, axis: usize) -> f32 {
    if axis == 0 { v.x } else { v.y }
}

fn set_component(v: &mut Vector, axis: usize, value: f32) {
    if axis == 0 {
        v.x = value;
    } else {
        v.y = value;
    }
}

/// Every instrument's reading of one term, and what each class predicts.
mod reading;
pub use reading::{Candidate, Reading, median, read, tolerance};

#[cfg(test)]
mod tests;
