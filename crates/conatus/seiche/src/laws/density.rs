// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! Density: Gastner–Newman's density-equalizing flow over the node bodies.
//!
//! When the flow begins, each node's mass is splatted onto a grid and
//! smoothed a little. From then on the grid is never re-splatted: the field
//! diffuses, `∂ρ/∂t = D∇²ρ`, and every node rides its velocity
//! `v = −D∇ρ/ρ`. That velocity carries density exactly as diffusion spreads
//! it, so the mass a node started amid travels with it, and when the field
//! is even each node's room is its share of the mass: the cartogram. Then
//! the flow is spent and the nodes rest.
//!
//! Point masses make the early field steep, so each physics tick is cut into
//! substeps no longer than it takes the fastest node to cross a fraction of
//! a cell (a CFL bound), the field diffusing by the same substep. Diffusion is
//! implicit, so the substep is for the nodes' sake, not the field's.
//!
//! The grid is the CPU tier. [`DensityMedium`] is the seam the GPU tier
//! replaces; the law keeps the substep loop, the walls and the pins.
//!
//! The flow writes positions (advection proper, as `CouplingForce`'s
//! FlowAdvect does), so a pinned (kinematic) body is skipped: its mass is
//! in the field but it is never moved, or its kinematic target would be
//! overwritten.

use std::collections::HashMap;
use std::sync::Mutex;

use rapier2d::prelude::*;

use crate::{Force, ForceContext, NODE_BODY_RADIUS, NodeKey};

use super::node_positions;

/// The square of world space the field covers, walled on every side.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DensityDomain {
    /// The lower-left corner.
    pub min: Vector,
    /// The side length, in world units.
    pub side: f32,
}

/// The density medium: the field the nodes ride. The CPU tier is
/// [`DensityGrid`]; the GPU tier implements the same seam.
pub trait DensityMedium: Send + std::fmt::Debug {
    /// Start a flow: splat `particles` (position, mass) over `domain` and
    /// smooth them to the implicit step of `blur` (world units), solved to
    /// convergence. `background` is added everywhere, mass per area.
    fn begin(
        &mut self,
        domain: DensityDomain,
        particles: &[(Vector, f32)],
        blur: f32,
        background: f32,
    );
    /// Diffuse the field one implicit step of diffusion length `length`
    /// (`sqrt(D·τ)` for a substep `τ`).
    fn advance(&mut self, length: f32);
    /// The flow `−∇ρ/ρ` (per world unit) at each position, in order.
    fn flow(&self, positions: &[Vector], out: &mut Vec<Vector>);
    /// The cell size, in world units: the CFL bound's yardstick.
    fn cell(&self) -> f32;
    /// The coefficient of variation of the field over its cells.
    fn field_cv(&self) -> f32;
}

/// The CPU tier: a square grid of cell-centred densities with walls, splatted
/// by cloud-in-cell weights and diffused by Jacobi sweeps of the implicit
/// step `(I − α∇²) u' = u`.
#[derive(Clone, Debug)]
pub struct DensityGrid {
    resolution: usize,
    /// Jacobi sweeps per unit of `α` (cells²) in an advance, at least four.
    pub sweeps_per_alpha: f32,
    domain: DensityDomain,
    /// The field's total at the flow's start: the walls conserve it, and a
    /// truncated Jacobi solve that drifts from it is rescaled back.
    total: f32,
    source: Vec<f32>,
    field: Vec<f32>,
    scratch: Vec<f32>,
}

impl DensityGrid {
    /// A `resolution`² grid.
    pub fn new(resolution: usize) -> Self {
        let resolution = resolution.max(4);
        let cells = resolution * resolution;
        Self {
            resolution,
            sweeps_per_alpha: 6.0,
            domain: DensityDomain {
                min: Vector::ZERO,
                side: 1.0,
            },
            total: 0.0,
            source: vec![0.0; cells],
            field: vec![0.0; cells],
            scratch: vec![0.0; cells],
        }
    }

    pub fn resolution(&self) -> usize {
        self.resolution
    }

    /// The splatted density, row-major (`y * n + x`), mass per world area.
    pub fn source(&self) -> &[f32] {
        &self.source
    }

    /// The field, row-major, mass per world area.
    pub fn field(&self) -> &[f32] {
        &self.field
    }

    /// A position's continuous cell coordinate: cell `i`'s centre is `i`.
    fn cell_coord(&self, p: Vector) -> (f32, f32) {
        let h = self.cell();
        (
            (p.x - self.domain.min.x) / h - 0.5,
            (p.y - self.domain.min.y) / h - 0.5,
        )
    }

    /// The four cloud-in-cell neighbours of a cell coordinate and their
    /// weights, clamped at the walls.
    fn corners(&self, gx: f32, gy: f32) -> [(usize, f32); 4] {
        let last = (self.resolution - 1) as f32;
        let gx = gx.clamp(0.0, last);
        let gy = gy.clamp(0.0, last);
        let (x0, y0) = (gx.floor() as usize, gy.floor() as usize);
        let (fx, fy) = (gx - x0 as f32, gy - y0 as f32);
        let x1 = (x0 + 1).min(self.resolution - 1);
        let y1 = (y0 + 1).min(self.resolution - 1);
        let n = self.resolution;
        [
            (y0 * n + x0, (1.0 - fx) * (1.0 - fy)),
            (y0 * n + x1, fx * (1.0 - fy)),
            (y1 * n + x0, (1.0 - fx) * fy),
            (y1 * n + x1, fx * fy),
        ]
    }

    /// Splat `particles` over `domain` onto the source grid, mass per area.
    pub fn splat(&mut self, domain: DensityDomain, particles: &[(Vector, f32)]) {
        self.domain = domain;
        self.source.iter_mut().for_each(|c| *c = 0.0);
        let per_area = 1.0 / (self.cell() * self.cell());
        for &(p, mass) in particles {
            let (gx, gy) = self.cell_coord(p);
            for (cell, w) in self.corners(gx, gy) {
                self.source[cell] += mass * w * per_area;
            }
        }
    }

    /// `iterations` Jacobi sweeps of `(I − α∇²) u = source` with mirror
    /// (wall) edges, from the current field. `alpha` is in cells².
    pub fn diffuse(&mut self, alpha: f32, iterations: usize) {
        let n = self.resolution;
        for _ in 0..iterations {
            for y in 0..n {
                for x in 0..n {
                    let mut sum = 0.0;
                    let mut inside = 0.0;
                    if x > 0 {
                        sum += self.field[y * n + x - 1];
                        inside += 1.0;
                    }
                    if x + 1 < n {
                        sum += self.field[y * n + x + 1];
                        inside += 1.0;
                    }
                    if y > 0 {
                        sum += self.field[(y - 1) * n + x];
                        inside += 1.0;
                    }
                    if y + 1 < n {
                        sum += self.field[(y + 1) * n + x];
                        inside += 1.0;
                    }
                    self.scratch[y * n + x] =
                        (self.source[y * n + x] + alpha * sum) / (1.0 + alpha * inside);
                }
            }
            std::mem::swap(&mut self.field, &mut self.scratch);
        }
    }

    /// The field at a cell, the wall's mirror outside.
    fn at(&self, x: i32, y: i32) -> f32 {
        let n = self.resolution as i32;
        let (cx, cy) = (x.clamp(0, n - 1), y.clamp(0, n - 1));
        self.field[(cy * n + cx) as usize]
    }

    /// The field and its gradient (per world unit) at `p`: the central
    /// difference at each cell centre, interpolated bilinearly with the same
    /// weights the splat used, so a lone particle feels no flow of its own.
    pub fn sample(&self, p: Vector) -> (f32, Vector) {
        let (gx, gy) = self.cell_coord(p);
        let n = self.resolution;
        let mut value = 0.0;
        let mut grad = Vector::ZERO;
        for (cell, w) in self.corners(gx, gy) {
            let (x, y) = ((cell % n) as i32, (cell / n) as i32);
            value += w * self.field[cell];
            grad += w * Vector::new(
                self.at(x + 1, y) - self.at(x - 1, y),
                self.at(x, y + 1) - self.at(x, y - 1),
            );
        }
        (value, grad / (2.0 * self.cell()))
    }

    fn sweeps(&self, alpha: f32) -> usize {
        ((alpha * self.sweeps_per_alpha).ceil() as usize).clamp(4, 4_000)
    }
}

impl DensityMedium for DensityGrid {
    fn begin(
        &mut self,
        domain: DensityDomain,
        particles: &[(Vector, f32)],
        blur: f32,
        background: f32,
    ) {
        self.splat(domain, particles);
        self.source.iter_mut().for_each(|c| *c += background);
        self.total = self.source.iter().sum();
        self.field.copy_from_slice(&self.source);
        let alpha = (blur / self.cell()).powi(2);
        // Converged once: the sweeps a fresh (not warm) solve needs.
        self.diffuse(alpha, self.sweeps(alpha).max((8.0 * alpha) as usize));
    }

    fn advance(&mut self, length: f32) {
        let alpha = (length / self.cell()).powi(2);
        self.source.copy_from_slice(&self.field);
        self.diffuse(alpha, self.sweeps(alpha));
        let now: f32 = self.field.iter().sum();
        if now > 0.0 {
            let scale = self.total / now;
            self.field.iter_mut().for_each(|u| *u *= scale);
        }
    }

    fn flow(&self, positions: &[Vector], out: &mut Vec<Vector>) {
        out.clear();
        for &p in positions {
            let (rho, grad) = self.sample(p);
            out.push(-grad / rho.max(f32::MIN_POSITIVE));
        }
    }

    fn cell(&self) -> f32 {
        self.domain.side / self.resolution as f32
    }

    fn field_cv(&self) -> f32 {
        let n = self.field.len() as f32;
        let mean = self.field.iter().sum::<f32>() / n;
        let var = self.field.iter().map(|u| (u - mean).powi(2)).sum::<f32>() / n;
        if mean > 0.0 { var.sqrt() / mean } else { 0.0 }
    }
}

/// A flow under way: its walls, diffusivity, pass, and the diagnostics.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DensityFlowState {
    pub domain: DensityDomain,
    /// Diffusivity `D`, world units² per second.
    pub diffusivity: f32,
    /// The mean node spacing at the target area: the yardstick of a shift.
    pub spacing: f32,
    /// Seconds of flow so far, in this pass.
    pub elapsed: f32,
    /// Which pass this is since the flow (re)armed, from one.
    pub pass: u32,
    /// Whether the passes have stopped (the test, or the cap).
    pub converged: bool,
    /// The substeps the last tick took, and the most any tick has taken.
    pub last_substeps: u32,
    pub max_substeps_seen: u32,
    /// Ticks that hit the substep cap (the CFL bound then not held).
    pub capped_ticks: u32,
    /// The fastest node speed on the last tick, world units per second.
    pub last_speed: f32,
}

/// One finished pass, for the convergence test and its receipts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DensityPass {
    pub pass: u32,
    /// Mean distance the moving nodes travelled over the pass, in spacings.
    pub shift: f32,
    /// The coefficient of variation of the nodes' field splatted afresh
    /// where the pass left them: how uneven the layout still is.
    pub field_cv: f32,
}

/// When repeated passes stop.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DensityStop {
    /// After `patience` passes in a row whose mean shift is under this many
    /// node spacings.
    Shift(f32),
    /// After `patience` passes in a row whose field CV changed by less than
    /// this fraction from the pass before.
    FieldCv(f32),
    /// Never by test: the pass cap alone.
    Cap,
}

#[derive(Debug)]
struct DensityState {
    medium: Box<dyn DensityMedium>,
    flow: Option<DensityFlowState>,
    /// Where the nodes stood when this pass began.
    pass_start: Vec<Vector>,
    history: Vec<DensityPass>,
    /// Passes in a row that met the stop test.
    calm: u32,
}

/// The Density law: nodes ride Gastner–Newman's diffusing field until mass
/// density is even.
#[derive(Debug)]
pub struct Density {
    masses: HashMap<NodeKey, f32>,
    state: Mutex<DensityState>,
    /// Target world area per unit mass: sets the walls' size.
    pub area_per_mass: f32,
    /// The walls' centre.
    pub centre: (f32, f32),
    /// The initial smoothing of each pass, in mean node spacings.
    pub initial_blur: f32,
    /// Seconds of one pass: the field's slowest mode evens to 1% in it.
    pub seconds: f32,
    /// The farthest a node moves in one substep, in cells.
    pub cfl: f32,
    /// The most substeps in one tick.
    pub max_substeps: u32,
    /// A uniform background density, as a fraction of the target density,
    /// so empty cells are not poles. It rides the flow like node mass.
    pub background: f32,
    /// The stop test for repeated passes, and how many passes in a row must
    /// meet it.
    pub stop: DensityStop,
    pub patience: u32,
    /// The pass cap: the fallback that ends the passes if the test never
    /// does. One is Gastner and Newman's single flow.
    pub max_passes: u32,
}

impl Density {
    /// The law over `masses` (node → mass; absent nodes weigh `1.0`) on the
    /// CPU grid at `resolution`².
    pub fn new(masses: impl IntoIterator<Item = (NodeKey, f32)>, resolution: usize) -> Self {
        Self::with_medium(masses, Box::new(DensityGrid::new(resolution)))
    }

    /// The law over any medium (the GPU tier's entry).
    pub fn with_medium(
        masses: impl IntoIterator<Item = (NodeKey, f32)>,
        medium: Box<dyn DensityMedium>,
    ) -> Self {
        Self {
            masses: masses.into_iter().collect(),
            state: Mutex::new(DensityState {
                medium,
                flow: None,
                pass_start: Vec::new(),
                history: Vec::new(),
                calm: 0,
            }),
            area_per_mass: 2_500.0,
            centre: (0.0, 0.0),
            initial_blur: 0.25,
            seconds: 1.0,
            cfl: 0.5,
            max_substeps: 64,
            background: 0.05,
            stop: DensityStop::Cap,
            patience: 1,
            max_passes: 1,
        }
    }

    pub fn mass(&self, key: &NodeKey) -> f32 {
        self.masses.get(key).copied().unwrap_or(1.0).max(0.01)
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, DensityState> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// The flow under way, if it has begun.
    pub fn flow_state(&self) -> Option<DensityFlowState> {
        self.lock().flow
    }

    /// Every finished pass since the flow first began, in order.
    pub fn pass_history(&self) -> Vec<DensityPass> {
        self.lock().history.clone()
    }

    /// Begin pass `pass` over the nodes where they stand: the walls around
    /// the target area, `D` from `seconds`, the splat smoothed.
    fn begin(&self, state: &mut DensityState, particles: &[(Vector, f32)], pass: u32) {
        let total: f32 = particles.iter().map(|(_, m)| m).sum();
        let side = (total * self.area_per_mass).sqrt();
        let spacing = (self.area_per_mass * total / particles.len() as f32).sqrt();
        let domain = DensityDomain {
            min: Vector::new(self.centre.0 - side / 2.0, self.centre.1 - side / 2.0),
            side,
        };
        let target = 1.0 / self.area_per_mass;
        state.medium.begin(
            domain,
            particles,
            self.initial_blur * spacing,
            self.background * target,
        );
        let carried = state.flow;
        state.flow = Some(DensityFlowState {
            domain,
            diffusivity: 4.6 * side * side / (std::f32::consts::PI.powi(2) * self.seconds.max(0.1)),
            spacing,
            elapsed: 0.0,
            pass,
            converged: false,
            last_substeps: 0,
            max_substeps_seen: carried.map_or(0, |f| f.max_substeps_seen),
            capped_ticks: carried.map_or(0, |f| f.capped_ticks),
            last_speed: 0.0,
        });
        state.pass_start = particles.iter().map(|(p, _)| *p).collect();
    }

    /// Close pass `pass`, after the next one has splatted the nodes where
    /// it left them: record it, apply the stop test, and say whether the
    /// passes go on.
    fn close_pass(&self, state: &mut DensityState, pass: u32, shift: f32) -> bool {
        let field_cv = state.medium.field_cv();
        let before = state.history.last().copied();
        state.history.push(DensityPass {
            pass,
            shift,
            field_cv,
        });
        let calm = match self.stop {
            DensityStop::Shift(limit) => shift < limit,
            DensityStop::FieldCv(limit) => before.is_some_and(|b| {
                b.pass + 1 == pass && (field_cv - b.field_cv).abs() < limit * b.field_cv
            }),
            DensityStop::Cap => false,
        };
        state.calm = if calm { state.calm + 1 } else { 0 };
        state.calm < self.patience.max(1) && pass < self.max_passes.max(1)
    }
}

impl Force for Density {
    fn apply(&self, ctx: &mut ForceContext<'_>, dt: f32) {
        let nodes = node_positions(ctx);
        if nodes.is_empty() {
            return;
        }
        // A pinned (kinematic) body belongs to the drag: its mass is in the
        // field but it is never moved.
        let dynamic: Vec<bool> = nodes
            .iter()
            .map(|(_, h, _)| ctx.bodies.get(*h).is_some_and(|b| b.is_dynamic()))
            .collect();
        let dragging = dynamic.iter().any(|moves| !moves);
        let mut positions: Vec<Vector> = nodes.iter().map(|(_, _, p)| *p).collect();
        let particles =
            || -> Vec<(Vector, f32)> { nodes.iter().map(|(k, _, p)| (*p, self.mass(k))).collect() };
        let mut state = self.lock();
        match state.flow {
            None => self.begin(&mut state, &particles(), 1),
            Some(flow) if flow.converged => {
                // A drag re-arms the passes; otherwise the flow is spent.
                if !dragging {
                    return;
                }
                state.calm = 0;
                self.begin(&mut state, &particles(), 1);
            },
            Some(flow) if flow.elapsed >= self.seconds - 1e-4 => {
                let moved: Vec<f32> = positions
                    .iter()
                    .zip(&state.pass_start)
                    .zip(&dynamic)
                    .filter(|(_, moves)| **moves)
                    .map(|((p, q), _)| (*p - *q).length())
                    .collect();
                let shift =
                    moved.iter().sum::<f32>() / moved.len().max(1) as f32 / flow.spacing.max(1e-3);
                self.begin(&mut state, &particles(), flow.pass + 1);
                let more = self.close_pass(&mut state, flow.pass, shift);
                if dragging {
                    // A held node keeps the passes going: it is still moving.
                    state.calm = 0;
                    if let Some(next) = state.flow.as_mut() {
                        next.pass = 1;
                    }
                } else if !more {
                    if let Some(spent) = state.flow.as_mut() {
                        spent.converged = true;
                    }
                    return;
                }
            },
            Some(_) => {},
        }
        let mut flow = state.flow.expect("a flow under way");
        let lo = flow.domain.min + Vector::splat(NODE_BODY_RADIUS);
        let hi = flow.domain.min + Vector::splat(flow.domain.side - NODE_BODY_RADIUS);
        let reach = self.cfl * state.medium.cell();
        let d = flow.diffusivity;
        let mut velocity = Vec::with_capacity(positions.len());
        let (mut remaining, mut substeps, mut fastest) = (dt, 0u32, 0.0f32);
        while remaining > 1e-7 {
            state.medium.flow(&positions, &mut velocity);
            let speed = velocity
                .iter()
                .zip(&dynamic)
                .filter(|(_, moves)| **moves)
                .map(|(v, _)| v.length() * d)
                .fold(0.0, f32::max);
            fastest = fastest.max(speed);
            substeps += 1;
            let last = substeps >= self.max_substeps.max(1);
            let tau = if last || speed * remaining <= reach {
                remaining
            } else {
                reach / speed
            };
            if last && speed * tau > reach {
                flow.capped_ticks += 1;
            }
            state.medium.advance((d * tau).sqrt());
            for ((p, v), moves) in positions.iter_mut().zip(&velocity).zip(&dynamic) {
                if *moves {
                    *p = (*p + *v * (d * tau)).clamp(lo, hi.max(lo));
                }
            }
            remaining -= tau;
        }
        flow.elapsed += dt;
        flow.last_substeps = substeps;
        flow.max_substeps_seen = flow.max_substeps_seen.max(substeps);
        flow.last_speed = fastest;
        state.flow = Some(flow);
        drop(state);
        for (((_, handle, _), p), moves) in nodes.iter().zip(positions).zip(dynamic) {
            if moves && let Some(body) = ctx.bodies.get_mut(*handle) {
                body.set_translation(p, true);
            }
        }
    }

    /// Until the passes stop, the law keeps the host ticking past its settle
    /// budget; a spent flow lets it rest.
    fn wants_tick(&self) -> bool {
        self.lock().flow.is_none_or(|flow| !flow.converged)
    }
}

#[cfg(test)]
mod tests;
