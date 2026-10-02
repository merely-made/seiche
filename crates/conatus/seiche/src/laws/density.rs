// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! Density: a Gastner–Newman density-equalizing flow over the node bodies.
//!
//! Each node's mass is splatted onto a grid, the grid is diffused, and every
//! node advects along `-∇ρ/ρ` of the diffused field: out of crowded regions
//! into empty ones, until the smoothed mass density is even. A heavy node's
//! splat counts for more, so its neighbours end farther away: at rest, the
//! room a node holds follows its mass, the cartogram reading. Read as
//! particles, it is an overdamped isothermal gas (pressure ∝ ρ) smoothed at
//! the diffusion length, which is why that length is set in node spacings.
//!
//! The grid is the CPU tier of the law. [`DensityMedium`] is the seam the
//! GPU tier replaces: the law asks a medium for each node's flow velocity and
//! does the advection itself, so pin handling and bounds stay in one place.
//!
//! The flow writes positions (advection proper, as `CouplingForce`'s
//! FlowAdvect does), so a pinned (kinematic) body is skipped: it stays in the
//! field as mass but is never moved, or its kinematic target would be
//! overwritten.

use std::collections::HashMap;
use std::sync::Mutex;

use rapier2d::prelude::*;

use crate::{Force, ForceContext, NODE_BODY_RADIUS, NodeKey};

use super::node_positions;

/// The square of world space a medium covers this step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DensityDomain {
    /// The lower-left corner.
    pub min: Vector,
    /// The side length, in world units.
    pub side: f32,
}

/// What lies past the domain's edge.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DensityEdge {
    /// A wall: no flux through the edge (a mirror ghost cell).
    Wall,
    /// A sea held at this density (Gastner–Newman's own boundary).
    Sea(f32),
}

/// How the field relates to the nodes from one step to the next.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DensityField {
    /// Splat the nodes afresh every step and smooth them at the diffusion
    /// length: the field is always the current nodes' density.
    Resplat,
    /// Gastner–Newman's own flow: on a fresh step splat the nodes and smooth
    /// them at the diffusion length; on every later step diffuse the field
    /// one implicit time step (`diffusion_length` is then `sqrt(D·dt)`) and
    /// never re-splat. The nodes ride the flow until the field is even.
    Evolve { fresh: bool },
}

/// One step's request to a medium.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DensityStep {
    pub domain: DensityDomain,
    pub edge: DensityEdge,
    /// Diffusion length, in world units: the smoothing scale of the field.
    pub diffusion_length: f32,
    /// Density floor added before dividing, so an empty cell is not a pole.
    pub floor: f32,
    pub field: DensityField,
}

/// The density medium: splat, diffuse, and report `-∇ρ/ρ` at each particle.
/// The CPU tier is [`DensityGrid`]; the GPU tier implements the same seam.
pub trait DensityMedium: Send + std::fmt::Debug {
    /// Rebuild the field from `particles` (position, mass) and write each
    /// particle's flow `-∇ρ/ρ` (per world unit) into `out`, in order.
    fn flow(&mut self, step: DensityStep, particles: &[(Vector, f32)], out: &mut Vec<Vector>);
}

/// The CPU tier: a square grid of cell-centred densities, splatted by
/// cloud-in-cell weights, diffused by Jacobi iterations of one implicit
/// step `(I − α∇²) u = ρ`, warm-started from the previous step's field.
///
/// With `max_diffusion_cells` set, the grid works at the finest power-of-two
/// division of its resolution at which the diffusion length spans at most
/// that many cells: a few Jacobi sweeps then reach the implicit step's
/// solution, where a length of dozens of cells leaves the field lagging the
/// nodes and a fast flow oscillating. `None` (the default) always works at
/// full resolution.
#[derive(Clone, Debug)]
pub struct DensityGrid {
    max_resolution: usize,
    resolution: usize,
    /// Jacobi iterations per step.
    pub iterations: usize,
    pub max_diffusion_cells: Option<f32>,
    source: Vec<f32>,
    field: Vec<f32>,
    scratch: Vec<f32>,
}

impl DensityGrid {
    /// A `resolution`² grid running `iterations` Jacobi sweeps per step.
    pub fn new(resolution: usize, iterations: usize) -> Self {
        let resolution = resolution.max(4);
        let cells = resolution * resolution;
        Self {
            max_resolution: resolution,
            resolution,
            iterations,
            max_diffusion_cells: None,
            source: vec![0.0; cells],
            field: vec![0.0; cells],
            scratch: vec![0.0; cells],
        }
    }

    /// The working resolution (cells per side) of the last step.
    pub fn resolution(&self) -> usize {
        self.resolution
    }

    /// Choose the working resolution for a step; a change resets the field.
    fn work_at(&mut self, step: &DensityStep) {
        let mut resolution = self.max_resolution;
        if let Some(max_cells) = self.max_diffusion_cells {
            while resolution / 2 >= 8
                && step.diffusion_length * resolution as f32 / step.domain.side > max_cells
            {
                resolution /= 2;
            }
        }
        if resolution != self.resolution {
            self.resolution = resolution;
            let cells = resolution * resolution;
            self.source = vec![0.0; cells];
            self.field = vec![0.0; cells];
            self.scratch = vec![0.0; cells];
        }
    }

    /// The splatted density, row-major (`y * n + x`), mass per world area.
    pub fn source(&self) -> &[f32] {
        &self.source
    }

    /// The diffused field, row-major, mass per world area.
    pub fn field(&self) -> &[f32] {
        &self.field
    }

    /// A particle's continuous cell coordinate: cell `i`'s centre is `i`.
    fn cell_coord(&self, domain: DensityDomain, p: Vector) -> (f32, f32) {
        let h = domain.side / self.resolution as f32;
        (
            (p.x - domain.min.x) / h - 0.5,
            (p.y - domain.min.y) / h - 0.5,
        )
    }

    /// The four cloud-in-cell neighbours of a cell coordinate and their
    /// weights, clamped at the edge (a particle within half a cell of the
    /// wall puts all its mass in the edge cell).
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

    /// Splat `particles` onto the source grid as mass per world area.
    pub fn splat(&mut self, domain: DensityDomain, particles: &[(Vector, f32)]) {
        self.source.iter_mut().for_each(|c| *c = 0.0);
        let h = domain.side / self.resolution as f32;
        let per_area = 1.0 / (h * h);
        for &(p, mass) in particles {
            let (gx, gy) = self.cell_coord(domain, p);
            for (cell, w) in self.corners(gx, gy) {
                self.source[cell] += mass * w * per_area;
            }
        }
    }

    /// `iterations` Jacobi sweeps of `(I − α∇²) u = source`, from the current
    /// field. `alpha` is in cells² (the diffusion length over the cell size,
    /// squared).
    pub fn diffuse(&mut self, alpha: f32, edge: DensityEdge, iterations: usize) {
        let n = self.resolution;
        for _ in 0..iterations {
            for y in 0..n {
                for x in 0..n {
                    let mut sum = 0.0;
                    let mut inside = 0.0;
                    for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                        let (nx, ny) = (x as i32 + dx, y as i32 + dy);
                        if nx >= 0 && ny >= 0 && (nx as usize) < n && (ny as usize) < n {
                            sum += self.field[ny as usize * n + nx as usize];
                            inside += 1.0;
                        } else if let DensityEdge::Sea(sea) = edge {
                            sum += sea;
                            inside += 1.0;
                        }
                    }
                    self.scratch[y * n + x] =
                        (self.source[y * n + x] + alpha * sum) / (1.0 + alpha * inside);
                }
            }
            std::mem::swap(&mut self.field, &mut self.scratch);
        }
    }

    /// The field's value at a cell, with the edge's ghost outside.
    fn at(&self, x: i32, y: i32, edge: DensityEdge) -> f32 {
        let n = self.resolution as i32;
        if x >= 0 && y >= 0 && x < n && y < n {
            return self.field[(y * n + x) as usize];
        }
        match edge {
            DensityEdge::Sea(sea) => sea,
            DensityEdge::Wall => {
                let (cx, cy) = (x.clamp(0, n - 1), y.clamp(0, n - 1));
                self.field[(cy * n + cx) as usize]
            },
        }
    }

    /// The field and its gradient (per world unit) at `p`: the central
    /// difference at each cell centre, interpolated bilinearly with the same
    /// weights the splat used, so a lone particle feels no force of its own.
    pub fn sample(&self, domain: DensityDomain, edge: DensityEdge, p: Vector) -> (f32, Vector) {
        let h = domain.side / self.resolution as f32;
        let (gx, gy) = self.cell_coord(domain, p);
        let n = self.resolution;
        let mut value = 0.0;
        let mut grad = Vector::ZERO;
        for (cell, w) in self.corners(gx, gy) {
            let (x, y) = ((cell % n) as i32, (cell / n) as i32);
            value += w * self.field[cell];
            grad += w * Vector::new(
                self.at(x + 1, y, edge) - self.at(x - 1, y, edge),
                self.at(x, y + 1, edge) - self.at(x, y - 1, edge),
            );
        }
        (value, grad / (2.0 * h))
    }
}

impl DensityMedium for DensityGrid {
    fn flow(&mut self, step: DensityStep, particles: &[(Vector, f32)], out: &mut Vec<Vector>) {
        out.clear();
        // An evolving field keeps the level it began at.
        if step.field != (DensityField::Evolve { fresh: false }) {
            self.work_at(&step);
        }
        let h = step.domain.side / self.resolution as f32;
        let alpha = (step.diffusion_length / h).powi(2);
        match step.field {
            DensityField::Resplat => {
                self.splat(step.domain, particles);
                self.diffuse(alpha, step.edge, self.iterations);
            },
            DensityField::Evolve { fresh: true } => {
                // The initial smoothing is solved once, to convergence.
                self.splat(step.domain, particles);
                self.field.copy_from_slice(&self.source);
                let sweeps = ((8.0 * alpha) as usize).clamp(self.iterations, 4_000);
                self.diffuse(alpha, step.edge, sweeps);
            },
            DensityField::Evolve { fresh: false } => {
                self.source.copy_from_slice(&self.field);
                self.diffuse(alpha, step.edge, self.iterations);
            },
        }
        for &(p, _) in particles {
            let (rho, grad) = self.sample(step.domain, step.edge, p);
            out.push(-grad / (rho + step.floor));
        }
    }
}

/// Where the Density law's domain sits and what bounds it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DensityBounds {
    /// Walls around a square of the target area, centred here: the nodes
    /// spread until they fill it evenly.
    Walls { centre: (f32, f32) },
    /// A square around the nodes' own extent, `margin` diffusion lengths past
    /// it, whose rim is a sea at the target density: the graph keeps its own
    /// outline and grows or shrinks toward the target area.
    Sea { margin: f32 },
}

/// Which field the Density law's nodes ride.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DensityFlow {
    /// The current nodes' density every step: a live law of positions.
    Resplat,
    /// Gastner–Newman's flow from the nodes as they were when the law began:
    /// the field diffuses to even over `seconds`, carrying the nodes, then
    /// rests. Walls only (the domain is fixed at the start).
    Gastner { seconds: f32 },
}

#[derive(Debug)]
struct DensityState {
    medium: Box<dyn DensityMedium>,
    /// The Gastner flow's fixed domain, once begun.
    begun: Option<(DensityDomain, DensityEdge, f32)>,
}

/// The Density law: nodes flow along `-∇ρ/ρ` until mass density is even.
#[derive(Debug)]
pub struct Density {
    masses: HashMap<NodeKey, f32>,
    state: Mutex<DensityState>,
    /// Target world area per unit mass: sets the size the layout fills.
    pub area_per_mass: f32,
    /// The field's diffusion length, in mean node spacings at the target
    /// area (`sqrt(area_per_mass · mean mass)`).
    pub diffusion_spacings: f32,
    /// The flow's diffusivity, world units² per second: speed per unit of
    /// relative density gradient.
    pub diffusivity: f32,
    /// The fastest a node is carried, world units per second.
    pub max_speed: f32,
    /// The density floor, as a fraction of the target density.
    pub floor: f32,
    pub bounds: DensityBounds,
    pub flow: DensityFlow,
}

impl Density {
    /// The law over `masses` (node → mass; absent nodes weigh `1.0`) on the
    /// CPU grid at `resolution`².
    pub fn new(masses: impl IntoIterator<Item = (NodeKey, f32)>, resolution: usize) -> Self {
        Self::with_medium(masses, Box::new(DensityGrid::new(resolution, 24)))
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
                begun: None,
            }),
            area_per_mass: 2_500.0,
            diffusion_spacings: 1.0,
            diffusivity: 12_000.0,
            max_speed: 400.0,
            floor: 0.05,
            bounds: DensityBounds::Walls { centre: (0.0, 0.0) },
            flow: DensityFlow::Resplat,
        }
    }

    pub fn mass(&self, key: &NodeKey) -> f32 {
        self.masses.get(key).copied().unwrap_or(1.0).max(0.01)
    }

    /// The step a set of particles asks of the medium: its domain, edge,
    /// diffusion length and floor.
    pub fn step_for(&self, particles: &[(Vector, f32)]) -> Option<DensityStep> {
        if particles.is_empty() {
            return None;
        }
        let total: f32 = particles.iter().map(|(_, m)| m).sum();
        let target_area = total * self.area_per_mass;
        let target = total / target_area;
        let spacing = (self.area_per_mass * total / particles.len() as f32).sqrt();
        let diffusion_length = self.diffusion_spacings * spacing;
        let (domain, edge) = match self.bounds {
            DensityBounds::Walls { centre } => {
                let side = target_area.sqrt();
                (
                    DensityDomain {
                        min: Vector::new(centre.0 - side / 2.0, centre.1 - side / 2.0),
                        side,
                    },
                    DensityEdge::Wall,
                )
            },
            DensityBounds::Sea { margin } => {
                let (mut lo, mut hi) = (particles[0].0, particles[0].0);
                for (p, _) in particles {
                    lo = lo.min(*p);
                    hi = hi.max(*p);
                }
                let centre = (lo + hi) / 2.0;
                let extent = (hi - lo).max_element().max(target_area.sqrt());
                let side = extent + 2.0 * margin * diffusion_length;
                (
                    DensityDomain {
                        min: centre - Vector::splat(side / 2.0),
                        side,
                    },
                    DensityEdge::Sea(target),
                )
            },
        };
        Some(DensityStep {
            domain,
            edge,
            diffusion_length,
            floor: self.floor * target,
            field: DensityField::Resplat,
        })
    }
}

impl Force for Density {
    fn apply(&self, ctx: &mut ForceContext<'_>, dt: f32) {
        let nodes = node_positions(ctx);
        let particles: Vec<(Vector, f32)> =
            nodes.iter().map(|(k, _, p)| (*p, self.mass(k))).collect();
        let Some(mut step) = self.step_for(&particles) else {
            return;
        };
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        // The flow's diffusivity: the configured one, or (Gastner) the one
        // that evens the domain's slowest mode to 1% in `seconds`.
        let mut diffusivity = self.diffusivity;
        if let DensityFlow::Gastner { seconds } = self.flow {
            match state.begun {
                None => {
                    let d = 4.6 * step.domain.side.powi(2)
                        / (std::f32::consts::PI.powi(2) * seconds.max(0.1));
                    state.begun = Some((step.domain, step.edge, d));
                    step.field = DensityField::Evolve { fresh: true };
                    diffusivity = d;
                },
                Some((domain, edge, d)) => {
                    step.domain = domain;
                    step.edge = edge;
                    step.diffusion_length = (d * dt).sqrt();
                    step.field = DensityField::Evolve { fresh: false };
                    diffusivity = d;
                },
            }
        }
        let mut flow = Vec::with_capacity(particles.len());
        state.medium.flow(step, &particles, &mut flow);
        drop(state);
        let walls = matches!(self.bounds, DensityBounds::Walls { .. });
        let lo = step.domain.min + Vector::splat(NODE_BODY_RADIUS);
        let hi = step.domain.min + Vector::splat(step.domain.side - NODE_BODY_RADIUS);
        for ((_, handle, position), velocity) in nodes.iter().zip(flow) {
            // A pinned (kinematic) body belongs to the drag: it weighs in the
            // field but is never moved.
            let Some(body) = ctx.bodies.get_mut(*handle) else {
                continue;
            };
            if !body.is_dynamic() {
                continue;
            }
            let mut v = velocity * diffusivity;
            let speed = v.length();
            if speed > self.max_speed {
                v *= self.max_speed / speed;
            }
            let mut next = *position + v * dt;
            if walls && lo.x < hi.x {
                next = next.clamp(lo, hi);
            }
            body.set_translation(next, true);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Simulation;
    use euclid::default::Point2D;

    /// The exact solution of `(I − α∇²) u = b` with a mirror edge, by the
    /// cosine transform the mirror Laplacian is diagonal in: the analytic
    /// reference the Jacobi sweeps must reach.
    fn spectral_reference(b: &[f32], n: usize, alpha: f64) -> Vec<f64> {
        let basis = |k: usize, i: usize| {
            (std::f64::consts::PI * k as f64 * (i as f64 + 0.5) / n as f64).cos()
        };
        let norm = |k: usize| {
            if k == 0 {
                1.0 / n as f64
            } else {
                2.0 / n as f64
            }
        };
        let eigen = |k: usize| 2.0 - 2.0 * (std::f64::consts::PI * k as f64 / n as f64).cos();
        let mut out = vec![0.0f64; n * n];
        for ky in 0..n {
            for kx in 0..n {
                let mut coeff = 0.0;
                for y in 0..n {
                    for x in 0..n {
                        coeff += b[y * n + x] as f64 * basis(kx, x) * basis(ky, y);
                    }
                }
                coeff *= norm(kx) * norm(ky) / (1.0 + alpha * (eigen(kx) + eigen(ky)));
                for y in 0..n {
                    for x in 0..n {
                        out[y * n + x] += coeff * basis(kx, x) * basis(ky, y);
                    }
                }
            }
        }
        out
    }

    fn unit_domain(n: usize) -> DensityDomain {
        DensityDomain {
            min: Vector::ZERO,
            side: n as f32,
        }
    }

    /// A known splat: one mass at a cell centre lands in that cell alone, one
    /// at a cell corner splits in quarters, and mass is conserved.
    #[test]
    fn the_splat_is_cloud_in_cell_and_conserves_mass() {
        let mut grid = DensityGrid::new(8, 0);
        let domain = unit_domain(8);
        grid.splat(domain, &[(Vector::new(2.5, 3.5), 4.0)]);
        assert_eq!(grid.source()[3 * 8 + 2], 4.0);
        grid.splat(domain, &[(Vector::new(3.0, 3.0), 4.0)]);
        for cell in [2 * 8 + 2, 2 * 8 + 3, 3 * 8 + 2, 3 * 8 + 3] {
            assert!((grid.source()[cell] - 1.0).abs() < 1e-6);
        }
        let total: f32 = grid.source().iter().sum();
        assert!((total - 4.0).abs() < 1e-5);
    }

    /// The diffused field matches the analytic (spectral) solution of the
    /// implicit step on a known splat, to 1e-4 relative, and diffusion
    /// conserves mass under walls.
    #[test]
    fn jacobi_diffusion_matches_the_spectral_reference() {
        let n = 24;
        let alpha = 6.0;
        let mut grid = DensityGrid::new(n, 0);
        let domain = unit_domain(n);
        grid.splat(
            domain,
            &[
                (Vector::new(7.3, 9.6), 3.0),
                (Vector::new(16.0, 15.0), 1.0),
                (Vector::new(0.2, 23.9), 2.0),
            ],
        );
        grid.diffuse(alpha, DensityEdge::Wall, 3_000);
        let reference = spectral_reference(grid.source(), n, alpha as f64);
        let peak = reference.iter().cloned().fold(0.0f64, f64::max);
        let worst = grid
            .field()
            .iter()
            .zip(&reference)
            .map(|(u, r)| (*u as f64 - r).abs())
            .fold(0.0f64, f64::max);
        assert!(
            worst / peak < 1e-4,
            "jacobi vs spectral: {worst:e} of {peak:e}"
        );
        let before: f32 = grid.source().iter().sum();
        let after: f32 = grid.field().iter().sum();
        assert!(
            (before - after).abs() / before < 1e-4,
            "walls conserve mass"
        );
        // And it is a blur, not a copy: the peak spread out.
        let src_peak = grid.source().iter().cloned().fold(0.0, f32::max);
        assert!((peak as f32) < src_peak * 0.5);
    }

    /// A field at the sea's own density is a fixed point of a sea edge.
    #[test]
    fn a_uniform_field_at_sea_density_stays_uniform() {
        let n = 16;
        let mut grid = DensityGrid::new(n, 0);
        grid.source.iter_mut().for_each(|c| *c = 0.25);
        grid.diffuse(9.0, DensityEdge::Sea(0.25), 400);
        assert!(grid.field().iter().all(|u| (u - 0.25).abs() < 1e-5));
    }

    /// A lone particle feels no flow of its own, wherever it sits in a cell.
    #[test]
    fn a_lone_particle_feels_no_self_flow() {
        let n = 32;
        let domain = unit_domain(n);
        for (x, y) in [(16.0, 16.0), (15.8, 16.3), (16.5, 16.5)] {
            let mut grid = DensityGrid::new(n, 400);
            let mut out = Vec::new();
            grid.flow(
                DensityStep {
                    domain,
                    edge: DensityEdge::Wall,
                    diffusion_length: 3.0,
                    floor: 1e-4,
                    field: DensityField::Resplat,
                },
                &[(Vector::new(x, y), 1.0)],
                &mut out,
            );
            assert!(
                out[0].length() < 1e-4,
                "self flow at ({x}, {y}): {:?}",
                out[0]
            );
        }
    }

    fn crowd(n: usize) -> Vec<(NodeKey, Point2D<f32>)> {
        (0..n)
            .map(|i| {
                let a = i as f32 * 2.399_963;
                let r = 6.0 * (i as f32).sqrt();
                (NodeKey::new(i), Point2D::new(r * a.cos(), r * a.sin()))
            })
            .collect()
    }

    /// The law's claim at small scale: a crowd spreads out, the heavy node
    /// ends with the most room (the nearest neighbour farthest away).
    #[test]
    fn a_crowd_spreads_and_the_heavy_node_gets_room() {
        let nodes = crowd(24);
        let heavy = NodeKey::new(0);
        let mut sim = Simulation::new();
        sim.sync_nodes(nodes.clone());
        sim.sync_edges(Vec::<(NodeKey, NodeKey)>::new());
        let masses = nodes
            .iter()
            .map(|(k, _)| (*k, if *k == heavy { 8.0 } else { 1.0 }));
        sim.set_forces(vec![Box::new(Density::new(masses, 64))]);
        for _ in 0..600 {
            sim.tick(1.0 / 60.0);
        }
        let at: Vec<(NodeKey, Point2D<f32>)> = sim.positions().collect();
        let nearest = |k: NodeKey| {
            let p = at.iter().find(|(q, _)| *q == k).unwrap().1;
            at.iter()
                .filter(|(q, _)| *q != k)
                .map(|(_, o)| (*o - p).length())
                .fold(f32::MAX, f32::min)
        };
        let heavy_room = nearest(heavy);
        let mut light: Vec<f32> = at
            .iter()
            .filter(|(k, _)| *k != heavy)
            .map(|(k, _)| nearest(*k))
            .collect();
        light.sort_by(f32::total_cmp);
        let median = light[light.len() / 2];
        assert!(
            median > 36.0,
            "the crowd spread: median nearest {median:.0}"
        );
        eprintln!("heavy room {heavy_room:.1}, light median {median:.1}");
        assert!(
            heavy_room > median * 1.4,
            "heavy {heavy_room:.0} vs median {median:.0}"
        );
    }

    /// A pinned (kinematic) body stays at its target while the rest flow.
    /// Positive control: without the dynamic-body skip this fails, since the
    /// advection writes the pinned body's translation.
    #[test]
    fn a_pinned_body_is_never_moved() {
        let nodes = crowd(16);
        let pinned = NodeKey::new(3);
        let mut sim = Simulation::new();
        sim.sync_nodes(nodes.clone());
        sim.sync_edges(Vec::<(NodeKey, NodeKey)>::new());
        sim.set_forces(vec![Box::new(Density::new(
            nodes.iter().map(|(k, _)| (*k, 1.0)),
            64,
        ))]);
        let target = Point2D::new(30.0, -20.0);
        sim.pin(pinned, target);
        let free_start = sim.position_of(NodeKey::new(9)).unwrap();
        for _ in 0..120 {
            sim.tick(1.0 / 60.0);
            let held = sim.position_of(pinned).unwrap();
            assert!(
                (held - target).length() < 1e-3,
                "the pinned body left its target: {held:?}"
            );
        }
        assert!(
            (sim.position_of(NodeKey::new(9)).unwrap() - free_start).length() > 5.0,
            "the free bodies still flow"
        );
    }
}
