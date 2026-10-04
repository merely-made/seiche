// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! Density's CPU tier: the walled grid the nodes' field lives on.

use rapier2d::prelude::*;

use super::{DensityDomain, DensityMedium};

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

    fn renew(&mut self, particles: &[(Vector, f32)], weight: f32, background: f32) {
        let domain = self.domain;
        self.splat(domain, particles);
        let w = weight.clamp(0.0, 1.0);
        for (u, s) in self.field.iter_mut().zip(&self.source) {
            *u = (1.0 - w) * *u + w * (*s + background);
        }
        self.total = self.field.iter().sum();
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
