// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! Bounded, view-local flexible paths for the independent P5 link-dynamics probe.
//!
//! This evaluator consumes ordinary endpoint coordinates and declared field
//! inputs. It has no graph, persistence, or host-rendering authority. Output
//! points are temporary geometry keyed by the caller's link id; endpoints and
//! identity are copied through unchanged.

use std::collections::HashSet;

use crate::{FieldDef, FieldId, FieldRegistry, ScalarField, VectorField, eval_vector};

pub const MAX_LINKS: usize = 64;
pub const MAX_FIELDS: usize = 8;
pub const MAX_CONTROL_POINTS: usize = 32;
pub const MAX_ITERATIONS: usize = 48;
pub const MAX_COMPATIBILITY_KEY_BYTES: usize = 128;
pub const MAX_FIELD_AST_NODES: usize = 256;
pub const MAX_FIELD_AST_DEPTH: usize = 16;
const MAX_WORK_UNITS: usize = 2_000_000;
const MAX_INPUT_MAGNITUDE: f32 = 10_000.0;

/// One fixed-endpoint link appearance. `id` must be unique in the input batch.
/// Endpoints retain caller identity. Compatible paths match samples by fixed
/// endpoint geometry, so endpoint labels do not affect the response.
#[derive(Clone, Copy, Debug)]
pub struct ResponsiveLink<'a> {
    pub id: u64,
    pub start_endpoint_id: u64,
    pub end_endpoint_id: u64,
    pub start: (f32, f32),
    pub end: (f32, f32),
    pub compatibility: Option<&'a str>,
}

/// A declared field contribution to the path response.
#[derive(Clone, Copy, Debug)]
pub struct PathField<'a> {
    pub field: &'a VectorField,
    pub strength: f32,
}

/// Independent view-local response controls. Values are validated before work
/// begins; disabling a response skips it without changing the other response.
#[derive(Clone, Copy, Debug)]
pub struct ResponsivePathConfig {
    pub control_points: usize,
    pub iterations: usize,
    pub time: f32,
    pub field_response: bool,
    pub compatible_attraction: bool,
    pub field_gain: f32,
    pub compatibility_gain: f32,
    pub smoothing: f32,
    pub max_step: f32,
}

impl Default for ResponsivePathConfig {
    fn default() -> Self {
        Self {
            control_points: 16,
            iterations: 12,
            time: 0.0,
            field_response: true,
            compatible_attraction: false,
            field_gain: 0.025,
            compatibility_gain: 0.08,
            smoothing: 0.12,
            max_step: 4.0,
        }
    }
}

/// Rebuildable geometry for one link appearance, retaining its caller identity.
#[derive(Clone, Debug, PartialEq)]
pub struct ResponsivePath {
    pub id: u64,
    pub start_endpoint_id: u64,
    pub end_endpoint_id: u64,
    pub points: Vec<(f32, f32)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResponsivePathError {
    TooManyLinks,
    TooManyFields,
    InvalidControlPointCount,
    TooManyIterations,
    WorkLimitExceeded,
    DuplicateLinkId,
    CompatibilityKeyTooLong,
    NonFiniteInput,
    InputOutOfRange,
    FieldExpressionTooLarge,
    FieldSampleCycle,
    NonFiniteFieldOutput,
}

/// Evaluate a deterministic batch of responsive paths under a fixed work cap.
/// Input order is output order and defines tie-free accumulation order.
pub fn evaluate_responsive_paths(
    links: &[ResponsiveLink<'_>],
    fields: &[PathField<'_>],
    registry: &FieldRegistry,
    config: ResponsivePathConfig,
) -> Result<Vec<ResponsivePath>, ResponsivePathError> {
    validate(links, fields, config)?;
    let mut field_cost = 0usize;
    let field_enabled = config.field_response && config.field_gain != 0.0;
    let attraction_enabled = config.compatible_attraction && config.compatibility_gain != 0.0;
    if field_enabled {
        for field in fields {
            let mut nodes = 0;
            let mut active = HashSet::new();
            let cost = validate_vector_field(field.field, registry, 0, &mut nodes, &mut active)?;
            field_cost = field_cost.saturating_add(cost);
        }
    }
    let work = links
        .len()
        .saturating_mul(config.control_points.saturating_sub(2))
        .saturating_mul(config.iterations)
        .saturating_mul(usize::from(attraction_enabled) * links.len() + field_cost + 1);
    if work > MAX_WORK_UNITS {
        return Err(ResponsivePathError::WorkLimitExceeded);
    }

    let mut paths: Vec<ResponsivePath> = links
        .iter()
        .map(|link| ResponsivePath {
            id: link.id,
            start_endpoint_id: link.start_endpoint_id,
            end_endpoint_id: link.end_endpoint_id,
            points: (0..config.control_points)
                .map(|i| {
                    if i == 0 {
                        return link.start;
                    }
                    if i + 1 == config.control_points {
                        return link.end;
                    }
                    let t = i as f32 / (config.control_points - 1) as f32;
                    (
                        link.start.0 + (link.end.0 - link.start.0) * t,
                        link.start.1 + (link.end.1 - link.start.1) * t,
                    )
                })
                .collect(),
        })
        .collect();

    for _ in 0..config.iterations {
        let mut next = paths.clone();
        for (link_index, link) in links.iter().enumerate() {
            for point_index in 1..config.control_points - 1 {
                let point = paths[link_index].points[point_index];
                let before = paths[link_index].points[point_index - 1];
                let after = paths[link_index].points[point_index + 1];
                let tangent = unit((after.0 - before.0, after.1 - before.1));
                let normal = (-tangent.1, tangent.0);
                let mut dx = 0.0;
                let mut dy = 0.0;

                if field_enabled {
                    for field in fields {
                        let vector =
                            eval_vector(field.field, registry, point.0, point.1, config.time);
                        if !vector.0.is_finite() || !vector.1.is_finite() {
                            return Err(ResponsivePathError::NonFiniteFieldOutput);
                        }
                        if vector.0.abs() > MAX_INPUT_MAGNITUDE
                            || vector.1.abs() > MAX_INPUT_MAGNITUDE
                        {
                            return Err(ResponsivePathError::InputOutOfRange);
                        }
                        let normal_strength = vector.0 * normal.0 + vector.1 * normal.1;
                        dx += normal.0 * normal_strength * field.strength * config.field_gain;
                        dy += normal.1 * normal_strength * field.strength * config.field_gain;
                    }
                }

                if attraction_enabled {
                    let mut pull = (0.0, 0.0);
                    let mut count = 0usize;
                    if let Some(key) = link.compatibility {
                        for (other_index, other) in links.iter().enumerate() {
                            if other_index == link_index || other.compatibility != Some(key) {
                                continue;
                            }
                            let reverse_index = correspondence_is_reversed(link, other);
                            let other_point_index = if reverse_index {
                                config.control_points - 1 - point_index
                            } else {
                                point_index
                            };
                            let other_point = paths[other_index].points[other_point_index];
                            pull.0 += other_point.0 - point.0;
                            pull.1 += other_point.1 - point.1;
                            count += 1;
                        }
                    }
                    if count > 0 {
                        let inv_count = 1.0 / count as f32;
                        let normal_pull =
                            (pull.0 * inv_count) * normal.0 + (pull.1 * inv_count) * normal.1;
                        dx += normal.0 * normal_pull * config.compatibility_gain;
                        dy += normal.1 * normal_pull * config.compatibility_gain;
                    }
                }

                let smooth = (
                    (before.0 + after.0) * 0.5 - point.0,
                    (before.1 + after.1) * 0.5 - point.1,
                );
                dx += smooth.0 * config.smoothing;
                dy += smooth.1 * config.smoothing;
                let step = clamp_length((dx, dy), config.max_step);
                next[link_index].points[point_index] = (point.0 + step.0, point.1 + step.1);
            }
        }
        paths = next;
    }

    Ok(paths)
}

fn validate(
    links: &[ResponsiveLink<'_>],
    fields: &[PathField<'_>],
    config: ResponsivePathConfig,
) -> Result<(), ResponsivePathError> {
    if links.len() > MAX_LINKS {
        return Err(ResponsivePathError::TooManyLinks);
    }
    if fields.len() > MAX_FIELDS {
        return Err(ResponsivePathError::TooManyFields);
    }
    if !(3..=MAX_CONTROL_POINTS).contains(&config.control_points) {
        return Err(ResponsivePathError::InvalidControlPointCount);
    }
    if config.iterations > MAX_ITERATIONS {
        return Err(ResponsivePathError::TooManyIterations);
    }
    let work = links
        .len()
        .saturating_mul(config.control_points.saturating_sub(2))
        .saturating_mul(config.iterations)
        .saturating_mul(
            usize::from(config.field_response && config.field_gain != 0.0) * fields.len()
                + usize::from(config.compatible_attraction && config.compatibility_gain != 0.0)
                    * links.len()
                + 1,
        );
    if work > MAX_WORK_UNITS {
        return Err(ResponsivePathError::WorkLimitExceeded);
    }
    let mut ids = HashSet::with_capacity(links.len());
    for link in links {
        if !ids.insert(link.id) {
            return Err(ResponsivePathError::DuplicateLinkId);
        }
        if link
            .compatibility
            .is_some_and(|key| key.len() > MAX_COMPATIBILITY_KEY_BYTES)
        {
            return Err(ResponsivePathError::CompatibilityKeyTooLong);
        }
        if !finite_pair(link.start) || !finite_pair(link.end) {
            return Err(ResponsivePathError::NonFiniteInput);
        }
        if link
            .start
            .0
            .abs()
            .max(link.start.1.abs())
            .max(link.end.0.abs())
            .max(link.end.1.abs())
            > MAX_INPUT_MAGNITUDE
        {
            return Err(ResponsivePathError::InputOutOfRange);
        }
    }
    for field in fields {
        if !field.strength.is_finite() {
            return Err(ResponsivePathError::NonFiniteInput);
        }
        if field.strength.abs() > MAX_INPUT_MAGNITUDE {
            return Err(ResponsivePathError::InputOutOfRange);
        }
    }
    let scalars = [
        config.time,
        config.field_gain,
        config.compatibility_gain,
        config.smoothing,
        config.max_step,
    ];
    if scalars.iter().any(|v| !v.is_finite()) {
        return Err(ResponsivePathError::NonFiniteInput);
    }
    if scalars.iter().any(|v| v.abs() > MAX_INPUT_MAGNITUDE) || config.max_step < 0.0 {
        return Err(ResponsivePathError::InputOutOfRange);
    }
    Ok(())
}

fn finite_pair(pair: (f32, f32)) -> bool {
    pair.0.is_finite() && pair.1.is_finite()
}

/// Choose the endpoint pairing with the lower total squared distance. Equal
/// pairings use endpoint-coordinate order for a stable direction tie-break.
/// This geometry rule is only a fixture correspondence; it declares no
/// semantic relation between the links.
fn correspondence_is_reversed(a: &ResponsiveLink<'_>, b: &ResponsiveLink<'_>) -> bool {
    let direct = distance_squared(a.start, b.start) + distance_squared(a.end, b.end);
    let crossed = distance_squared(a.start, b.end) + distance_squared(a.end, b.start);
    if crossed < direct {
        true
    } else if direct < crossed {
        false
    } else {
        geometric_reverse(a) != geometric_reverse(b)
    }
}

fn distance_squared(a: (f32, f32), b: (f32, f32)) -> f64 {
    let dx = f64::from(a.0) - f64::from(b.0);
    let dy = f64::from(a.1) - f64::from(b.1);
    dx * dx + dy * dy
}

fn geometric_reverse(link: &ResponsiveLink<'_>) -> bool {
    (link.start.0, link.start.1) > (link.end.0, link.end.1)
}

fn count_ast_node(nodes: &mut usize, depth: usize) -> Result<(), ResponsivePathError> {
    *nodes += 1;
    if *nodes > MAX_FIELD_AST_NODES || depth > MAX_FIELD_AST_DEPTH {
        return Err(ResponsivePathError::FieldExpressionTooLarge);
    }
    Ok(())
}

/// Return a conservative number of primitive field operations for one vector
/// evaluation, expanding registry samples while detecting cycles.
fn validate_vector_field(
    field: &VectorField,
    registry: &FieldRegistry,
    depth: usize,
    nodes: &mut usize,
    active: &mut HashSet<FieldId>,
) -> Result<usize, ResponsivePathError> {
    count_ast_node(nodes, depth)?;
    let cost = match field {
        VectorField::ConstVec { x, y } => {
            if !x.is_finite() || !y.is_finite() {
                return Err(ResponsivePathError::NonFiniteInput);
            }
            if x.abs() > MAX_INPUT_MAGNITUDE || y.abs() > MAX_INPUT_MAGNITUDE {
                return Err(ResponsivePathError::InputOutOfRange);
            }
            1
        },
        VectorField::Coord => 1,
        VectorField::Gradient(scalar) => {
            validate_scalar_field(scalar, registry, depth + 1, nodes, active)?.1
        },
        VectorField::Perp(inner) | VectorField::ScaleConst(inner, _) => {
            if let VectorField::ScaleConst(_, scale) = field {
                validate_scalar(*scale)?;
            }
            sat_add(
                1,
                validate_vector_field(inner, registry, depth + 1, nodes, active)?,
            )
        },
        VectorField::Add(a, b) => {
            let a = validate_vector_field(a, registry, depth + 1, nodes, active)?;
            let b = validate_vector_field(b, registry, depth + 1, nodes, active)?;
            sat_add(1, sat_add(a, b))
        },
        VectorField::Scale(vector, scalar) => {
            let vector = validate_vector_field(vector, registry, depth + 1, nodes, active)?;
            let scalar = validate_scalar_field(scalar, registry, depth + 1, nodes, active)?.0;
            sat_add(1, sat_add(vector, scalar))
        },
        VectorField::Sample(id) => {
            if let Some(FieldDef::Vector(sample)) = registry.get(*id) {
                enter_sample(*id, active)?;
                let result = validate_vector_field(sample, registry, depth + 1, nodes, active);
                active.remove(id);
                sat_add(1, result?)
            } else {
                1
            }
        },
    };
    Ok(cost)
}

/// Return `(eval_scalar_cost, grad_scalar_cost)` while checking every stored
/// child and sampled registry definition. A finite-difference gradient costs
/// four complete scalar evaluations; analytic cases use their actual child
/// expansion, including Gaussian center sampling.
fn validate_scalar_field(
    field: &ScalarField,
    registry: &FieldRegistry,
    depth: usize,
    nodes: &mut usize,
    active: &mut HashSet<FieldId>,
) -> Result<(usize, usize), ResponsivePathError> {
    count_ast_node(nodes, depth)?;
    let costs = match field {
        ScalarField::Const(value) => {
            validate_scalar(*value)?;
            (1, 1)
        },
        ScalarField::CoordX | ScalarField::CoordY | ScalarField::Time => (1, 1),
        ScalarField::Gaussian { center, sigma } => {
            validate_scalar(*sigma)?;
            if *sigma < 0.0 {
                return Err(ResponsivePathError::InputOutOfRange);
            }
            let center = validate_vector_field(center, registry, depth + 1, nodes, active)?;
            // `eval_scalar(Gaussian)` samples its center once. The analytic
            // gradient samples it once directly and once through eval_scalar.
            let evaluate = sat_add(1, center);
            (evaluate, sat_add(1, sat_add(center, evaluate)))
        },
        ScalarField::Disk { center, radius, .. } => {
            validate_scalar(*radius)?;
            if *radius < 0.0 {
                return Err(ResponsivePathError::InputOutOfRange);
            }
            let center = validate_vector_field(center, registry, depth + 1, nodes, active)?;
            let evaluate = sat_add(1, center);
            (evaluate, sat_mul(4, evaluate))
        },
        ScalarField::Linear { normal, offset } => {
            validate_scalar(*offset)?;
            let normal = validate_vector_field(normal, registry, depth + 1, nodes, active)?;
            (sat_add(1, normal), normal)
        },
        ScalarField::Add(a, b) | ScalarField::Mul(a, b) => {
            let a = validate_scalar_field(a, registry, depth + 1, nodes, active)?;
            let b = validate_scalar_field(b, registry, depth + 1, nodes, active)?;
            let evaluate = sat_add(1, sat_add(a.0, b.0));
            let gradient = if matches!(field, ScalarField::Mul(_, _)) {
                sat_mul(4, evaluate)
            } else {
                sat_add(a.1, b.1)
            };
            (evaluate, gradient)
        },
        ScalarField::Scale(inner, coefficient) => {
            validate_scalar(*coefficient)?;
            let inner = validate_scalar_field(inner, registry, depth + 1, nodes, active)?;
            (sat_add(1, inner.0), inner.1)
        },
        ScalarField::Dot(a, b) => {
            let a = validate_vector_field(a, registry, depth + 1, nodes, active)?;
            let b = validate_vector_field(b, registry, depth + 1, nodes, active)?;
            let evaluate = sat_add(1, sat_add(a, b));
            (evaluate, sat_mul(4, evaluate))
        },
        ScalarField::Negate(inner) => {
            let inner = validate_scalar_field(inner, registry, depth + 1, nodes, active)?;
            (sat_add(1, inner.0), inner.1)
        },
        ScalarField::Sample(id) => {
            if let Some(FieldDef::Scalar(sample)) = registry.get(*id) {
                enter_sample(*id, active)?;
                let result = validate_scalar_field(sample, registry, depth + 1, nodes, active);
                active.remove(id);
                let sampled = result?;
                let evaluate = sat_add(1, sampled.0);
                (evaluate, sat_mul(4, evaluate))
            } else {
                (1, 4)
            }
        },
    };
    Ok(costs)
}

fn enter_sample(id: FieldId, active: &mut HashSet<FieldId>) -> Result<(), ResponsivePathError> {
    if active.insert(id) {
        Ok(())
    } else {
        Err(ResponsivePathError::FieldSampleCycle)
    }
}

fn validate_scalar(value: f32) -> Result<(), ResponsivePathError> {
    if !value.is_finite() {
        return Err(ResponsivePathError::NonFiniteInput);
    }
    if value.abs() > MAX_INPUT_MAGNITUDE {
        return Err(ResponsivePathError::InputOutOfRange);
    }
    Ok(())
}

fn sat_add(a: usize, b: usize) -> usize {
    a.saturating_add(b)
}

fn sat_mul(a: usize, b: usize) -> usize {
    a.saturating_mul(b)
}

fn unit(vector: (f32, f32)) -> (f32, f32) {
    let length = vector.0.hypot(vector.1);
    if length > f32::EPSILON {
        (vector.0 / length, vector.1 / length)
    } else {
        (1.0, 0.0)
    }
}

fn clamp_length(vector: (f32, f32), max: f32) -> (f32, f32) {
    let length = vector.0.hypot(vector.1);
    if length > max && length > f32::EPSILON {
        (vector.0 * max / length, vector.1 * max / length)
    } else {
        vector
    }
}

#[cfg(test)]
#[path = "responsive_path/tests.rs"]
mod tests;
