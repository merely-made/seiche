// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! `CouplingForce` — the aether→seiche seam (field-system Phase 3b).
//!
//! A [`numen::Coupling`] is field-algebra truth: a node selector of
//! targets, a field id, a [`CouplingResponse`](numen::CouplingResponse), and a
//! strength. This adapter compiles one into a [`Force`] so the same rapier tick
//! that runs the built-in [`NodeExclusion`](crate::NodeExclusion) /
//! [`EdgeSpring`](crate::EdgeSpring) / [`Boundary`](crate::Boundary) forces also
//! runs scriptable couplings. numen evaluates the field (closed forms +
//! finite-difference gradients); seiche supplies bodies and integration.
//!
//! Build it from an already-resolved field + target set with
//! [`CouplingForce::new`] (the host resolves a coupling's field and selector
//! against its own graph — see mere-canvas's `build::coupling_force_from_graph`).
//! The captured target set is a snapshot: rebuild the force when the graph's
//! nodes or the selector's matches change.
//!
//! Equivalence: `Boundary` is exactly `AttractToMin` on the radial paraboloid
//! `½(x² + y²)` (its gradient is `(x, y)`, so `-grad · strength = -pos · strength`).
//! The built-ins stay as the fast native path; couplings are the general one.

use crate::NodeKey;
use numen::{
    CouplingResponse, FieldDefinition, FieldRegistry, ScalarField, VectorField, eval_scalar,
    eval_vector, grad_scalar,
};
use rapier2d::prelude::*;

use crate::{
    Class, Currency, Declared, Force, ForceContext, Kernel, Layout, Observable, State, Term,
    Topology,
};

/// A [`numen::Coupling`] compiled to a [`Force`]: evaluate the field at each
/// target node's position and apply the response.
pub struct CouplingForce {
    pub(crate) binding: std::sync::Arc<crate::CouplingBinding>,
}

/// One response's proposed write. This is not an aggregate force or proof that
/// a controller applied it. Execution consumes the same value after resolving
/// the body; a read-only caller may evaluate its own supplied point instead.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CouplingEffect {
    AddForce([f32; 2]),
    ScaleVelocity(f32),
    SetVelocity([f32; 2]),
    Displace([f32; 2]),
    Inert(CouplingInertReason),
}

/// Why this response proposes no physical write under the current executor.
/// These are observations of existing behavior, not new admission policies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CouplingInertReason {
    NotInResolvedTargets,
    FieldTypeMismatch,
    OutsidePositiveRegion,
    NonDynamicBody,
    UnsupportedResponse,
    OutsideExtent,
    UnsupportedExtent,
    InvalidReferencePoint,
    InvalidSampleTime,
}

/// A report-only point evaluation. `sample_time` is the executor's configured
/// field sampling time, including when an inert result did not sample a field.
/// It is not wall time, a simulation tick or an event timestamp.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CouplingEvaluation {
    pub sample_time: f32,
    pub effect: CouplingEffect,
}

// Preserve the current field execution contract. A real clock remains an
// owner-policy checkpoint; exposing this value must not silently advance it.

impl CouplingForce {
    /// From an already-resolved field definition and target set. Needs no
    /// a graph; the field must be self-contained or carry its registry via
    /// [`Self::with_registry`].
    pub fn new(
        response: CouplingResponse,
        strength: f32,
        targets: impl IntoIterator<Item = NodeKey>,
        field: FieldDefinition,
    ) -> Self {
        let targets: Vec<_> = targets.into_iter().collect();
        let target_identities = vec![None; targets.len()];
        Self {
            binding: std::sync::Arc::new(crate::CouplingBinding {
                response,
                strength,
                targets,
                selector: None,
                animated: crate::field_clock::animated(&field, &FieldRegistry::new()),
                field,
                extent: numen::FieldExtent::Global,
                registry: FieldRegistry::new(),
                source: None,
                target_identities,
            }),
        }
    }

    /// Supply a registry so the field's `Sample` references resolve.
    pub fn with_registry(mut self, registry: FieldRegistry) -> Self {
        let binding = std::sync::Arc::make_mut(&mut self.binding);
        binding.animated = crate::field_clock::animated(&binding.field, &registry);
        binding.registry = registry;
        self
    }

    /// Capture the owner's Numen selector when available. Candidate snapshots
    /// still drive execution; this distinguishes selector edits with equal matches.
    pub fn with_selector(mut self, selector: numen::NodeSelector) -> Self {
        std::sync::Arc::make_mut(&mut self.binding).selector = Some(selector);
        self
    }

    /// World-space scope for each body reference point. Attached extents are
    /// explicitly unsupported until their owner supplies resolved geometry.
    pub fn with_extent(mut self, extent: numen::FieldExtent) -> Self {
        std::sync::Arc::make_mut(&mut self.binding).extent = extent;
        self
    }

    /// Capture owner IDs and optional stable identities for each target slot.
    /// The callback runs once per slot, retaining duplicate occurrence mappings.
    /// Seiche records these supplied identities without asserting their authority.
    pub fn with_provenance(
        mut self,
        source: crate::CouplingSource,
        mut identify: impl FnMut(usize, NodeKey) -> Option<crate::CouplingTargetIdentity>,
    ) -> Self {
        let binding = std::sync::Arc::make_mut(&mut self.binding);
        binding.source = Some(source);
        binding.target_identities = binding
            .targets
            .iter()
            .enumerate()
            .map(|(slot, &node)| identify(slot, node))
            .collect();
        self
    }

    /// Immutable snapshot of the exact compiled inputs, including its registry.
    pub fn binding(&self) -> &crate::CouplingBinding {
        &self.binding
    }

    // `from_coupling(coupling, &Graph)` — which resolved a coupling's field and
    // selector against a mere `Graph` — left when seiche became a portable,
    // kernel-free crate. Graph resolution is a host concern: the host reads its
    // graph and calls [`Self::new`] / [`Self::with_registry`] with the resolved
    // field + target set (mere-canvas's `build::coupling_force_from_graph`).

    /// Number of nodes this force currently acts on.
    pub fn target_count(&self) -> usize {
        self.binding.targets.len()
    }

    /// Evaluate one response at caller-supplied body inputs without mutating a
    /// body, invoking an applet or advancing a controller. Candidates are the captured selector result; eligibility tests the body
    /// reference point against the bound extent, including its boundary.
    ///
    /// This convenience probe explicitly samples at `0.0`; use `evaluate_at_time`
    /// with a runtime clock when inspecting a running field. It describes one application;
    /// it does not sum duplicate targets, other couplings, contacts or anchored
    /// returns. Missing `Sample` references retain numen's existing zero fallback.
    /// The caller owns body presence, provenance and input validity; this method
    /// checks finite reference points for scope, but is not a complete input
    /// validator and does not certify that the reported write ran.
    pub fn evaluate_at(
        &self,
        node: NodeKey,
        position: [f32; 2],
        is_dynamic: bool,
        dt: f32,
    ) -> CouplingEvaluation {
        self.evaluate_at_time(node, position, is_dynamic, dt, 0.0)
    }

    /// Read-only probe at a caller-supplied clock, with reference-point scope.
    /// It does not advance the runtime or certify body presence/owner authority.
    pub fn evaluate_at_time(
        &self,
        node: NodeKey,
        position: [f32; 2],
        is_dynamic: bool,
        dt: f32,
        sample_time: f32,
    ) -> CouplingEvaluation {
        let effect = if self.binding.targets.contains(&node) {
            self.evaluate_target(position, is_dynamic, dt, sample_time)
        } else {
            CouplingEffect::Inert(CouplingInertReason::NotInResolvedTargets)
        };
        CouplingEvaluation {
            sample_time,
            effect,
        }
    }

    // Execution has already resolved each target, so it avoids a membership
    // scan per body. Keeping the original snapshot point also preserves repeated
    // targets: advection offsets accumulate; damping repeatedly scales velocity.
    pub(crate) fn evaluate_target(
        &self,
        [x, y]: [f32; 2],
        is_dynamic: bool,
        dt: f32,
        sample_time: f32,
    ) -> CouplingEffect {
        use CouplingEffect::{AddForce, Displace, Inert, ScaleVelocity, SetVelocity};
        use CouplingInertReason::{
            FieldTypeMismatch, NonDynamicBody, OutsidePositiveRegion, UnsupportedResponse,
        };
        if !sample_time.is_finite() {
            return Inert(CouplingInertReason::InvalidSampleTime);
        }
        match crate::coupling_membership::eligibility(&self.binding, [x, y]) {
            crate::CouplingEligibility::Inside => {},
            crate::CouplingEligibility::Outside => {
                return Inert(CouplingInertReason::OutsideExtent);
            },
            crate::CouplingEligibility::UnsupportedExtent => {
                return Inert(CouplingInertReason::UnsupportedExtent);
            },
            _ => return Inert(CouplingInertReason::InvalidReferencePoint),
        }
        match &self.binding.response {
            CouplingResponse::AttractToMin | CouplingResponse::RepelFromMax => {
                let Some(scalar) = self.scalar() else {
                    return Inert(FieldTypeMismatch);
                };
                let sign = if matches!(self.binding.response, CouplingResponse::AttractToMin) {
                    -1.0
                } else {
                    1.0
                };
                let (gx, gy) = grad_scalar(scalar, &self.binding.registry, x, y, sample_time);
                AddForce([
                    sign * gx * self.binding.strength,
                    sign * gy * self.binding.strength,
                ])
            },
            CouplingResponse::ContainmentWall => {
                let Some(scalar) = self.scalar() else {
                    return Inert(FieldTypeMismatch);
                };
                let depth = eval_scalar(scalar, &self.binding.registry, x, y, sample_time);
                if depth > 0.0 {
                    let (gx, gy) = grad_scalar(scalar, &self.binding.registry, x, y, sample_time);
                    let force = Vector::new(gx, gy) * (self.binding.strength * depth);
                    AddForce([force.x, force.y])
                } else {
                    Inert(OutsidePositiveRegion)
                }
            },
            CouplingResponse::DampenInside { factor } => {
                let Some(scalar) = self.scalar() else {
                    return Inert(FieldTypeMismatch);
                };
                if eval_scalar(scalar, &self.binding.registry, x, y, sample_time) > 0.0 {
                    // The existing response ignores coupling strength.
                    ScaleVelocity(*factor)
                } else {
                    Inert(OutsidePositiveRegion)
                }
            },
            CouplingResponse::AlignVelocity => {
                let Some(field) = self.vector() else {
                    return Inert(FieldTypeMismatch);
                };
                let (vx, vy) = eval_vector(field, &self.binding.registry, x, y, sample_time);
                SetVelocity([vx * self.binding.strength, vy * self.binding.strength])
            },
            CouplingResponse::FlowAdvect => {
                let Some(field) = self.vector() else {
                    return Inert(FieldTypeMismatch);
                };
                if !is_dynamic {
                    return Inert(NonDynamicBody);
                }
                let (vx, vy) = eval_vector(field, &self.binding.registry, x, y, sample_time);
                let offset = Vector::new(vx, vy) * (self.binding.strength * dt);
                Displace([offset.x, offset.y])
            },
            CouplingResponse::Open { .. } => Inert(UnsupportedResponse),
        }
    }

    fn scalar(&self) -> Option<&ScalarField> {
        match &self.binding.field {
            FieldDefinition::Scalar(s) => Some(s),
            FieldDefinition::Vector(_) => None,
        }
    }

    fn vector(&self) -> Option<&VectorField> {
        match &self.binding.field {
            FieldDefinition::Vector(v) => Some(v),
            FieldDefinition::Scalar(_) => None,
        }
    }
}

impl CouplingForce {
    pub(crate) fn write_effect(body: &mut RigidBody, effect: CouplingEffect) {
        match effect {
            CouplingEffect::AddForce([fx, fy]) => body.add_force(Vector::new(fx, fy), true),
            CouplingEffect::ScaleVelocity(factor) => {
                let velocity = body.linvel();
                body.set_linvel(velocity * factor, true);
            },
            CouplingEffect::SetVelocity([vx, vy]) => body.set_linvel(Vector::new(vx, vy), true),
            CouplingEffect::Displace([dx, dy]) => {
                let next = body.translation() + Vector::new(dx, dy);
                body.set_translation(next, true);
            },
            CouplingEffect::Inert(_) => {},
        }
    }
}

impl CouplingForce {
    pub(crate) fn sample_points(
        &self,
        ctx: &ForceContext<'_>,
    ) -> Vec<(Option<RigidBodyHandle>, Option<[f32; 2]>)> {
        self.binding
            .targets
            .iter()
            .map(|node| {
                ctx.bodies_by_node
                    .get(node)
                    .and_then(|&handle| {
                        ctx.bodies.get(handle).map(|body| {
                            let p = body.translation();
                            (Some(handle), Some([p.x, p.y]))
                        })
                    })
                    .unwrap_or((None, None))
            })
            .collect()
    }

    pub(crate) fn apply_samples(
        &self,
        ctx: &mut ForceContext<'_>,
        dt: f32,
        samples: &[(Option<RigidBodyHandle>, Option<[f32; 2]>)],
    ) {
        for &(handle, point) in samples {
            if let (Some(handle), Some(point)) = (handle, point)
                && let Some(body) = ctx.bodies.get_mut(handle)
            {
                let effect = self.evaluate_target(
                    point,
                    body.is_dynamic(),
                    dt,
                    ctx.field_clock.sample_time(),
                );
                Self::write_effect(body, effect);
            }
        }
    }
}
impl Force for CouplingForce {
    fn wants_tick(&self) -> bool {
        let typed = match &self.binding.response {
            CouplingResponse::AlignVelocity | CouplingResponse::FlowAdvect => {
                self.vector().is_some()
            },
            CouplingResponse::Open { .. } => false,
            _ => self.scalar().is_some(),
        };
        self.binding.animated
            && typed
            && !self.binding.targets.is_empty()
            && !matches!(self.binding.extent, numen::FieldExtent::AttachedToNode(_))
    }

    fn apply(&self, ctx: &mut ForceContext<'_>, dt: f32) {
        let samples = self.sample_points(ctx);
        self.apply_samples(ctx, dt, &samples);
    }
}

/// Global autonomous gradients are E over the field potential. Time-dependent
/// or extent-clipped gradients conservatively declare N: the timeless energy
/// interface cannot prove their global potential. State-writing responses stay
/// K. The open tail applies no physical term.
impl Declared for CouplingForce {
    fn terms(&self) -> Vec<Term> {
        let field = |class| {
            Term::force(
                "coupling",
                Topology::Unary,
                Kernel::Field,
                class,
                Observable::Residual,
            )
        };
        match &self.binding.response {
            CouplingResponse::AttractToMin
            | CouplingResponse::RepelFromMax
            | CouplingResponse::ContainmentWall => vec![field(
                if self.binding.animated
                    || !matches!(self.binding.extent, numen::FieldExtent::Global)
                {
                    Class::N
                } else {
                    Class::E
                },
            )],
            CouplingResponse::DampenInside { .. } | CouplingResponse::AlignVelocity => {
                vec![field(Class::K).moving(State::Velocity, Currency::Kinematic)]
            },
            CouplingResponse::FlowAdvect => {
                vec![field(Class::K).moving(State::Position, Currency::Kinematic)]
            },
            CouplingResponse::Open { .. } => Vec::new(),
        }
    }

    /// `∓s·φ` for attract and repel, `−(s/2)·φ²` inside the wall.
    fn energy(&self, _term: usize, layout: &Layout<'_>) -> Option<f64> {
        let scalar = self.scalar()?;
        let s = f64::from(self.binding.strength);
        let at = |i: usize| {
            let p = layout.nodes[i].1;
            f64::from(eval_scalar(scalar, &self.binding.registry, p.x, p.y, 0.0))
        };
        let targets: Vec<usize> = {
            let index = layout.index();
            self.binding
                .targets
                .iter()
                .filter_map(|k| index.get(k).copied())
                .collect()
        };
        let per_node: Box<dyn Fn(f64) -> f64> = match self.binding.response {
            CouplingResponse::AttractToMin => Box::new(move |phi| s * phi),
            CouplingResponse::RepelFromMax => Box::new(move |phi| -s * phi),
            CouplingResponse::ContainmentWall => {
                Box::new(move |phi: f64| if phi > 0.0 { -0.5 * s * phi * phi } else { 0.0 })
            },
            _ => return None,
        };
        Some(targets.into_iter().map(|i| per_node(at(i))).sum())
    }
}

// The graph-resolution tests (`from_coupling_resolves_field_and_selector` /
// `_resolves_sample_through_registry` / `_returns_none_for_unknown_field`) left
// with `from_coupling`: that logic is now mere-canvas's
// `build::coupling_force_from_graph`, and its tests live there. What stays here
// is the kernel-free physics of a resolved [`CouplingForce`].
#[cfg(test)]
#[path = "coupling_effect_tests.rs"]
mod effect_tests;

#[cfg(test)]
#[path = "coupling_force_tests.rs"]
mod tests;
