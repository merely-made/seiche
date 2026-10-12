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
    response: CouplingResponse,
    strength: f32,
    /// Nodes the coupling acts on, resolved from the selector at build time.
    targets: Vec<NodeKey>,
    /// The field to sample. Scalar responses (attract/repel/wall/dampen) need a
    /// scalar; flow responses (align/advect) need a vector. A mismatch is inert.
    field: FieldDefinition,
    /// Registry for the field's `Sample` references. Empty by default — a
    /// self-contained field needs none, and a missing sample evaluates to zero.
    registry: FieldRegistry,
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
const SAMPLE_TIME: f32 = 0.0;

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
        Self {
            response,
            strength,
            targets: targets.into_iter().collect(),
            field,
            registry: FieldRegistry::new(),
        }
    }

    /// Supply a registry so the field's `Sample` references resolve.
    pub fn with_registry(mut self, registry: FieldRegistry) -> Self {
        self.registry = registry;
        self
    }

    // `from_coupling(coupling, &Graph)` — which resolved a coupling's field and
    // selector against a mere `Graph` — left when seiche became a portable,
    // kernel-free crate. Graph resolution is a host concern: the host reads its
    // graph and calls [`Self::new`] / [`Self::with_registry`] with the resolved
    // field + target set (mere-canvas's `build::coupling_force_from_graph`).

    /// Number of nodes this force currently acts on.
    pub fn target_count(&self) -> usize {
        self.targets.len()
    }

    /// Evaluate one response at caller-supplied body inputs without mutating a
    /// body, invoking an applet or advancing a controller. Target membership is
    /// the captured selector result, not membership in the field's extent.
    ///
    /// Field sampling remains at `0.0`. The result describes one application;
    /// it does not sum duplicate targets, other couplings, contacts or anchored
    /// returns. Missing `Sample` references retain numen's existing zero fallback.
    /// The caller owns body presence, provenance and input validity; this method
    /// does not validate finite inputs or certify that the reported write ran.
    pub fn evaluate_at(
        &self,
        node: NodeKey,
        position: [f32; 2],
        is_dynamic: bool,
        dt: f32,
    ) -> CouplingEvaluation {
        let effect = if self.targets.contains(&node) {
            self.evaluate_target(position, is_dynamic, dt)
        } else {
            CouplingEffect::Inert(CouplingInertReason::NotInResolvedTargets)
        };
        CouplingEvaluation {
            sample_time: SAMPLE_TIME,
            effect,
        }
    }

    // Execution has already resolved each target, so it avoids a membership
    // scan per body. Keeping the original snapshot point also preserves repeated
    // targets: advection offsets accumulate; damping repeatedly scales velocity.
    fn evaluate_target(&self, [x, y]: [f32; 2], is_dynamic: bool, dt: f32) -> CouplingEffect {
        use CouplingEffect::{AddForce, Displace, Inert, ScaleVelocity, SetVelocity};
        use CouplingInertReason::{
            FieldTypeMismatch, NonDynamicBody, OutsidePositiveRegion, UnsupportedResponse,
        };
        match &self.response {
            CouplingResponse::AttractToMin | CouplingResponse::RepelFromMax => {
                let Some(scalar) = self.scalar() else {
                    return Inert(FieldTypeMismatch);
                };
                let sign = if matches!(self.response, CouplingResponse::AttractToMin) {
                    -1.0
                } else {
                    1.0
                };
                let (gx, gy) = grad_scalar(scalar, &self.registry, x, y, SAMPLE_TIME);
                AddForce([sign * gx * self.strength, sign * gy * self.strength])
            },
            CouplingResponse::ContainmentWall => {
                let Some(scalar) = self.scalar() else {
                    return Inert(FieldTypeMismatch);
                };
                let depth = eval_scalar(scalar, &self.registry, x, y, SAMPLE_TIME);
                if depth > 0.0 {
                    let (gx, gy) = grad_scalar(scalar, &self.registry, x, y, SAMPLE_TIME);
                    let force = Vector::new(gx, gy) * (self.strength * depth);
                    AddForce([force.x, force.y])
                } else {
                    Inert(OutsidePositiveRegion)
                }
            },
            CouplingResponse::DampenInside { factor } => {
                let Some(scalar) = self.scalar() else {
                    return Inert(FieldTypeMismatch);
                };
                if eval_scalar(scalar, &self.registry, x, y, SAMPLE_TIME) > 0.0 {
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
                let (vx, vy) = eval_vector(field, &self.registry, x, y, SAMPLE_TIME);
                SetVelocity([vx * self.strength, vy * self.strength])
            },
            CouplingResponse::FlowAdvect => {
                let Some(field) = self.vector() else {
                    return Inert(FieldTypeMismatch);
                };
                if !is_dynamic {
                    return Inert(NonDynamicBody);
                }
                let (vx, vy) = eval_vector(field, &self.registry, x, y, SAMPLE_TIME);
                let offset = Vector::new(vx, vy) * (self.strength * dt);
                Displace([offset.x, offset.y])
            },
            CouplingResponse::Open { .. } => Inert(UnsupportedResponse),
        }
    }

    fn scalar(&self) -> Option<&ScalarField> {
        match &self.field {
            FieldDefinition::Scalar(s) => Some(s),
            FieldDefinition::Vector(_) => None,
        }
    }

    fn vector(&self) -> Option<&VectorField> {
        match &self.field {
            FieldDefinition::Vector(v) => Some(v),
            FieldDefinition::Scalar(_) => None,
        }
    }
}

impl Force for CouplingForce {
    fn apply(&self, ctx: &mut ForceContext<'_>, dt: f32) {
        // Snapshot field sample points before any writes, as the original
        // executor did. Duplicate target applications stay in their input order.
        let mut samples: Vec<(RigidBodyHandle, f32, f32)> = Vec::with_capacity(self.targets.len());
        for &node in &self.targets {
            if let Some(&handle) = ctx.bodies_by_node.get(&node) {
                if let Some(body) = ctx.bodies.get(handle) {
                    let t = body.translation();
                    samples.push((handle, t.x, t.y));
                }
            }
        }
        for (handle, x, y) in samples {
            if let Some(body) = ctx.bodies.get_mut(handle) {
                match self.evaluate_target([x, y], body.is_dynamic(), dt) {
                    CouplingEffect::AddForce([fx, fy]) => body.add_force(Vector::new(fx, fy), true),
                    CouplingEffect::ScaleVelocity(factor) => {
                        let velocity = body.linvel();
                        body.set_linvel(velocity * factor, true);
                    },
                    CouplingEffect::SetVelocity([vx, vy]) => {
                        body.set_linvel(Vector::new(vx, vy), true)
                    },
                    CouplingEffect::Displace([dx, dy]) => {
                        let next = body.translation() + Vector::new(dx, dy);
                        body.set_translation(next, true);
                    },
                    CouplingEffect::Inert(_) => {},
                }
            }
        }
    }
}

/// Gradient responses (attract, repel, the wall) are E over the field's
/// potential; the damping, alignment and advection responses write state
/// (K). The open tail applies nothing, so it declares nothing.
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
        match &self.response {
            CouplingResponse::AttractToMin
            | CouplingResponse::RepelFromMax
            | CouplingResponse::ContainmentWall => vec![field(Class::E)],
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
        let s = f64::from(self.strength);
        let at = |i: usize| {
            let p = layout.nodes[i].1;
            f64::from(eval_scalar(scalar, &self.registry, p.x, p.y, 0.0))
        };
        let targets: Vec<usize> = {
            let index = layout.index();
            self.targets
                .iter()
                .filter_map(|k| index.get(k).copied())
                .collect()
        };
        let per_node: Box<dyn Fn(f64) -> f64> = match self.response {
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
mod tests {
    use super::*;
    use crate::{Boundary, NodeKey, Simulation};
    use euclid::default::Point2D;

    /// Kernel-free node-list builder (seiche drives physics from `(NodeKey,
    /// position)` pairs).
    #[derive(Default)]
    struct Nodes {
        list: Vec<(NodeKey, Point2D<f32>)>,
    }

    impl Nodes {
        fn at(&mut self, x: f32, y: f32) -> NodeKey {
            let key = NodeKey::new(self.list.len());
            self.list.push((key, Point2D::new(x, y)));
            key
        }
        fn sync(&self, sim: &mut Simulation) {
            sim.sync_nodes(self.list.iter().copied());
        }
    }

    fn radius(sim: &Simulation, k: NodeKey) -> f32 {
        sim.position_of(k).unwrap().to_vector().length()
    }

    /// `½(x² + y²)` — gradient `(x, y)`, the potential whose `AttractToMin`
    /// coupling reproduces `Boundary`.
    pub(super) fn paraboloid() -> ScalarField {
        ScalarField::Scale(
            Box::new(ScalarField::Add(
                Box::new(ScalarField::Mul(
                    Box::new(ScalarField::CoordX),
                    Box::new(ScalarField::CoordX),
                )),
                Box::new(ScalarField::Mul(
                    Box::new(ScalarField::CoordY),
                    Box::new(ScalarField::CoordY),
                )),
            )),
            0.5,
        )
    }

    #[test]
    fn attract_to_min_centers_like_boundary() {
        // Two sims, same seed: Boundary vs an AttractToMin coupling on the
        // paraboloid. The paraboloid's gradient is (x, y), so the coupling force
        // is -pos·strength — Boundary exactly, up to numen's finite-difference
        // step. The bodies track each other across the whole settle (a ~580-unit
        // journey leaves them within a couple of units).
        let mut g = Nodes::default();
        let a = g.at(500.0, -300.0);

        let mut boundary_sim = Simulation::new();
        g.sync(&mut boundary_sim);
        boundary_sim.add_force(Boundary { strength: 1.5 });

        let mut coupling_sim = Simulation::new();
        g.sync(&mut coupling_sim);
        coupling_sim.add_force(CouplingForce::new(
            CouplingResponse::AttractToMin,
            1.5,
            [a],
            FieldDefinition::Scalar(paraboloid()),
        ));

        for _ in 0..120 {
            boundary_sim.tick(1.0 / 60.0);
            coupling_sim.tick(1.0 / 60.0);
        }

        let pb = boundary_sim.position_of(a).unwrap();
        let pc = coupling_sim.position_of(a).unwrap();
        assert!(
            radius(&coupling_sim, a) < 500.0,
            "coupling should pull inward"
        );
        assert!(
            (pb - pc).length() < 5.0,
            "coupling diverged from Boundary: {pb:?} vs {pc:?}"
        );
    }

    #[test]
    fn open_response_applies_no_force() {
        // An open-tail response (e.g. a visual coupling) is not a force: the
        // integrator must leave the body where it is, even on a field whose
        // AttractToMin response would otherwise pull it to the origin.
        let mut g = Nodes::default();
        let a = g.at(300.0, 0.0);

        let mut sim = Simulation::new();
        g.sync(&mut sim);
        sim.add_force(CouplingForce::new(
            CouplingResponse::open(
                "https://mere.computer/ns/coupling#visual/highlight".to_string(),
            ),
            1.0,
            [a],
            FieldDefinition::Scalar(paraboloid()),
        ));
        let p0 = sim.position_of(a).unwrap();
        for _ in 0..120 {
            sim.tick(1.0 / 60.0);
        }
        assert!(
            (sim.position_of(a).unwrap() - p0).length() < 1.0e-3,
            "an open-tail response must apply no force"
        );
    }

    #[test]
    fn flow_advect_leaves_a_pinned_body_at_its_kinematic_target() {
        // FlowAdvect writes positions; a pinned (kinematic) body belongs to
        // the drag, so it stays on its target while a free one is advected.
        let mut g = Nodes::default();
        let pinned = g.at(0.0, 0.0);
        let free = g.at(100.0, 0.0);
        let mut sim = Simulation::new();
        g.sync(&mut sim);
        sim.add_force(CouplingForce::new(
            CouplingResponse::FlowAdvect,
            1.0,
            [pinned, free],
            FieldDefinition::Vector(VectorField::ConstVec { x: 120.0, y: 0.0 }),
        ));
        let target = Point2D::new(-200.0, 50.0);
        sim.pin(pinned, target);
        let free_start = sim.position_of(free).unwrap();
        for _ in 0..60 {
            sim.tick(1.0 / 60.0);
        }
        let held = sim.position_of(pinned).unwrap();
        assert!(
            (held - target).length() < 1e-3,
            "the pinned body left its target: {held:?}"
        );
        assert!(
            sim.position_of(free).unwrap().x > free_start.x + 10.0,
            "the free body is still advected"
        );
    }
}
