// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

use super::*;

impl Simulation {
    pub fn new() -> Self {
        let mut parameters = IntegrationParameters::default();
        // Mere lives at world-coordinate scale where 1 unit ≈ 1
        // pixel — the default 1/60s timestep is fine, but we also
        // expose `tick(dt)` so callers can vary it.
        parameters.dt = 1.0 / 60.0;
        Self {
            pipeline: PhysicsPipeline::new(),
            parameters,
            // No global gravity — forces supply directional forces
            // when they want them. A "gravity" force is just another
            // Force impl.
            gravity: Vector::ZERO,
            islands: IslandManager::new(),
            broad_phase: DefaultBroadPhase::new(),
            narrow_phase: NarrowPhase::new(),
            bodies: RigidBodySet::new(),
            colliders: ColliderSet::new(),
            impulse_joints: ImpulseJointSet::new(),
            multibody_joints: MultibodyJointSet::new(),
            soft_bodies: SoftBodySet::new(),
            ccd_solver: CCDSolver::new(),
            bodies_by_node: HashMap::new(),
            static_obstacles: Vec::new(),
            edges: EdgeTopology::default(),
            forces: Vec::new(),
            coupling_forces: Vec::new(),
            coupling_trace_session: CouplingTraceSession::new(),
            coupling_trace_limit: None,
            coupling_trace: None,
            coupling_binding_generation: 0,
            field_clock: FieldClock::default(),
            coupling_membership: Default::default(),
            coupling_membership_frame: None,
            coupling_entry_handler: None,
            coupling_entry_relay: false,
            coupling_entry_events: Vec::new(),
            affinity_force: None,
            anchor_force: None,
            repulsion: None,
            steps: 0,
            linear_damping: DEFAULT_LINEAR_DAMPING,
            scene_bodies: HashMap::new(),
            scene_sprites: HashMap::new(),
            next_scene_id: 0,
            scene_perpetual: false,
            fluid: None,
            nodes_tangible: false,
            sift: sift::Sift::default(),
            node_materials: HashMap::new(),
            scene_field: None,
            emitters: Vec::new(),
        }
    }

    /// The linear damping every node body carries (the "inertia" tunable).
    pub fn linear_damping(&self) -> f32 {
        self.linear_damping
    }

    /// Set the linear damping for new node bodies and re-apply it to every live
    /// one, so a settings change takes effect immediately rather than only for
    /// nodes added afterward. (Physics settings.)
    pub fn set_linear_damping(&mut self, damping: f32) {
        self.linear_damping = damping;
        for (_, body) in self.bodies.iter_mut() {
            body.set_linear_damping(damping);
        }
    }

    /// Register a force. Forces apply in registration order on every
    /// tick.
    pub fn add_force<F: Force + 'static>(&mut self, force: F) {
        self.forces.push(Box::new(force));
    }

    /// Replace the built-in force list wholesale — the law switch of the
    /// physics catalog. Couplings, the affinity force and the anchor force
    /// keep their own slots and are untouched; bodies keep their positions and
    /// velocities, so the new law takes over from where the old one left the
    /// graph rather than from a reseed. Forces run in the order given.
    pub fn set_forces(&mut self, forces: Vec<Box<dyn Force>>) {
        self.forces = forces;
    }

    /// Remove every built-in force (the catalog's `Still`): bodies coast to
    /// rest under damping alone, and anchored items' springs, if installed,
    /// are all that pulls.
    pub fn clear_forces(&mut self) {
        self.forces.clear();
    }

    pub fn force_count(&self) -> usize {
        self.forces.len()
    }

    /// A node body's current linear velocity, if it has a body.
    pub fn velocity_of(&self, node: NodeKey) -> Option<Vector> {
        self.bodies_by_node
            .get(&node)
            .and_then(|&handle| self.bodies.get(handle))
            .map(|body| body.linvel())
    }

    /// The node bodies' rms speed, `√(Σ |v|² / n)`, in world units a second:
    /// a settle is this falling under a floor, whatever the bodies' masses
    /// or count (dynamics grammar plan, F46). Zero with no bodies.
    pub fn rms_speed(&self) -> f32 {
        let (sum, n) = self
            .bodies_by_node
            .values()
            .filter_map(|&handle| self.bodies.get(handle))
            .fold((0.0_f32, 0_u32), |(sum, n), body| {
                let v = body.linvel();
                (sum + v.x * v.x + v.y * v.y, n + 1)
            });
        if n == 0 { 0.0 } else { (sum / n as f32).sqrt() }
    }

    /// Total kinetic energy of the node bodies, `Σ ½ m v²` — the number a
    /// receipt reads to say a restless law is restless and a settling one
    /// settled.
    pub fn kinetic_energy(&self) -> f32 {
        self.bodies_by_node
            .values()
            .filter_map(|&handle| self.bodies.get(handle))
            .map(|body| {
                let v = body.linvel();
                0.5 * body.mass() * (v.x * v.x + v.y * v.y)
            })
            .sum()
    }

    /// Replace the field-coupling forces wholesale (the built-in layout forces in
    /// `forces` are untouched). The host re-resolves every coupling against the
    /// current graph + node set and hands the fresh set here — the move / resize /
    /// new-node rebuild. Position-preserving: forces never touch body state, so
    /// swapping them leaves the layout exactly where it is. (Field regions.)
    pub fn set_coupling_forces(&mut self, forces: Vec<CouplingForce>) {
        self.coupling_binding_generation = self
            .coupling_binding_generation
            .checked_add(1)
            .expect("coupling binding generation exhausted");
        self.coupling_membership.retain_bindings(
            forces
                .iter()
                .map(|force| std::sync::Arc::clone(&force.binding)),
        );
        self.coupling_forces = forces;
        self.coupling_membership_frame = None;
        self.coupling_trace = None;
    }

    /// Number of field-coupling forces currently applied.
    pub fn coupling_force_count(&self) -> usize {
        self.coupling_forces.len()
    }

    /// Install (or clear, with `None`) the host's staging repulsion solver and its
    /// routing threshold. The closure receives the complete [`RepulsionRequest`]
    /// and must return validated output; failure falls back to the native CPU law.
    /// This does not make this Rapier simulation GPU-resident. See
    /// [`RepulsionSolver`].
    pub fn set_repulsion_solver(&mut self, solver: Option<RepulsionSolver>, threshold: usize) {
        self.repulsion = solver.map(|solver| Repulsion {
            route: RepulsionRoute::Sync(solver),
            threshold: threshold.max(1),
        });
    }

    /// Install (or clear, with `None`) a lagged evaluator: at or above
    /// `threshold` nodes, [`NodeExclusion`] applies the newest answer that is
    /// at most `max_stale_steps` old and for the same bodies, and runs its CPU
    /// law on every other step. Replaces any synchronous solver. `threshold`
    /// 0 behaves as 1: every non-empty graph uses the device.
    pub fn set_lagged_repulsion(
        &mut self,
        solver: Option<Box<dyn LaggedRepulsion>>,
        threshold: usize,
        max_stale_steps: u32,
    ) {
        self.repulsion = solver.map(|solver| Repulsion {
            route: RepulsionRoute::Lagged(LaggedLane::new(solver, max_stale_steps)),
            threshold: threshold.max(1),
        });
    }

    /// The installed lagged lane's counts (`None` when no lane is installed).
    pub fn repulsion_stats(&self) -> Option<LaggedStats> {
        self.repulsion.as_ref().and_then(Repulsion::stats)
    }

    /// Install (or clear, with `None`) the pairwise **affinity** force wholesale — the rebuild a
    /// fresh affinity signal triggers. Position-preserving (forces never touch body state), so
    /// swapping it leaves the layout exactly where it is; the host follows with a `settle` to let
    /// the new equilibrium take. (Graph signals — P4.)
    pub fn set_affinity_force(&mut self, force: Option<AffinitySpring>) {
        self.affinity_force = force;
    }

    /// Install (or clear, with `None`) per-node **anchor** springs toward the
    /// anchored items' arrangement positions: they return there while
    /// repulsion, edge springs, collisions, coupled fields and drag keep
    /// acting. Position-preserving, like every other force swap; the host
    /// follows with a `settle` to let the new equilibrium take.
    pub fn set_anchor_force(&mut self, force: Option<AnchorSpring>) {
        self.anchor_force = force;
    }

    /// How many nodes the installed anchor force pulls (`0` when none is set).
    pub fn anchor_count(&self) -> usize {
        self.anchor_force.as_ref().map_or(0, AnchorSpring::len)
    }

    /// Number of affinity pairs the installed affinity force pulls along (`0` when none is set).
    pub fn affinity_pair_count(&self) -> usize {
        self.affinity_force
            .as_ref()
            .map_or(0, AffinitySpring::pair_count)
    }

    /// A rapier-free [`LayoutView`] over the current layout: the live positions,
    /// the spring edge topology, and the node radius the picks/cull use. The host
    /// reads node picks, edge picks, and cull from this rather than the rapier
    /// query index, so those reads can run on the UI thread while the simulation
    /// itself ticks elsewhere (the always-offload physics actor, P6).
    pub fn view(&self) -> LayoutView {
        let mut view =
            LayoutView::from_parts(self.positions(), std::iter::empty(), NODE_BODY_RADIUS);
        view.set_edge_topology(self.edges.clone());
        view
    }

    /// A `Send` [`LayoutSnapshot`] of the current positions, stamped with
    /// `generation` — the payload a physics actor emits each tick for the host to
    /// fold into its [`LayoutView`]. Positions only; edges stay with the host's
    /// graph, so they need not cross the actor boundary every frame.
    pub fn snapshot(&self, generation: u64) -> LayoutSnapshot {
        LayoutSnapshot {
            positions: self.positions().collect(),
            scene: self.scene_bodies().collect(),
            fluid: self
                .fluid
                .as_ref()
                .map(|f| f.positions().collect())
                .unwrap_or_default(),
            fluid_radius: self
                .fluid
                .as_ref()
                .map(|f| f.params().particle_radius)
                .unwrap_or(0.0),
            energy: self.kinetic_energy(),
            speed: self.rms_speed(),
            generation,
        }
    }

    /// Retained for call-site compatibility; a no-op now. [`Self::hit_test`] and
    /// [`Self::cull_aabb`] read live body translations directly (the rapier
    /// `QueryPipeline` went ephemeral in rapier 0.33), so there is no separate
    /// index to refresh after moving bodies outside a tick.
    pub fn refresh_spatial_index(&mut self) {}

    /// Hit-test a world-space point: the node whose body lies within
    /// [`NODE_BODY_RADIUS`] of it, or `None`. Reads live positions, so it always
    /// reflects the most recent [`Self::tick`]. Node bodies are kept separated,
    /// so on the rare overlap this returns one of the hits, not a defined
    /// "topmost".
    ///
    /// Position-and-radius, matching the rapier-free [`LayoutView`] the orrery
    /// actually picks from on the UI thread — the two stay consistent. (Restore
    /// shape-accurate, collider-true picking in `LayoutView` if a non-ball node
    /// face ever needs it; that is the live path, not this in-thread helper.)
    pub fn hit_test(&self, point: Point2D<f32>) -> Option<NodeKey> {
        let r2 = NODE_BODY_RADIUS * NODE_BODY_RADIUS;
        self.positions()
            .find(|(_, p)| (*p - point).square_length() <= r2)
            .map(|(node, _)| node)
    }

    /// Frustum cull: every node whose body (center ± [`NODE_BODY_RADIUS`])
    /// intersects `region` (world space). Reads live positions. Order is
    /// unspecified.
    pub fn cull_aabb(&self, region: Box2D<f32>) -> Vec<NodeKey> {
        let r = NODE_BODY_RADIUS;
        self.positions()
            .filter(|(_, p)| {
                Box2D::new(
                    Point2D::new(p.x - r, p.y - r),
                    Point2D::new(p.x + r, p.y + r),
                )
                .intersects(&region)
            })
            .map(|(node, _)| node)
            .collect()
    }

    /// Advance the simulation by `dt` seconds. Walks every registered
    /// [`Force`] first (so forces accumulate), then steps the world. Reading
    /// the settled layout back is a separate call — [`Simulation::positions`]
    /// — so callers use the positions for whatever they like (write them into
    /// their own graph, draw debug overlays, and so on).
    pub fn tick(&mut self, dt: f32) {
        if !dt.is_finite() || dt <= 0.0 {
            return;
        }
        self.coupling_trace = self.coupling_trace_limit.map(|limit| CouplingTrace {
            session: self.coupling_trace_session.clone(),
            step: self.steps,
            clock: self.field_clock,
            dt,
            binding_generation: self.coupling_binding_generation,
            records: Vec::new(),
            omitted: 0,
            limit,
        });
        self.coupling_membership_frame = Some(CouplingMembershipFrame {
            session: self.coupling_trace_session.clone(),
            step: self.steps,
            clock: self.field_clock,
            binding_generation: self.coupling_binding_generation,
            reports: Vec::new(),
        });
        self.parameters.dt = dt;

        // Emitters spawn / reap scene bodies before the step, so fresh droplets are simulated this
        // frame. (Physics scenes — emitters.)
        self.step_emitters(dt);

        // rapier's `add_force` is a *persistent* force (it survives across
        // steps until reset), so clear last tick's force forces before this
        // tick's forces set fresh ones. Without this, per-tick forces compound
        // and the layout goes unstable.
        if !self.forces.is_empty()
            || !self.coupling_forces.is_empty()
            || self.affinity_force.is_some()
            || self.anchor_force.is_some()
            || self.scene_field.is_some()
        {
            for (_, body) in self.bodies.iter_mut() {
                body.reset_forces(false);
            }
            // Preserve the public ForceContext slice contract for every law
            // and external Force implementation. This transient compatibility
            // read is the sole expanded topology at the evaluation boundary;
            // simulation/view/actor retain only counted runs.
            let edges: Vec<_> = self.edges.iter().copied().collect();
            let mut ctx = ForceContext {
                bodies: &mut self.bodies,
                colliders: &self.colliders,
                joints: &mut self.impulse_joints,
                bodies_by_node: &self.bodies_by_node,
                edges: &edges,
                repulsion: self.repulsion.as_mut(),
                step: self.steps,
                field_clock: self.field_clock,
            };
            for force in &self.forces {
                force.apply(&mut ctx, dt);
            }
            // Field couplings apply after the built-ins, in the same reset window.
            for (index, force) in self.coupling_forces.iter().enumerate() {
                let samples = force.sample_points(&ctx);
                if let Some(frame) = &mut self.coupling_membership_frame {
                    let first = frame.reports.len();
                    self.coupling_membership.observe(
                        &force.binding,
                        &samples,
                        ctx.bodies_by_node,
                        frame,
                        self.coupling_entry_handler.as_ref(),
                        self.coupling_entry_relay,
                    );
                    if self.coupling_entry_relay {
                        self.coupling_entry_events.extend(
                            frame.reports[first..]
                                .iter()
                                .filter(|report| {
                                    report.entry_action == crate::CouplingEntryOutcome::Pending
                                })
                                .map(|report| report.entry_event(index)),
                        );
                    }
                }
                if let Some(trace) = &mut self.coupling_trace {
                    force.apply_traced(&mut ctx, dt, index, trace, &samples);
                } else {
                    force.apply_samples(&mut ctx, dt, &samples);
                }
            }
            // The affinity force (if installed) applies last, over the same context — a weighted
            // clustering pull on top of the topology springs. (Graph signals — P4.)
            if let Some(force) = &self.affinity_force {
                force.apply(&mut ctx, dt);
            }
            // Anchor springs last: an anchored item's return sits on top of
            // the graph's own forces. (Arrangement roles, G7.)
            if let Some(force) = &self.anchor_force {
                force.apply(&mut ctx, dt);
            }
        }

        // The scene force-field (a whirlpool / well) drives the dynamic scene bodies — applied after
        // the force loops (so the body borrow is free), still before the step. (Physics scenes P4.)
        self.apply_scene_field();

        let physics_hooks = ();
        let event_handler = ();
        self.pipeline.step(
            self.gravity,
            &self.parameters,
            &mut self.islands,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.bodies,
            &mut self.colliders,
            &mut self.impulse_joints,
            &mut self.multibody_joints,
            &mut self.soft_bodies,
            &mut self.ccd_solver,
            &physics_hooks,
            &event_handler,
        );

        // Step the liquid pool (if any) after the rigid world; it reads the same dt and folds into
        // the same snapshot. (Physics scenes P4c.)
        if let Some(fluid) = &mut self.fluid {
            fluid.step(dt);
        }
        // Two-way coupling: the pool bounces off the scene bodies + (when the graph is tangible) the
        // node bodies, and shoves the dynamic ones back. (Physics scenes P4c.)
        self.couple_fluid_to_bodies();
        self.steps += 1;
        if !self.field_clock.paused {
            self.field_clock.elapsed += f64::from(dt);
        }
        self.coupling_membership.finish_tick();
    }

    // `write_positions_to(&mut Graph)` — which copied each body's translation back
    // onto its graph node — left when seiche became kernel-free (it had no mere
    // consumer). The host reads [`Self::positions`] and writes them into its own
    // graph; mere's position write-back is a host concern (and a node-dissolution
    // target).

    /// True when every body's linear velocity is below
    /// `velocity_epsilon`. Useful for "suspend the tick when the
    /// graph is at rest" idle-detection.
    pub fn is_at_rest(&self, velocity_epsilon: f32) -> bool {
        let eps_sq = velocity_epsilon * velocity_epsilon;
        for (_, body) in self.bodies.iter() {
            let v = body.linvel();
            if v.x * v.x + v.y * v.y > eps_sq {
                return false;
            }
        }
        true
    }

    /// Pin a node's body to a position via a kinematic-position
    /// override. Use this during a drag: call `pin` while the user
    /// holds the mouse, then `unpin` on release so other forces can
    /// react to the final position again.
    pub fn pin(&mut self, node: NodeKey, position: Point2D<f32>) {
        let Some(handle) = self.bodies_by_node.get(&node).copied() else {
            return;
        };
        let Some(body) = self.bodies.get_mut(handle) else {
            return;
        };
        body.set_body_type(RigidBodyType::KinematicPositionBased, true);
        body.set_next_kinematic_translation(Vector::new(position.x, position.y));
    }

    pub fn unpin(&mut self, node: NodeKey) {
        let Some(handle) = self.bodies_by_node.get(&node).copied() else {
            return;
        };
        let Some(body) = self.bodies.get_mut(handle) else {
            return;
        };
        body.set_body_type(RigidBodyType::Dynamic, true);
    }
}
