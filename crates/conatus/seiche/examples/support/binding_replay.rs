// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! Shared native/Wasm replay implementation. This deliberately records only
//! the bounded state needed to localize deterministic replay divergence.

use std::num::NonZeroUsize;

use euclid::default::Point2D;
use numen::{CouplingResponse, FieldDefinition, ScalarField, VectorField};
use seiche::spec::{Node, PresetAt, Realization};
use seiche::{
    Anneal, CouplingAppliedEffect, CouplingForce, DynamicsSpec, Force, NodeKey, Simulation,
    SpecCatalog, SpecError,
};
use serde::{Deserialize, Serialize};
use state_witness::{Trace, Witness, divergences, first_trace_divergence};

const MAX_BODIES: usize = 64;
const MAX_TICKS: usize = 600;
const FORCE_ORDER: &[&str] = &["anneal-seeded", "time-flow"];

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReplayInput {
    spec: DynamicsSpec,
    seed: u64,
    bodies: Vec<BodyInput>,
    force_order: Vec<String>,
    ticks: Vec<TickInput>,
    #[serde(default)]
    perturb: Option<PerturbInput>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BodyInput {
    label: String,
    position: [f32; 2],
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TickInput {
    tick: u64,
    dt: f32,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PerturbInput {
    tick: u64,
    dt: f32,
}

#[derive(Serialize)]
struct ReplayOutput {
    protocol: u16,
    spec_admission: String,
    tick_dts: Vec<TickMetadata>,
    trace: Trace,
    perturbation: Option<PerturbationOutput>,
}

#[derive(Serialize)]
struct TickMetadata {
    tick: u64,
    dt_bits: u32,
}

#[derive(Serialize)]
struct PerturbationOutput {
    tick: u64,
    baseline_dt_bits: u32,
    candidate_dt_bits: u32,
    trace: Trace,
    first_divergence: Option<state_witness::TraceDivergence>,
    physical_divergence_at_tick: state_witness::Divergence,
}

struct ReplayCatalog {
    seed: u64,
}

impl SpecCatalog for ReplayCatalog {
    fn preset(&self, id: &str, at: PresetAt) -> Option<Vec<Box<dyn Force>>> {
        if at != PresetAt::Law || id != "binding-replay" {
            return None;
        }
        Some(vec![Box::new(Anneal::seeded(self.seed))])
    }
}

/// Execute one input document. Empty input selects the checked-in canonical
/// fixture; a perturbation runs a second pass and reports its first witness
/// divergence from the canonical pass.
pub fn run(input: &str) -> Result<String, String> {
    let source = if input.trim().is_empty() {
        include_str!("../binding_replay.json")
    } else {
        input
    };
    let fixture: ReplayInput =
        serde_json::from_str(source).map_err(|error| format!("invalid replay fixture: {error}"))?;
    validate(&fixture)?;

    let baseline = run_trace(&fixture, None)?;
    let perturbation = match fixture.perturb {
        Some(perturb) => {
            let candidate = run_trace(&fixture, Some(perturb))?;
            let first_divergence = first_trace_divergence(&baseline, &candidate);
            let baseline_tick = baseline
                .get(perturb.tick)
                .ok_or_else(|| "missing baseline tick".to_owned())?;
            let candidate_tick = candidate
                .get(perturb.tick)
                .ok_or_else(|| "missing candidate tick".to_owned())?;
            let physical_divergence_at_tick = divergences(baseline_tick, candidate_tick)
                .find(|divergence| divergence.label.starts_with("position:") || divergence.label.starts_with("velocity:"))
                .ok_or_else(|| format!("dt perturb at tick {} produced no position/velocity divergence at that tick", perturb.tick))?;
            let baseline_dt_bits = fixture
                .ticks
                .iter()
                .find(|tick| tick.tick == perturb.tick)
                .unwrap()
                .dt
                .to_bits();
            Some(PerturbationOutput {
                tick: perturb.tick,
                baseline_dt_bits,
                candidate_dt_bits: perturb.dt.to_bits(),
                trace: candidate,
                first_divergence,
                physical_divergence_at_tick,
            })
        },
        None => None,
    };
    let output = ReplayOutput {
        protocol: 1,
        spec_admission: "harness-catalog:binding-replay".into(),
        tick_dts: fixture
            .ticks
            .iter()
            .map(|tick| TickMetadata {
                tick: tick.tick,
                dt_bits: tick.dt.to_bits(),
            })
            .collect(),
        trace: baseline,
        perturbation,
    };
    serde_json::to_string(&output).map_err(|error| format!("encode replay output: {error}"))
}

fn validate(input: &ReplayInput) -> Result<(), String> {
    if input.seed != input.spec.seed {
        return Err("fixture seed must equal dynamics spec seed".into());
    }
    if input.bodies.is_empty() || input.bodies.len() > MAX_BODIES {
        return Err(format!("body count must be 1..={MAX_BODIES}"));
    }
    if input.ticks.is_empty() || input.ticks.len() > MAX_TICKS {
        return Err(format!("tick count must be 1..={MAX_TICKS}"));
    }
    if !input
        .force_order
        .iter()
        .map(String::as_str)
        .eq(FORCE_ORDER.iter().copied())
    {
        return Err(format!("force_order must be {}", FORCE_ORDER.join(",")));
    }
    let mut labels = std::collections::HashSet::new();
    for body in &input.bodies {
        if body.label.is_empty() || body.label.contains('\0') || !labels.insert(&body.label) {
            return Err(format!("invalid or duplicate body label {:?}", body.label));
        }
        if !body.position.iter().all(|value| value.is_finite()) {
            return Err(format!("invalid position for body {:?}", body.label));
        }
    }
    for (index, tick) in input.ticks.iter().enumerate() {
        if tick.tick != index as u64 + 1 {
            return Err("tick transcript must be ordered and start at tick 1".into());
        }
        if !tick.dt.is_finite() || !(0.0..=0.1).contains(&tick.dt) || tick.dt == 0.0 {
            return Err(format!(
                "tick {} dt must be finite and in (0, 0.1]",
                tick.tick
            ));
        }
    }
    if let Some(perturb) = input.perturb {
        if !input.ticks.iter().any(|tick| tick.tick == perturb.tick) {
            return Err("perturbation tick must occur in transcript".into());
        }
        if !perturb.dt.is_finite() || !(0.0..=0.1).contains(&perturb.dt) || perturb.dt == 0.0 {
            return Err("perturbation dt must be finite and in (0, 0.1]".into());
        }
    }
    Ok(())
}

fn run_trace(input: &ReplayInput, perturb: Option<PerturbInput>) -> Result<Trace, String> {
    let expected_spec = DynamicsSpec {
        version: seiche::spec::DYNAMICS_SPEC_VERSION,
        seed: input.seed,
        root: Node::preset("binding-replay"),
        channels: Default::default(),
        realization: Realization::Integrate { damping: Some(0.7) },
        target: None,
        bars: Vec::new(),
    };
    if input.spec != expected_spec {
        return Err("fixture spec differs from the single supported binding-replay spec".into());
    }
    let nodes: Vec<_> = (0..input.bodies.len()).map(NodeKey::new).collect();
    let catalog = ReplayCatalog { seed: input.seed };
    input
        .spec
        .derive(&catalog)
        .map_err(|error: SpecError| format!("dynamics spec refused: {error:?}"))?;

    let mut simulation = Simulation::new();
    simulation.sync_nodes(
        nodes.iter().copied().zip(
            input
                .bodies
                .iter()
                .map(|body| Point2D::new(body.position[0], body.position[1])),
        ),
    );
    let forces = catalog
        .preset("binding-replay", PresetAt::Law)
        .ok_or_else(|| "replay preset missing".to_owned())?;
    simulation.set_forces(forces);
    if let seiche::spec::Realization::Integrate {
        damping: Some(damping),
    } = input.spec.realization
    {
        simulation.set_linear_damping(damping as f32);
    }

    let timed_field = FieldDefinition::Vector(VectorField::Scale(
        Box::new(VectorField::const_vec(8.0, 0.0)),
        Box::new(ScalarField::Time),
    ));
    let timed_coupling = CouplingForce::new(
        CouplingResponse::FlowAdvect,
        1.0,
        nodes.iter().copied(),
        timed_field,
    );
    simulation.set_coupling_forces(vec![timed_coupling]);
    simulation.set_coupling_trace_limit(NonZeroUsize::new(input.bodies.len()));

    let mut trace = Trace::new();
    for tick in &input.ticks {
        let dt = perturb
            .filter(|candidate| candidate.tick == tick.tick)
            .map_or(tick.dt, |candidate| candidate.dt);
        simulation.tick(dt);
        let clock = simulation.field_clock();
        let mut witness = Witness::new();
        witness
            .insert("clock:elapsed-f64-bits", &clock.elapsed.to_bits())
            .map_err(|error| error.to_string())?;
        witness
            .insert("clock:epoch", &clock.epoch)
            .map_err(|error| error.to_string())?;
        witness
            .insert("clock:paused", &clock.paused)
            .map_err(|error| error.to_string())?;
        let coupling_trace = simulation
            .coupling_trace()
            .ok_or_else(|| "coupling trace missing after tick".to_owned())?;
        if coupling_trace.omitted != 0 {
            return Err("coupling trace was truncated".into());
        }
        for record in &coupling_trace.records {
            if !record.binding.animated {
                return Err("time FlowAdvect field was not classified as animated".into());
            }
            let body_index = record.node.index();
            let label = input
                .bodies
                .get(body_index)
                .ok_or_else(|| "coupling trace contains an unknown body".to_owned())?
                .label
                .as_str();
            let evaluation = record
                .evaluation
                .ok_or_else(|| format!("missing animated field evaluation for {label}"))?;
            witness
                .insert(
                    format!("field:{label}/sample-time-f32-bits"),
                    &evaluation.sample_time.to_bits(),
                )
                .map_err(|error| error.to_string())?;
            let CouplingAppliedEffect::Write { before, after } = &record.applied else {
                return Err(format!("animated field did not write body {label}"));
            };
            witness
                .insert(
                    format!("field:{label}/displacement-x-bits"),
                    &(after.position[0] - before.position[0]).to_bits(),
                )
                .map_err(|error| error.to_string())?;
            witness
                .insert(
                    format!("field:{label}/displacement-y-bits"),
                    &(after.position[1] - before.position[1]).to_bits(),
                )
                .map_err(|error| error.to_string())?;
        }
        for (label, node) in input
            .bodies
            .iter()
            .map(|body| body.label.as_str())
            .zip(nodes.iter())
        {
            let position = simulation
                .positions()
                .find(|(key, _)| key == node)
                .map(|(_, point)| point)
                .ok_or_else(|| format!("missing position for body {label}"))?;
            let velocity = simulation
                .velocity_of(*node)
                .ok_or_else(|| format!("missing velocity for body {label}"))?;
            witness
                .insert(format!("position:{label}/x-bits"), &position.x.to_bits())
                .map_err(|error| error.to_string())?;
            witness
                .insert(format!("position:{label}/y-bits"), &position.y.to_bits())
                .map_err(|error| error.to_string())?;
            witness
                .insert(format!("velocity:{label}/x-bits"), &velocity.x.to_bits())
                .map_err(|error| error.to_string())?;
            witness
                .insert(format!("velocity:{label}/y-bits"), &velocity.y.to_bits())
                .map_err(|error| error.to_string())?;
        }
        trace
            .push(tick.tick, witness)
            .map_err(|error| error.to_string())?;
    }
    Ok(trace)
}
