// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! One term's readings on a fixture, the tolerances they are judged by, and
//! what each declared class predicts of them. Split from the instruments
//! themselves to keep both files under the workspace's size ceiling.

use super::{
    Probe, Vector, balance, descent, gradient_error, isolated, jacobian_asymmetry, loop_work,
    persistent_motion, velocity_read,
};
use crate::Force;

/// Tolerances, each a fraction of the instrument's own scale.
pub mod tolerance {
    /// The largest single rise of an energy, against its whole fall.
    pub const RISE: f64 = 1e-3;
    /// `‖M·F + ∇U‖ / ‖∇U‖`.
    pub const GRADIENT: f64 = 1e-2;
    /// `|Σ wᵢFᵢ| / Σ wᵢ|Fᵢ|`.
    pub const BALANCE: f64 = 1e-4;
    /// Net work around a loop, against its absolute work.
    pub const LOOP_WORK: f64 = 1e-3;
    /// The Jacobian's antisymmetric part, against the whole.
    pub const JACOBIAN: f64 = 2e-2;
    /// Late kinetic energy against the early peak.
    pub const MOTION: f32 = 1e-3;
    /// Change in force when the bodies move.
    pub const VELOCITY: f64 = 1e-6;
}

/// The loop's size in world units, per body.
const LOOP_AMPLITUDE: f32 = 40.0;

/// A test of conservativeness for a term with no energy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Candidate {
    LoopWork,
    Jacobian,
    Motion,
}

impl Candidate {
    pub const ALL: [Candidate; 3] = [Candidate::LoopWork, Candidate::Jacobian, Candidate::Motion];
}

/// One term's readings on one fixture: the worst over the starts, except the
/// two finite-difference candidates, kept per start and judged by their
/// median, since a start that puts a pair across a cutoff or a cell boundary
/// reads that kink rather than the term.
#[derive(Clone, Debug, PartialEq)]
pub struct Reading {
    pub term: crate::Term,
    pub rise: Option<f64>,
    pub gradient: Option<f64>,
    /// In the term's metric, for an internal term.
    pub balance: Option<f64>,
    /// The same with every weight one.
    pub identity_balance: Option<f64>,
    pub loop_work: Vec<f64>,
    pub jacobian: Vec<f64>,
    pub motion: f32,
    pub velocity: f64,
}

/// The middle value (the upper of the two middles for an even count).
pub fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted.get(sorted.len() / 2).copied().unwrap_or(0.0)
}

/// Read term `term` of the force `make` builds, at each of `starts`. `make`
/// is called afresh wherever a reading would otherwise inherit state (a
/// clock, a phase, a kick).
pub fn read(
    probe: &mut Probe,
    make: &dyn Fn() -> Box<dyn Force>,
    term: usize,
    starts: &[Vec<Vector>],
) -> Reading {
    let declared = make().terms()[term];
    let subject = isolated(make(), term);
    let force = subject.as_ref();
    let weights = {
        probe.place(&starts[0]);
        probe.weights(force, term)
    };
    let ones = vec![1.0; probe.len()];
    let worst = |a: Option<f64>, b: Option<f64>| match (a, b) {
        (Some(a), Some(b)) => Some(a.max(b)),
        (a, b) => a.or(b),
    };
    let mut reading = Reading {
        term: declared,
        rise: None,
        gradient: None,
        balance: None,
        identity_balance: None,
        loop_work: Vec::new(),
        jacobian: Vec::new(),
        motion: 0.0,
        velocity: 0.0,
    };
    for (k, start) in starts.iter().enumerate() {
        let fall = descent(probe, force, term, start, 300, 0.5);
        reading.rise = worst(reading.rise, fall.map(|d| d.relative_rise()));
        reading.gradient = worst(
            reading.gradient,
            gradient_error(probe, force, term, start, 0.05),
        );
        if declared.topology.is_internal() {
            let metric = balance(probe, force, &weights, start).relative();
            let identity = balance(probe, force, &ones, start).relative();
            reading.balance = worst(reading.balance, Some(metric));
            reading.identity_balance = worst(reading.identity_balance, Some(identity));
        }
        reading.loop_work.push(loop_work(
            probe,
            force,
            &weights,
            start,
            LOOP_AMPLITUDE,
            k as u64 + 1,
            720,
        ));
        reading
            .jacobian
            .push(jacobian_asymmetry(probe, force, &weights, start, 0.02));
        reading.velocity =
            reading
                .velocity
                .max(velocity_read(probe, force, start, 30.0, k as u64 + 1));
    }
    reading.motion =
        persistent_motion(probe, isolated(make(), term), &starts[0], 1800, 120).ratio();
    reading
}

impl Reading {
    /// Whether `candidate` calls the term conservative.
    pub fn conservative(&self, candidate: Candidate) -> bool {
        match candidate {
            Candidate::LoopWork => median(&self.loop_work) <= tolerance::LOOP_WORK,
            Candidate::Jacobian => median(&self.jacobian) <= tolerance::JACOBIAN,
            Candidate::Motion => self.motion <= tolerance::MOTION,
        }
    }

    /// Whether the readings agree with the declared class, conservativeness
    /// judged by `candidate`. A K term writes state and is not read here.
    pub fn agrees(&self, candidate: Candidate) -> Result<(), String> {
        use crate::Class;
        let name = self.term.name;
        let fail = |what: &str| Err(format!("{name} ({:?}): {what}: {self:?}", self.term.class));
        let reads_velocity = self.velocity > tolerance::VELOCITY;
        let balanced = self.balance.is_none_or(|b| b <= tolerance::BALANCE);
        match self.term.class {
            Class::E | Class::Em => {
                if self.rise.is_none_or(|r| r > tolerance::RISE) {
                    return fail("its energy rose, or it has none");
                }
                if self.gradient.is_none_or(|g| g > tolerance::GRADIENT) {
                    return fail("its forces are not its energy's gradient");
                }
                if !balanced || reads_velocity || !self.conservative(candidate) {
                    return fail("not conservative and balanced");
                }
            },
            Class::H => {
                if !balanced || reads_velocity || !self.conservative(candidate) {
                    return fail("not conservative and balanced at a held clock");
                }
            },
            Class::N => {
                if !reads_velocity && self.conservative(candidate) {
                    return fail("reads as conservative");
                }
            },
            Class::K => return fail("a K term writes state; it is not read by force"),
        }
        Ok(())
    }
}
