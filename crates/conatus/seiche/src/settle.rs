// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! Bounded specification execution (F195). Hosts resolve catalog terms and
//! channels, and supply their existing fixed-step controller. This loop owns
//! the step count and result ordering; it reads no clock, draws no frames and
//! makes no claim that reaching a bound means the layout rested.

use euclid::default::Point2D;

use crate::{NodeKey, spec::DynamicsSpec};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SettleEnd {
    Rested,
    LawFinished,
    StepLimit,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SettleReport {
    pub steps: u32,
    pub end: SettleEnd,
    /// Stable key order, in the arrangement's local coordinates.
    pub positions: Vec<(NodeKey, Point2D<f32>)>,
}

/// The binding host's already resolved inputs and controller. `step_fixed`
/// advances exactly one nominal simulation tick, including role returns and
/// schedule accounting. Its stopping decision must not depend on host time.
pub trait DynamicsRunner {
    type Error;

    fn prepare(&mut self, spec: &DynamicsSpec) -> Result<(), Self::Error>;
    fn step_fixed(&mut self) -> Result<Option<SettleEnd>, Self::Error>;
    fn positions(&self) -> Vec<(NodeKey, Point2D<f32>)>;
}

#[derive(Debug)]
pub enum SettleError<E> {
    Binding(E),
    NonFinite(NodeKey),
    Duplicate(NodeKey),
    InvalidStop,
}

impl<E: std::fmt::Display> std::fmt::Display for SettleError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Binding(error) => error.fmt(f),
            Self::NonFinite(key) => write!(f, "settle.positions[{key:?}]: non-finite position"),
            Self::Duplicate(key) => write!(f, "settle.positions[{key:?}]: duplicate body"),
            Self::InvalidStop => write!(f, "settle: the binding cannot report a step-limit stop"),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for SettleError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Binding(error) => Some(error),
            _ => None,
        }
    }
}

/// Execute at most `bound` fixed steps. Zero returns the bound input positions
/// without advancing. The caller supplies the bound; no unmeasured default is
/// installed. A living law's bounded snapshot remains usable by a live host.
pub fn settle<R: DynamicsRunner>(
    spec: &DynamicsSpec,
    inputs: &mut R,
    bound: u32,
) -> Result<SettleReport, SettleError<R::Error>> {
    inputs.prepare(spec).map_err(SettleError::Binding)?;
    let mut steps = 0;
    let mut end = SettleEnd::StepLimit;
    while steps < bound {
        let stop = inputs.step_fixed().map_err(SettleError::Binding)?;
        steps += 1;
        if let Some(stop) = stop {
            if stop == SettleEnd::StepLimit {
                return Err(SettleError::InvalidStop);
            }
            end = stop;
            break;
        }
    }
    let mut positions = inputs.positions();
    positions.sort_by_key(|(key, _)| *key);
    for (key, at) in &positions {
        if !at.x.is_finite() || !at.y.is_finite() {
            return Err(SettleError::NonFinite(*key));
        }
    }
    for pair in positions.windows(2) {
        if pair[0].0 == pair[1].0 {
            return Err(SettleError::Duplicate(pair[0].0));
        }
    }
    Ok(SettleReport {
        steps,
        end,
        positions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Probe {
        ticks: u32,
        done: Option<u32>,
    }
    impl DynamicsRunner for Probe {
        type Error = String;
        fn prepare(&mut self, _: &DynamicsSpec) -> Result<(), String> {
            self.ticks = 0;
            Ok(())
        }
        fn step_fixed(&mut self) -> Result<Option<SettleEnd>, String> {
            self.ticks += 1;
            Ok((self.done == Some(self.ticks)).then_some(SettleEnd::LawFinished))
        }
        fn positions(&self) -> Vec<(NodeKey, Point2D<f32>)> {
            vec![
                (NodeKey::new(2), Point2D::new(self.ticks as f32, 1.0)),
                (NodeKey::new(0), Point2D::new(0.0, 0.0)),
            ]
        }
    }

    #[test]
    fn a_continuous_run_is_bounded_and_a_law_stop_is_distinct() {
        let spec = DynamicsSpec::new(crate::spec::Node::preset("test"));
        let mut probe = Probe {
            ticks: 0,
            done: None,
        };
        let zero = settle(&spec, &mut probe, 0).unwrap();
        assert_eq!(zero.steps, 0);
        assert_eq!(zero.end, SettleEnd::StepLimit);
        let capped = settle(&spec, &mut probe, 7).unwrap();
        assert_eq!(capped.steps, 7);
        assert_eq!(capped.end, SettleEnd::StepLimit);
        assert_eq!(capped.positions[0].0, NodeKey::new(0));
        probe.done = Some(3);
        let done = settle(&spec, &mut probe, 7).unwrap();
        assert_eq!(done.steps, 3);
        assert_eq!(done.end, SettleEnd::LawFinished);
        assert_ne!(done.positions, capped.positions);
    }
}
