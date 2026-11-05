// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

use super::*;

/// One leaf's derived fields from its built forces, with its overrides
/// checked against its term names.
pub(super) fn leaf(
    path: &str,
    kind: LeafKind,
    forces: &[Box<dyn Force>],
    overrides: &BTreeMap<String, f64>,
) -> Result<DerivedLeaf, SpecError> {
    let mut terms = Vec::new();
    for force in forces {
        for (i, term) in force.terms().into_iter().enumerate() {
            terms.push(DerivedTerm {
                reading: match scale::family(&term) {
                    Some(family) => WeightReading::Absolute(family),
                    None => WeightReading::Multiplier,
                },
                resident: force.resident(i),
                weight: None,
                term,
            });
        }
    }
    let name = match &kind {
        LeafKind::Preset(id) => id.clone(),
        LeafKind::Raw(kind) => (*kind).to_string(),
    };
    for (term_name, weight) in overrides {
        check_number(Some(*weight), path, "term weight")?;
        let mut named = terms.iter_mut().filter(|t| t.term.name == term_name);
        let Some(found) = named.next() else {
            return Err(SpecError::UnknownTerm {
                path: path.into(),
                kind: name,
                name: term_name.clone(),
            });
        };
        found.weight = Some(*weight);
        if named.next().is_some() {
            return Err(invalid(
                path,
                &format!("{name} has two terms named {term_name}"),
            ));
        }
    }
    let declared: Vec<Term> = terms.iter().map(|t| t.term).collect();
    Ok(DerivedLeaf {
        path: path.into(),
        leaf: kind,
        currency: compose::law_currency(&declared),
        terms,
    })
}

pub(super) fn reserved(rung: &Option<String>, path: &str) -> Result<(), SpecError> {
    match rung {
        Some(_) => Err(SpecError::Reserved {
            path: path.into(),
            field: "rung",
        }),
        None => Ok(()),
    }
}

/// A number the spec gives is finite, and a weight, an override, a pull or
/// a damping is not negative: a negative one would otherwise be clamped,
/// silently.
pub(super) fn check_number(value: Option<f64>, path: &str, field: &str) -> Result<(), SpecError> {
    match value {
        Some(x) if !x.is_finite() || x < 0.0 => Err(invalid(
            path,
            &format!("{field} is a finite number, not negative; this one is {x}"),
        )),
        _ => Ok(()),
    }
}

pub(super) fn invalid(path: &str, reason: &str) -> SpecError {
    SpecError::Invalid {
        path: path.into(),
        reason: reason.into(),
    }
}

pub(super) fn by_currency(path: &str, reason: &'static str) -> SpecError {
    SpecError::Currency {
        path: path.into(),
        reason,
    }
}
