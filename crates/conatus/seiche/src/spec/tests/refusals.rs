// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

use super::*;

#[test]
fn a_required_parameter_or_input_left_out_is_refused_by_name() {
    for (raw, _) in filled() {
        let kind = raw.kind();
        for slot in raw.inputs().into_iter().filter(|s| s.required) {
            let mut bare = raw.clone();
            match &mut bare {
                RawTerm::StressSpring(p) => p.pairs = None,
                RawTerm::Gravity(p) => p.masses = None,
                RawTerm::ParticleLife(p) => p.kinds = None,
                RawTerm::Density(p) => p.masses = None,
                RawTerm::DomainCluster(p) => p.groups = None,
                RawTerm::DepthGravity(p) => p.depths = None,
                _ => unreachable!("{kind} has no required input"),
            }
            assert_eq!(
                bare.build("root.parts[0]", 0).err(),
                Some(SpecError::Missing {
                    path: "root.parts[0]".into(),
                    field: slot.slot,
                }),
                "{kind}"
            );
        }
    }
    for (raw, field) in [
        (RawTerm::StressSpring(Default::default()), "unit_length"),
        (
            RawTerm::Gravity(GravityParams {
                masses: Some("mass.degree".into()),
                ..Default::default()
            }),
            "counter_damping",
        ),
        (
            RawTerm::Density(DensityParams {
                masses: Some("mass.degree".into()),
                ..Default::default()
            }),
            "resolution",
        ),
        (RawTerm::GravityLocus(Default::default()), "target"),
    ] {
        assert!(
            matches!(raw.build("root", 0), Err(SpecError::Missing { field: f, .. }) if f == field || f == "pairs"),
            "{}: {field}",
            raw.kind()
        );
    }
    let optional: Vec<_> = RawTerm::defaults()
        .iter()
        .flat_map(|r| r.inputs())
        .filter(|s| !s.required)
        .map(|s| s.slot)
        .collect();
    assert_eq!(optional, ["radii", "masses", "masses"]);
}

#[test]
fn unknown_presets_and_terms_fail_by_path() {
    let unknown = derive(schedule(vec![
        Node::preset("springs"),
        mix(vec![Node::preset("springs"), Node::preset("orbitz")]),
    ]))
    .unwrap_err();
    assert_eq!(
        unknown,
        SpecError::UnknownPreset {
            path: "root.stages[1].node.parts[1]".into(),
            id: "orbitz".into(),
            at: PresetAt::Law,
        }
    );
    // A law id where an overlay goes is unknown there.
    let misplaced = derive(with_overlays(
        Node::preset("springs"),
        vec![Node::preset("still")],
    ))
    .unwrap_err();
    assert_eq!(
        misplaced,
        SpecError::UnknownPreset {
            path: "root.overlays[0]".into(),
            id: "still".into(),
            at: PresetAt::Overlay,
        }
    );
    let mut terms = BTreeMap::new();
    terms.insert("exclusoin".to_string(), 1.0);
    let term = derive(Node::Preset {
        id: "springs".into(),
        weight: 1.0,
        terms,
        rung: None,
        overlays: Vec::new(),
    })
    .unwrap_err();
    assert_eq!(
        term,
        SpecError::UnknownTerm {
            path: "root".into(),
            kind: "springs".into(),
            name: "exclusoin".into(),
        }
    );
    assert_eq!(
        unknown.to_string(),
        "at root.stages[1].node.parts[1]: unknown law preset orbitz"
    );
}

#[test]
fn a_rung_is_reserved() {
    let node = Node::Preset {
        id: "springs".into(),
        weight: 1.0,
        terms: BTreeMap::new(),
        rung: Some("cpu-exact".into()),
        overlays: Vec::new(),
    };
    assert_eq!(
        derive(mix(vec![Node::preset("springs"), node])).unwrap_err(),
        SpecError::Reserved {
            path: "root.parts[1]".into(),
            field: "rung"
        }
    );
}

#[test]
fn a_law_that_writes_state_is_refused_where_weights_scale_force() {
    for (root, path) in [
        (
            mix(vec![Node::preset("springs"), Node::preset("still")]),
            "root.parts[1]",
        ),
        (
            grouped(Node::preset("still"), Node::preset("springs")),
            "root.outer",
        ),
        (
            grouped(Node::preset("springs"), Node::preset("still")),
            "root.inner",
        ),
    ] {
        assert_eq!(
            derive(root).unwrap_err(),
            SpecError::Currency {
                path: path.into(),
                reason: compose::UNWEIGHTED
            }
        );
    }
    // The control: Still alone, and Still as a stage, are runnable.
    assert!(derive(Node::preset("still")).is_ok());
    // An overlay the catalog refuses on a law is refused, never dropped.
    assert_eq!(
        derive(with_overlays(
            Node::preset("still"),
            vec![Node::preset("grid")]
        ))
        .unwrap_err(),
        SpecError::Currency {
            path: "root.overlays[0]".into(),
            reason: "still refuses grid"
        }
    );
    // And an overlay that writes state on a law that does: two writers.
    assert_eq!(
        derive(with_overlays(
            Node::preset("still"),
            vec![Node::raw(RawTerm::Hold(HoldParams {}))]
        ))
        .unwrap_err(),
        SpecError::Currency {
            path: "root.overlays[0]".into(),
            reason: compose::TWO_WRITERS
        }
    );
}

#[test]
fn shapes_the_canvas_cannot_run_are_refused_by_path() {
    let law = || Node::preset("springs");
    let raw = || Node::raw(RawTerm::Boundary(Default::default()));
    let cases: Vec<(Node, &str, &str)> = vec![
        (raw(), "root", RAW_UNBOUND),
        (mix(vec![law(), raw()]), "root.parts[1]", RAW_UNBOUND),
        (
            mix(vec![law(), mix(vec![law()])]),
            "root.parts[1]",
            NESTED_UNBOUND,
        ),
        (
            mix(vec![grouped(law(), law())]),
            "root.parts[0]",
            NESTED_UNBOUND,
        ),
        (
            grouped(law(), mix(vec![law()])),
            "root.inner",
            NESTED_UNBOUND,
        ),
        (
            grouped(law(), grouped(law(), law())),
            "root.inner",
            NESTED_UNBOUND,
        ),
        (
            mix(vec![schedule(vec![law()])]),
            "root.parts[0]",
            SCHEDULE_NESTED,
        ),
        (
            schedule(vec![schedule(vec![law()])]),
            "root.stages[0].node",
            SCHEDULE_NESTED,
        ),
        (
            mix(vec![with_overlays(law(), vec![Node::preset("centre")])]),
            "root.parts[0]",
            OVERLAYS_UNBOUND,
        ),
        (
            with_overlays(schedule(vec![law()]), vec![Node::preset("centre")]),
            "root",
            SCHEDULE_OVERLAYS,
        ),
        (
            with_overlays(law(), vec![mix(vec![law()])]),
            "root.overlays[0]",
            OVERLAY_UNBOUND,
        ),
        (weighted(law(), 2.0), "root", WEIGHT_UNBOUND),
        (
            grouped(law(), weighted(law(), 2.0)),
            "root.inner",
            WEIGHT_UNBOUND,
        ),
    ];
    for (root, path, reason) in cases {
        assert_eq!(
            derive(root.clone()).unwrap_err(),
            SpecError::Unrunnable {
                path: path.into(),
                reason
            },
            "{root:?}"
        );
    }
}

#[test]
fn numbers_out_of_range_are_refused_not_clamped() {
    assert!(matches!(
        derive(mix(vec![weighted(Node::preset("springs"), -1.0)])),
        Err(SpecError::Invalid { path, .. }) if path == "root.parts[0]"
    ));
    assert!(matches!(
        derive(mix(vec![weighted(Node::preset("springs"), f64::NAN)])),
        Err(SpecError::Invalid { .. })
    ));
    assert!(matches!(
        derive(mix(Vec::new())),
        Err(SpecError::Invalid { .. })
    ));
    let mut spec = DynamicsSpec::new(Node::preset("springs"));
    spec.realization = Realization::Integrate {
        damping: Some(-0.5),
    };
    assert!(matches!(
        spec.derive(&Catalog),
        Err(SpecError::Invalid { path, .. }) if path == "realization"
    ));
}
