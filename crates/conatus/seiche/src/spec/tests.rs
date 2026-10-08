// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! The dynamics spec's receipts (G4a): every raw kind against its force,
//! derive's refusals each with its positive control, and (with `serde`) the
//! wire's version, strictness and round trips.

use super::raw::*;
use super::*;
use crate::{
    Anneal, BarnesHutRepulsion, Boids, Boundary, Class, CounterDamping, Density, DepthGravity,
    DomainCluster, EdgeSpring, Gravity, GravityLocus, GridSnap, Hold, HubGravity, Kuramoto,
    LinLogForce, MagneticSpring, NodeExclusion, ParticleLife, StressSpring,
};

/// A small catalog: two laws and two overlays, enough for derive's paths.
struct Catalog;

impl SpecCatalog for Catalog {
    fn preset(&self, id: &str, at: PresetAt) -> Option<Vec<Box<dyn Force>>> {
        match (at, id) {
            (PresetAt::Law, "springs") => Some(vec![
                Box::new(NodeExclusion::default()),
                Box::new(EdgeSpring::default()),
                Box::new(Boundary::default()),
            ]),
            (PresetAt::Law, "still") => Some(vec![Box::new(Hold)]),
            (PresetAt::Overlay, "centre") => Some(vec![Box::new(GravityLocus::at((0.0, 0.0)))]),
            (PresetAt::Overlay, "grid") => Some(vec![Box::new(GridSnap::default())]),
            _ => None,
        }
    }

    fn admits(&self, law: &str, overlay: &str) -> Option<Admission> {
        (law == "still" && overlay == "grid").then_some(Admission::Refuse("still refuses grid"))
    }
}

fn with_overlays(mut node: Node, overlays: Vec<Node>) -> Node {
    *node.overlays_mut() = overlays;
    node
}

fn weighted(mut node: Node, w: f64) -> Node {
    match &mut node {
        Node::Preset { weight, .. }
        | Node::Raw { weight, .. }
        | Node::Mix { weight, .. }
        | Node::Grouped { weight, .. }
        | Node::Schedule { weight, .. } => *weight = w,
    }
    node
}

fn mix(parts: Vec<Node>) -> Node {
    Node::Mix {
        parts,
        weight: 1.0,
        overlays: Vec::new(),
    }
}

fn grouped(outer: Node, inner: Node) -> Node {
    Node::Grouped {
        partition: "groups.meaning".into(),
        outer: Box::new(outer),
        inner: Box::new(inner),
        weight: 1.0,
        overlays: Vec::new(),
    }
}

fn schedule(nodes: Vec<Node>) -> Node {
    Node::Schedule {
        stages: nodes
            .into_iter()
            .map(|node| Stage {
                node,
                stop: Stop::Rest,
                capture: None,
            })
            .collect(),
        weight: 1.0,
        overlays: Vec::new(),
    }
}

fn derive(root: Node) -> Result<Derived, SpecError> {
    DynamicsSpec::new(root).derive(&Catalog)
}

/// Every raw kind with its required fields filled, beside the force the
/// constructor builds with the same values: the declared rows must match.
fn filled() -> Vec<(RawTerm, Box<dyn Force>)> {
    let ch = || Some("test.channel".to_string());
    vec![
        (
            RawTerm::NodeExclusion(Default::default()),
            Box::new(NodeExclusion::default()),
        ),
        (
            RawTerm::EdgeSpring(Default::default()),
            Box::new(EdgeSpring::default()),
        ),
        (
            RawTerm::Boundary(Default::default()),
            Box::new(Boundary::default()),
        ),
        (
            RawTerm::BarnesHutRepulsion(Default::default()),
            Box::new(BarnesHutRepulsion::default()),
        ),
        (
            RawTerm::StressSpring(StressSpringParams {
                pairs: ch(),
                unit_length: Some(170.0),
                ..Default::default()
            }),
            Box::new(StressSpring::from_weighted_distances(
                std::iter::empty(),
                170.0,
            )),
        ),
        (
            RawTerm::Linlog(Default::default()),
            Box::new(LinLogForce::default()),
        ),
        (
            RawTerm::Gravity(GravityParams {
                masses: ch(),
                counter_damping: Some(CounterDampingParam::Tangential),
                ..Default::default()
            }),
            Box::new(Gravity::new(std::iter::empty(), CounterDamping::Tangential)),
        ),
        (
            RawTerm::ParticleLife(ParticleLifeParams {
                kinds: ch(),
                ..Default::default()
            }),
            Box::new(ParticleLife::seeded(std::iter::empty(), 2, DEFAULT_SEED)),
        ),
        (
            RawTerm::Boids(Default::default()),
            Box::new(Boids::default()),
        ),
        (
            RawTerm::Kuramoto(Default::default()),
            Box::new(Kuramoto::new(std::iter::empty())),
        ),
        (
            RawTerm::MagneticSpring(Default::default()),
            Box::new(MagneticSpring::default()),
        ),
        (
            RawTerm::Anneal(Default::default()),
            Box::new(Anneal::seeded(DEFAULT_SEED)),
        ),
        (RawTerm::Hold(HoldParams {}), Box::new(Hold)),
        (
            RawTerm::Density(DensityParams {
                masses: ch(),
                resolution: Some(64),
                ..Default::default()
            }),
            Box::new(Density::new(std::iter::empty(), 64)),
        ),
        (
            RawTerm::DegreeRepulsion(Default::default()),
            Box::new(crate::DegreeRepulsion::default()),
        ),
        (
            RawTerm::DomainCluster(DomainClusterParams {
                groups: ch(),
                ..Default::default()
            }),
            Box::new(DomainCluster::new(std::iter::empty())),
        ),
        (
            RawTerm::HubGravity(Default::default()),
            Box::new(HubGravity::default()),
        ),
        (
            RawTerm::DepthGravity(DepthGravityParams {
                depths: ch(),
                ..Default::default()
            }),
            Box::new(DepthGravity::new(std::iter::empty())),
        ),
        (
            RawTerm::GridSnap(Default::default()),
            Box::new(GridSnap::default()),
        ),
        (
            RawTerm::GravityLocus(GravityLocusParams {
                target: Some((0.0, 0.0)),
                ..Default::default()
            }),
            Box::new(GravityLocus::at((0.0, 0.0))),
        ),
    ]
}

#[test]
fn every_raw_kind_declares_what_its_force_declares() {
    let filled = filled();
    assert_eq!(filled.len(), 20, "one raw kind per force type (F107)");
    for ((raw, force), kind) in filled.iter().zip(RawTerm::KINDS) {
        assert_eq!(raw.kind(), kind, "KINDS follows declaration order");
        let built = raw
            .build("root", DEFAULT_SEED)
            .expect("filled, so it builds");
        assert_eq!(built.terms(), force.terms(), "{kind}: the declared rows");
        let derived = DynamicsSpec::new(Node::raw(raw.clone()))
            .derive(&NoPresets)
            .err();
        assert_eq!(
            derived,
            Some(SpecError::Unrunnable {
                path: "root".into(),
                reason: RAW_UNBOUND
            }),
            "{kind}: derives, and is refused only for want of a runner"
        );
    }
    let mut kinds = RawTerm::KINDS.to_vec();
    kinds.sort();
    kinds.dedup();
    assert_eq!(kinds.len(), 20, "kind ids are distinct");
    for (default, kind) in RawTerm::defaults().iter().zip(RawTerm::KINDS) {
        assert_eq!(default.kind(), kind);
    }
}

#[test]
fn omitted_parameters_take_the_constructors_defaults() {
    assert_eq!(
        NodeExclusionParams::default(),
        NodeExclusionParams::of(&NodeExclusion::default())
    );
    assert_eq!(NodeExclusionParams::default().strength, 220_000.0);
    assert_eq!(StressSpringParams::default().stiffness, 40.0);
    assert_eq!(StressSpringParams::default().unit_length, None);
    assert_eq!(GravityParams::default().strength, 9_000.0);
    assert_eq!(GravityParams::default().counter_damping, None);
    assert_eq!(DensityParams::default().stop, DensityStopParam::Cap);
    assert_eq!(DensityParams::default().resolution, None);
    assert_eq!(KuramotoParams::default().default_radius, 200.0);
    assert_eq!(AnnealParams::default().cooling, f64::from(0.995f32));
    assert_eq!(GravityLocusParams::default().target, None);
    // A parameter the spec gives is the force's parameter.
    let p = NodeExclusionParams {
        cutoff: 144.0,
        ..Default::default()
    };
    let built = RawTerm::NodeExclusion(p.clone())
        .build("root", 0)
        .unwrap()
        .terms();
    assert_eq!(
        built[0].topology,
        crate::Topology::AllPairs {
            cutoff: Some(144.0)
        }
    );
}

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
fn a_seeded_kind_matrix_derives_n_and_a_symmetric_one_e() {
    let kinds = Some("groups.site".to_string());
    let class = |matrix| {
        RawTerm::ParticleLife(ParticleLifeParams {
            kinds: kinds.clone(),
            matrix,
            ..Default::default()
        })
        .build("root", DEFAULT_SEED)
        .map(|f| f.terms()[0].class)
    };
    assert_eq!(class(None), Ok(Class::N));
    assert_eq!(class(Some(vec![1.0, -0.5, -0.5, 1.0])), Ok(Class::E));
    assert_eq!(class(Some(vec![1.0, 0.5, -0.5, 1.0])), Ok(Class::N));
    assert!(matches!(
        class(Some(vec![1.0, 0.5, 0.5])),
        Err(SpecError::Invalid { .. })
    ));
}

#[test]
fn the_shapes_the_canvas_runs_derive() {
    let law = || Node::preset("springs");
    for root in [
        with_overlays(law(), vec![Node::preset("centre")]),
        with_overlays(
            mix(vec![weighted(law(), 0.25), weighted(law(), 2.0)]),
            vec![Node::preset("grid")],
        ),
        grouped(weighted(law(), 16.0), law()),
        schedule(vec![
            with_overlays(law(), vec![Node::preset("centre")]),
            mix(vec![law(), law()]),
            grouped(law(), law()),
            Node::preset("still"),
        ]),
    ] {
        let derived = derive(root.clone()).unwrap_or_else(|e| panic!("{e}: {root:?}"));
        assert!(derived.leaves.iter().all(|l| !l.terms.is_empty()));
    }
}

#[test]
fn derived_fields_come_from_the_declarations() {
    let d = derive(with_overlays(
        Node::preset("springs"),
        vec![Node::preset("centre")],
    ))
    .unwrap();
    assert_eq!(d.depth, 2);
    let paths: Vec<_> = d.leaves.iter().map(|l| l.path.as_str()).collect();
    assert_eq!(paths, ["root", "root.overlays[0]"]);
    let springs: Vec<Term> = Catalog
        .preset("springs", PresetAt::Law)
        .unwrap()
        .iter()
        .flat_map(|f| f.terms())
        .collect();
    let derived: Vec<Term> = d.leaves[0].terms.iter().map(|t| t.term).collect();
    assert_eq!(derived, springs);
    assert_eq!(d.leaves[0].currency, Currency::Force);
    let still = derive(Node::preset("still")).unwrap();
    assert_eq!(still.leaves[0].currency, Currency::Kinematic);
}

#[test]
fn a_term_override_reads_absolute_with_a_reference_and_a_multiplier_without() {
    let mut terms = BTreeMap::new();
    terms.insert("exclusion".to_string(), 3.0);
    terms.insert("edge spring".to_string(), 2.0);
    let node = Node::Preset {
        id: "springs".into(),
        weight: 1.0,
        terms: terms.clone(),
        rung: None,
        overlays: Vec::new(),
    };
    // At the root a per-term weight has no runner yet; inside a schedule
    // likewise. Its reading is derived all the same.
    let err = derive(node.clone()).unwrap_err();
    assert_eq!(
        err,
        SpecError::Unrunnable {
            path: "root".into(),
            reason: OVERRIDE_UNBOUND
        }
    );
    let mut leaves = Vec::new();
    node.declare("root", PresetAt::Law, 0, &Catalog, &mut leaves)
        .unwrap();
    let reading = |name: &str| {
        leaves[0]
            .terms
            .iter()
            .find(|t| t.term.name == name)
            .map(|t| (t.reading, t.weight))
            .unwrap()
    };
    assert_eq!(
        reading("exclusion"),
        (WeightReading::Absolute(Family::Contact), Some(3.0))
    );
    assert_eq!(
        reading("edge spring"),
        (WeightReading::Absolute(Family::Stretch), Some(2.0))
    );
    let mut leaves = Vec::new();
    Node::raw(RawTerm::Boids(Default::default()))
        .declare("root", PresetAt::Law, 0, &Catalog, &mut leaves)
        .unwrap();
    let alignment = leaves[0]
        .terms
        .iter()
        .find(|t| t.term.name == "alignment")
        .unwrap();
    assert_eq!(alignment.reading, WeightReading::Multiplier);
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

/// A chain `depth` nodes deep: nested schedules, the deepest wire per level,
/// ending in a law.
pub(crate) fn chain(depth: usize) -> Node {
    let mut node = Node::preset("springs");
    for _ in 1..depth {
        node = schedule(vec![node]);
    }
    node
}

#[test]
fn depth_32_is_taken_and_33_refused() {
    assert_eq!(chain(32).depth(), 32);
    // Nested schedules are unrunnable, so the cap is read before them.
    assert!(matches!(
        DynamicsSpec::new(chain(32)).derive(&Catalog),
        Err(SpecError::Unrunnable {
            reason: SCHEDULE_NESTED,
            ..
        })
    ));
    assert_eq!(
        DynamicsSpec::new(chain(33)).derive(&Catalog),
        Err(SpecError::Depth { depth: 33, max: 32 })
    );
    // A runnable shape at the cap: overlays count as children.
    let mut deep = Node::preset("centre");
    for _ in 1..31 {
        deep = with_overlays(Node::preset("centre"), vec![deep]);
    }
    let root = with_overlays(Node::preset("springs"), vec![deep]);
    assert_eq!(root.depth(), 32);
    assert!(matches!(
        DynamicsSpec::new(root).derive(&Catalog),
        Err(SpecError::Unrunnable {
            reason: OVERLAYS_UNBOUND,
            ..
        })
    ));
}

#[test]
fn a_target_keys_group_roles_by_a_groups_channel() {
    let mut spec = DynamicsSpec::new(Node::preset("springs"));
    spec.target = Some(Target {
        arrangement: "spiral.recency".into(),
        anchored_pull: 12.0,
        default_role: Role::Seeded,
        groups: Some(GroupRoles {
            channel: "kind.site".into(),
            roles: BTreeMap::new(),
        }),
        items: BTreeMap::new(),
    });
    assert!(matches!(
        spec.derive(&Catalog),
        Err(SpecError::Invalid { path, .. }) if path == "target.groups"
    ));
    if let Some(GroupRoles { channel, .. }) = spec.target.as_mut().unwrap().groups.as_mut() {
        *channel = "groups.site".into();
    }
    assert!(spec.derive(&Catalog).is_ok());
}

#[cfg(feature = "serde")]
mod wire;
