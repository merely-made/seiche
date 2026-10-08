// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! The spec's wire (F98, F104): JSON and TOML round trips, the version
//! read in `sceno::Score`'s pattern, and unknown names refused.

use super::*;

/// A spec using every part of the type: a schedule whose stages hold a
/// mix and a grouping, raw terms with parameters and overrides, channels,
/// a target and bars.
pub(crate) fn rich() -> DynamicsSpec {
    let mut overrides = BTreeMap::new();
    overrides.insert("exclusion".to_string(), 2.5);
    let raw_gravity = Node::Raw {
        term: RawTerm::Gravity(GravityParams {
            masses: Some("mass.pagerank".into()),
            counter_damping: Some(CounterDampingParam::Tangential),
            strength: 4_500.0,
            ..Default::default()
        }),
        weight: 0.5,
        terms: BTreeMap::new(),
        rung: None,
        overlays: Vec::new(),
    };
    let kinds = Node::raw(RawTerm::ParticleLife(ParticleLifeParams {
        kinds: Some("groups.meaning".into()),
        matrix: Some(vec![1.0, -0.25, 0.75, 1.0]),
        ..Default::default()
    }));
    let stage_mix = with_overlays(
        mix(vec![
            Node::Preset {
                id: "springs".into(),
                weight: 0.75,
                terms: overrides,
                rung: None,
                overlays: Vec::new(),
            },
            raw_gravity,
        ]),
        vec![Node::preset("centre")],
    );
    let stage_grouped = grouped(weighted(Node::preset("springs"), 16.0), mix(vec![kinds]));
    let root = Node::Schedule {
        stages: vec![
            Stage {
                node: stage_mix,
                stop: Stop::Frames(240),
                capture: Some(Role::Anchored),
            },
            Stage {
                node: stage_grouped,
                stop: Stop::LawDone,
                capture: None,
            },
            Stage {
                node: Node::raw(RawTerm::Density(DensityParams {
                    masses: Some("mass.degree".into()),
                    resolution: Some(64),
                    stop: DensityStopParam::Shift(0.05),
                    renew: Some(0.5),
                    ..Default::default()
                })),
                stop: Stop::Rest,
                capture: None,
            },
        ],
        weight: 1.0,
        overlays: Vec::new(),
    };
    let mut spec = DynamicsSpec::new(root);
    spec.seed = 42;
    spec.channels.insert("kind".into(), "groups.site".into());
    spec.channels.insert("mass".into(), "mass.pagerank".into());
    spec.realization = Realization::Integrate { damping: Some(0.7) };
    let mut items = BTreeMap::new();
    items.insert(
        "00000000-0000-0000-0000-000000000007".to_string(),
        Role::Pinned,
    );
    let mut group_roles = BTreeMap::new();
    group_roles.insert("example.test".to_string(), Role::Anchored);
    spec.target = Some(Target {
        arrangement: "spiral.recency".into(),
        anchored_pull: 0.5,
        default_role: Role::Seeded,
        groups: Some(GroupRoles {
            channel: "groups.site".into(),
            roles: group_roles,
        }),
        items,
    });
    spec.bars = vec![
        Bar {
            observable: Observable::MassAreaRank,
            at_least: Some(0.8),
            at_most: None,
        },
        Bar {
            observable: Observable::Overlaps,
            at_least: None,
            at_most: Some(4.0),
        },
    ];
    spec
}

#[test]
fn json_round_trips_byte_for_byte_and_writes_no_derived_field() {
    let spec = rich();
    let json = serde_json::to_string(&spec).unwrap();
    let back: DynamicsSpec = serde_json::from_str(&json).unwrap();
    assert_eq!(back, spec);
    assert_eq!(serde_json::to_string(&back).unwrap(), json, "byte for byte");
    for derived in ["currency", "\"class\"", "metric", "\"reading\""] {
        assert!(
            !json.contains(derived),
            "{derived} is derived, never written"
        );
    }
    // Through a JSON value, as the scene's facet carries it.
    let value = serde_json::to_value(&spec).unwrap();
    let back: DynamicsSpec = serde_json::from_value(value.clone()).unwrap();
    assert_eq!(serde_json::to_value(&back).unwrap(), value);
}

#[test]
fn toml_round_trips_a_tree_four_deep() {
    let spec = rich();
    assert!(spec.depth() >= 4, "schedule, grouping, mix, term");
    let text = toml::to_string(&spec).unwrap();
    let back: DynamicsSpec = toml::from_str(&text).unwrap();
    assert_eq!(back, spec, "{text}");
    assert_eq!(toml::to_string(&back).unwrap(), text, "byte for byte");
}

#[test]
fn a_hand_written_toml_recipe_reads_with_defaults() {
    let recipe = r#"
version = 1

[root]
node = "preset"
id = "springs"

[[root.overlays]]
node = "raw"

[root.overlays.term.gravity-locus]
target = [0.0, 0.0]
oscillation = [240.0, 24.0]
"#;
    let spec: DynamicsSpec = toml::from_str(recipe).unwrap();
    assert_eq!(
        spec.seed, DEFAULT_SEED,
        "F101: a spec without one runs on LAW_SEED"
    );
    assert_eq!(spec.realization, Realization::Integrate { damping: None });
    let Node::Raw {
        term: RawTerm::GravityLocus(locus),
        ..
    } = &spec.root.overlays()[0]
    else {
        panic!("{spec:?}");
    };
    assert_eq!(locus.strength, 0.3_f32 as f64, "omitted: the constructor's");
    assert_eq!(locus.oscillation, Some((240.0, 24.0)));
    let tide = locus.clone();
    let mut tide_built = RawTerm::GravityLocus(tide)
        .build("root.overlays[0]", 0)
        .unwrap()
        .terms();
    assert_eq!(tide_built.remove(0).class, Class::H, "a moving target");
}

#[test]
fn a_newer_spec_is_refused_and_an_older_one_stamped() {
    let mut value = serde_json::to_value(DynamicsSpec::new(Node::preset("springs"))).unwrap();
    value["version"] = serde_json::json!(2);
    let err = serde_json::from_value::<DynamicsSpec>(value.clone()).unwrap_err();
    assert!(
        err.to_string()
            .contains("dynamics spec version 2 is newer than this reader's 1"),
        "{err}"
    );
    value["version"] = serde_json::json!(1);
    assert!(serde_json::from_value::<DynamicsSpec>(value.clone()).is_ok());
    value["version"] = serde_json::json!(0);
    let older: DynamicsSpec = serde_json::from_value(value).unwrap();
    assert_eq!(older.version, DYNAMICS_SPEC_VERSION, "stamped on read");
}

#[test]
fn unknown_names_are_refused_everywhere() {
    let base = serde_json::to_value(rich()).unwrap();
    let refused = |edit: &dyn Fn(&mut serde_json::Value), needle: &str| {
        let mut value = base.clone();
        edit(&mut value);
        let err = serde_json::from_value::<DynamicsSpec>(value)
            .expect_err(needle)
            .to_string();
        assert!(err.contains(needle), "{needle}: {err}");
    };
    // The positive control: the unedited spec reads.
    assert!(serde_json::from_value::<DynamicsSpec>(base.clone()).is_ok());
    refused(
        &|v| v["extra"] = serde_json::json!(1),
        "unknown field `extra`",
    );
    refused(
        &|v| v["root"]["stages"][0]["node"]["colour"] = serde_json::json!("red"),
        "unknown field `colour`",
    );
    refused(
        &|v| v["root"]["stages"][0]["stopp"] = serde_json::json!("rest"),
        "unknown field `stopp`",
    );
    refused(
        &|v| {
            v["root"]["stages"][2]["node"]["term"]["density"]["resolutoin"] = serde_json::json!(32)
        },
        "unknown field `resolutoin`",
    );
    refused(
        &|v| {
            let term = v["root"]["stages"][2]["node"]["term"]["density"].take();
            v["root"]["stages"][2]["node"]["term"] = serde_json::json!({ "densty": term });
        },
        "unknown variant `densty`",
    );
    refused(
        &|v| v["root"]["node"] = serde_json::json!("sequence"),
        "unknown variant `sequence`",
    );
    refused(
        &|v| v["target"]["default_role"] = serde_json::json!("tethered"),
        "unknown variant `tethered`",
    );
    refused(
        &|v| v["target"]["arrangment"] = serde_json::json!("grid"),
        "unknown field `arrangment`",
    );
    refused(
        &|v| v["bars"][0]["observable"] = serde_json::json!("beauty"),
        "unknown variant `beauty`",
    );
    refused(
        &|v| v["realization"] = serde_json::json!({ "anneal": {} }),
        "unknown variant `anneal`",
    );
}

#[test]
fn a_depth_32_chain_round_trips_through_json_text() {
    for depth in [32, 33] {
        let spec = DynamicsSpec::new(super::chain(depth));
        let json = serde_json::to_string(&spec).unwrap();
        let nesting = json
            .chars()
            .scan(0i32, |d, c| {
                match c {
                    '{' | '[' => *d += 1,
                    '}' | ']' => *d -= 1,
                    _ => {},
                }
                Some(*d)
            })
            .max()
            .unwrap();
        println!("depth {depth}: JSON nesting {nesting}");
        let back: DynamicsSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(back, spec);
    }
}
