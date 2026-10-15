// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

use super::*;
use crate::{ScalarField, VectorField};

fn link(id: u64, y: f32, compatibility: Option<&'static str>) -> ResponsiveLink<'static> {
    ResponsiveLink {
        id,
        start_endpoint_id: 1,
        end_endpoint_id: 2,
        start: (-60.0, y),
        end: (60.0, y),
        compatibility,
    }
}

fn config() -> ResponsivePathConfig {
    ResponsivePathConfig {
        control_points: 17,
        iterations: 24,
        smoothing: 0.08,
        ..ResponsivePathConfig::default()
    }
}

fn vertical_excursion(path: &ResponsivePath, baseline: f32) -> f32 {
    path.points
        .iter()
        .map(|point| (point.1 - baseline).abs())
        .fold(0.0, f32::max)
}

#[test]
fn attractor_flow_and_vortex_each_change_the_path_shape() {
    let registry = FieldRegistry::new();
    let attractor = VectorField::gradient_of(ScalarField::gaussian_at(0.0, 45.0, 100.0));
    let flow = VectorField::const_vec(0.0, 1.0);
    let vortex = VectorField::Perp(Box::new(VectorField::gradient_of(
        ScalarField::gaussian_at(0.0, 0.0, 100.0),
    )));
    let input = [link(41, 0.0, None)];
    let base = config();

    for (field, strength, name) in [
        (&attractor, 100.0, "attractor"),
        (&flow, 1.0, "flow"),
        (&vortex, 100.0, "vortex"),
    ] {
        let result =
            evaluate_responsive_paths(&input, &[PathField { field, strength }], &registry, base)
                .unwrap();
        assert!(
            vertical_excursion(&result[0], 0.0) > 0.05,
            "{name} should produce an observable path response"
        );
    }
}

#[test]
fn field_and_compatible_link_responses_are_independently_switchable() {
    let registry = FieldRegistry::new();
    let field = VectorField::const_vec(0.0, 1.0);
    let links = [
        link(10, 0.0, Some("shared")),
        link(11, 40.0, Some("shared")),
    ];

    let field_only = evaluate_responsive_paths(
        &links,
        &[PathField {
            field: &field,
            strength: 1.0,
        }],
        &registry,
        ResponsivePathConfig {
            compatible_attraction: false,
            ..config()
        },
    )
    .unwrap();
    assert!(vertical_excursion(&field_only[0], 0.0) > 0.05);
    assert!(vertical_excursion(&field_only[1], 40.0) > 0.05);

    let link_only = evaluate_responsive_paths(
        &links,
        &[PathField {
            field: &field,
            strength: 1.0,
        }],
        &registry,
        ResponsivePathConfig {
            field_response: false,
            compatible_attraction: true,
            ..config()
        },
    )
    .unwrap();
    assert!(link_only[0].points[8].1 > 0.0);
    assert!(link_only[1].points[8].1 < 40.0);
    assert_eq!(link_only[0].points[0], links[0].start);
    assert_eq!(link_only[0].points[16], links[0].end);
}

#[test]
fn incompatible_links_do_not_attract_and_ids_and_input_order_survive() {
    let registry = FieldRegistry::new();
    let links = [
        link(20, 0.0, Some("compatible")),
        link(21, 40.0, Some("compatible")),
        link(22, 80.0, Some("other")),
    ];
    let settings = ResponsivePathConfig {
        field_response: false,
        compatible_attraction: true,
        ..config()
    };
    let a = evaluate_responsive_paths(&links, &[], &registry, settings).unwrap();
    let b = evaluate_responsive_paths(&links, &[], &registry, settings).unwrap();
    assert_eq!(a, b);
    assert_eq!(
        a.iter().map(|path| path.id).collect::<Vec<_>>(),
        [20, 21, 22]
    );
    assert_eq!(a[2].points[8].1, 80.0);
    assert!(a[0].points[8].1 > 0.0);
    assert!(a[1].points[8].1 < 40.0);
    for (path, source) in a.iter().zip(links) {
        assert_eq!(path.points.first().copied(), Some(source.start));
        assert_eq!(path.points.last().copied(), Some(source.end));
    }
}

#[test]
fn finite_work_and_input_limits_fail_closed() {
    let registry = FieldRegistry::new();
    let too_many_points = ResponsivePathConfig {
        control_points: MAX_CONTROL_POINTS + 1,
        ..config()
    };
    assert_eq!(
        evaluate_responsive_paths(&[link(1, 0.0, None)], &[], &registry, too_many_points),
        Err(ResponsivePathError::InvalidControlPointCount)
    );
    let duplicate_ids = [link(1, 0.0, None), link(1, 10.0, None)];
    assert_eq!(
        evaluate_responsive_paths(&duplicate_ids, &[], &registry, config()),
        Err(ResponsivePathError::DuplicateLinkId)
    );
    let long_key = "x".repeat(MAX_COMPATIBILITY_KEY_BYTES + 1);
    let overlong_key = [ResponsiveLink {
        id: 1,
        start_endpoint_id: 1,
        end_endpoint_id: 2,
        start: (-1.0, 0.0),
        end: (1.0, 0.0),
        compatibility: Some(&long_key),
    }];
    assert_eq!(
        evaluate_responsive_paths(&overlong_key, &[], &registry, config()),
        Err(ResponsivePathError::CompatibilityKeyTooLong)
    );
    let bad_endpoint = [ResponsiveLink {
        id: 1,
        start_endpoint_id: 1,
        end_endpoint_id: 2,
        start: (f32::NAN, 0.0),
        end: (1.0, 0.0),
        compatibility: None,
    }];
    assert_eq!(
        evaluate_responsive_paths(&bad_endpoint, &[], &registry, config()),
        Err(ResponsivePathError::NonFiniteInput)
    );
    let max_work = ResponsivePathConfig {
        control_points: MAX_CONTROL_POINTS,
        iterations: MAX_ITERATIONS,
        compatible_attraction: true,
        ..config()
    };
    let many_links = (0..MAX_LINKS)
        .map(|n| link(n as u64, n as f32, None))
        .collect::<Vec<_>>();
    assert_eq!(
        evaluate_responsive_paths(&many_links, &[], &registry, max_work),
        Err(ResponsivePathError::WorkLimitExceeded)
    );
}

#[test]
fn non_finite_field_outputs_are_refused() {
    let registry = FieldRegistry::new();
    let field = (0..10).fold(VectorField::const_vec(1.0, 0.0), |inner, _| {
        VectorField::ScaleConst(Box::new(inner), 10_000.0)
    });
    assert_eq!(
        evaluate_responsive_paths(
            &[link(1, 0.0, None)],
            &[PathField {
                field: &field,
                strength: 1.0
            }],
            &registry,
            config(),
        ),
        Err(ResponsivePathError::NonFiniteFieldOutput)
    );
}

#[test]
fn recursive_field_samples_are_bounded_and_cycles_refused() {
    let mut registry = FieldRegistry::new();
    let id = FieldId::from_uuid(uuid::Uuid::from_u128(77));
    registry.insert_with_id(id, FieldDef::Vector(VectorField::Sample(id)));
    let field = VectorField::Sample(id);
    assert_eq!(
        evaluate_responsive_paths(
            &[link(1, 0.0, None)],
            &[PathField {
                field: &field,
                strength: 1.0,
            }],
            &registry,
            config(),
        ),
        Err(ResponsivePathError::FieldSampleCycle)
    );
}

#[test]
fn a_cycle_hidden_behind_scalar_scale_is_rejected_before_evaluation() {
    let mut registry = FieldRegistry::new();
    let id = FieldId::from_uuid(uuid::Uuid::from_u128(78));
    registry.insert_with_id(
        id,
        FieldDef::Scalar(ScalarField::Scale(Box::new(ScalarField::Sample(id)), 2.0)),
    );
    let field = VectorField::gradient_of(ScalarField::Sample(id));
    assert_eq!(
        evaluate_responsive_paths(
            &[link(1, 0.0, None)],
            &[PathField {
                field: &field,
                strength: 1.0,
            }],
            &registry,
            config(),
        ),
        Err(ResponsivePathError::FieldSampleCycle)
    );
}

#[test]
fn field_expression_depth_is_limited_before_evaluation() {
    let registry = FieldRegistry::new();
    let field = (0..=MAX_FIELD_AST_DEPTH).fold(VectorField::const_vec(1.0, 0.0), |inner, _| {
        VectorField::ScaleConst(Box::new(inner), 1.0)
    });
    assert_eq!(
        evaluate_responsive_paths(
            &[link(1, 0.0, None)],
            &[PathField {
                field: &field,
                strength: 1.0,
            }],
            &registry,
            config(),
        ),
        Err(ResponsivePathError::FieldExpressionTooLarge)
    );
}

#[test]
fn nested_gradient_expansion_is_included_in_the_work_budget() {
    let registry = FieldRegistry::new();
    let mut scalar = ScalarField::CoordX;
    for _ in 0..6 {
        scalar = ScalarField::Dot(
            Box::new(VectorField::gradient_of(scalar)),
            Box::new(VectorField::const_vec(1.0, 0.0)),
        );
    }
    let field = VectorField::gradient_of(scalar);
    assert_eq!(
        evaluate_responsive_paths(
            &[link(1, 0.0, None)],
            &[PathField {
                field: &field,
                strength: 1.0,
            }],
            &registry,
            config(),
        ),
        Err(ResponsivePathError::WorkLimitExceeded)
    );
}

#[test]
fn disabled_responses_return_the_straight_fixed_endpoint_path() {
    let registry = FieldRegistry::new();
    let unused_field = VectorField::const_vec(f32::INFINITY, 0.0);
    let links = [link(3, 12.0, Some("same")), link(4, 30.0, Some("same"))];
    let paths = evaluate_responsive_paths(
        &links,
        &[PathField {
            field: &unused_field,
            strength: 1.0,
        }],
        &registry,
        ResponsivePathConfig {
            field_response: false,
            compatible_attraction: false,
            smoothing: 0.0,
            ..config()
        },
    )
    .unwrap();
    for (path, link) in paths.iter().zip(links) {
        assert!(path.points.iter().all(|point| point.1 == link.start.1));
    }
}

#[test]
fn zero_response_gains_disable_each_influence_independently() {
    let registry = FieldRegistry::new();
    let field = VectorField::const_vec(0.0, 1.0);
    let links = [link(31, 0.0, Some("same")), link(32, 40.0, Some("same"))];

    let no_field_motion = evaluate_responsive_paths(
        &links,
        &[PathField {
            field: &field,
            strength: 1.0,
        }],
        &registry,
        ResponsivePathConfig {
            field_response: true,
            field_gain: 0.0,
            compatible_attraction: false,
            smoothing: 0.0,
            ..config()
        },
    )
    .unwrap();
    assert_eq!(no_field_motion[0].points[8].1, 0.0);

    let no_link_motion = evaluate_responsive_paths(
        &links,
        &[],
        &registry,
        ResponsivePathConfig {
            field_response: false,
            compatible_attraction: true,
            compatibility_gain: 0.0,
            smoothing: 0.0,
            ..config()
        },
    )
    .unwrap();
    assert_eq!(no_link_motion[0].points[8].1, 0.0);
    assert_eq!(no_link_motion[1].points[8].1, 40.0);
}

#[test]
fn geometric_correspondence_ignores_labels_and_handles_reversed_direction() {
    let registry = FieldRegistry::new();
    let vortex = VectorField::Perp(Box::new(VectorField::gradient_of(
        ScalarField::gaussian_at(0.0, 0.0, 100.0),
    )));
    let links = [
        ResponsiveLink {
            id: 40,
            start_endpoint_id: 101,
            end_endpoint_id: 102,
            start: (-60.0, 0.0),
            end: (60.0, 0.0),
            compatibility: Some("same"),
        },
        ResponsiveLink {
            id: 41,
            start_endpoint_id: 201,
            end_endpoint_id: 202,
            start: (60.0, 40.0),
            end: (-60.0, 40.0),
            compatibility: Some("same"),
        },
    ];
    let settings = ResponsivePathConfig {
        field_response: true,
        compatible_attraction: true,
        ..config()
    };
    let paths = evaluate_responsive_paths(
        &links,
        &[PathField {
            field: &vortex,
            strength: 100.0,
        }],
        &registry,
        settings,
    )
    .unwrap();

    // Identity labels change while the fixed geometry stays the same.
    let relabeled = [
        ResponsiveLink {
            start_endpoint_id: 909,
            end_endpoint_id: 808,
            ..links[0]
        },
        ResponsiveLink {
            start_endpoint_id: 707,
            end_endpoint_id: 606,
            ..links[1]
        },
    ];
    let relabeled_paths = evaluate_responsive_paths(
        &relabeled,
        &[PathField {
            field: &vortex,
            strength: 100.0,
        }],
        &registry,
        settings,
    )
    .unwrap();
    for (path, relabeled_path) in paths.iter().zip(&relabeled_paths) {
        assert_eq!(path.points, relabeled_path.points);
    }

    // The second link is reversed at input, with the same distinct endpoint
    // identities attached to their actual geometric positions.
    let forward = [
        links[0],
        ResponsiveLink {
            start_endpoint_id: 202,
            end_endpoint_id: 201,
            start: (-60.0, 40.0),
            end: (60.0, 40.0),
            ..links[1]
        },
    ];
    let forward_paths = evaluate_responsive_paths(
        &forward,
        &[PathField {
            field: &vortex,
            strength: 100.0,
        }],
        &registry,
        settings,
    )
    .unwrap();
    assert!(vertical_excursion(&paths[0], 0.0) > 0.05);
    assert!(vertical_excursion(&paths[1], 40.0) > 0.05);
    for (reverse_point, forward_point) in paths[1]
        .points
        .iter()
        .zip(forward_paths[1].points.iter().rev())
    {
        assert!((reverse_point.0 - forward_point.0).abs() < 1.0e-5);
        assert!((reverse_point.1 - forward_point.1).abs() < 1.0e-5);
    }
    assert_eq!(paths[1].start_endpoint_id, 201);
    assert_eq!(paths[1].end_endpoint_id, 202);
    assert_eq!(paths[1].points[0], links[1].start);
    assert_eq!(paths[1].points[16], links[1].end);
}
