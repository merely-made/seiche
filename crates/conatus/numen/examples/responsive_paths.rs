// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! Inspectable deterministic P5 fixture: attractor, flow and vortex on fixed
//! endpoint links, with compatible-link attraction toggled independently.

use numen::{
    FieldRegistry, PathField, ResponsiveLink, ResponsivePathConfig, ScalarField, VectorField,
    evaluate_responsive_paths,
};

fn main() {
    let registry = FieldRegistry::new();
    let attractor = VectorField::gradient_of(ScalarField::gaussian_at(0.0, 45.0, 100.0));
    let flow = VectorField::const_vec(0.0, 1.0);
    let vortex = VectorField::Perp(Box::new(VectorField::gradient_of(
        ScalarField::gaussian_at(0.0, 0.0, 100.0),
    )));
    let links = [
        ResponsiveLink {
            id: 101,
            start_endpoint_id: 1,
            end_endpoint_id: 2,
            start: (-60.0, 0.0),
            end: (60.0, 0.0),
            compatibility: Some("cites"),
        },
        ResponsiveLink {
            id: 102,
            start_endpoint_id: 1,
            end_endpoint_id: 2,
            start: (-60.0, 30.0),
            end: (60.0, 30.0),
            compatibility: Some("cites"),
        },
        ResponsiveLink {
            id: 103,
            start_endpoint_id: 1,
            end_endpoint_id: 2,
            start: (-60.0, 65.0),
            end: (60.0, 65.0),
            compatibility: Some("contradicts"),
        },
    ];

    for (name, field, strength) in [
        ("attractor", &attractor, 100.0),
        ("flow", &flow, 1.0),
        ("vortex", &vortex, 100.0),
    ] {
        for (field_response, compatible_attraction) in [(true, false), (false, true), (true, true)]
        {
            let settings = ResponsivePathConfig {
                control_points: 17,
                iterations: 24,
                field_response,
                compatible_attraction,
                ..ResponsivePathConfig::default()
            };
            let paths = evaluate_responsive_paths(
                &links,
                &[PathField { field, strength }],
                &registry,
                settings,
            )
            .expect("fixture inputs are within the evaluator bounds");
            println!(
                "{name} strength={strength} settings={settings:?} quarter/mid/three-quarter: {}",
                paths
                    .iter()
                    .map(|path| {
                        let quarter = path.points.len() / 4;
                        let middle = path.points.len() / 2;
                        let three_quarters = path.points.len() * 3 / 4;
                        format!(
                            "{}=[{:?}, {:?}, {:?}]",
                            path.id,
                            path.points[quarter],
                            path.points[middle],
                            path.points[three_quarters]
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
    }
}
