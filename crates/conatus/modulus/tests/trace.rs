// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! `BrickMap::trace` through modulus's public surface alone. The crate's own
//! `traversal_tests` measure the same walk against an exact one, with the
//! walk it replaced as their positive control.

use std::collections::BTreeMap;

use modulus::{
    AtlasLimits, BRICK_EDGE, BrickHit, BrickKey, BrickMap, BrickProjectionRevision, BrickTrace,
};

/// A map of `keys` over a pointer volume `extent` bricks wide from the
/// lowest key, each voxel filled by `fill` from its world coordinates.
fn map(keys: &[BrickKey], extent: [u32; 3], fill: impl Fn([i32; 3]) -> u8) -> BrickMap {
    let edge = BRICK_EDGE as i32;
    let bricks: BTreeMap<BrickKey, Vec<u8>> = keys
        .iter()
        .map(|key| {
            let mut bytes = vec![0; BRICK_EDGE.pow(3) as usize];
            for y in 0..edge {
                for z in 0..edge {
                    for x in 0..edge {
                        let at = [0, 1, 2].map(|i| i32::from(key[i]) * edge + [x, y, z][i]);
                        bytes[((y * edge + z) * edge + x) as usize] = fill(at);
                    }
                }
            }
            (*key, bytes)
        })
        .collect();
    let revision = BrickProjectionRevision(0);
    let mut map = BrickMap::with_limits(revision, keys.len(), extent, AtlasLimits::DEFAULT)
        .expect("the test map fits the default limits");
    map.retarget(BrickProjectionRevision(1), keys.iter().copied(), |key| {
        bricks.get(&key).map(Vec::as_slice)
    })
    .expect("the keys fit the volume");
    map
}

fn hit(trace: BrickTrace) -> BrickHit {
    match trace {
        BrickTrace::Hit(hit) => hit,
        other => panic!("expected a hit, got {other:?}"),
    }
}

#[test]
fn a_walk_that_starts_in_a_solid_voxel_hits_it_with_the_default_normal() {
    let map = map(&[[0, 0, 0]], [1, 1, 1], |_| 7);
    let found = hit(map.trace([4.5, 100.0, 3.5], [0.0, -1.0, 0.0], 1_000.0));
    assert_eq!((found.voxel, found.material), ([4, 7, 3], 7));
    assert_eq!(found.normal, [0.0, 1.0, 0.0]);
    assert!((found.t - 92.0).abs() < 1e-3, "entered at the top face");
}

#[test]
fn a_walk_that_steps_into_a_voxel_reports_the_face_it_came_through() {
    let map = map(&[[0, 0, 0]], [1, 1, 1], |[x, ..]| u8::from(x >= 4) * 9);
    let found = hit(map.trace([-50.0, 4.5, 4.5], [1.0, 0.0, 0.0], 1_000.0));
    assert_eq!((found.voxel, found.material), ([4, 4, 4], 9));
    assert_eq!(found.normal, [-1.0, 0.0, 0.0]);
    assert_eq!(found.t, 54.0);
}

#[test]
fn air_a_miss_and_the_far_cut_are_clear() {
    let map = map(&[[0, 0, 0]], [1, 1, 1], |[x, ..]| u8::from(x >= 4) * 9);
    // Over the top of the volume, through its air half and out the side,
    // and short of the solid half by the far cut.
    assert_eq!(
        map.trace([-50.0, 20.0, 4.5], [1.0, 0.0, 0.0], 1_000.0),
        BrickTrace::Clear
    );
    assert_eq!(
        map.trace([2.5, 50.0, 4.5], [0.0, -1.0, 0.0], 1_000.0),
        BrickTrace::Clear
    );
    assert_eq!(
        map.trace([-50.0, 4.5, 4.5], [1.0, 0.0, 0.0], 53.0),
        BrickTrace::Clear
    );
}

#[test]
fn a_walk_past_its_budget_is_exhausted_not_clear() {
    // 1,600 voxels of air, then one solid brick: more cells than the
    // shader's 1,024 from the near end, fewer from the middle.
    let map = map(&[[0, 0, 0], [200, 0, 0]], [201, 1, 1], |[x, ..]| {
        u8::from(x >= 1_600) * 3
    });
    assert_eq!(
        map.trace([-10.0, 4.5, 4.5], [1.0, 0.0, 0.0], 5_000.0),
        BrickTrace::Exhausted
    );
    let found = hit(map.trace([1_000.5, 4.5, 4.5], [1.0, 0.0, 0.0], 5_000.0));
    assert_eq!(
        (found.voxel, found.normal),
        ([1_600, 4, 4], [-1.0, 0.0, 0.0])
    );
}

#[test]
fn every_hit_is_a_voxel_the_map_holds() {
    // Rolling ground over 4 by 2 by 4 bricks, and a fan of rays onto it
    // from every side and from inside.
    let keys: Vec<BrickKey> = (0..4)
        .flat_map(|x| (0..2).flat_map(move |y| (0..4).map(move |z| [x, y, z])))
        .collect();
    let map = map(&keys, [4, 2, 4], |[x, y, z]| {
        u8::from(y <= (x * 7 + z * 3).rem_euclid(11)) * (1 + (x + z).rem_euclid(5) as u8)
    });
    let mut hits = 0;
    for index in 0..4_096 {
        let angle = index as f32 * 0.618_034 * std::f32::consts::TAU;
        let rise = (index % 17) as f32 / 8.0 - 1.0;
        let direction = [angle.cos(), rise, angle.sin()];
        let eye = [
            16.0 - direction[0] * 60.0,
            12.0 - rise * 30.0,
            16.0 - direction[2] * 60.0,
        ];
        if let BrickTrace::Hit(found) = map.trace(eye, direction, 500.0) {
            hits += 1;
            assert_ne!(found.material, 0);
            assert_eq!(map.material_at(found.voxel), found.material, "{found:?}");
        }
    }
    assert!(hits > 1_000, "the fan is not vacuous: {hits} of it lands");
}

#[test]
fn raising_the_volume_s_top_moves_no_hit() {
    // Flat ground whose top layer is y = 1, 16 bricks square, under one,
    // two and three layers of pointer volume, seen as Isometry's board sees
    // its ground: along its dimetric forward, from about 1,380 units back.
    let keys: Vec<BrickKey> = (0..16)
        .flat_map(|x| (0..16).map(move |z| [x, 0, z]))
        .collect();
    let maps: Vec<BrickMap> = (1..=3)
        .map(|layers| map(&keys, [16, layers, 16], |[_, y, _]| u8::from(y <= 1) * 2))
        .collect();
    let forward = [-0.612_372_46, -0.5, -0.612_372_46];
    let mut landed = 0;
    for row in 0..64 {
        for column in 0..64 {
            let target = [4.0 + column as f32 * 1.87, 2.0, 4.0 + row as f32 * 1.93];
            let eye = [0, 1, 2].map(|i| target[i] - forward[i] * 1_380.0);
            let traces: Vec<BrickTrace> = maps
                .iter()
                .map(|map| map.trace(eye, forward, 3_000.0))
                .collect();
            assert!(traces.iter().all(|trace| *trace == traces[0]), "{traces:?}");
            landed += usize::from(matches!(traces[0], BrickTrace::Hit(_)));
        }
    }
    assert_eq!(landed, 64 * 64);
}
