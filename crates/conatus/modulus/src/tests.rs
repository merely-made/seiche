// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! Unit tests for the brick map and its shared shader. Split out of
//! `lib.rs` to keep both files under the workspace's per-file size ceiling.

use super::*;

fn solid(material: u8) -> [u8; BRICK_EDGE.pow(3) as usize] {
    [material; BRICK_EDGE.pow(3) as usize]
}

#[test]
fn map_order_and_atlas_layout_are_source_order_independent() {
    let low = solid(2);
    let high = solid(3);
    let source = |key: BrickKey| match key {
        [-1, 0, 0] => Some(low.as_slice()),
        [1, 0, 0] => Some(high.as_slice()),
        _ => None,
    };
    let first = BrickMap::from_keys(
        BrickProjectionRevision(7),
        [[1, 0, 0], [-1, 0, 0], [1, 0, 0]],
        source,
    )
    .unwrap();
    let second =
        BrickMap::from_keys(BrickProjectionRevision(7), [[-1, 0, 0], [1, 0, 0]], source).unwrap();

    assert_eq!(first.origin(), [-1, 0, 0]);
    assert_eq!(first.pointer_extent(), [3, 1, 1]);
    assert_eq!(first.pointers(), second.pointers());
    assert_eq!(first.atlas(), second.atlas());
    assert_eq!(first.material_at([-8, 0, 0]), 2);
    assert_eq!(first.material_at([8, 0, 0]), 3);
    assert_eq!(first.material_at([0, 0, 0]), 0);
    assert_eq!(
        BrickTraceSpace::from_map(&first).world_min,
        [-8.0, 0.0, 0.0, 0.0]
    );
}

#[test]
fn missing_and_malformed_bricks_are_refused() {
    assert!(matches!(
        BrickMap::from_keys(BrickProjectionRevision(0), [[0, 0, 0]], |_| None),
        Err(BrickMapError::MissingBrick { .. })
    ));
    let malformed = [1u8; 8];
    assert!(matches!(
        BrickMap::from_keys(BrickProjectionRevision(0), [[0, 0, 0]], |_| {
            Some(malformed.as_slice())
        }),
        Err(BrickMapError::InvalidBrickLength { .. })
    ));
}

#[test]
fn refused_refresh_writes_nothing() {
    let first = solid(1);
    let replacement = solid(2);
    let mut map = BrickMap::from_keys(BrickProjectionRevision(0), [[0, 0, 0]], |_| {
        Some(first.as_slice())
    })
    .unwrap();
    let pointers = map.pointers.clone();
    let atlas = map.atlas.clone();

    assert!(matches!(
        map.refresh([[0, 0, 0], [1, 0, 0]], |_| Some(replacement.as_slice())),
        Err(BrickMapError::UnknownKey { .. })
    ));
    assert_eq!(map.pointers, pointers);
    assert_eq!(map.atlas, atlas);
}

#[test]
fn a_retarget_retains_slots_and_recycles_evictions_deterministically() {
    let a = solid(2);
    let b = solid(3);
    let c = solid(4);
    let source = |key: BrickKey| match key {
        [0, 0, 0] => Some(a.as_slice()),
        [1, 0, 0] => Some(b.as_slice()),
        [2, 0, 0] => Some(c.as_slice()),
        _ => None,
    };
    let mut map = BrickMap::with_capacity(BrickProjectionRevision(0), 1, [4, 1, 4]).unwrap();
    assert_eq!(map.capacity(), 255);
    let first = map
        .retarget(BrickProjectionRevision(1), [[0, 0, 0], [1, 0, 0]], source)
        .unwrap();
    assert_eq!(first.loaded_slots, vec![1, 2]);
    let b_slot = map.key_slots[&[1, 0, 0]];

    let second = map
        .retarget(BrickProjectionRevision(2), [[1, 0, 0], [2, 0, 0]], source)
        .unwrap();
    assert_eq!(
        map.key_slots[&[1, 0, 0]],
        b_slot,
        "a kept brick keeps its slot"
    );
    assert_eq!(second.evicted, 1);
    assert_eq!(second.retained, 1);
    assert_eq!(
        second.loaded_slots,
        vec![map.key_slots[&[2, 0, 0]]],
        "only the loaded brick's slot was written"
    );
    assert_eq!(
        map.key_slots[&[2, 0, 0]],
        1,
        "the evicted slot recycles to the loaded key"
    );
    assert_eq!(map.material_at([8, 0, 0]), 3);
    assert_eq!(map.material_at([16, 0, 0]), 4);
    assert_eq!(map.material_at([0, 0, 0]), 0, "the evicted brick is gone");
    assert_eq!(map.pointer_extent(), [4, 1, 4], "the extent never moves");

    // The same sequence from scratch lands byte-identically.
    let mut again = BrickMap::with_capacity(BrickProjectionRevision(0), 1, [4, 1, 4]).unwrap();
    again
        .retarget(BrickProjectionRevision(1), [[0, 0, 0], [1, 0, 0]], source)
        .unwrap();
    again
        .retarget(BrickProjectionRevision(2), [[1, 0, 0], [2, 0, 0]], source)
        .unwrap();
    assert_eq!(map.pointers(), again.pointers());
    assert_eq!(map.atlas(), again.atlas());
}

#[test]
fn a_refused_retarget_writes_nothing() {
    let a = solid(2);
    let source = |_: BrickKey| Some(a.as_slice());
    let mut map = BrickMap::with_capacity(BrickProjectionRevision(0), 1, [2, 1, 2]).unwrap();
    map.retarget(BrickProjectionRevision(1), [[0, 0, 0]], source)
        .unwrap();
    let pointers = map.pointers.clone();
    let atlas = map.atlas.clone();
    let held_origin = map.origin();

    assert!(matches!(
        map.retarget(BrickProjectionRevision(2), [[0, 0, 0], [4, 0, 0]], source),
        Err(BrickMapError::ExtentExceeded { .. })
    ));
    assert!(matches!(
        map.retarget(BrickProjectionRevision(1), [[1, 0, 0]], source),
        Err(BrickMapError::ProjectionNotAdvanced { .. })
    ));
    assert!(matches!(
        map.retarget(BrickProjectionRevision(2), [[1, 0, 0]], |_| None),
        Err(BrickMapError::MissingBrick { .. })
    ));
    assert_eq!(map.pointers, pointers);
    assert_eq!(map.atlas, atlas);
    assert_eq!(map.origin(), held_origin);

    // An unchanged selection is a no-op that needs no revision advance.
    let unchanged = map
        .retarget(BrickProjectionRevision(1), [[0, 0, 0]], source)
        .unwrap();
    assert!(unchanged.loaded_slots.is_empty());
    assert_eq!(unchanged.retained, 1);
}

#[test]
fn a_retargeted_map_reads_like_a_rebuilt_one() {
    let low = solid(2);
    let high = solid(3);
    let source = |key: BrickKey| match key {
        [-1, 0, 0] => Some(low.as_slice()),
        [1, 0, 1] => Some(high.as_slice()),
        _ => None,
    };
    let mut travelled = BrickMap::with_capacity(BrickProjectionRevision(0), 1, [4, 1, 4]).unwrap();
    travelled
        .retarget(BrickProjectionRevision(1), [[-1, 0, 0]], source)
        .unwrap();
    travelled
        .retarget(BrickProjectionRevision(2), [[-1, 0, 0], [1, 0, 1]], source)
        .unwrap();
    let rebuilt =
        BrickMap::from_keys(BrickProjectionRevision(2), [[-1, 0, 0], [1, 0, 1]], source).unwrap();
    for at in [[-8, 0, 0], [-1, 7, 7], [8, 0, 8], [15, 7, 15], [0, 0, 0]] {
        assert_eq!(travelled.material_at(at), rebuilt.material_at(at), "{at:?}");
    }
    assert_eq!(travelled.origin(), rebuilt.origin());
}

/// Three bricks shrunk to the last, which keeps slot 3 while only one key
/// stays resident: the case the paging lane caught.
fn shrunk_to_a_high_slot() -> BrickMap {
    let bricks = [solid(2), solid(3), solid(4)];
    let source = |key: BrickKey| bricks.get(key[0] as usize).map(|brick| brick.as_slice());
    let mut map = BrickMap::with_capacity(BrickProjectionRevision(0), 1, [4, 1, 1]).unwrap();
    map.retarget(
        BrickProjectionRevision(1),
        [[0, 0, 0], [1, 0, 0], [2, 0, 0]],
        source,
    )
    .unwrap();
    let shrink = map
        .retarget(BrickProjectionRevision(2), [[2, 0, 0]], source)
        .unwrap();
    assert_eq!((shrink.evicted, shrink.retained), (2, 1));
    assert_eq!(
        map.key_slots[&[2, 0, 0]],
        3,
        "the kept brick keeps its slot"
    );
    map
}

/// What the shared shader's `brick_material_at` reads: the pointer volume,
/// then that slot's atlas box, with none of the map's key bookkeeping.
fn traced_material_at(map: &BrickMap, at: [i32; 3]) -> u8 {
    let edge = BRICK_EDGE as i32;
    let local = [0, 1, 2].map(|axis| at[axis] - i32::from(map.origin()[axis]) * edge);
    if local.iter().any(|axis| *axis < 0) {
        return 0;
    }
    let slot = map.pointer_at(local.map(|axis| (axis / edge) as u32));
    let Some(index) = slot.and_then(|slot| slot.checked_sub(1)) else {
        return 0;
    };
    let [sx, _, sz] = map.slots();
    let spot = [index % sx, index / (sx * sz), (index / sx) % sz];
    let texel = [0, 1, 2].map(|axis| spot[axis] * BRICK_EDGE + (local[axis] % edge) as u32);
    let [width, height, _] = map.atlas_extent();
    map.atlas()[((texel[2] * height + texel[1]) * width + texel[0]) as usize]
}

/// The first solid voxel straight down one column: the simplest pick ray.
fn first_solid_down(read: impl Fn([i32; 3]) -> u8, x: i32, z: i32) -> Option<(i32, u8)> {
    (0..2 * BRICK_EDGE as i32).rev().find_map(|y| {
        let material = read([x, y, z]);
        (material != 0).then_some((y, material))
    })
}

#[test]
fn a_brick_kept_through_a_shrink_refreshes_in_its_slot() {
    let mut map = shrunk_to_a_high_slot();
    let replacement = solid(9);
    let changed = map
        .refresh([[2, 0, 0]], |_| Some(replacement.as_slice()))
        .unwrap();
    assert_eq!(changed, vec![3]);
    assert_eq!(
        map.pointer_at([0, 0, 0]),
        Some(3),
        "the pointer still names it"
    );
    assert_eq!(map.slot_texels(3), Some(replacement.to_vec()));
    assert_eq!(map.material_at([16, 0, 0]), 9);
}

#[test]
fn a_pick_after_a_shrink_meets_the_ground_the_tracer_draws() {
    let map = shrunk_to_a_high_slot();
    let drawn = first_solid_down(|at| traced_material_at(&map, at), 20, 3);
    assert_eq!(drawn, Some((7, 4)), "the tracer draws the kept brick");
    assert_eq!(first_solid_down(|at| map.material_at(at), 20, 3), drawn);

    // Evicted ground, the kept brick and the empty pointer cells beyond it
    // read alike through the map and through the shader's path.
    let edge = BRICK_EDGE as i32;
    for x in 0..6 * edge {
        for y in 0..edge {
            for z in 0..edge {
                let at = [x, y, z];
                assert_eq!(map.material_at(at), traced_material_at(&map, at), "{at:?}");
            }
        }
    }
}

#[test]
fn every_slot_the_atlas_holds_has_its_own_box_inside_it() {
    let map = BrickMap::with_capacity(BrickProjectionRevision(0), 2, [1, 1, 1]).unwrap();
    let extent = map.atlas_extent();
    let mut origins = BTreeSet::new();
    for slot in 1..=map.capacity() as u32 {
        let origin = map.atlas_slot_origin(slot).expect("a slot within capacity");
        assert!(
            (0..3).all(|axis| origin[axis] + BRICK_EDGE <= extent[axis]),
            "slot {slot} at {origin:?}"
        );
        assert!(origins.insert(origin), "slot {slot} shares {origin:?}");
    }
    assert_eq!(map.atlas_slot_origin(0), None, "slot 0 is air");
    assert_eq!(map.atlas_slot_origin(map.capacity() as u32 + 1), None);
}

#[test]
fn shared_shader_stops_before_product_policy() {
    assert!(BRICK_DDA_WGSL.contains("fn brick_dda"));
    for forbidden in ["camera", "fog", "light", "material_colour", "critter"] {
        assert!(!BRICK_DDA_WGSL.contains(forbidden), "found {forbidden}");
    }
}
