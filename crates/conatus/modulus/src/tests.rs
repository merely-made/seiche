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

/// The recommended 8 MiB host budget on a device at wgpu's default limits.
const CARD: AtlasLimits = AtlasLimits {
    max_texture_dimension_3d: 2048,
    max_atlas_bytes: 8 << 20,
};

/// `width` by `depth` bricks on the ground layer, in key order.
fn grid(width: i16, depth: i16) -> Vec<BrickKey> {
    (0..width)
        .flat_map(|x| (0..depth).map(move |z| [x, 0, z]))
        .collect()
}

/// A solid brick per key, numbered from 2 so neighbours differ.
fn solids(keys: &[BrickKey]) -> BTreeMap<BrickKey, [u8; BRICK_EDGE.pow(3) as usize]> {
    keys.iter()
        .enumerate()
        .map(|(index, key)| (*key, solid(index as u8 % 250 + 2)))
        .collect()
}

/// A brick's lowest voxel.
fn voxel_origin(key: BrickKey) -> [i32; 3] {
    key.map(|axis| i32::from(axis) * BRICK_EDGE as i32)
}

/// A selection shrunk to its last key, which keeps the highest slot while
/// only one key stays resident: the case the paging lane caught.
struct Shrunk {
    map: BrickMap,
    kept: BrickKey,
    slot: u32,
    material: u8,
}

/// The paging lane's case over `keys`, in whatever atlas `map` carries.
fn shrink_to_the_last(mut map: BrickMap, keys: &[BrickKey]) -> Shrunk {
    let bricks = solids(keys);
    let source = |key: BrickKey| bricks.get(&key).map(|brick| brick.as_slice());
    map.retarget(BrickProjectionRevision(1), keys.iter().copied(), source)
        .unwrap();
    let (&kept, brick) = bricks.last_key_value().expect("a selection");
    let shrink = map
        .retarget(BrickProjectionRevision(2), [kept], source)
        .unwrap();
    assert_eq!((shrink.evicted, shrink.retained), (keys.len() - 1, 1));
    let slot = map.key_slots[&kept];
    assert_eq!(
        slot as usize,
        keys.len(),
        "the kept brick keeps the top slot"
    );
    Shrunk {
        map,
        kept,
        slot,
        material: brick[0],
    }
}

/// Three bricks in one atlas row, and 3,000 in twelve rows of a card-sized
/// atlas, past the default cap.
fn shrunk_cases() -> [Shrunk; 2] {
    let row = BrickMap::with_capacity(BrickProjectionRevision(0), 1, [4, 1, 1]).unwrap();
    let card = BrickMap::with_limits(BrickProjectionRevision(0), 3_000, [60, 1, 50], CARD).unwrap();
    [
        shrink_to_the_last(row, &[[0, 0, 0], [1, 0, 0], [2, 0, 0]]),
        shrink_to_the_last(card, &grid(60, 50)),
    ]
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
    for Shrunk {
        mut map,
        kept,
        slot,
        ..
    } in shrunk_cases()
    {
        let replacement = solid(255);
        let changed = map
            .refresh([kept], |_| Some(replacement.as_slice()))
            .unwrap();
        assert_eq!(changed, vec![slot]);
        assert_eq!(
            map.pointer_at([0, 0, 0]),
            Some(slot),
            "the pointer names it"
        );
        assert_eq!(map.slot_texels(slot), Some(replacement.to_vec()));
        assert_eq!(map.material_at(voxel_origin(kept)), 255);
    }
}

#[test]
fn a_pick_after_a_shrink_meets_the_ground_the_tracer_draws() {
    for Shrunk {
        map,
        kept,
        material,
        ..
    } in shrunk_cases()
    {
        let [x, _, z] = voxel_origin(kept);
        let drawn = first_solid_down(|at| traced_material_at(&map, at), x + 4, z + 3);
        assert_eq!(
            drawn,
            Some((7, material)),
            "the tracer draws the kept brick"
        );
        assert_eq!(
            first_solid_down(|at| map.material_at(at), x + 4, z + 3),
            drawn
        );

        // Evicted ground, the kept brick and the empty pointer cells around
        // it read alike through the map and through the shader's path.
        let edge = BRICK_EDGE as i32;
        for dx in -2 * edge..3 * edge {
            for y in 0..edge {
                for dz in -2 * edge..3 * edge {
                    let at = [x + dx, y, z + dz];
                    assert_eq!(map.material_at(at), traced_material_at(&map, at), "{at:?}");
                }
            }
        }
    }
}

#[test]
fn every_slot_the_atlas_holds_has_its_own_box_inside_it() {
    let rows = BrickMap::with_capacity(BrickProjectionRevision(0), 2, [1, 1, 1]).unwrap();
    let card =
        BrickMap::with_limits(BrickProjectionRevision(0), CARD.max_bricks(), [1; 3], CARD).unwrap();
    for map in [rows, card] {
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
}

#[test]
fn default_limits_keep_the_historical_cap() {
    assert_eq!(AtlasLimits::default(), AtlasLimits::DEFAULT);
    assert_eq!(
        (AtlasLimits::DEFAULT.max_bricks(), MAX_BRICKS),
        (2_047, 2_047)
    );
    let revision = BrickProjectionRevision(0);
    let too_many = |actual| BrickMapError::TooManyBricks {
        actual,
        maximum: 2_047,
    };

    // Rows under the default cap, refused past it as before.
    let eight = BrickMap::with_capacity(revision, 8, [1, 1, 1]).unwrap();
    assert_eq!((eight.capacity(), eight.atlas().len()), (2_047, 1 << 20));
    let zero = BrickMap::with_capacity(revision, 0, [1, 1, 1]).unwrap();
    assert_eq!(zero.slots(), [16, 1, 16], "zero rows still build one");
    let nine = BrickMap::with_capacity(revision, 9, [1, 1, 1]);
    assert_eq!(nine.unwrap_err(), too_many(2_303));

    // Asking by bricks under the default limits builds the same atlas.
    let by_bricks = BrickMap::with_limits(revision, 2_047, [1; 3], AtlasLimits::DEFAULT).unwrap();
    assert_eq!(by_bricks.slots(), eight.slots());
    let past = BrickMap::with_limits(revision, 2_048, [1; 3], AtlasLimits::DEFAULT);
    assert_eq!(past.unwrap_err(), too_many(2_048));

    // from_keys has no limits of its own and keeps the default cap.
    let keys: Vec<BrickKey> = (0..2_048).map(|x| [x, 0, 0]).collect();
    let whole = BrickMap::from_keys(revision, keys, |_| None);
    assert_eq!(whole.unwrap_err(), too_many(2_048));
}

#[test]
fn a_card_sized_map_holds_more_than_the_default_cap() {
    assert_eq!(CARD.max_bricks(), 16_383, "8 MiB is 64 rows");
    let keys = grid(60, 50);
    let bricks = solids(&keys);
    let mut map =
        BrickMap::with_limits(BrickProjectionRevision(0), keys.len(), [60, 1, 50], CARD).unwrap();
    assert_eq!(
        map.slots(),
        [16, 12, 16],
        "3,000 bricks round up to 12 rows"
    );
    assert_eq!(map.capacity(), 3_071);
    assert_eq!(BrickTraceSpace::from_map(&map).atlas_slots, [16, 12, 16, 0]);
    let delta = map
        .retarget(BrickProjectionRevision(1), keys.iter().copied(), |key| {
            bricks.get(&key).map(|brick| brick.as_slice())
        })
        .unwrap();
    assert_eq!(delta.loaded_slots.len(), 3_000);

    // Every brick, in rows past the old cap too, reads its own material
    // through the map and through the shader's path.
    for (key, brick) in &bricks {
        let low = voxel_origin(*key);
        for at in [low, low.map(|axis| axis + BRICK_EDGE as i32 - 1)] {
            assert_eq!(map.material_at(at), brick[0], "{key:?}");
            assert_eq!(traced_material_at(&map, at), brick[0], "{key:?}");
        }
    }
}

#[test]
fn card_limits_refuse_past_the_budget_and_the_texture_edge() {
    let revision = BrickProjectionRevision(0);
    // The budget binds first: 8 MiB is 64 of the 256 rows a 2,048 edge allows.
    let full = BrickMap::with_limits(revision, 16_383, [1, 1, 1], CARD).unwrap();
    assert_eq!(full.atlas().len(), 8 << 20);
    assert_eq!(
        BrickMap::with_limits(revision, 16_384, [1, 1, 1], CARD).unwrap_err(),
        BrickMapError::TooManyBricks {
            actual: 16_384,
            maximum: 16_383,
        }
    );

    // The edge binds first: the web tier's 256 texels are 32 rows, whatever
    // the budget, and bound every pointer volume axis too.
    let web = AtlasLimits {
        max_texture_dimension_3d: 256,
        max_atlas_bytes: u64::MAX,
    };
    let tall = BrickMap::with_limits(revision, 8_191, [256, 1, 1], web).unwrap();
    assert_eq!(tall.atlas_extent(), [128, 256, 128]);
    assert_eq!(
        BrickMap::with_limits(revision, 8_192, [1, 1, 1], web).unwrap_err(),
        BrickMapError::TooManyBricks {
            actual: 8_192,
            maximum: 8_191,
        }
    );
    assert_eq!(
        BrickMap::with_limits(revision, 1, [1, 257, 1], web).unwrap_err(),
        BrickMapError::TextureDimensionExceeded {
            pointer_extent: [1, 257, 1],
            maximum: 256,
        }
    );

    // Limits too small for one row refuse even an empty map, and no budget
    // takes the atlas past 4 GiB.
    for cramped in [
        AtlasLimits {
            max_texture_dimension_3d: 64,
            ..web
        },
        AtlasLimits {
            max_atlas_bytes: 100_000,
            ..CARD
        },
    ] {
        assert_eq!(cramped.max_bricks(), 0);
        assert!(matches!(
            BrickMap::with_limits(revision, 0, [1, 1, 1], cramped),
            Err(BrickMapError::TooManyBricks {
                actual: 0,
                maximum: 0
            })
        ));
    }
    let boundless = AtlasLimits {
        max_texture_dimension_3d: u32::MAX,
        max_atlas_bytes: u64::MAX,
    };
    assert_eq!(boundless.max_bricks(), 32_767 * 256 - 1);
}

#[test]
fn shared_shader_stops_before_product_policy() {
    assert!(BRICK_DDA_WGSL.contains("fn brick_dda"));
    for forbidden in ["camera", "fog", "light", "material_colour", "critter"] {
        assert!(!BRICK_DDA_WGSL.contains(forbidden), "found {forbidden}");
    }
}
