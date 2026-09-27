// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! The rays the traversal is measured on: Isometry's board frame, rebuilt
//! bit for bit from its probe's camera, and a seeded spread of terrain boxes
//! and rays in every direction, near the origin and far from it.

use super::*;

/// A ray as the shader receives it: an eye and a direction, both f32.
pub(super) type Ray = ([f32; 3], [f32; 3]);

/// The recommended 8 MiB host budget on a device at wgpu's default limits.
const CARD: AtlasLimits = AtlasLimits {
    max_texture_dimension_3d: 2048,
    max_atlas_bytes: 8 << 20,
};

// Isometry's board camera, from the setup line of the `headroom_bands`
// probe on its `lane-e-paging` branch (`bands.md`, 2026-09-26): a 2:1
// dimetric slab whose rays begin on its front wall, about 1,380 units
// behind the ground they reach.
const ORIGIN: [f32; 3] = [297.708_25, 690.0, 277.613_95];
const FORWARD: [f32; 3] = [-0.612_372_46, -0.5, -0.612_372_46];
const RIGHT: [f32; 3] = [74.242_27, 0.0, -74.242_27];
const UP: [f32; 3] = [-31.333_334, 76.750_69, -31.333_334];
const WALL: [f32; 3] = [-0.0, -51.167_118, -212.867_86];
pub(super) const BOARD_FAR: f32 = 3_177.735_8;
pub(super) const TEXTURE: [u32; 2] = [890, 752];

/// Texel `(px, py)`'s ray, in f32 and in the order isometer's
/// `TraceCamera::ray_at` and the tracer's `camera_ray` take it: the near
/// plane point, then the slide along forward onto the front wall.
pub(super) fn board_ray(px: u32, py: u32) -> Ray {
    let nx = 2.0 * (px as f32 + 0.5) / TEXTURE[0] as f32 - 1.0;
    let ny = 1.0 - 2.0 * (py as f32 + 0.5) / TEXTURE[1] as f32;
    let advance = WALL[0] * nx + WALL[1] * ny + WALL[2];
    let eye = [0, 1, 2].map(|i| ORIGIN[i] + (RIGHT[i] * nx + UP[i] * ny) + FORWARD[i] * advance);
    (eye, FORWARD)
}

/// The whole frame, row by row.
pub(super) fn board_frame() -> Vec<(u32, u32)> {
    (0..TEXTURE[1])
        .flat_map(|py| (0..TEXTURE[0]).map(move |px| (px, py)))
        .collect()
}

/// Every row of the probe's two banded columns and their neighbours, and
/// every fifth texel of every fifth row elsewhere: quick enough for the
/// default test run, with the frame's worst rays all in it.
pub(super) fn board_sample() -> Vec<(u32, u32)> {
    board_frame()
        .into_iter()
        .filter(|&(px, py)| {
            (440..=450).contains(&px) || (866..=876).contains(&px) || (px % 5 == 0 && py % 5 == 0)
        })
        .collect()
}

/// A material that tells every voxel from its neighbours, so a hit one
/// voxel over reads as a different material, on the GPU too.
pub(super) fn material_of([x, y, z]: [i32; 3]) -> u8 {
    (1 + x.rem_euclid(5) + 5 * z.rem_euclid(5) + 25 * y.rem_euclid(3)) as u8
}

/// The probe's pointer box, x and z from -640 to -128, over flat ground
/// whose top layer is y = 1, as under both banded columns, with `headroom`
/// spare brick layers above it.
pub(super) fn board_map(headroom: u32) -> BrickMap {
    let keys: Vec<BrickKey> = (-80..-16)
        .flat_map(|x| (-80..-16).map(move |z| [x, 0, z]))
        .collect();
    build(&keys, [64, 1 + headroom, 64], |[_, y, _]| y <= 1)
}

/// A capacity-fixed map over `extent` holding `keys`, whose bricks are
/// solid wherever `solid` says, in [`material_of`]'s materials.
fn build(keys: &[BrickKey], extent: [u32; 3], solid: impl Fn([i32; 3]) -> bool) -> BrickMap {
    let edge = BRICK_EDGE as i32;
    let bricks: BTreeMap<BrickKey, Vec<u8>> = keys
        .iter()
        .map(|key| {
            let mut bytes = vec![0; BRICK_EDGE.pow(3) as usize];
            for y in 0..edge {
                for z in 0..edge {
                    for x in 0..edge {
                        let at = [0, 1, 2].map(|i| i32::from(key[i]) * edge + [x, y, z][i]);
                        if solid(at) {
                            bytes[((y * edge + z) * edge + x) as usize] = material_of(at);
                        }
                    }
                }
            }
            (*key, bytes)
        })
        .collect();
    let mut map = BrickMap::with_limits(BrickProjectionRevision(0), keys.len(), extent, CARD)
        .expect("the scene fits the card");
    map.retarget(BrickProjectionRevision(1), keys.iter().copied(), |key| {
        bricks.get(&key).map(Vec::as_slice)
    })
    .expect("the scene's keys fit its box");
    map
}

/// splitmix64, so every run draws the same scenes.
pub(super) struct Stream(pub(super) u64);

impl Stream {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn within(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * self.unit()
    }

    fn below(&mut self, bound: u64) -> u64 {
        self.next() % bound
    }
}

/// One seeded terrain box and the rays aimed into it.
pub(super) struct Scene {
    pub(super) map: BrickMap,
    pub(super) far: f32,
    pub(super) rays: Vec<Ray>,
}

/// A box of up to 24 by 4 by 24 bricks whose origin lies up to `reach`
/// bricks from the world's, over a ground of 3-voxel columns (some holes,
/// some with a floating slab), and `count` rays into it: from eyes near,
/// far and inside it, grazing its floor, nearly parallel to an axis, and
/// in orthographic bundles whose x and z components are equal, as the
/// board's are.
pub(super) fn scene(seed: u64, reach: i32, count: usize) -> Scene {
    let mut stream = Stream(seed);
    let extent = [
        1 + stream.below(24) as u32,
        1 + stream.below(4) as u32,
        1 + stream.below(24) as u32,
    ];
    let origin = [0, 1, 2].map(|i| {
        let room = i64::from(reach) - i64::from(extent[i]);
        let offset = stream.below(2 * room.max(0) as u64 + 1) as i64 - room.max(0);
        offset as i16
    });
    let edge = BRICK_EDGE as i32;
    let low = origin.map(|v| i32::from(v) * edge);
    let size = extent.map(|v| v as i32 * edge);
    // Each 3-voxel column: its height above the box floor, three in ten
    // holes, and one in ten carrying a two-voxel slab above a gap.
    let tiles = [(size[0] + 2) / 3, (size[2] + 2) / 3];
    let columns: Vec<(i32, i32)> = (0..tiles[0] * tiles[1])
        .map(|_| {
            let height = if stream.below(10) < 3 {
                0
            } else {
                stream.below(size[1] as u64) as i32
            };
            let slab = if stream.below(10) == 0 {
                height + 2
            } else {
                i32::MAX
            };
            (height, slab)
        })
        .collect();
    let solid = |[x, y, z]: [i32; 3]| {
        let (height, slab) = columns[((x - low[0]) / 3 * tiles[1] + (z - low[2]) / 3) as usize];
        let up = y - low[1];
        up < height || (up >= slab && up < slab.saturating_add(2))
    };
    let keys: Vec<BrickKey> = (0..extent[0] as i16)
        .flat_map(|x| {
            (0..extent[1] as i16).flat_map(move |y| (0..extent[2] as i16).map(move |z| [x, y, z]))
        })
        .map(|local| [0, 1, 2].map(|i| origin[i] + local[i]))
        .filter(|key| {
            let base = key.map(|v| i32::from(v) * edge);
            (0..edge).any(|x| {
                (0..edge).any(|z| {
                    let (height, slab) = columns[((base[0] + x - low[0]) / 3 * tiles[1]
                        + (base[2] + z - low[2]) / 3)
                        as usize];
                    let floor = base[1] - low[1];
                    height > floor || (slab < floor + edge && slab.saturating_add(2) > floor)
                })
            })
        })
        .collect();
    let map = if keys.is_empty() {
        BrickMap::with_limits(BrickProjectionRevision(0), 1, extent, CARD).expect("an empty box")
    } else {
        build(&keys, extent, solid)
    };
    let diagonal = f64::from(size.iter().map(|v| v * v).sum::<i32>()).sqrt();
    let centre = [0, 1, 2].map(|i| f64::from(low[i]) + f64::from(size[i]) / 2.0);
    let inside = |stream: &mut Stream| {
        [0, 1, 2].map(|i| stream.within(f64::from(low[i]), f64::from(low[i] + size[i])))
    };
    let rays = (0..count)
        .map(|index| {
            let target = inside(&mut stream);
            let (eye, direction) = match index % 6 {
                0 | 1 => {
                    let [least, most] = if index % 6 == 0 {
                        [0.6, 3.0]
                    } else {
                        [5.0, 60.0]
                    };
                    let radius = diagonal * stream.within(least, most);
                    aim(sphere(&mut stream, centre, radius), target)
                },
                2 => (inside(&mut stream), unit_vector(&mut stream)),
                3 => {
                    let mut direction = unit_vector(&mut stream);
                    direction[1] = -stream.within(0.001, 0.05);
                    let target = [
                        target[0],
                        f64::from(low[1]) + stream.within(0.0, 3.0),
                        target[2],
                    ];
                    let back = diagonal * stream.within(1.0, 20.0);
                    let direction = normal(direction);
                    (
                        [0, 1, 2].map(|i| target[i] - direction[i] * back),
                        direction,
                    )
                },
                4 => {
                    let mut direction = unit_vector(&mut stream);
                    for _ in 0..1 + stream.below(2) {
                        let axis = stream.below(3) as usize;
                        let tiny = [0.0, 1e-7, 3e-6, 1e-4][stream.below(4) as usize];
                        direction[axis] = if stream.below(2) == 0 { tiny } else { -tiny };
                    }
                    let direction = normal(direction);
                    let back = diagonal * stream.within(0.2, 30.0);
                    (
                        [0, 1, 2].map(|i| target[i] - direction[i] * back),
                        direction,
                    )
                },
                _ => {
                    let sign = |stream: &mut Stream| if stream.below(2) == 0 { 1.0 } else { -1.0 };
                    let across = stream.within(0.2, 1.0) * sign(&mut stream);
                    let down = stream.within(0.2, 1.0) * sign(&mut stream);
                    let direction = normal([across, down, across]);
                    let back = diagonal * stream.within(2.0, 200.0);
                    (
                        [0, 1, 2].map(|i| target[i] - direction[i] * back),
                        direction,
                    )
                },
            };
            (eye.map(|v| v as f32), direction.map(|v| v as f32))
        })
        .collect();
    Scene {
        map,
        far: 1.0e6,
        rays,
    }
}

fn normal(v: [f64; 3]) -> [f64; 3] {
    let length = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    v.map(|c| c / length)
}

fn unit_vector(stream: &mut Stream) -> [f64; 3] {
    loop {
        let v = [0; 3].map(|_| stream.within(-1.0, 1.0));
        let length = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if length > 0.05 && length <= 1.0 {
            return normal(v);
        }
    }
}

fn sphere(stream: &mut Stream, centre: [f64; 3], radius: f64) -> [f64; 3] {
    let direction = unit_vector(stream);
    [0, 1, 2].map(|i| centre[i] + direction[i] * radius)
}

fn aim(eye: [f64; 3], target: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    (eye, normal([0, 1, 2].map(|i| target[i] - eye[i])))
}

/// Four reaches, from boxes about the origin to boxes 30,000 bricks out,
/// where an f32 coordinate keeps only six bits below the voxel.
pub(super) fn stress_scenes(per_reach: u64, count: usize) -> Vec<Scene> {
    [0, 100, 3_000, 30_000]
        .into_iter()
        .enumerate()
        .flat_map(|(band, reach)| {
            (0..per_reach)
                .map(move |index| scene(0x5EED_0000 + band as u64 * 1_000 + index, reach, count))
        })
        .collect()
}
