// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! The shared traversal measured against an exact walk of the same rays.
//!
//! [`mirror`] is `brick_dda.wgsl` in f32 on the CPU; [`exact`] walks the
//! same f32 ray in f64 with every crossing taken afresh from the eye, and
//! records how near each ray came to a tie. The rays are Isometry's board
//! frame (its probe's camera, rebuilt bit for bit, over the probe's box with
//! zero to two spare layers of headroom) and a seeded spread of boxes and
//! rays from near the origin to 240,000 voxels out.

use std::collections::BTreeMap;

use super::*;

mod exact;
mod mirror;
mod scenes;

use exact::{Exact, exact};
pub(crate) use mirror::brick_material_at;
use mirror::{Hit, brick_dda};
use scenes::{
    BOARD_FAR, Ray, TEXTURE, board_frame, board_map, board_ray, board_sample, stress_scenes,
};

/// A ray whose walks disagree within this many f32 ulps of a tie is at
/// f32's resolution: a crossing taken afresh from the eye rounds twice, a
/// subtraction and a division, each within an ulp of the crossing, so two
/// crossings closer than three ulps can come out in either order.
const TIE: f64 = 3.0;

/// The closest-call bands a tally sorts disagreements into, in ulps.
const BANDS: [f64; 6] = [1.0, 2.0, 3.0, 4.0, 8.0, 16.0];

/// What a pixel shows: the voxel, its material and the face entered by.
fn landing(hit: Option<Hit>) -> Option<([i32; 3], u8, [f32; 3])> {
    hit.map(|hit| (hit.voxel, hit.material, hit.normal))
}

/// One walk's hits against the exact walk's, over a set of rays.
#[derive(Debug, Default)]
struct Tally {
    rays: usize,
    /// Unlike the exact walk, on a ray within [`TIE`] ulps of a tie.
    ties: usize,
    /// Unlike the exact walk, on a ray that never came within [`TIE`].
    faults: usize,
    /// Rays whose exact walk ran out of budget, left out of the count.
    exhausted: usize,
    /// The widest closest call among the faults, in ulps.
    widest: f64,
    /// Disagreements by closest call: under each of [`BANDS`], then over.
    bands: [usize; 7],
}

impl Tally {
    fn add(&mut self, walked: Option<Hit>, truth: &Exact) {
        self.rays += 1;
        if truth.exhausted {
            self.exhausted += 1;
        } else if landing(walked) != landing(truth.hit) {
            self.bands[BANDS.iter().filter(|band| truth.margin >= **band).count()] += 1;
            if truth.margin < TIE {
                self.ties += 1;
            } else {
                self.faults += 1;
                self.widest = self.widest.max(truth.margin);
            }
        }
    }
}

/// The shader's walk and the exact walk of every ray into `map`.
fn walks(map: &BrickMap, far: f32, rays: &[Ray]) -> Vec<(Option<Hit>, Exact)> {
    let space = BrickTraceSpace::from_map(map);
    let material = |voxel| brick_material_at(map, voxel);
    rays.iter()
        .map(|&(eye, direction)| {
            (
                brick_dda(&space, far, eye, direction, material),
                exact(&space, far, eye, direction, material),
            )
        })
        .collect()
}

/// The board's `texels` at each headroom, walked both ways.
fn board(texels: &[(u32, u32)]) -> [Vec<(Option<Hit>, Exact)>; 3] {
    let rays: Vec<Ray> = texels.iter().map(|&(px, py)| board_ray(px, py)).collect();
    [0, 1, 2].map(|headroom| walks(&board_map(headroom), BOARD_FAR, &rays))
}

/// How many rays land differently under two headrooms.
fn moved(a: &[(Option<Hit>, Exact)], b: &[(Option<Hit>, Exact)]) -> usize {
    a.iter()
        .zip(b)
        .filter(|(a, b)| landing(a.0) != landing(b.0))
        .count()
}

fn tally(walked: &[(Option<Hit>, Exact)]) -> Tally {
    let mut tally = Tally::default();
    for (hit, truth) in walked {
        tally.add(*hit, truth);
    }
    tally
}

#[test]
fn the_board_rays_are_the_probe_s_bit_for_bit() {
    // Isometry's probe logged these f32 eyes for the first banded texel of
    // column 445 and the one three rows below it.
    let probe: [((u32, u32), [f64; 3]); 2] = [
        (
            (445, 153),
            [
                428.146_057_128_906_25,
                856.990_722_656_25,
                407.884_948_730_468_75,
            ],
        ),
        (
            (445, 156),
            [
                428.146_057_128_906_25,
                856.174_255_371_093_8,
                407.884_948_730_468_75,
            ],
        ),
    ];
    for ((px, py), eye) in probe {
        assert_eq!(
            board_ray(px, py).0,
            eye.map(|v| v as f32),
            "texel ({px}, {py})"
        );
    }
    assert_eq!(board_frame().len(), (TEXTURE[0] * TEXTURE[1]) as usize);
}

#[test]
fn the_exact_walk_does_not_depend_on_the_headroom() {
    let [zero, one, two] = board(&board_sample());
    let exact = |walked: &[(Option<Hit>, Exact)]| -> Vec<_> {
        walked.iter().map(|(_, truth)| landing(truth.hit)).collect()
    };
    assert_eq!(exact(&zero), exact(&one));
    assert_eq!(exact(&zero), exact(&two));
    assert!(zero.iter().all(|(_, truth)| !truth.exhausted));
}

#[test]
fn the_probe_s_texel_lands_where_the_probe_saw_it() {
    // Texel (445, 153): the probe's f32 walk lands on (-619, 1, -640) at
    // headroom 0 and one voxel over in x at headroom 1, while its exact walk
    // lands on (-619, 1, -640) whatever the box.
    let [zero, one, two] = board(&[(445, 153)]);
    let voxel = |walked: &[(Option<Hit>, Exact)]| walked[0].0.map(|hit| hit.voxel);
    assert_eq!(voxel(&zero), Some([-619, 1, -640]));
    assert_eq!(voxel(&one), Some([-620, 1, -640]));
    assert!(voxel(&two).is_some());
    for walked in [&zero, &one, &two] {
        assert_eq!(walked[0].1.hit.map(|hit| hit.voxel), Some([-619, 1, -640]));
    }
}

#[test]
fn today_s_walk_drifts_from_the_exact_walk() {
    let [zero, one, two] = board(&board_sample());
    let stress = stress_scenes(3, 512);
    let mut spread = Tally::default();
    for scene in &stress {
        for (hit, truth) in walks(&scene.map, scene.far, &scene.rays) {
            spread.add(hit, &truth);
        }
    }
    let (zero_one, zero_two) = (moved(&zero, &one), moved(&zero, &two));
    println!(
        "board headroom 0: {:?}\nboard headroom 1: {:?}\nboard headroom 2: {:?}\n\
         moved by headroom 1: {zero_one}, by headroom 2: {zero_two}\nspread: {spread:?}",
        tally(&zero),
        tally(&one),
        tally(&two),
    );
    assert!(
        zero_one > 0 && zero_two > zero_one,
        "one spare layer moves texels, two more"
    );
    assert!(tally(&one).faults > tally(&zero).faults);
    assert!(spread.faults > 0);
}

/// The whole frame and a wider spread, run by hand in release:
/// `cargo test -p modulus --release -- --ignored --nocapture receipt`.
#[test]
#[ignore = "a receipt run by hand"]
fn receipt() {
    let [zero, one, two] = board(&board_frame());
    println!(
        "board frame, {} texels\n headroom 0: {:?}\n headroom 1: {:?}\n headroom 2: {:?}\n \
         moved by headroom 1: {}, by headroom 2: {}",
        zero.len(),
        tally(&zero),
        tally(&one),
        tally(&two),
        moved(&zero, &one),
        moved(&zero, &two),
    );
    for (reach, scenes) in stress_scenes(12, 2048).chunks(12).enumerate() {
        let mut spread = Tally::default();
        for scene in scenes {
            for (hit, truth) in walks(&scene.map, scene.far, &scene.rays) {
                spread.add(hit, &truth);
            }
        }
        println!("spread, reach band {reach}: {spread:?}");
    }
}
