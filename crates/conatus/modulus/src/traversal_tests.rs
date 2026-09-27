// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! The shared traversal measured against an exact walk of the same rays.
//!
//! [`mirror`] is `brick_dda.wgsl` in f32 on the CPU, beside the walk it
//! replaced on 2026-09-27, which added `1 / |direction|` to each crossing
//! per step and is kept as the positive control; [`exact`] walks the same
//! f32 ray in f64 with every crossing taken afresh from the eye, and records
//! how near each ray came to a tie. The rays are Isometry's board frame (its
//! probe's camera, rebuilt bit for bit, over the probe's box with zero to
//! two spare layers of headroom) and a seeded spread of boxes and rays from
//! near the origin to 240,000 voxels out.

use std::collections::BTreeMap;
use std::sync::OnceLock;

use super::*;

mod exact;
mod mirror;
mod scenes;

use exact::{Exact, exact};
pub(crate) use mirror::brick_material_at;
use mirror::{Crossings, Hit, brick_dda};
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

/// One ray walked three ways: by the shader, by the accumulated walk it
/// replaced, and exactly.
struct Walked {
    direct: Option<Hit>,
    accumulated: Option<Hit>,
    exact: Exact,
}

impl Walked {
    fn by(&self, crossings: Crossings) -> Option<Hit> {
        match crossings {
            Crossings::Direct => self.direct,
            Crossings::Accumulated => self.accumulated,
        }
    }
}

/// Every ray into `map`, walked three ways.
fn walks(map: &BrickMap, far: f32, rays: &[Ray]) -> Vec<Walked> {
    let space = BrickTraceSpace::from_map(map);
    let material = |voxel| brick_material_at(map, voxel);
    rays.iter()
        .map(|&(eye, direction)| Walked {
            direct: brick_dda(&space, far, eye, direction, Crossings::Direct, material),
            accumulated: brick_dda(
                &space,
                far,
                eye,
                direction,
                Crossings::Accumulated,
                material,
            ),
            exact: exact(&space, far, eye, direction, material),
        })
        .collect()
}

/// The board's `texels` at headroom 0, 1 and 2.
fn board(texels: &[(u32, u32)]) -> [Vec<Walked>; 3] {
    let rays: Vec<Ray> = texels.iter().map(|&(px, py)| board_ray(px, py)).collect();
    [0, 1, 2].map(|headroom| walks(&board_map(headroom), BOARD_FAR, &rays))
}

/// [`board_sample`], walked once for every test that reads it.
fn sampled() -> &'static [Vec<Walked>; 3] {
    static SAMPLE: OnceLock<[Vec<Walked>; 3]> = OnceLock::new();
    SAMPLE.get_or_init(|| board(&board_sample()))
}

/// The seeded spread, three boxes a reach.
fn spread() -> Vec<Walked> {
    stress_scenes(3, 512)
        .iter()
        .flat_map(|scene| walks(&scene.map, scene.far, &scene.rays))
        .collect()
}

/// How many rays one walk lands differently under two headrooms.
fn moved(a: &[Walked], b: &[Walked], crossings: Crossings) -> usize {
    a.iter()
        .zip(b)
        .filter(|(a, b)| landing(a.by(crossings)) != landing(b.by(crossings)))
        .count()
}

fn tally(walked: &[Walked], crossings: Crossings) -> Tally {
    let mut tally = Tally::default();
    for ray in walked {
        tally.add(ray.by(crossings), &ray.exact);
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
    let [zero, one, two] = sampled();
    let exact =
        |walked: &[Walked]| -> Vec<_> { walked.iter().map(|ray| landing(ray.exact.hit)).collect() };
    assert_eq!(exact(zero), exact(one));
    assert_eq!(exact(zero), exact(two));
    assert!(zero.iter().all(|ray| !ray.exact.exhausted));
}

#[test]
fn the_probe_s_texel_lands_on_the_exact_voxel_at_any_headroom() {
    // Texel (445, 153). The accumulated walk lands on (-619, 1, -640) at
    // headroom 0 and one voxel over in x at headroom 1, as the probe saw;
    // the shader's walk and the exact walk land on (-619, 1, -640) whatever
    // the box.
    let exact_voxel = Some([-619, 1, -640]);
    let walked = board(&[(445, 153)]);
    let voxel =
        |headroom: usize, crossings| walked[headroom][0].by(crossings).map(|hit: Hit| hit.voxel);
    assert_eq!(voxel(0, Crossings::Accumulated), exact_voxel);
    assert_eq!(voxel(1, Crossings::Accumulated), Some([-620, 1, -640]));
    for (headroom, walked) in walked.iter().enumerate() {
        assert_eq!(voxel(headroom, Crossings::Direct), exact_voxel);
        assert_eq!(walked[0].exact.hit.map(|hit| hit.voxel), exact_voxel);
    }
}

#[test]
fn headroom_no_longer_moves_a_texel() {
    let [zero, one, two] = sampled();
    // The positive control: the accumulated walk moves the probe's own 330
    // texels at one spare layer and 660 at two.
    let control = Crossings::Accumulated;
    assert_eq!(
        (moved(zero, one, control), moved(zero, two, control)),
        (330, 660)
    );
    let shader = Crossings::Direct;
    assert_eq!((moved(zero, one, shader), moved(zero, two, shader)), (0, 0));
}

#[test]
fn the_shader_s_walk_leaves_the_exact_walk_only_at_ties() {
    let [zero, one, two] = sampled();
    let spread = spread();
    for crossings in [Crossings::Accumulated, Crossings::Direct] {
        println!(
            "{crossings:?}\n board headroom 0: {:?}\n board headroom 1: {:?}\n \
             board headroom 2: {:?}\n spread: {:?}",
            tally(zero, crossings),
            tally(one, crossings),
            tally(two, crossings),
            tally(&spread, crossings),
        );
    }
    let control = Crossings::Accumulated;
    assert!(tally(one, control).faults > tally(zero, control).faults);
    assert!(tally(&spread, control).faults > 0, "the control drifts");
    for walked in [zero, one, two, &spread] {
        let shader = tally(walked, Crossings::Direct);
        assert_eq!((shader.faults, shader.exhausted), (0, 0), "{shader:?}");
    }
}

/// The whole frame and a wider spread, run by hand in release:
/// `cargo test -p modulus --release -- --ignored --nocapture receipt`.
#[test]
#[ignore = "a receipt run by hand"]
fn receipt() {
    let [zero, one, two] = board(&board_frame());
    let bands: Vec<Vec<Walked>> = stress_scenes(12, 2048)
        .chunks(12)
        .map(|band| {
            band.iter()
                .flat_map(|scene| walks(&scene.map, scene.far, &scene.rays))
                .collect()
        })
        .collect();
    for crossings in [Crossings::Accumulated, Crossings::Direct] {
        println!(
            "{crossings:?}, board frame of {} texels\n headroom 0: {:?}\n headroom 1: {:?}\n \
             headroom 2: {:?}\n moved by headroom 1: {}, by headroom 2: {}",
            zero.len(),
            tally(&zero, crossings),
            tally(&one, crossings),
            tally(&two, crossings),
            moved(&zero, &one, crossings),
            moved(&zero, &two, crossings),
        );
        for (reach, walked) in bands.iter().enumerate() {
            println!(
                " spread, reach band {reach}: {:?}",
                tally(walked, crossings)
            );
        }
    }
}
