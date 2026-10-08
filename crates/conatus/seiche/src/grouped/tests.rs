// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! The grouped combinator's receipts: built from energy terms it passes G1's
//! descent and reciprocity instruments, and the full-force spread (F9's
//! rejected option) fails reciprocity on groups of unequal size, in the same
//! run, with equal groups as the control on why.

use super::*;
use crate::instruments::{Probe, Vector, balance, read};
use crate::laws::Rng;
use crate::{BarnesHutConfig, BarnesHutRepulsion, Boundary, EdgeSpring, NodeExclusion};

/// A fixture's keys, edges and groups.
type Fixture = (Vec<NodeKey>, Vec<(NodeKey, NodeKey)>, Vec<(NodeKey, u32)>);

/// Forty nodes in five groups of 3, 5, 8, 10 and 14, joined by a seeded
/// tree and twenty chords, so edges run within and across groups.
fn fixture(sizes: &[usize]) -> Fixture {
    let n: usize = sizes.iter().sum();
    let keys: Vec<NodeKey> = (0..n).map(NodeKey::new).collect();
    let mut rng = Rng::new(7);
    let mut edges = Vec::new();
    for i in 1..n {
        let j = ((rng.unit() * i as f32) as usize).min(i - 1);
        edges.push((keys[j], keys[i]));
    }
    for _ in 0..20 {
        let (a, b) = (
            ((rng.unit() * n as f32) as usize).min(n - 1),
            ((rng.unit() * n as f32) as usize).min(n - 1),
        );
        if a != b {
            edges.push((keys[a], keys[b]));
        }
    }
    let mut groups = Vec::new();
    let mut next = 0;
    for (g, &size) in sizes.iter().enumerate() {
        for _ in 0..size {
            groups.push((keys[next], g as u32));
            next += 1;
        }
    }
    (keys, edges, groups)
}

/// Charge's repulsion between groups at weight 16 (F71), Springs within
/// each.
fn charge_between_springs_within(
    groups: &[(NodeKey, u32)],
    edges: &[(NodeKey, NodeKey)],
    theta: f32,
    spread: Spread,
) -> Grouped {
    let partition = Partition::new(groups.iter().copied(), edges);
    let charge = BarnesHutRepulsion {
        strength: 6_000.0,
        config: BarnesHutConfig {
            theta,
            ..BarnesHutConfig::default()
        },
        ..BarnesHutRepulsion::default()
    };
    let outer: Vec<Box<dyn Force>> = vec![Box::new(
        crate::Weighted::new(Box::new(charge), 16.0).unwrap(),
    )];
    let inner = (0..partition.len())
        .map(|_| {
            vec![
                Box::new(NodeExclusion::default()) as Box<dyn Force>,
                Box::new(EdgeSpring::default()),
                Box::new(Boundary::default()),
            ]
        })
        .collect();
    Grouped::new(partition, outer, inner).with_spread(spread)
}

fn starts(probe: &Probe) -> Vec<Vec<Vector>> {
    (1..=3).map(|seed| probe.scatter(seed, 300.0)).collect()
}

#[test]
fn the_partition_masks_edges_and_builds_the_groups_graph() {
    let (_, edges, groups) = fixture(&[3, 5, 8, 10, 14]);
    let partition = Partition::new(groups.iter().copied(), &edges);
    assert_eq!(partition.len(), 5);
    let inside: usize = (0..5).map(|g| partition.group(g).1.len()).sum();
    let (keys, outer) = partition.outer_graph();
    assert_eq!(keys.len(), 5);
    assert!(
        inside > 0 && !outer.is_empty(),
        "{inside} inside, {outer:?}"
    );
    let of: HashMap<NodeKey, u32> = groups.iter().copied().collect();
    for g in 0..5 {
        for (a, b) in partition.group(g).1 {
            assert_eq!(of[a], of[b]);
        }
    }
    let crossing = edges.iter().filter(|(a, b)| of[a] != of[b]).count();
    assert_eq!(
        inside + crossing,
        edges.len(),
        "every edge is inside or across"
    );
}

/// Weight share: every term agrees with its declared class (E) by G1's
/// instruments, the outer charge read on the exact rung (θ 0) with θ 0.5
/// printed beside it (F16). Full force: the outer term's forces no longer
/// sum to zero on groups of unequal size, and do on equal ones.
#[test]
fn weight_share_passes_descent_and_reciprocity_and_full_force_fails_on_unequal_groups() {
    let (keys, edges, groups) = fixture(&[3, 5, 8, 10, 14]);
    let mut probe = Probe::new(&keys, &edges);
    let starts = starts(&probe);
    let exact = || -> Box<dyn Force> {
        Box::new(charge_between_springs_within(
            &groups,
            &edges,
            0.0,
            Spread::WeightShare,
        ))
    };
    let terms = exact().terms();
    assert_eq!(terms.len(), 4);
    assert_eq!(terms[0].topology, Topology::Groups);
    for term in 0..terms.len() {
        let reading = read(&mut probe, &exact, term, &starts);
        println!(
            "weight share, term {term} {}: rise {:?} gradient {:?} balance {:?}",
            reading.term.name, reading.rise, reading.gradient, reading.balance
        );
        reading.agrees().unwrap();
    }
    let approximate = || -> Box<dyn Force> {
        Box::new(charge_between_springs_within(
            &groups,
            &edges,
            0.5,
            Spread::WeightShare,
        ))
    };
    let rung = read(&mut probe, &approximate, 0, &starts);
    println!(
        "weight share, outer charge at θ 0.5: gradient {:?} balance {:?}",
        rung.gradient, rung.balance
    );

    let ones = vec![1.0; probe.len()];
    let full = charge_between_springs_within(&groups, &edges, 0.0, Spread::FullForce);
    let share = charge_between_springs_within(&groups, &edges, 0.0, Spread::WeightShare);
    let outer_full = full.isolate(0).unwrap();
    let outer_share = share.isolate(0).unwrap();
    let (mut worst_full, mut worst_share) = (0.0f64, 0.0f64);
    for start in &starts {
        worst_full =
            worst_full.max(balance(&mut probe, outer_full.as_ref(), &ones, start).relative());
        worst_share =
            worst_share.max(balance(&mut probe, outer_share.as_ref(), &ones, start).relative());
    }
    println!("outer charge balance: weight share {worst_share:.3e}, full force {worst_full:.3e}");
    assert!(worst_share <= crate::instruments::tolerance::BALANCE);
    assert!(
        worst_full > 100.0 * crate::instruments::tolerance::BALANCE,
        "full force should fail reciprocity on unequal groups: {worst_full}"
    );

    let (keys, edges, equal) = fixture(&[8, 8, 8, 8, 8]);
    let mut probe = Probe::new(&keys, &edges);
    let starts = (1..=3).map(|s| probe.scatter(s, 300.0)).collect::<Vec<_>>();
    let full = charge_between_springs_within(&equal, &edges, 0.0, Spread::FullForce);
    let outer = full.isolate(0).unwrap();
    let ones = vec![1.0; probe.len()];
    let worst = starts
        .iter()
        .map(|s| balance(&mut probe, outer.as_ref(), &ones, s).relative())
        .fold(0.0f64, f64::max);
    println!("outer charge balance, full force on equal groups: {worst:.3e}");
    assert!(worst <= crate::instruments::tolerance::BALANCE);
}

/// The mask: an inner law sees only its group, so a pair split across two
/// groups feels no inner repulsion, and two members of one group do.
#[test]
fn the_inner_law_acts_only_within_a_group() {
    let keys: Vec<NodeKey> = (0..3).map(NodeKey::new).collect();
    let groups = vec![(keys[0], 0), (keys[1], 0), (keys[2], 1)];
    let partition = Partition::new(groups, &[]);
    let inner = (0..2)
        .map(|_| vec![Box::new(NodeExclusion::default()) as Box<dyn Force>])
        .collect();
    let grouped = Grouped::new(partition, Vec::new(), inner);
    let mut probe = Probe::new(&keys, &[]);
    probe.place(&[
        Vector::new(0.0, 0.0),
        Vector::new(50.0, 0.0),
        Vector::new(0.0, 50.0),
    ]);
    let f = probe.forces(&grouped);
    assert!(
        f[0].x < 0.0 && f[0].y.abs() < 1e-6,
        "pushed from its group mate only: {:?}",
        f[0]
    );
    assert!(f[2].length() < 1e-6, "alone in its group: {:?}", f[2]);
}
