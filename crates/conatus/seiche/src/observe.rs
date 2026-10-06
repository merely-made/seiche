// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! Observables over a layout: the numbers a composition's receipt reads.
//!
//! [`separation`] is [`Observable::Separation`](crate::Observable): the mean
//! distance between group centroids over the mean spread within a group, the
//! statistic G2 measured Group pull with (G2's `separation`, mean pairwise
//! centroid distance over mean per-group RMS radius). [`group_stress`] is
//! how well a layout keeps each group's own edge structure: Kamada–Kawai
//! stress over the pairs a group's internal edges connect, at the scale that
//! fits best, so a layout drawn larger or smaller is not penalized for it.
//!
//! Plan: `design_docs/mere_docs/implementation_strategy/2026-10-02_dynamics_grammar_plan.md`, G3.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

use crate::NodeKey;

/// Groups apart against groups spread, as one ratio and its parts.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Separation {
    /// Mean distance between two groups' centroids, over every pair.
    pub gap: f64,
    /// Mean RMS distance of a group's members from its centroid.
    pub spread: f64,
    /// `gap / spread`: above 1 when groups sit farther apart than they spread.
    pub ratio: f64,
    /// The closest pair of centroids, a stricter reading printed beside it.
    pub nearest_gap: f64,
}

/// [`Separation`] of `groups` (a group per node) at `positions`. Nodes with
/// no group, and groups with no member placed, are left out.
pub fn separation(
    positions: &[(NodeKey, (f64, f64))],
    groups: &HashMap<NodeKey, u32>,
) -> Separation {
    let mut members: BTreeMap<u32, Vec<(f64, f64)>> = BTreeMap::new();
    for (key, at) in positions {
        if let Some(group) = groups.get(key) {
            members.entry(*group).or_default().push(*at);
        }
    }
    let centres: Vec<((f64, f64), f64)> = members
        .values()
        .map(|points| {
            let n = points.len() as f64;
            let c = points
                .iter()
                .fold((0.0, 0.0), |c, p| (c.0 + p.0 / n, c.1 + p.1 / n));
            let spread = (points
                .iter()
                .map(|p| (p.0 - c.0).powi(2) + (p.1 - c.1).powi(2))
                .sum::<f64>()
                / n)
                .sqrt();
            (c, spread)
        })
        .collect();
    let (mut gap, mut pairs, mut nearest) = (0.0, 0.0, f64::INFINITY);
    for i in 0..centres.len() {
        for j in (i + 1)..centres.len() {
            let (a, b) = (centres[i].0, centres[j].0);
            let d = (a.0 - b.0).hypot(a.1 - b.1);
            gap += d;
            pairs += 1.0;
            nearest = nearest.min(d);
        }
    }
    if pairs == 0.0 {
        return Separation::default();
    }
    let gap = gap / pairs;
    let spread = centres.iter().map(|(_, s)| s).sum::<f64>() / centres.len() as f64;
    Separation {
        gap,
        spread,
        ratio: gap / spread.max(1e-9),
        nearest_gap: nearest,
    }
}

/// Scale-free stress inside groups: over every pair a group's internal edges
/// connect, `Σ (α·d/h − 1)² / N` with `h` the hop distance inside the group,
/// `d` the layout distance and `α` the best-fitting scale, one `α` for the
/// whole layout. Zero is every pair at its hop distance times one length;
/// higher is worse. `None` with no connected pair.
pub fn group_stress(
    positions: &[(NodeKey, (f64, f64))],
    edges: &[(NodeKey, NodeKey)],
    groups: &HashMap<NodeKey, u32>,
) -> Option<f64> {
    let pairs: Vec<(f64, f64)> = group_pairs(positions, edges, groups)
        .into_values()
        .flatten()
        .collect();
    scaled_stress(&pairs)
}

/// The same with each group fitting its own `α`, averaged over groups with
/// a connected pair: a group drawn larger or smaller than another is not
/// penalized for it.
pub fn group_stress_each(
    positions: &[(NodeKey, (f64, f64))],
    edges: &[(NodeKey, NodeKey)],
    groups: &HashMap<NodeKey, u32>,
) -> Option<f64> {
    let each: Vec<f64> = group_pairs(positions, edges, groups)
        .values()
        .filter_map(|pairs| scaled_stress(pairs))
        .collect();
    (!each.is_empty()).then(|| each.iter().sum::<f64>() / each.len() as f64)
}

/// `Σ (α·d/h − 1)² / N` at the `α` that minimizes it, `α = Σ(d/h) / Σ(d/h)²`.
fn scaled_stress(pairs: &[(f64, f64)]) -> Option<f64> {
    if pairs.is_empty() {
        return None;
    }
    let (s1, s2) = pairs.iter().fold((0.0, 0.0), |(s1, s2), (d, h)| {
        (s1 + d / h, s2 + (d / h).powi(2))
    });
    let alpha = if s2 > 0.0 { s1 / s2 } else { 0.0 };
    Some(
        pairs
            .iter()
            .map(|(d, h)| (alpha * d / h - 1.0).powi(2))
            .sum::<f64>()
            / pairs.len() as f64,
    )
}

/// Every pair a group's internal edges connect, as (layout distance, hops
/// inside the group), by group.
fn group_pairs(
    positions: &[(NodeKey, (f64, f64))],
    edges: &[(NodeKey, NodeKey)],
    groups: &HashMap<NodeKey, u32>,
) -> BTreeMap<u32, Vec<(f64, f64)>> {
    let at: HashMap<NodeKey, (f64, f64)> = positions.iter().copied().collect();
    let mut adjacency: HashMap<NodeKey, Vec<NodeKey>> = HashMap::new();
    for &(a, b) in edges {
        if a != b && groups.get(&a).is_some() && groups.get(&a) == groups.get(&b) {
            adjacency.entry(a).or_default().push(b);
            adjacency.entry(b).or_default().push(a);
        }
    }
    let mut out: BTreeMap<u32, Vec<(f64, f64)>> = BTreeMap::new();
    let mut sources: Vec<NodeKey> = adjacency.keys().copied().collect();
    sources.sort_by_key(|k| k.index());
    for source in sources {
        let mut hops: HashMap<NodeKey, u32> = HashMap::from([(source, 0)]);
        let mut queue = VecDeque::from([source]);
        let mut seen = HashSet::from([source]);
        while let Some(node) = queue.pop_front() {
            for &next in adjacency.get(&node).into_iter().flatten() {
                if seen.insert(next) {
                    hops.insert(next, hops[&node] + 1);
                    queue.push_back(next);
                }
            }
        }
        for (target, h) in hops {
            if target.index() <= source.index() {
                continue;
            }
            let (Some(p), Some(q)) = (at.get(&source), at.get(&target)) else {
                continue;
            };
            out.entry(groups[&source])
                .or_default()
                .push(((p.0 - q.0).hypot(p.1 - q.1), f64::from(h)));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keys(n: usize) -> Vec<NodeKey> {
        (0..n).map(NodeKey::new).collect()
    }

    /// Two tight groups far apart read a high ratio; the same points with the
    /// groups dealt across them read near zero.
    #[test]
    fn separation_reads_apart_groups_high_and_mixed_groups_low() {
        let k = keys(4);
        let at = vec![
            (k[0], (0.0, 0.0)),
            (k[1], (10.0, 0.0)),
            (k[2], (1000.0, 0.0)),
            (k[3], (1010.0, 0.0)),
        ];
        let apart: HashMap<_, _> = [(k[0], 0), (k[1], 0), (k[2], 1), (k[3], 1)].into();
        let mixed: HashMap<_, _> = [(k[0], 0), (k[1], 1), (k[2], 0), (k[3], 1)].into();
        let a = separation(&at, &apart);
        let m = separation(&at, &mixed);
        assert!((a.gap - 1000.0).abs() < 1e-9 && (a.spread - 5.0).abs() < 1e-9);
        assert!(a.ratio > 100.0 && m.ratio < 0.1, "{a:?} {m:?}");
    }

    /// A path drawn at even spacing has zero stress at any scale; folded back
    /// on itself it does not.
    #[test]
    fn group_stress_is_zero_for_a_straight_path_and_not_for_a_folded_one() {
        let k = keys(3);
        let edges = vec![(k[0], k[1]), (k[1], k[2])];
        let one: HashMap<_, _> = k.iter().map(|&key| (key, 0)).collect();
        let straight = vec![
            (k[0], (0.0, 0.0)),
            (k[1], (50.0, 0.0)),
            (k[2], (100.0, 0.0)),
        ];
        let folded = vec![(k[0], (0.0, 0.0)), (k[1], (50.0, 0.0)), (k[2], (1.0, 0.0))];
        assert!(group_stress(&straight, &edges, &one).unwrap() < 1e-12);
        assert!(group_stress(&folded, &edges, &one).unwrap() > 0.1);
        let split: HashMap<_, _> = [(k[0], 0), (k[1], 1), (k[2], 1)].into();
        assert!(
            group_stress(&straight, &edges, &split).is_some(),
            "one internal edge"
        );
        let alone: HashMap<_, _> = [(k[0], 0), (k[1], 1), (k[2], 2)].into();
        assert_eq!(group_stress(&straight, &edges, &alone), None);
    }

    /// Two straight paths drawn at different spacings: each group alone is
    /// stress-free, and one scale for both is not.
    #[test]
    fn per_group_stress_forgives_a_group_drawn_at_its_own_scale() {
        let k = keys(6);
        let edges = vec![(k[0], k[1]), (k[1], k[2]), (k[3], k[4]), (k[4], k[5])];
        let groups: HashMap<_, _> = k
            .iter()
            .enumerate()
            .map(|(i, &key)| (key, (i / 3) as u32))
            .collect();
        let at = vec![
            (k[0], (0.0, 0.0)),
            (k[1], (50.0, 0.0)),
            (k[2], (100.0, 0.0)),
            (k[3], (0.0, 500.0)),
            (k[4], (200.0, 500.0)),
            (k[5], (400.0, 500.0)),
        ];
        assert!(group_stress_each(&at, &edges, &groups).unwrap() < 1e-12);
        assert!(group_stress(&at, &edges, &groups).unwrap() > 0.1);
    }
}
