// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! The resident graph's receipts: the hashed grid computes
//! `seiche::NodeExclusion`'s law at any extent, weighted springs match a host
//! loop of the same law, pins hold, slots are generational, growth keeps live
//! state, and the bounded reads return exactly what they name.
//!
//! Needs a real adapter, so each test skips where there is none.

#![cfg(feature = "resident")]

use std::time::Instant;

use conatus::resident::{
    ExclusionParams, GraphError, GraphParams, ResidentClient, ResidentGraph, SLOT_ALIVE,
    SLOT_PINNED,
};
use seiche::{NodeExclusion, NodeExclusionParams, node_exclusion_reference};

fn client() -> Option<ResidentClient> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        flags: wgpu::InstanceFlags::default(),
        memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
        backend_options: wgpu::BackendOptions::default(),
        display: None,
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))
    .ok()?;
    let backend = adapter.get_info().backend;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("conatus graph receipt"),
        ..Default::default()
    }))
    .ok()?;
    Some(ResidentClient::init(cubecl::wgpu::WgpuSetup {
        instance,
        adapter,
        device,
        queue,
        backend,
    }))
}

macro_rules! graph_or_skip {
    ($capacity:expr, $params:expr) => {{
        let Some(client) = client() else {
            eprintln!("no wgpu adapter: skipping");
            return;
        };
        ResidentGraph::new(&client, $capacity, $params)
    }};
}

fn exclusion_only() -> GraphParams {
    GraphParams::default()
}

/// Springs and centering off, so the exclusion is the whole force.
fn law() -> ExclusionParams {
    let d = NodeExclusion::default();
    ExclusionParams {
        strength: d.strength,
        cutoff: d.cutoff,
        min_distance: d.min_distance,
    }
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

/// A scatter at a settled layout's density, every twentieth body given a twin
/// a few units away (the `min_distance` floor) and every fiftieth a coincident
/// one (a zero displacement).
fn scatter(n: usize, spacing: f32) -> Vec<[f32; 4]> {
    let mut rng = Lcg(0x2026_1125);
    let extent = (n as f32).sqrt() * spacing * 0.5;
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let p = [rng.next() * extent, rng.next() * extent, 0.0, 0.0];
        out.push(p);
        if out.len() % 20 == 0 && out.len() < n {
            out.push([p[0] + 3.0, p[1] - 2.0, 0.0, 0.0]);
        }
        if out.len() % 50 == 0 && out.len() < n {
            out.push(p);
        }
    }
    out
}

fn reference(positions: &[[f32; 4]], p: ExclusionParams) -> (Vec<f32>, Vec<f32>) {
    let xs: Vec<f32> = positions.iter().map(|p| p[0]).collect();
    let ys: Vec<f32> = positions.iter().map(|p| p[1]).collect();
    node_exclusion_reference(
        &xs,
        &ys,
        NodeExclusionParams {
            strength: p.strength,
            cutoff: p.cutoff,
            min_distance: p.min_distance,
        },
    )
    .expect("equal lengths")
}

/// Per-body relative error of `gpu` against the CPU law: (mean, worst).
fn agreement(gpu: &[[f32; 4]], fx: &[f32], fy: &[f32]) -> (f64, f64) {
    let (mut worst, mut mean) = (0.0f64, 0.0f64);
    for (i, force) in gpu.iter().enumerate() {
        let magnitude = (fx[i] as f64).hypot(fy[i] as f64).max(1e-6);
        let relative = ((force[0] - fx[i]) as f64).hypot((force[1] - fy[i]) as f64) / magnitude;
        worst = worst.max(relative);
        mean += relative;
    }
    (mean / gpu.len() as f64, worst)
}

fn check_exclusion(graph: &mut ResidentGraph, positions: &[[f32; 4]], what: &str) {
    let params = graph.params().exclusion;
    let (fx, fy) = reference(positions, params);
    graph.insert_many(positions).expect("insert");
    graph.compute_forces();
    let gpu = graph.read_forces().expect("forces");
    let (mean, worst) = agreement(&gpu, &fx, &fy);
    println!("{what}: relative error mean {mean:.2e} worst {worst:.2e}");
    assert!(worst < 1e-3, "{what}: mean {mean:.2e}, worst {worst:.2e}");
}

fn exclusion_params() -> GraphParams {
    GraphParams {
        spring_k: 0.0,
        centering: 0.0,
        exclusion: law(),
        ..exclusion_only()
    }
}

#[test]
fn the_hashed_grid_computes_node_exclusions_law_at_1k_10k_and_50k() {
    let Some(client) = client() else {
        return;
    };
    for n in [1_000usize, 10_000, 50_000] {
        let mut graph = ResidentGraph::new(&client, n, exclusion_params());
        check_exclusion(&mut graph, &scatter(n, 140.0), &format!("n={n}"));
    }
}

#[test]
fn a_sparse_layout_of_huge_extent_is_still_exact() {
    let mut graph = graph_or_skip!(2_048, exclusion_params());
    let mut rng = Lcg(0x5eed);
    // A wide sparse field, three dense knots far from it, and two outliers
    // at a hundred million units.
    let mut positions: Vec<[f32; 4]> = (0..1_500)
        .map(|_| [rng.next() * 5.0e6, rng.next() * 5.0e6, 0.0, 0.0])
        .collect();
    for centre in [[-4.0e6f32, 3.0e6], [2.5e6, -1.0e6], [0.0, 0.0]] {
        for _ in 0..150 {
            positions.push([
                centre[0] + rng.next() * 600.0,
                centre[1] + rng.next() * 600.0,
                0.0,
                0.0,
            ]);
        }
    }
    positions.push([1.0e8, 1.0e8, 0.0, 0.0]);
    positions.push([1.0e8 + 400.0, 1.0e8 - 300.0, 0.0, 0.0]);
    check_exclusion(&mut graph, &positions, "huge extent");
}

fn spring_reference(
    positions: &[[f32; 4]],
    pairs: &[(usize, usize, f32)],
    k: f32,
    rest: f32,
) -> Vec<[f32; 3]> {
    let mut f = vec![[0.0f32; 3]; positions.len()];
    for &(a, b, w) in pairs {
        let d: Vec<f32> = (0..3).map(|c| positions[b][c] - positions[a][c]).collect();
        let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-4);
        let scale = k * w * (len - rest) / len;
        for c in 0..3 {
            f[a][c] += d[c] * scale;
            f[b][c] -= d[c] * scale;
        }
    }
    f
}

#[test]
fn weighted_springs_match_a_host_loop_of_the_same_law() {
    let mut params = exclusion_params();
    params.exclusion.strength = 0.0;
    let mut graph = graph_or_skip!(512, params);
    let positions = scatter(400, 60.0);
    let slots = graph.insert_many(&positions).expect("insert");
    let mut rng = Lcg(77);
    let pairs: Vec<(usize, usize, f32)> = (0..900)
        .map(|_| {
            let a = ((rng.next() * 0.5 + 0.5) * 399.0) as usize;
            let b = ((rng.next() * 0.5 + 0.5) * 399.0) as usize;
            (a, b, 0.25 + (rng.next() + 1.0))
        })
        .filter(|(a, b, _)| a != b)
        .collect();
    let set: Vec<_> = pairs
        .iter()
        .map(|&(a, b, w)| (slots[a], slots[b], w))
        .collect();
    graph.set_springs(41, &set).expect("springs");
    assert_eq!(graph.spring_revision(), 41);
    graph.compute_forces();
    let gpu = graph.read_forces().expect("forces");
    let cpu = spring_reference(&positions, &pairs, params.spring_k, params.rest_length);
    for (i, (g, c)) in gpu.iter().zip(&cpu).enumerate() {
        let magnitude = c.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-3);
        let error = (0..3).map(|k| (g[k] - c[k]).powi(2)).sum::<f32>().sqrt();
        assert!(error / magnitude < 1e-3, "body {i}: gpu {g:?} cpu {c:?}");
    }
    // One stale endpoint refuses the whole call and leaves the set as it was.
    graph.remove(slots[0]).expect("remove");
    let refused = graph.set_springs(42, &[(slots[1], slots[2], 1.0), (slots[0], slots[3], 1.0)]);
    assert_eq!(
        refused,
        Err(GraphError::SpringPair {
            pair: 1,
            slot: slots[0]
        })
    );
    assert_eq!(graph.spring_revision(), 41);
}

#[test]
fn a_pinned_node_holds_its_target_while_neighbours_move() {
    let mut graph = graph_or_skip!(64, exclusion_params());
    let slots = graph
        .insert_many(&[
            [0.0, 0.0, 0.0, 0.0],
            [20.0, 0.0, 0.0, 0.0],
            [0.0, 25.0, 0.0, 0.0],
        ])
        .expect("insert");
    let target = [5.0, 7.0, 0.0];
    graph.pin(slots[0], target).expect("pin");
    let before = graph.read_positions().expect("positions");
    for _ in 0..40 {
        graph.step();
    }
    let after = graph.read_positions().expect("positions");
    assert_eq!(after[0][..3], target, "the pin did not hold");
    assert!(graph.read_flags().expect("flags")[0] & SLOT_PINNED != 0);
    assert_ne!(after[1], before[1], "the neighbour did not move");
    assert_ne!(after[2], before[2]);
    // Re-targeting is one write; unpinning releases it to the forces.
    graph.pin(slots[0], [-5.0, 0.0, 0.0]).expect("retarget");
    graph.step();
    assert_eq!(
        graph.read_positions().expect("positions")[0][..3],
        [-5.0, 0.0, 0.0]
    );
    graph.unpin(slots[0]).expect("unpin");
    for _ in 0..5 {
        graph.step();
    }
    assert_ne!(
        graph.read_positions().expect("positions")[0][..3],
        [-5.0, 0.0, 0.0]
    );
}

#[test]
fn removed_slots_are_reused_with_a_new_generation_and_stale_writes_are_refused() {
    let mut graph = graph_or_skip!(64, exclusion_params());
    let slots = graph
        .insert_many(&[
            [0.0, 0.0, 0.0, 0.0],
            [30.0, 0.0, 0.0, 0.0],
            [0.0, 400.0, 0.0, 0.0],
        ])
        .expect("insert");
    graph.compute_forces();
    let both = graph.read_forces().expect("forces");
    assert!(both[0][0] != 0.0, "the neighbour exerted nothing");

    graph.remove(slots[1]).expect("remove");
    graph.compute_forces();
    let alone = graph.read_forces().expect("forces");
    assert_eq!(alone[1], [0.0; 4], "a dead slot felt a force");
    // The only body left within the cutoff of slot 0 is far above it.
    let (fx, fy) = reference(&[[0.0, 0.0, 0.0, 0.0], [0.0, 400.0, 0.0, 0.0]], law());
    assert!((alone[0][0] - fx[0]).abs() < 1e-3 && (alone[0][1] - fy[0]).abs() < 1e-3);
    assert_eq!(graph.read_flags().expect("flags")[1], 0);

    assert_eq!(
        graph.pin(slots[1], [1.0; 3]),
        Err(GraphError::StaleSlot(slots[1]))
    );
    assert_eq!(graph.remove(slots[1]), Err(GraphError::StaleSlot(slots[1])));
    assert!(graph.read_slots(&[slots[1]]).is_err());

    let again = graph.insert([30.0, 0.0, 0.0, 0.0]).expect("reinsert");
    assert_eq!(again.index, slots[1].index);
    assert_eq!(again.generation, slots[1].generation + 1);
    assert_eq!(
        graph.pin(slots[1], [1.0; 3]),
        Err(GraphError::StaleSlot(slots[1]))
    );
    assert_eq!(graph.slot_at(again.index), Some(again));
    assert_eq!(graph.len(), 3);
    assert_eq!(graph.read_flags().expect("flags")[1], SLOT_ALIVE);
    graph.compute_forces();
    assert_eq!(graph.read_forces().expect("forces")[0][0], both[0][0]);

    // A spring set from before the removal does not follow the new occupant.
    graph
        .set_springs(1, &[(slots[0], again, 1.0)])
        .expect("springs");
    graph.remove(again).expect("remove");
    let third = graph.insert([30.0, 0.0, 0.0, 0.0]).expect("reinsert");
    assert_eq!(third.index, again.index);
    let mut quiet = exclusion_params();
    quiet.exclusion.strength = 0.0;
    graph.set_params(quiet);
    graph.compute_forces();
    assert_eq!(graph.read_forces().expect("forces")[0], [0.0; 4]);
}

#[test]
fn growth_changes_the_epoch_and_keeps_live_state() {
    let mut graph = graph_or_skip!(8, exclusion_params());
    assert!(graph.capacity() >= 8);
    let first = scatter(graph.capacity(), 50.0);
    let slots = graph.insert_many(&first).expect("insert");
    graph.pin(slots[2], [9.0, 9.0, 0.0]).expect("pin");
    graph.remove(slots[3]).expect("remove");
    for _ in 0..5 {
        graph.step();
    }
    let (epoch, capacity) = (graph.allocation_epoch(), graph.capacity());
    let (positions, velocities) = (
        graph.read_positions().expect("p"),
        graph.read_velocities().expect("v"),
    );
    assert!(velocities.iter().any(|v| v[0] != 0.0));
    let more = graph.insert_many(&scatter(capacity, 50.0)).expect("grow");
    assert_eq!(graph.allocation_epoch(), epoch + 1);
    assert_eq!(graph.capacity(), capacity * 2);
    assert_eq!(graph.positions_lease().allocation_epoch, epoch + 1);
    let (grown_p, grown_v) = (
        graph.read_positions().expect("p"),
        graph.read_velocities().expect("v"),
    );
    // The removed slot's index was reused first; compare every other old slot.
    for (i, slot) in slots.iter().enumerate() {
        if i != 3 {
            assert_eq!(grown_p[slot.index as usize], positions[i], "position {i}");
            assert_eq!(grown_v[slot.index as usize], velocities[i], "velocity {i}");
        }
    }
    assert_eq!(more.len(), capacity);
    // The pin and the generations survived the move.
    let flags = graph.read_flags().expect("flags");
    assert!(flags[2] & SLOT_PINNED != 0);
    graph.step();
    assert_eq!(graph.read_positions().expect("p")[2][..3], [9.0, 9.0, 0.0]);
    assert_eq!(
        graph.pin(slots[3], [0.0; 3]),
        Err(GraphError::StaleSlot(slots[3]))
    );
}

#[test]
fn pick_and_read_slots_return_exactly_what_they_name() {
    let mut graph = graph_or_skip!(64, exclusion_params());
    let positions: Vec<[f32; 4]> = (0..10)
        .map(|i| [i as f32 * 100.0, -(i as f32) * 50.0, 0.0, 0.0])
        .collect();
    let slots = graph.insert_many(&positions).expect("insert");
    assert_eq!(
        graph.pick([302.0, -148.0, 0.0], 10.0).expect("pick"),
        Some(slots[3])
    );
    assert_eq!(graph.pick([350.0, -50.0, 0.0], 10.0).expect("pick"), None);
    // Of two inside the radius, the nearer wins.
    assert_eq!(
        graph.pick([440.0, -200.0, 0.0], 100.0).expect("pick"),
        Some(slots[4])
    );
    let order = [slots[7], slots[0], slots[7], slots[5]];
    let gathered = graph
        .read_slots(&order)
        .expect("gather")
        .wait()
        .expect("read");
    let expected: Vec<_> = order.iter().map(|s| positions[s.index as usize]).collect();
    assert_eq!(gathered, expected);
    assert!(
        graph
            .read_slots(&[])
            .expect("empty")
            .wait()
            .expect("read")
            .is_empty()
    );
    // A removed slot is never picked, and the next pick sees the survivor.
    graph.remove(slots[3]).expect("remove");
    assert_eq!(graph.pick([302.0, -148.0, 0.0], 10.0).expect("pick"), None);
    let reused = graph.insert([302.0, -148.0, 0.0, 0.0]).expect("insert");
    assert_eq!(
        graph.pick([302.0, -148.0, 0.0], 10.0).expect("pick"),
        Some(reused)
    );
    assert_eq!(reused.generation, slots[3].generation + 1);
    assert!(graph.max_speed().expect("settle") >= 0.0);
}

#[test]
fn the_flags_lease_and_the_positions_lease_share_a_contract() {
    let mut graph = graph_or_skip!(64, exclusion_params());
    graph.insert_many(&scatter(10, 50.0)).expect("insert");
    let (p, f) = (graph.positions_lease(), graph.flags_lease());
    assert_eq!(p.lease.shape, [10, 4, 1]);
    assert_eq!(f.lease.shape, [10, 1, 1]);
    assert_eq!(p.allocation_epoch, f.allocation_epoch);
    assert_eq!(p.lease.stamp.revision, f.lease.stamp.revision);
    assert!(p.lease.byte_len() <= p.lease.size);
    let revision = p.lease.stamp.revision;
    graph.step();
    assert_eq!(graph.positions_lease().lease.stamp.revision, revision + 1);
    graph.flush().expect("flush");
}

/// Step cost and steady-state traffic. Opt in: `-- --ignored --nocapture`.
#[test]
#[ignore = "cost probe"]
fn step_cost_and_traffic() {
    let Some(client) = client() else {
        return;
    };
    for n in [2_000usize, 10_000, 50_000] {
        let mut graph = ResidentGraph::new(&client, n, GraphParams::default());
        let positions = scatter(n, 140.0);
        let slots = graph.insert_many(&positions).expect("insert");
        let pairs: Vec<_> = (1..n)
            .step_by(2)
            .map(|i| (slots[i - 1], slots[i], 1.0))
            .collect();
        graph.set_springs(1, &pairs).expect("springs");
        for _ in 0..4 {
            graph.step();
        }
        graph.max_speed().expect("warm");
        let steps = 30;
        let before = graph.stats();
        let started = Instant::now();
        for frame in 0..steps {
            graph.pin(slots[0], [frame as f32, 0.0, 0.0]).expect("pin");
            graph.step();
            graph.flush().expect("flush");
            graph.max_speed().expect("settle");
        }
        let elapsed = started.elapsed();
        let after = graph.stats();
        println!(
            "n={n}: {:.2} ms/step; per step {} B up, {} B read back",
            elapsed.as_secs_f64() * 1e3 / steps as f64,
            (after.uploaded_bytes - before.uploaded_bytes) / steps,
            (after.read_bytes - before.read_bytes) / steps,
        );
    }
}
