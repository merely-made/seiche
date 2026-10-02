// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! The exclusion lane's receipts (physics catalog plan, P5a): the kernel
//! computes `seiche::NodeExclusion`'s law, a sign-flipped kernel is caught by
//! the overlap check, and the cost per call at the plan's sizes.
//!
//! The anchor is `seiche::node_exclusion_reference`, the naive host loop of
//! the same law. Needs a real adapter, so each test skips where there is none.

#![cfg(feature = "resident")]

use std::sync::{Arc, Mutex};
use std::time::Instant;

use conatus::resident::{Exclusion, ExclusionParams, ResidentClient};
use seiche::{
    Boundary, EdgeSpring, NODE_BODY_RADIUS, NodeExclusion, NodeExclusionParams, NodeKey,
    RepulsionForces, RepulsionRequest, RepulsionSolver, RepulsionSolverError, Simulation,
    node_exclusion_reference,
};

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
    println!("adapter: {} ({backend:?})", adapter.get_info().name);
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("conatus exclusion lane receipt"),
        ..Default::default()
    }))
    .ok()?;
    Some(ResidentClient::init(burn::backend::wgpu::WgpuSetup {
        instance,
        adapter,
        device,
        queue,
        backend,
    }))
}

/// A seeded scatter at a settled layout's density (about one body per
/// `spacing`² of plane), so `cutoff` prunes real pairs; every twentieth body
/// gets a twin a few units away, so the `min_distance` floor is exercised.
fn scatter(n: usize, spacing: f32) -> Vec<[f32; 4]> {
    let mut state = 0x2026_1002u64;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
    };
    let extent = (n as f32).sqrt() * spacing * 0.5;
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let p = [next() * extent, next() * extent, 0.0, 0.0];
        out.push(p);
        if out.len() % 20 == 0 && out.len() < n {
            out.push([p[0] + 3.0, p[1] - 2.0, 0.0, 0.0]);
        }
    }
    out
}

fn law() -> ExclusionParams {
    let d = NodeExclusion::default();
    ExclusionParams {
        strength: d.strength,
        cutoff: d.cutoff,
        min_distance: d.min_distance,
    }
}

fn reference(positions: &[[f32; 4]], params: ExclusionParams) -> (Vec<f32>, Vec<f32>) {
    let xs: Vec<f32> = positions.iter().map(|p| p[0]).collect();
    let ys: Vec<f32> = positions.iter().map(|p| p[1]).collect();
    node_exclusion_reference(
        &xs,
        &ys,
        NodeExclusionParams {
            strength: params.strength,
            cutoff: params.cutoff,
            min_distance: params.min_distance,
        },
    )
    .expect("equal lengths")
}

#[test]
fn the_kernel_computes_node_exclusions_law_at_1k_and_10k() {
    let Some(client) = client() else {
        eprintln!("no wgpu adapter: skipping the exclusion receipt");
        return;
    };
    let mut lane = Exclusion::new(&client);
    for n in [1_000usize, 10_000] {
        let positions = scatter(n, 140.0);
        let params = law();
        let gpu = lane.submit(&positions, params).wait().expect("readback");
        let (fx, fy) = reference(&positions, params);

        let mut worst = 0.0f64;
        let mut mean = 0.0f64;
        let mut beyond_cutoff_pairs = 0usize;
        for (i, force) in gpu.iter().enumerate() {
            let magnitude = (fx[i] as f64).hypot(fy[i] as f64).max(1e-6);
            let dx = (force[0] - fx[i]) as f64;
            let dy = (force[1] - fy[i]) as f64;
            let relative = dx.hypot(dy) / magnitude;
            worst = worst.max(relative);
            mean += relative;
            assert_eq!(force[2], 0.0, "force leaked into the padded axis");
        }
        mean /= n as f64;
        // The cutoff must actually prune here, or this test says nothing
        // about it.
        let cut2 = params.cutoff * params.cutoff;
        for j in 1..positions.len().min(200) {
            let d = (positions[0][0] - positions[j][0]).powi(2)
                + (positions[0][1] - positions[j][1]).powi(2);
            if d > cut2 {
                beyond_cutoff_pairs += 1;
            }
        }
        println!(
            "n={n}: relative error mean {mean:.2e} worst {worst:.2e}; \
             body 0 has {beyond_cutoff_pairs} of its first 199 pairs past the cutoff"
        );
        assert!(
            beyond_cutoff_pairs > 0,
            "the scatter never reaches the cutoff"
        );
        assert!(
            worst < 1e-3,
            "the kernel disagrees with NodeExclusion at n={n}: mean {mean:.2e}, worst {worst:.2e}"
        );
    }
    assert_eq!(lane.dispatches(), 2);
}

/// A [`RepulsionSolver`] over the lane, blocking (the synchronous seam the
/// tests keep). `sign` multiplies the law's strength: -1 is the
/// sign-flipped kernel of the positive control.
fn solver(lane: Arc<Mutex<Exclusion>>, sign: f32) -> RepulsionSolver {
    Arc::new(move |xs: &[f32], ys: &[f32], request: RepulsionRequest| {
        let positions: Vec<[f32; 4]> = xs.iter().zip(ys).map(|(&x, &y)| [x, y, 0.0, 0.0]).collect();
        let forces = lane
            .lock()
            .expect("lane")
            .submit(
                &positions,
                ExclusionParams {
                    strength: request.strength * sign,
                    cutoff: request.cutoff,
                    min_distance: request.min_distance,
                },
            )
            .wait()
            .map_err(|error| RepulsionSolverError::Backend(error.to_string()))?;
        RepulsionForces::new(
            xs.len(),
            forces.iter().map(|f| f[0]).collect(),
            forces.iter().map(|f| f[1]).collect(),
        )
    })
}

/// `n` unlinked bodies under exclusion and the boundary pull (the springs need
/// no device), seeded in a loose spiral and settled through the given solver
/// (or the CPU law when `None`). Returns the overlapping pairs (closer than two
/// node radii, `Canvas::layout_stats`'s definition) and the spread. With no
/// edges nothing can tangle or buckle, so a settled CPU run has no overlaps.
fn settle(n: usize, solver: Option<RepulsionSolver>, ticks: usize) -> (usize, f32) {
    let mut sim = Simulation::new();
    sim.add_force(NodeExclusion::default());
    sim.add_force(EdgeSpring::default());
    sim.add_force(Boundary::default());
    let nodes: Vec<(NodeKey, euclid::default::Point2D<f32>)> = (0..n)
        .map(|i| {
            let angle = i as f32 * 2.399_963;
            let radius = 60.0 * (i as f32).sqrt();
            (
                NodeKey::new(i),
                euclid::default::Point2D::new(radius * angle.cos(), radius * angle.sin()),
            )
        })
        .collect();
    sim.sync_nodes(nodes.iter().copied());
    if let Some(solver) = solver {
        sim.set_repulsion_solver(Some(solver), 0);
    }
    for _ in 0..ticks {
        sim.tick(1.0 / 60.0);
    }
    let positions: Vec<_> = nodes
        .iter()
        .filter_map(|(key, _)| sim.position_of(*key))
        .collect();
    let mut overlaps = 0;
    for i in 0..positions.len() {
        for j in (i + 1)..positions.len() {
            if (positions[i] - positions[j]).length() < 2.0 * NODE_BODY_RADIUS {
                overlaps += 1;
            }
        }
    }
    let count = positions.len() as f32;
    let centroid = positions
        .iter()
        .fold(euclid::default::Vector2D::<f32>::zero(), |acc, p| {
            acc + p.to_vector()
        })
        / count;
    let spread = (positions
        .iter()
        .map(|p| (p.to_vector() - centroid).square_length())
        .sum::<f32>()
        / count)
        .sqrt();
    (overlaps, spread)
}

#[test]
fn a_sign_flipped_kernel_fails_the_overlap_check() {
    let Some(client) = client() else {
        eprintln!("no wgpu adapter: skipping the positive control");
        return;
    };
    let lane = Arc::new(Mutex::new(Exclusion::new(&client)));
    let n = 200;
    let ticks = 600;
    let (cpu_overlaps, cpu_spread) = settle(n, None, ticks);
    let (gpu_overlaps, gpu_spread) = settle(n, Some(solver(lane.clone(), 1.0)), ticks);
    let dispatched = lane.lock().unwrap().dispatches();
    let (flipped_overlaps, flipped_spread) = settle(n, Some(solver(lane.clone(), -1.0)), ticks);
    println!(
        "n={n}, {ticks} ticks: cpu {cpu_overlaps} overlaps spread {cpu_spread:.1}; \
         gpu {gpu_overlaps} overlaps spread {gpu_spread:.1} ({dispatched} dispatches); \
         sign-flipped {flipped_overlaps} overlaps spread {flipped_spread:.1}"
    );
    assert_eq!(cpu_overlaps, 0, "the CPU law itself overlaps");
    assert_eq!(gpu_overlaps, 0, "the device law overlaps");
    assert!(
        dispatched >= ticks as u64,
        "the device did not run every tick"
    );
    assert!(
        (gpu_spread - cpu_spread).abs() / cpu_spread < 0.05,
        "the device settle spread {gpu_spread} differs from the CPU's {cpu_spread}"
    );
    assert!(
        flipped_overlaps > 0,
        "the overlap check passed a sign-flipped kernel: it cannot see a broken device law"
    );
}

/// Cost per call, device (upload, dispatch, readback) against the CPU law.
/// `cargo test -p conatus --features resident --release --test exclusion -- --ignored --nocapture`
#[test]
#[ignore]
fn exclusion_cost_per_call() {
    let Some(client) = client() else {
        eprintln!("no wgpu adapter");
        return;
    };
    let mut lane = Exclusion::new(&client);
    let params = law();
    for n in [500usize, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000] {
        let positions = scatter(n, 140.0);
        lane.submit(&positions, params).wait().expect("warm");
        let calls = if n <= 10_000 { 20 } else { 5 };
        let start = Instant::now();
        for _ in 0..calls {
            lane.submit(&positions, params).wait().expect("readback");
        }
        let gpu_ms = start.elapsed().as_secs_f64() * 1e3 / calls as f64;
        let cpu_calls = if n <= 2_000 { 10 } else { 1 };
        let start = Instant::now();
        for _ in 0..cpu_calls {
            std::hint::black_box(reference(&positions, params));
        }
        let cpu_ms = start.elapsed().as_secs_f64() * 1e3 / cpu_calls as f64;
        println!("n={n}: device {gpu_ms:.2} ms/call, cpu {cpu_ms:.2} ms/call");
    }
}
