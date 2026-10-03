// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! The exclusion lane's receipts (physics catalog plan, P5a): both passes,
//! every pair and the cell list, compute `seiche::NodeExclusion`'s law; the
//! binning's scan is exact; a sign-flipped kernel is caught by the overlap
//! check on either pass; and the cost per call at the plan's sizes.
//!
//! The anchor is `seiche::node_exclusion_reference`, the naive host loop of
//! the same law. Needs a real adapter, so each test skips where there is none.

#![cfg(feature = "resident")]

use std::sync::{Arc, Mutex};
use std::time::Instant;

use conatus::resident::{Exclusion, ExclusionParams, ResidentClient, binning};
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

/// Per-body relative error of `gpu` against the CPU law: (mean, worst).
fn agreement(gpu: &[[f32; 4]], fx: &[f32], fy: &[f32]) -> (f64, f64) {
    let mut worst = 0.0f64;
    let mut mean = 0.0f64;
    for (i, force) in gpu.iter().enumerate() {
        let magnitude = (fx[i] as f64).hypot(fy[i] as f64).max(1e-6);
        let dx = (force[0] - fx[i]) as f64;
        let dy = (force[1] - fy[i]) as f64;
        let relative = dx.hypot(dy) / magnitude;
        worst = worst.max(relative);
        mean += relative;
        assert_eq!(force[2], 0.0, "force leaked into the padded axis");
    }
    (mean / gpu.len() as f64, worst)
}

#[test]
fn both_passes_compute_node_exclusions_law_at_1k_10k_and_50k() {
    let Some(client) = client() else {
        eprintln!("no wgpu adapter: skipping the exclusion receipt");
        return;
    };
    let mut lane = Exclusion::new(&client);
    let params = law();
    for n in [1_000usize, 10_000, 50_000] {
        let positions = scatter(n, 140.0);
        let (fx, fy) = reference(&positions, params);
        // The cutoff must actually prune here, or this test says nothing
        // about it.
        let cut2 = params.cutoff * params.cutoff;
        let beyond_cutoff_pairs = (1..positions.len().min(200))
            .filter(|&j| {
                (positions[0][0] - positions[j][0]).powi(2)
                    + (positions[0][1] - positions[j][1]).powi(2)
                    > cut2
            })
            .count();
        assert!(
            beyond_cutoff_pairs > 0,
            "the scatter never reaches the cutoff"
        );
        for (pass, threshold) in [("pairs", usize::MAX), ("cells", 0)] {
            lane.set_cell_threshold(threshold);
            let pending = lane.submit(&positions, params);
            assert_eq!(
                pending.used_cells(),
                pass == "cells",
                "{pass} pass not taken"
            );
            let gpu = pending.wait().expect("readback");
            let (mean, worst) = agreement(&gpu, &fx, &fy);
            println!(
                "n={n} {pass}: relative error mean {mean:.2e} worst {worst:.2e}; \
                 body 0 has {beyond_cutoff_pairs} of its first 199 pairs past the cutoff"
            );
            assert!(
                worst < 1e-3,
                "the {pass} pass disagrees with NodeExclusion at n={n}: mean {mean:.2e}, worst {worst:.2e}"
            );
        }
    }
    assert_eq!(lane.dispatches(), 6);
    assert_eq!(lane.cell_dispatches(), 3);
}

#[test]
fn the_binning_scan_is_exact_across_its_levels() {
    let Some(client) = client() else {
        eprintln!("no wgpu adapter: skipping the scan receipt");
        return;
    };
    let compute = client.compute_client();
    // One block, a block edge either side, two levels, and three levels
    // (256 * 256 = 65,536 is where a third level appears).
    for len in [1usize, 255, 256, 257, 4_097, 70_001] {
        let data: Vec<u32> = (0..len as u32).map(|i| (i * 2_654_435_761) % 7).collect();
        let handle = compute.create_from_slice(bytemuck::cast_slice(&data));
        binning::exclusive_scan(compute, &handle, len);
        let bytes = compute.read_one(handle).expect("scan readback");
        let gpu: &[u32] = bytemuck::cast_slice(&bytes[..len * 4]);
        let mut running = 0u32;
        for (i, (&got, &value)) in gpu.iter().zip(&data).enumerate() {
            assert_eq!(got, running, "scan of {len} wrong at {i}");
            running += value;
        }
    }
}

#[test]
fn a_sparse_layout_stays_on_every_pair() {
    let Some(client) = client() else {
        eprintln!("no wgpu adapter: skipping the sparse receipt");
        return;
    };
    let mut lane = Exclusion::new(&client);
    lane.set_cell_threshold(0);
    // Four bodies a million units apart: a cutoff-wide grid would hold about
    // a million cells for four bodies.
    let positions = vec![
        [0.0, 0.0, 0.0, 0.0],
        [1.0e6, 0.0, 0.0, 0.0],
        [0.0, 1.0e6, 0.0, 0.0],
        [5.0, 5.0, 0.0, 0.0],
    ];
    let pending = lane.submit(&positions, law());
    assert!(!pending.used_cells(), "a sparse layout binned anyway");
    let gpu = pending.wait().expect("readback");
    let (fx, fy) = reference(&positions, law());
    let (_, worst) = agreement(&gpu, &fx, &fy);
    assert!(worst < 1e-3, "sparse fallback disagrees: {worst:.2e}");
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
    let n = 200;
    let ticks = 600;
    let (cpu_overlaps, cpu_spread) = settle(n, None, ticks);
    assert_eq!(cpu_overlaps, 0, "the CPU law itself overlaps");
    for (pass, threshold) in [("pairs", usize::MAX), ("cells", 0)] {
        let lane = Arc::new(Mutex::new(Exclusion::new(&client)));
        lane.lock().unwrap().set_cell_threshold(threshold);
        let (gpu_overlaps, gpu_spread) = settle(n, Some(solver(lane.clone(), 1.0)), ticks);
        let dispatched = lane.lock().unwrap().dispatches();
        let cells = lane.lock().unwrap().cell_dispatches();
        let (flipped_overlaps, flipped_spread) = settle(n, Some(solver(lane.clone(), -1.0)), ticks);
        println!(
            "n={n}, {ticks} ticks, {pass}: cpu {cpu_overlaps} overlaps spread {cpu_spread:.1}; \
             gpu {gpu_overlaps} overlaps spread {gpu_spread:.1} ({dispatched} dispatches, \
             {cells} on cells); sign-flipped {flipped_overlaps} overlaps spread {flipped_spread:.1}"
        );
        assert_eq!(gpu_overlaps, 0, "the device law overlaps ({pass})");
        assert!(
            dispatched >= ticks as u64,
            "the device did not run every tick ({pass})"
        );
        assert_eq!(cells, if pass == "cells" { dispatched } else { 0 });
        assert!(
            (gpu_spread - cpu_spread).abs() / cpu_spread < 0.05,
            "the device settle spread {gpu_spread} differs from the CPU's {cpu_spread} ({pass})"
        );
        assert!(
            flipped_overlaps > 0,
            "the overlap check passed a sign-flipped {pass} kernel: it cannot see a broken device law"
        );
    }
}

/// Cost per call on the device (upload, dispatch, readback), every pair
/// against the cell list, beside the CPU law.
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
    for n in [
        500usize, 1_000, 2_000, 4_096, 5_000, 10_000, 20_000, 50_000, 100_000,
    ] {
        let positions = scatter(n, 140.0);
        let calls = if n <= 10_000 { 20 } else { 5 };
        let mut timed = |threshold: usize| {
            lane.set_cell_threshold(threshold);
            lane.submit(&positions, params).wait().expect("warm");
            let start = Instant::now();
            for _ in 0..calls {
                lane.submit(&positions, params).wait().expect("readback");
            }
            start.elapsed().as_secs_f64() * 1e3 / calls as f64
        };
        let pairs_ms = timed(usize::MAX);
        let cells_ms = timed(0);
        if n > 50_000 {
            println!("n={n}: pairs {pairs_ms:.2} ms/call, cells {cells_ms:.2} ms/call");
            continue;
        }
        let cpu_calls = if n <= 2_000 { 10 } else { 1 };
        let start = Instant::now();
        for _ in 0..cpu_calls {
            std::hint::black_box(reference(&positions, params));
        }
        let cpu_ms = start.elapsed().as_secs_f64() * 1e3 / cpu_calls as f64;
        println!(
            "n={n}: pairs {pairs_ms:.2} ms/call, cells {cells_ms:.2} ms/call, cpu {cpu_ms:.2} ms/call"
        );
    }
}

/// Submit-to-ready time for a caller that sleeps 1 ms between looks, as a
/// frame loop does, with a 16 ms gap between submissions. Before
/// `ResidentClient::poll_device` this took 14 ms at the median; with it,
/// the answer is there on the second look.
/// `cargo test -p conatus --features resident --release --test exclusion -- --ignored readback_latency --nocapture`
#[test]
#[ignore]
fn readback_latency_with_a_sleeping_caller() {
    let Some(client) = client() else { return };
    let mut lane = Exclusion::new(&client);
    for n in [200usize, 2_000] {
        let positions = scatter(n, 140.0);
        lane.submit(&positions, law()).wait().unwrap();
        let mut samples = Vec::new();
        for _ in 0..40 {
            let start = Instant::now();
            let mut pending = lane.submit(&positions, law());
            let mut polls = 0;
            loop {
                polls += 1;
                if pending.try_take().is_some() {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            samples.push((start.elapsed().as_secs_f64() * 1e3, polls));
            std::thread::sleep(std::time::Duration::from_millis(16));
        }
        let mut ms: Vec<f64> = samples.iter().map(|s| s.0).collect();
        ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let looks: Vec<u32> = samples.iter().map(|s| s.1).collect();
        println!(
            "n={n}: submit-to-ready p50 {:.2} ms p90 {:.2} max {:.2}; looks {looks:?}",
            ms[20], ms[36], ms[39]
        );
    }
}

/// A reader dropped mid-flight (a law switch, a closed canvas) must not leave
/// a mapped staging buffer for the next submit to trip on. Before readbacks
/// were adopted by the client, the per-tick bench failed wgpu validation
/// ("Buffer ... is still mapped" in `Queue::submit`) on exactly this.
#[test]
fn a_dropped_reader_leaves_no_mapped_buffer_behind() {
    let Some(client) = client() else {
        eprintln!("no wgpu adapter: skipping the dropped-reader receipt");
        return;
    };
    let mut lane = Exclusion::new(&client);
    let positions = scatter(500, 140.0);
    for _ in 0..50 {
        drop(lane.submit(&positions, law()));
    }
    for _ in 0..20 {
        lane.submit(&positions, law())
            .wait()
            .expect("readback after drops");
    }
    let start = Instant::now();
    while client.orphans() > 0 && start.elapsed().as_secs() < 5 {
        client.poll_device();
        std::thread::yield_now();
    }
    assert_eq!(client.orphans(), 0, "adopted readbacks never finished");
}
