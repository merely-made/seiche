// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! The lagged seam on a real device (physics catalog plan, P5b): a settle
//! through `seiche::gpu` meets the CPU settle's bounds with the device
//! counted as running; a sign-flipped device law fails the overlap check; a
//! device lost mid-run hands the steps back to the CPU; and the per-tick cost
//! that sets a native host's threshold.
//!
//! Needs a real adapter, so each test skips where there is none.

#![cfg(feature = "gpu")]

use std::time::Instant;

use euclid::default::Point2D;
use seiche::gpu::{DeviceRepulsion, PhysicsDevice};
use seiche::{
    Boundary, EdgeSpring, LaggedRepulsion, NODE_BODY_RADIUS, NodeExclusion, NodeKey,
    RepulsionForces, RepulsionRequest, RepulsionSolverError, Simulation,
};

fn device() -> Option<PhysicsDevice> {
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
    println!(
        "adapter: {} ({:?})",
        adapter.get_info().name,
        adapter.get_info().backend
    );
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("seiche lagged repulsion receipt"),
        ..Default::default()
    }))
    .ok()?;
    Some(PhysicsDevice::from_wgpu(instance, adapter, device, queue))
}

fn spiral(n: usize) -> Vec<(NodeKey, Point2D<f32>)> {
    (0..n)
        .map(|i| {
            let angle = i as f32 * 2.399_963;
            let radius = 60.0 * (i as f32).sqrt();
            (
                NodeKey::new(i),
                Point2D::new(radius * angle.cos(), radius * angle.sin()),
            )
        })
        .collect()
}

fn trio(nodes: &[(NodeKey, Point2D<f32>)]) -> Simulation {
    let mut sim = Simulation::new();
    sim.add_force(NodeExclusion::default());
    sim.add_force(EdgeSpring::default());
    sim.add_force(Boundary::default());
    sim.sync_nodes(nodes.iter().copied());
    sim
}

/// Overlapping pairs (closer than two node radii) and spread, the receipts'
/// signature numbers. The overlap scan is a uniform grid, so it stays cheap
/// at sizes the pairwise `layout_stats` would not.
fn signature(sim: &Simulation, nodes: &[(NodeKey, Point2D<f32>)]) -> (usize, f32) {
    let positions: Vec<Point2D<f32>> = nodes
        .iter()
        .filter_map(|(key, _)| sim.position_of(*key))
        .collect();
    let cell = 2.0 * NODE_BODY_RADIUS;
    let mut grid: std::collections::HashMap<(i32, i32), Vec<usize>> = Default::default();
    for (i, p) in positions.iter().enumerate() {
        grid.entry(((p.x / cell).floor() as i32, (p.y / cell).floor() as i32))
            .or_default()
            .push(i);
    }
    let mut overlaps = 0;
    for (i, p) in positions.iter().enumerate() {
        let (cx, cy) = ((p.x / cell).floor() as i32, (p.y / cell).floor() as i32);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for &j in grid.get(&(cx + dx, cy + dy)).into_iter().flatten() {
                    if j > i && (positions[j] - *p).length() < cell {
                        overlaps += 1;
                    }
                }
            }
        }
    }
    let n = positions.len() as f32;
    let centroid = positions
        .iter()
        .fold(euclid::default::Vector2D::zero(), |acc, p| {
            acc + p.to_vector()
        })
        / n;
    let spread = (positions
        .iter()
        .map(|p| (p.to_vector() - centroid).square_length())
        .sum::<f32>()
        / n)
        .sqrt();
    (overlaps, spread)
}

/// A device lane with a fault: the law's strength negated (the positive
/// control), or every submission refused after `lose_after` of them (a
/// device lost mid-run).
struct Faulty {
    inner: DeviceRepulsion,
    sign: f32,
    lose_after: Option<u64>,
    submitted: u64,
}

impl LaggedRepulsion for Faulty {
    fn submit(
        &mut self,
        xs: &[f32],
        ys: &[f32],
        request: RepulsionRequest,
    ) -> Result<(), RepulsionSolverError> {
        if self.lose_after.is_some_and(|after| self.submitted >= after) {
            return Err(RepulsionSolverError::Backend("device lost".into()));
        }
        self.submitted += 1;
        self.inner.submit(
            xs,
            ys,
            RepulsionRequest {
                strength: request.strength * self.sign,
                ..request
            },
        )
    }

    fn poll(&mut self) -> Option<Result<RepulsionForces, RepulsionSolverError>> {
        self.inner.poll()
    }

    fn in_flight(&self) -> usize {
        self.inner.in_flight()
    }
}

fn settle(sim: &mut Simulation, ticks: usize) {
    for _ in 0..ticks {
        sim.tick(1.0 / 60.0);
    }
}

/// Tick at 60 Hz, as a native physics actor paces itself, so an answer has
/// the frame a real host gives it. Returns the busy time per tick in ms, the
/// sleeps excluded.
fn paced(sim: &mut Simulation, ticks: usize) -> f64 {
    let frame = std::time::Duration::from_secs_f64(1.0 / 60.0);
    let mut busy = std::time::Duration::ZERO;
    for _ in 0..ticks {
        let start = Instant::now();
        sim.tick(1.0 / 60.0);
        let spent = start.elapsed();
        busy += spent;
        if spent < frame {
            std::thread::sleep(frame - spent);
        }
    }
    busy.as_secs_f64() * 1e3 / ticks as f64
}

#[test]
fn a_two_thousand_node_settle_on_the_device_meets_the_cpu_bounds() {
    let Some(device) = device() else {
        eprintln!("no wgpu adapter: skipping the lagged settle receipt");
        return;
    };
    let n = 2_000;
    let ticks = 600;
    let nodes = spiral(n);

    let mut cpu = trio(&nodes);
    let start = Instant::now();
    settle(&mut cpu, ticks);
    let cpu_ms = start.elapsed().as_secs_f64() * 1e3 / ticks as f64;
    let (cpu_overlaps, cpu_spread) = signature(&cpu, &nodes);
    let cpu_energy = cpu.kinetic_energy();

    let device = device.with_threshold(0).with_max_stale_steps(1);
    let mut gpu = trio(&nodes);
    device.install(&mut gpu);
    let gpu_ms = paced(&mut gpu, ticks);
    let (gpu_overlaps, gpu_spread) = signature(&gpu, &nodes);
    let gpu_energy = gpu.kinetic_energy();
    let stats = gpu.repulsion_stats().unwrap();

    println!(
        "n={n}, {ticks} ticks: cpu {cpu_ms:.2} ms/tick, overlaps {cpu_overlaps}, \
         spread {cpu_spread:.1}, energy {cpu_energy:.2}; device {gpu_ms:.2} ms/tick, \
         overlaps {gpu_overlaps}, spread {gpu_spread:.1}, energy {gpu_energy:.2}; \
         {stats:?}; device submissions {} answers {}",
        device.submissions(),
        device.answers()
    );
    assert_eq!(cpu_overlaps, 0, "the CPU settle overlaps");
    assert_eq!(gpu_overlaps, 0, "the device settle overlaps");
    assert!(
        (gpu_spread - cpu_spread).abs() / cpu_spread < 0.05,
        "spread {gpu_spread} against the CPU's {cpu_spread}"
    );
    assert!(
        gpu_energy <= cpu_energy.max(1.0) * 2.0,
        "energy {gpu_energy} against the CPU's {cpu_energy}"
    );
    // The device ran: answers counted on the device, and most steps used one.
    assert!(device.answers() > 0);
    assert!(
        stats.device_steps * 2 >= ticks as u64,
        "the device served fewer than half the steps: {stats:?}"
    );
    assert_eq!(stats.failures, 0, "{stats:?}");
}

#[test]
fn a_sign_flipped_device_law_fails_the_overlap_check() {
    let Some(device) = device() else {
        eprintln!("no wgpu adapter: skipping the positive control");
        return;
    };
    let nodes = spiral(200);
    let mut sim = trio(&nodes);
    sim.set_lagged_repulsion(
        Some(Box::new(Faulty {
            inner: device.repulsion(),
            sign: -1.0,
            lose_after: None,
            submitted: 0,
        })),
        0,
        1,
    );
    paced(&mut sim, 600);
    let (overlaps, spread) = signature(&sim, &nodes);
    let stats = sim.repulsion_stats().unwrap();
    println!("sign-flipped: {overlaps} overlaps, spread {spread:.1}; {stats:?}");
    assert!(stats.device_steps > 0, "the flipped law never applied");
    assert!(
        overlaps > 0,
        "the overlap check passed a sign-flipped device law"
    );
}

#[test]
fn a_device_lost_mid_run_hands_the_steps_to_the_cpu() {
    let Some(device) = device() else {
        eprintln!("no wgpu adapter: skipping the lost-device receipt");
        return;
    };
    let nodes = spiral(500);
    let ticks = 600;
    let mut cpu = trio(&nodes);
    settle(&mut cpu, ticks);
    let (cpu_overlaps, cpu_spread) = signature(&cpu, &nodes);

    let mut sim = trio(&nodes);
    sim.set_lagged_repulsion(
        Some(Box::new(Faulty {
            inner: device.repulsion(),
            sign: 1.0,
            lose_after: Some(100),
            submitted: 0,
        })),
        0,
        1,
    );
    paced(&mut sim, ticks);
    let (overlaps, spread) = signature(&sim, &nodes);
    let stats = sim.repulsion_stats().unwrap();
    println!(
        "lost after 100: {overlaps} overlaps, spread {spread:.1} (cpu {cpu_overlaps}, \
         {cpu_spread:.1}); {stats:?}"
    );
    // At most one device step per accepted submission, and every step after
    // the loss refused and run on the CPU.
    assert!(
        stats.device_steps > 0 && stats.device_steps <= 100,
        "{stats:?}"
    );
    assert_eq!(stats.submissions, 100, "{stats:?}");
    assert!(stats.failures > 0, "{stats:?}");
    assert_eq!(
        stats.device_steps + stats.cpu_steps,
        ticks as u64,
        "{stats:?}"
    );
    assert_eq!(overlaps, 0);
    assert!((spread - cpu_spread).abs() / cpu_spread < 0.05);
}

/// Physics cost per tick, the CPU law against the lagged device lane with
/// N = 1, from 100 to 10,000 nodes: the crossover is a native host's
/// threshold. The device run is paced at 60 Hz, as a native actor is, and
/// both report busy time per tick.
/// `cargo test -p seiche --features gpu --release --test gpu_repulsion -- --ignored --nocapture`
#[test]
#[ignore]
fn lagged_cost_per_tick() {
    let Some(device) = device() else {
        eprintln!("no wgpu adapter");
        return;
    };
    let device = device.with_threshold(0).with_max_stale_steps(1);
    for n in [100usize, 200, 300, 500, 750, 1_000, 2_000, 5_000, 10_000] {
        let nodes = spiral(n);
        let ticks = if n <= 2_000 { 120 } else { 40 };
        let mut cpu = trio(&nodes);
        settle(&mut cpu, 5);
        let start = Instant::now();
        settle(&mut cpu, ticks);
        let cpu_ms = start.elapsed().as_secs_f64() * 1e3 / ticks as f64;

        let mut gpu = trio(&nodes);
        device.install(&mut gpu);
        paced(&mut gpu, 5);
        let before = gpu.repulsion_stats().unwrap();
        let gpu_ms = paced(&mut gpu, ticks);
        let after = gpu.repulsion_stats().unwrap();
        println!(
            "n={n}: cpu {cpu_ms:.2} ms/tick, lagged device {gpu_ms:.2} ms/tick \
             ({} of {ticks} steps on the device)",
            after.device_steps - before.device_steps
        );
    }
}
