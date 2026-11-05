// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

use super::*;

/// Run `parts` on `fixture` from a spiral seed for `seconds`, printing one
/// line per sample time: shape, energy and the window's term ledger.
pub(in super::super) fn run(
    title: &str,
    fixture: &Fixture,
    spacing: f32,
    damping: f32,
    parts: Vec<Part>,
    seconds: u32,
) -> Report {
    run_quietly(title, fixture, spacing, damping, parts, seconds, true)
}

pub(in super::super) fn run_quietly(
    title: &str,
    fixture: &Fixture,
    spacing: f32,
    damping: f32,
    parts: Vec<Part>,
    seconds: u32,
    verbose: bool,
) -> Report {
    let mut sim = Simulation::new();
    sim.set_linear_damping(damping);
    sim.sync_nodes(fixture.spiral(spacing));
    sim.sync_edges(fixture.edges.clone());
    let labels: Vec<&str> = parts.iter().map(|p| p.label).collect();
    let mut ledgers = Vec::new();
    let mut forces: Vec<Box<dyn Force>> = Vec::new();
    for part in parts {
        let ledger = Arc::new(Mutex::new(Ledger::default()));
        ledgers.push(ledger.clone());
        forces.push(Box::new(Metered {
            inner: part.force,
            ledger,
            writes_only: part.writes_only,
        }));
    }
    let snap = Arc::new(Mutex::new(HashMap::new()));
    forces.push(Box::new(Snapshot(snap.clone())));
    sim.set_forces(forces);
    let handles: Vec<RigidBodyHandle> =
        fixture.keys.iter().map(|k| sim.bodies_by_node[k]).collect();
    let n = handles.len();
    let exclusion_at = labels.iter().position(|l| *l == "exclusion");
    let counter_at = labels.iter().position(|l| *l == "counter-damping");
    let exclusion_energy = |sim: &Simulation| {
        let nodes: Vec<(NodeKey, Vector)> = fixture
            .keys
            .iter()
            .map(|k| {
                let p = sim.position_of(*k).unwrap();
                (*k, Vector::new(p.x, p.y))
            })
            .collect();
        NodeExclusion::default()
            .energy(
                0,
                &crate::Layout {
                    nodes: &nodes,
                    edges: &fixture.edges,
                },
            )
            .unwrap()
    };
    let u_exclusion_0 = exclusion_energy(&sim);
    let mut exclusion_work_total = 0.0;

    if verbose {
        println!(
            "{title} [{} {} nodes, spiral {spacing}, damping {damping}]",
            fixture.name, n
        );
        println!(
            "  {:>5} {:>7} {:>6} {:>9} {:>4} {:>5} {:>5} {:>6} | work in the window (share): {} written damping contacts; counter-damping net of damping | mean outward push: {}",
            "t s",
            "extent",
            "x1s",
            "KE",
            "ovl",
            "tang",
            "coh",
            "m~r",
            labels.join(" "),
            labels.join(" "),
        );
    }
    let k = 1.0 / (1.0 + f64::from(damping) * f64::from(TICK_DT));
    let mut window = Window {
        work: vec![0.0; labels.len()],
        outward: vec![0.0; labels.len()],
        ..Window::default()
    };
    let mut written_seen = vec![0.0f64; labels.len()];
    let mut angle: Vec<f32> = vec![0.0; n];
    let mut turned: Vec<f32> = vec![0.0; n];
    let mut extent_1s = 0.0f32;
    let (mut max_ratio, mut end_ratio) = (0.0f32, 0.0f32);
    let (mut min_energy, mut end_energy) = (f32::INFINITY, 0.0f32);
    let (mut tangential, mut coherence, mut mass_radius, mut counted) = (0.0, 0.0, 0.0, 0);
    let mut max_overlaps = 0usize;
    let read = |sim: &Simulation| -> (Vec<Vector>, Vec<Vector>) {
        let at = handles
            .iter()
            .map(|h| sim.bodies[*h].translation())
            .collect();
        let vel = handles.iter().map(|h| sim.bodies[*h].linvel()).collect();
        (at, vel)
    };
    {
        let (at, _) = read(&sim);
        let c = centroid(&at);
        for i in 0..n {
            let r = at[i] - c;
            angle[i] = r.y.atan2(r.x);
        }
    }
    let ticks = seconds * 60;
    for tick in 1..=ticks {
        let (at0, _) = read(&sim);
        let c0 = centroid(&at0);
        sim.tick(TICK_DT);
        let (at1, v1) = read(&sim);
        let v0 = snap.lock().unwrap().clone();
        let mut tick_work = 0.0;
        for (t, ledger) in ledgers.iter().enumerate() {
            let ledger = ledger.lock().unwrap();
            let fresh = ledger.written - written_seen[t];
            written_seen[t] = ledger.written;
            window.written += fresh;
            for (i, h) in handles.iter().enumerate() {
                let Some(push) = ledger.push.get(h) else {
                    continue;
                };
                let vbar = (v0[h] + v1[i]) * 0.5;
                let w = f64::from(push.x * vbar.x + push.y * vbar.y) * f64::from(TICK_DT) * k;
                window.work[t] += w;
                tick_work += w;
                if Some(t) == exclusion_at {
                    exclusion_work_total += w;
                }
                let r = at0[i] - c0;
                let rl = r.length();
                if rl > 1e-3 {
                    window.outward[t] += f64::from((push.x * r.x + push.y * r.y) / rl) / n as f64;
                }
            }
        }
        let (mut ke0, mut ke1, mut damp) = (0.0f64, 0.0f64, 0.0f64);
        for (i, h) in handles.iter().enumerate() {
            let m = f64::from(sim.bodies[*h].mass());
            let (a, b) = (v0[h], v1[i]);
            let vbar = (a + b) * 0.5;
            ke0 += 0.5 * m * f64::from(a.length_squared());
            ke1 += 0.5 * m * f64::from(b.length_squared());
            damp -= m
                * f64::from(damping)
                * f64::from(TICK_DT)
                * k
                * f64::from(a.x * vbar.x + a.y * vbar.y);
        }
        window.damping += damp;
        window.contacts += (ke1 - ke0) - tick_work - damp;
        window.ticks += 1;
        let c1 = centroid(&at1);
        for i in 0..n {
            let r = at1[i] - c1;
            let a = r.y.atan2(r.x);
            let mut d = a - angle[i];
            while d > std::f32::consts::PI {
                d -= std::f32::consts::TAU;
            }
            while d < -std::f32::consts::PI {
                d += std::f32::consts::TAU;
            }
            turned[i] += d;
            angle[i] = a;
        }
        let second = tick / 60;
        if tick % 60 != 0 || !SAMPLES.contains(&second) || second > seconds {
            continue;
        }
        let s = shape(&at1, &v1, &fixture.masses);
        let energy = sim.kinetic_energy();
        if second >= 1 {
            max_overlaps = max_overlaps.max(s.overlaps);
        }
        if second == 1 {
            extent_1s = s.extent;
        }
        let ratio = s.extent / extent_1s.max(1.0);
        if second >= 1 {
            max_ratio = max_ratio.max(ratio);
            end_ratio = ratio;
            end_energy = energy;
            min_energy = min_energy.min(energy);
        }
        if second >= 6 {
            tangential += s.tangential;
            coherence += s.coherence;
            mass_radius += s.mass_radius;
            counted += 1;
        }
        let total: f64 = window.work.iter().map(|w| w.abs()).sum::<f64>()
            + window.written.abs()
            + window.damping.abs()
            + window.contacts.abs();
        let share = |w: f64| {
            if total > 0.0 {
                100.0 * w.abs() / total
            } else {
                0.0
            }
        };
        let works: Vec<String> = window
            .work
            .iter()
            .map(|w| format!("{w:.0}({:.0}%)", share(*w)))
            .collect();
        let outward: Vec<String> = window
            .outward
            .iter()
            .map(|o| format!("{:.1}", o / f64::from(window.ticks)))
            .collect();
        if verbose {
            println!(
                "  {second:>5} {:>7.0} {ratio:>6.2} {energy:>9.0} {:>4} {:>5.2} {:>5.2} {:>6.2} | {} {:.0}({:.0}%) {:.0}({:.0}%) {:.0}({:.0}%); {:.0} | {}",
                s.extent,
                s.overlaps,
                s.tangential,
                s.coherence,
                s.mass_radius,
                works.join(" "),
                window.written,
                share(window.written),
                window.damping,
                share(window.damping),
                window.contacts,
                share(window.contacts),
                counter_at.map_or(0.0, |c| window.work[c] + window.damping),
                outward.join(" "),
            );
        }
        window = Window {
            work: vec![0.0; labels.len()],
            outward: vec![0.0; labels.len()],
            ..Window::default()
        };
    }
    let revolutions =
        turned.iter().map(|t| t.abs()).sum::<f32>() / n as f32 / std::f32::consts::TAU;
    let exclusion_check =
        exclusion_at.map(|_| (exclusion_work_total, u_exclusion_0 - exclusion_energy(&sim)));
    if let Some((w, du)) = exclusion_check.filter(|_| verbose) {
        println!("  instrument: exclusion's booked work {w:.0} against its energy drop {du:.0}");
    }
    let counted = counted.max(1) as f32;
    let report = Report {
        extent_1s,
        max_ratio,
        end_ratio,
        min_energy,
        end_energy,
        mean_tangential: tangential / counted,
        mean_coherence: coherence / counted,
        mean_mass_radius: mass_radius / counted,
        revolutions,
        exclusion_check,
        max_overlaps,
    };
    if verbose {
        println!(
            "  summary: extent 1 s {:.0}, max x{:.2}, end x{:.2}; KE min {:.0}, end {:.0}; from 6 s tangential {:.2}, coherence {:.2}, mass~radius {:.2}; mean revolutions {:.2}",
            report.extent_1s,
            report.max_ratio,
            report.end_ratio,
            report.min_energy,
            report.end_energy,
            report.mean_tangential,
            report.mean_coherence,
            report.mean_mass_radius,
            report.revolutions,
        );
    }
    report
}
