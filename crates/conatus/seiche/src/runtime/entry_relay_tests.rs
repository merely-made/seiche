// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! The coupling-entry relay against the inline backend: the same scenario
//! drives an inline `Physics` (the handler runs inside the tick) and an
//! offloaded one (the simulation reports events and the handler runs when the
//! update is folded, one round trip later). What the host can see is the same:
//! the handler's reports, the outcome in the membership frame, the positions.

use std::sync::{Arc, Mutex};

use numen::{
    CouplingId, CouplingResponse, FieldDefinition, FieldExtent, FieldId, VectorField,
};

use super::*;
use crate::{
    CouplingEntryOutcome, CouplingMembershipChange, CouplingSource, CouplingTargetIdentity,
    TICK_DT,
};

const OUT: f32 = 1000.0;

/// What a handler saw of one report, less the process-local session token and
/// the binding's `Arc` identity.
#[derive(Debug, PartialEq)]
struct Seen {
    step: u64,
    elapsed: u64,
    generation: u64,
    node: usize,
    identity: Option<CouplingTargetIdentity>,
    target_index: Option<usize>,
    point: Option<[u32; 2]>,
    previous: Option<crate::CouplingEligibility>,
    eligibility: crate::CouplingEligibility,
    change: CouplingMembershipChange,
    cause: crate::CouplingMembershipCause,
    action: CouplingEntryOutcome,
    targets: Vec<usize>,
    source: Option<CouplingSource>,
}

fn force(nodes: &[NodeKey]) -> CouplingForce {
    // Flowing, so a body inside the region moves and the positions tell.
    CouplingForce::new(
        CouplingResponse::FlowAdvect,
        1.0,
        nodes.iter().copied(),
        FieldDefinition::Vector(VectorField::ConstVec { x: 0.0, y: 12.0 }),
    )
    .with_extent(FieldExtent::Region {
        min_x: -100.0,
        min_y: -100.0,
        max_x: 100.0,
        max_y: 100.0,
    })
    .with_provenance(
        CouplingSource {
            field: FieldId::from_uuid(uuid::Uuid::from_u128(100)),
            coupling: CouplingId::from_uuid(uuid::Uuid::from_u128(101)),
            revision: Some(3),
        },
        |_, node| {
            Some(CouplingTargetIdentity {
                member: node.index() as u128 + 1000,
                occurrence: None,
            })
        },
    )
}

type Log = Arc<Mutex<Vec<Seen>>>;

fn handler(log: &Log, refuse_node: Option<usize>) -> crate::CouplingEntryHandler {
    let log = Arc::clone(log);
    Arc::new(move |report| {
        log.lock().unwrap().push(Seen {
            step: report.step,
            elapsed: report.clock.elapsed.to_bits(),
            generation: report.binding_generation,
            node: report.node.index(),
            identity: report.identity,
            target_index: report.target_index,
            point: report.reference_point.map(|p| p.map(f32::to_bits)),
            previous: report.previous,
            eligibility: report.eligibility,
            change: report.change,
            cause: report.cause,
            action: report.entry_action.clone(),
            targets: report.binding.targets.iter().map(|n| n.index()).collect(),
            source: report.binding.source,
        });
        if Some(report.node.index()) == refuse_node {
            Err("owner declined".to_string())
        } else {
            Ok(())
        }
    })
}

#[derive(Clone, Copy, Debug)]
enum Run {
    Inline,
    /// Offloaded with the handler installed before the move.
    OffloadedAfterInstall,
    /// Offloaded, then the handler installed over the command channel.
    OffloadedBeforeInstall,
}

/// Run one tick (or wait for the actor's) and fold it.
fn tick(physics: &mut Physics, view: &mut LayoutView, ticks_so_far: &mut u64) {
    physics.settle(1);
    *ticks_so_far += 1;
    match physics {
        Physics::Inline(_) => {
            physics.advance_frame(view);
        },
        Physics::Actor(_) | Physics::Remote(_) => {
            let want = *ticks_so_far as f64 * f64::from(TICK_DT) - 1e-9;
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            while physics.field_clock().elapsed < want {
                assert!(std::time::Instant::now() < deadline, "the actor did not tick");
                physics.refresh(view);
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        },
    }
}

struct Outcome {
    log: Vec<Seen>,
    frame: Vec<(usize, CouplingMembershipChange, CouplingEntryOutcome)>,
    positions: Vec<(usize, u32, u32)>,
}

fn scenario(run: Run, refuse_node: Option<usize>) -> Outcome {
    let (a, b) = (NodeKey::new(0), NodeKey::new(1));
    let mut sim = Simulation::new();
    sim.sync_nodes([(a, Point2D::new(OUT, 0.0)), (b, Point2D::new(OUT, 40.0))]);
    let mut physics = Physics::inline(sim, 0);
    physics.set_coupling_forces(vec![force(&[a, b])]);
    let log: Log = Arc::default();
    if matches!(run, Run::Inline | Run::OffloadedAfterInstall) {
        physics.set_coupling_entry_handler(Some(handler(&log, refuse_node)));
    }
    if !matches!(run, Run::Inline) {
        physics.offload(Arc::new(|| {}));
    }
    if matches!(run, Run::OffloadedBeforeInstall) {
        physics.set_coupling_entry_handler(Some(handler(&log, refuse_node)));
    }
    let mut view = LayoutView::default();
    let mut ticks = 0;
    // Outside, both out of the region: initial observations, no entries.
    tick(&mut physics, &mut view, &mut ticks);
    // Both in: two entries.
    physics.seed(vec![(a, Point2D::new(0.0, 0.0)), (b, Point2D::new(10.0, 0.0))]);
    tick(&mut physics, &mut view, &mut ticks);
    // One out and back in; a replaced coupling list; then an entry under it.
    physics.seed(vec![(a, Point2D::new(OUT, 0.0)), (b, Point2D::new(10.0, 0.0))]);
    tick(&mut physics, &mut view, &mut ticks);
    physics.set_coupling_forces(vec![force(&[a, b])]);
    tick(&mut physics, &mut view, &mut ticks);
    physics.seed(vec![(a, Point2D::new(-20.0, 5.0)), (b, Point2D::new(10.0, 0.0))]);
    tick(&mut physics, &mut view, &mut ticks);
    // One more tick with a frame that carries an entry: a leaves and re-enters
    // in two seeds around two ticks.
    physics.seed(vec![(a, Point2D::new(OUT, 0.0)), (b, Point2D::new(10.0, 0.0))]);
    tick(&mut physics, &mut view, &mut ticks);
    physics.seed(vec![(a, Point2D::new(30.0, 30.0)), (b, Point2D::new(10.0, 0.0))]);
    tick(&mut physics, &mut view, &mut ticks);
    let frame = physics
        .coupling_membership()
        .expect("a frame after the last tick")
        .reports
        .iter()
        .map(|r| (r.node.index(), r.change, r.entry_action.clone()))
        .collect();
    let mut positions: Vec<_> = view
        .positions()
        .map(|(key, p)| (key.index(), p.x.to_bits(), p.y.to_bits()))
        .collect();
    positions.sort_unstable();
    let log = std::mem::take(&mut *log.lock().unwrap());
    Outcome {
        log,
        frame,
        positions,
    }
}

#[test]
fn the_relay_shows_the_host_what_the_inline_backend_does() {
    let inline = scenario(Run::Inline, None);
    assert!(
        inline.log.len() >= 4,
        "the scenario has entries to compare: {:?}",
        inline.log
    );
    assert!(
        inline
            .log
            .iter()
            .all(|seen| seen.change == CouplingMembershipChange::Entered
                && seen.action == CouplingEntryOutcome::Pending),
        "a handler sees the entry pending, as inline"
    );
    assert!(
        inline.log.iter().any(|seen| seen.generation == 1)
            && inline.log.iter().any(|seen| seen.generation == 2),
        "entries under both coupling lists are among them"
    );
    assert!(
        inline
            .frame
            .iter()
            .any(|row| row.2 == CouplingEntryOutcome::Completed),
        "the last frame carries a completed entry: {:?}",
        inline.frame
    );
    for run in [Run::OffloadedAfterInstall, Run::OffloadedBeforeInstall] {
        let offloaded = scenario(run, None);
        assert_eq!(offloaded.log, inline.log, "{run:?}: the handler's reports");
        assert_eq!(offloaded.frame, inline.frame, "{run:?}: the frame's outcomes");
        assert_eq!(offloaded.positions, inline.positions, "{run:?}: positions");
    }
}

#[test]
fn a_refusal_reaches_the_frame_through_the_relay_as_it_does_inline() {
    let inline = scenario(Run::Inline, Some(0));
    assert!(
        inline
            .frame
            .iter()
            .any(|row| matches!(row.2, CouplingEntryOutcome::Refused(_))),
        "{:?}",
        inline.frame
    );
    for run in [Run::OffloadedAfterInstall, Run::OffloadedBeforeInstall] {
        let offloaded = scenario(run, Some(0));
        assert_eq!(offloaded.log, inline.log, "{run:?}");
        assert_eq!(offloaded.frame, inline.frame, "{run:?}");
        assert_eq!(offloaded.positions, inline.positions, "{run:?}");
    }
}

/// The simulation, handed no host handler, reports nothing and runs nothing
/// when the relay is off, and a relayed entry is the only thing a relay adds.
#[test]
fn without_a_handler_there_is_no_relay_and_the_frame_says_not_registered() {
    let (a, _) = (NodeKey::new(0), NodeKey::new(1));
    let mut sim = Simulation::new();
    sim.sync_nodes([(a, Point2D::new(OUT, 0.0))]);
    let mut physics = Physics::inline(sim, 0);
    physics.set_coupling_forces(vec![force(&[a])]);
    physics.offload(Arc::new(|| {}));
    let mut view = LayoutView::default();
    let mut ticks = 0;
    tick(&mut physics, &mut view, &mut ticks);
    physics.seed(vec![(a, Point2D::new(0.0, 0.0))]);
    tick(&mut physics, &mut view, &mut ticks);
    let frame = physics.coupling_membership().unwrap();
    assert_eq!(frame.reports[0].change, CouplingMembershipChange::Entered);
    assert_eq!(frame.reports[0].entry_action, CouplingEntryOutcome::NotRegistered);
}
