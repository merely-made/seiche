// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! `Physics::Remote` over an in-process loopback: a [`PhysicsWorker`] on the
//! far side of a transport that moves the same bytes a Web Worker's
//! `postMessage` would, pumped by the test one interval a frame. The remote
//! backend must land on the inline backend's trajectory bit for bit, over every
//! law, and keep the order of commands issued before the far side is ready.

use std::{
    cell::RefCell,
    collections::VecDeque,
    rc::Rc,
    sync::{Arc, Mutex},
    time::Duration,
};

use numen::{CouplingId, CouplingResponse, FieldDefinition, FieldExtent, FieldId, VectorField};
use seiche_model::{
    PhysicsLaw, PhysicsOverlay,
    wire::{FromWorker, PROTOCOL_VERSION, ToWorker, decode, encode},
};

use super::*;
use crate::{CouplingSource, CouplingTargetIdentity, TICK_DT, catalog::LawInputs};

fn clock() -> Duration {
    Duration::ZERO
}

#[derive(Default)]
struct Hub {
    worker: Option<PhysicsWorker>,
    inbox: VecDeque<Vec<u8>>,
    /// Messages the far side has been given, decoded to their kind.
    sent: Vec<String>,
    /// The far side answers `Init` only once released.
    hold_ready: bool,
    held: Vec<Vec<u8>>,
    version: Option<u32>,
}

/// The test's end of the loopback, and the host's [`Transport`].
#[derive(Clone)]
struct Loopback(Rc<RefCell<Hub>>);

impl Loopback {
    fn new(hold_ready: bool) -> Self {
        Self(Rc::new(RefCell::new(Hub {
            worker: Some(PhysicsWorker::new(clock)),
            hold_ready,
            ..Hub::default()
        })))
    }

    /// One interval of the far side, as its timer would run it.
    fn pump(&self) {
        let mut hub = self.0.borrow_mut();
        let update = hub.worker.as_mut().unwrap().interval();
        if let Some(bytes) = update {
            hub.inbox.push_back(bytes);
        }
    }

    fn release(&self) {
        let mut hub = self.0.borrow_mut();
        hub.hold_ready = false;
        for bytes in std::mem::take(&mut hub.held) {
            hub.inbox.push_back(bytes);
        }
    }
}

impl Transport for Loopback {
    fn send(&mut self, message: Vec<u8>) {
        let mut hub = self.0.borrow_mut();
        let kind = match decode::<ToWorker>(&message).unwrap() {
            ToWorker::Init { .. } => "init".to_string(),
            ToWorker::Command(command) => format!("{command:?}")
                .split(['(', ' ', '{'])
                .next()
                .unwrap()
                .to_string(),
        };
        hub.sent.push(kind);
        let replies = hub.worker.as_mut().unwrap().receive(&message).unwrap();
        for reply in replies {
            let reply = match hub.version {
                Some(version) => encode(&FromWorker::Ready { version }),
                None => reply,
            };
            if hub.hold_ready {
                hub.held.push(reply);
            } else {
                hub.inbox.push_back(reply);
            }
        }
    }

    fn receive(&mut self) -> Option<Vec<u8>> {
        self.0.borrow_mut().inbox.pop_front()
    }
}

fn key(i: usize) -> NodeKey {
    NodeKey::new(i)
}

fn positions() -> Vec<(NodeKey, Point2D<f32>)> {
    (0..12)
        .map(|i| {
            (
                key(i),
                Point2D::new(i as f32 * 9.0, ((i * i) % 7) as f32 * 11.0),
            )
        })
        .collect()
}

fn edges() -> Vec<(NodeKey, NodeKey)> {
    (0..11).map(|i| (key(i), key(i + 1))).chain([(key(0), key(5))]).collect()
}

fn descriptor(
    law: PhysicsLaw,
    overlays: &[PhysicsOverlay],
    revision: Option<u64>,
) -> seiche_model::ForceSetDescriptor {
    let sites = (0..12).map(|i| (key(i), format!("site{}", i % 3))).collect();
    LawInputs::from_parts((0..12).map(key).collect(), edges(), sites)
        .with_revision(revision)
        .describe(law, None, overlays, crate::catalog::LawSources::bare())
}

const TICKS: u32 = 40;

fn bits(view: &LayoutView) -> Vec<(usize, u32, u32)> {
    let mut out: Vec<_> = view
        .positions()
        .map(|(k, p)| (k.index(), p.x.to_bits(), p.y.to_bits()))
        .collect();
    out.sort_unstable();
    out
}

fn inline_run(descriptor: &seiche_model::ForceSetDescriptor) -> Vec<(usize, u32, u32)> {
    let mut sim = Simulation::new();
    sim.sync_nodes(positions());
    sim.sync_edges(edges());
    let mut physics = Physics::inline(sim, 0);
    physics.set_force_set(descriptor.clone());
    physics.settle(TICKS);
    let mut view = LayoutView::default();
    while physics.advance_frame(&mut view) {}
    bits(&view)
}

fn remote_run(descriptor: &seiche_model::ForceSetDescriptor) -> Vec<(usize, u32, u32)> {
    let hub = Loopback::new(false);
    let mut physics = Physics::remote(Box::new(hub.clone()));
    assert!(!physics.runs_inline());
    physics.sync_nodes(positions());
    physics.sync_edges(edges());
    physics.set_force_set(descriptor.clone());
    physics.settle(TICKS);
    let mut view = LayoutView::default();
    for _ in 0..10_000 {
        hub.pump();
        if !physics.advance_frame(&mut view) {
            return bits(&view);
        }
    }
    panic!("the remote backend did not settle");
}

#[test]
fn a_remote_backend_follows_the_inline_trajectory_over_every_law() {
    for law in PhysicsLaw::ALL {
        let d = descriptor(law, &[], Some(1));
        let inline = inline_run(&d);
        assert_eq!(inline.len(), 12);
        assert_eq!(remote_run(&d), inline, "{law:?}");
    }
}

#[test]
fn a_remote_backend_follows_the_inline_trajectory_under_every_overlay() {
    for overlay in PhysicsOverlay::ALL {
        let d = descriptor(PhysicsLaw::Springs, &[overlay], Some(2));
        assert_eq!(remote_run(&d), inline_run(&d), "{overlay:?}");
    }
}

#[test]
fn commands_before_ready_queue_in_order_and_replay_after_it() {
    let hub = Loopback::new(true);
    let mut physics = Physics::remote(Box::new(hub.clone()));
    let d = descriptor(PhysicsLaw::Charge, &[], Some(3));
    physics.sync_nodes(positions());
    physics.sync_edges(edges());
    physics.set_force_set(d.clone());
    physics.settle(TICKS);
    let mut view = LayoutView::default();
    assert!(!physics.is_ready());
    assert_eq!(hub.0.borrow().sent, ["init"], "only the handshake went out");
    // Frames before `Ready` fold nothing and the backend keeps asking for them.
    hub.pump();
    assert!(physics.advance_frame(&mut view));
    assert!(view.positions().next().is_none());
    hub.release();
    for _ in 0..10_000 {
        hub.pump();
        if !physics.advance_frame(&mut view) {
            break;
        }
    }
    assert!(physics.is_ready());
    let sent = hub.0.borrow().sent.clone();
    assert_eq!(
        &sent[..5],
        ["init", "SyncNodes", "SyncEdgeTopology", "SetForceSet", "Settle"],
        "the queue replays in the order issued"
    );
    assert_eq!(bits(&view), inline_run(&d));
}

#[test]
fn a_far_side_on_another_version_is_refused_and_nothing_is_sent() {
    let hub = Loopback::new(false);
    hub.0.borrow_mut().version = Some(PROTOCOL_VERSION + 1);
    let mut physics = Physics::remote(Box::new(hub.clone()));
    let mut view = LayoutView::default();
    physics.sync_nodes(positions());
    physics.settle(5);
    physics.refresh(&mut view);
    physics.sync_edges(edges());
    assert!(!physics.is_ready());
    assert_eq!(hub.0.borrow().sent, ["init"]);
}

#[test]
fn commands_with_no_wire_form_are_dropped_not_sent() {
    let hub = Loopback::new(false);
    let mut physics = Physics::remote(Box::new(hub.clone()));
    let mut view = LayoutView::default();
    physics.refresh(&mut view); // reads Ready
    physics.set_forces(vec![Box::new(crate::NodeExclusion::default())]);
    physics.set_lagged_repulsion(None, 0, 1);
    assert_eq!(hub.0.borrow().sent, ["init"]);
}

fn flow(nodes: &[NodeKey]) -> CouplingForce {
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

/// What the entry handler was shown, less the process-local parts.
fn entry_log(remote: bool) -> (Vec<(u64, usize, u64, Option<usize>)>, Vec<(usize, u32, u32)>) {
    const OUT: f32 = 1000.0;
    let (a, b) = (key(0), key(1));
    let log: Arc<Mutex<Vec<(u64, usize, u64, Option<usize>)>>> = Arc::default();
    let handler: crate::CouplingEntryHandler = {
        let log = Arc::clone(&log);
        Arc::new(move |report| {
            log.lock().unwrap().push((
                report.step,
                report.node.index(),
                report.binding_generation,
                report.target_index,
            ));
            Ok(())
        })
    };
    let hub = Loopback::new(false);
    let mut physics = if remote {
        Physics::remote(Box::new(hub.clone()))
    } else {
        Physics::inline(Simulation::new(), 0)
    };
    physics.sync_nodes(vec![(a, Point2D::new(OUT, 0.0)), (b, Point2D::new(OUT, 40.0))]);
    physics.set_coupling_forces(vec![flow(&[a, b])]);
    physics.set_coupling_entry_handler(Some(handler));
    let mut view = LayoutView::default();
    let mut ticks = 0u64;
    let mut tick = |physics: &mut Physics, view: &mut LayoutView| {
        physics.settle(1);
        ticks += 1;
        let want = ticks as f64 * f64::from(TICK_DT) - 1e-9;
        for _ in 0..10_000 {
            hub.pump();
            physics.advance_frame(view);
            if physics.field_clock().elapsed >= want || !remote {
                break;
            }
        }
    };
    tick(&mut physics, &mut view);
    physics.seed(vec![(a, Point2D::new(0.0, 0.0)), (b, Point2D::new(10.0, 0.0))]);
    tick(&mut physics, &mut view);
    physics.seed(vec![(a, Point2D::new(OUT, 0.0)), (b, Point2D::new(10.0, 0.0))]);
    tick(&mut physics, &mut view);
    physics.set_coupling_forces(vec![flow(&[a, b])]);
    tick(&mut physics, &mut view);
    physics.seed(vec![(a, Point2D::new(-20.0, 5.0)), (b, Point2D::new(10.0, 0.0))]);
    tick(&mut physics, &mut view);
    physics.refresh(&mut view);
    let log = std::mem::take(&mut *log.lock().unwrap());
    (log, bits(&view))
}

#[test]
fn the_entry_handler_runs_on_the_host_for_a_remote_backend_as_for_inline() {
    let (inline_log, inline_positions) = entry_log(false);
    assert!(inline_log.len() >= 3, "{inline_log:?}");
    let (remote_log, remote_positions) = entry_log(true);
    assert_eq!(remote_log, inline_log, "the handler's reports");
    assert_eq!(remote_positions, inline_positions);
}

#[test]
fn the_worker_computes_each_channel_once_per_revision() {
    use crate::catalog::ChannelRuns;

    let mut worker = PhysicsWorker::new(clock);
    let cache = worker.channel_cache();
    let send = |worker: &mut PhysicsWorker, d: seiche_model::ForceSetDescriptor| {
        let bytes = encode(&ToWorker::Command(seiche_model::DataCommand::SetForceSet(d)));
        worker.receive(&bytes).unwrap();
    };
    let orbit_pagerank = |law, revision| {
        let mut d = descriptor(law, &[PhysicsOverlay::HubGravity], revision);
        d.sources.mass = seiche_model::PhysicsMassSource::PageRank;
        d
    };
    send(&mut worker, orbit_pagerank(PhysicsLaw::Orbit, Some(10)));
    let first = cache.borrow().runs();
    assert!(first.pagerank >= 1, "the first build computes PageRank: {first:?}");
    // A second rebuild at the same revision (another law, same graph) computes nothing.
    send(&mut worker, orbit_pagerank(PhysicsLaw::Springs, Some(10)));
    send(&mut worker, orbit_pagerank(PhysicsLaw::Orbit, Some(10)));
    assert_eq!(cache.borrow().runs(), first, "no recomputation at revision 10");
    // A new revision computes again.
    send(&mut worker, orbit_pagerank(PhysicsLaw::Orbit, Some(11)));
    let after = cache.borrow().runs();
    assert_eq!(after.pagerank, first.pagerank * 2);
    // A reused token over other edges recomputes, never serves the old values.
    let mut moved = orbit_pagerank(PhysicsLaw::Orbit, Some(11));
    moved.inputs.edges.pop();
    send(&mut worker, moved);
    assert_eq!(cache.borrow().runs().pagerank, after.pagerank + first.pagerank);
    // No revision: uncached, and the cache is not touched.
    let before = cache.borrow().runs();
    send(&mut worker, orbit_pagerank(PhysicsLaw::Orbit, None));
    assert_eq!(cache.borrow().runs(), before);
    assert_ne!(before, ChannelRuns::default());
}

#[test]
fn a_cached_build_makes_the_same_forces_as_an_uncached_one() {
    for law in PhysicsLaw::ALL {
        let mut d = descriptor(law, &[PhysicsOverlay::HubGravity, PhysicsOverlay::DepthGravity, PhysicsOverlay::Skeleton], Some(5));
        d.sources.mass = seiche_model::PhysicsMassSource::PageRank;
        d.sources.depth = seiche_model::PhysicsDepthSource::Layers;
        let run = |cached: bool| {
            let mut sim = Simulation::new();
            sim.sync_nodes(positions());
            sim.sync_edges(edges());
            let cache = crate::catalog::ChannelCache::shared();
            // Twice, so the second build reads the cache.
            let mut forces = crate::catalog::build_force_set(&d);
            if cached {
                crate::catalog::build_force_set_cached(&d, &cache);
                forces = crate::catalog::build_force_set_cached(&d, &cache);
            }
            sim.set_forces(forces);
            for _ in 0..20 {
                sim.tick(TICK_DT);
            }
            bits(&sim.view())
        };
        assert_eq!(run(true), run(false), "{law:?}");
    }
}
