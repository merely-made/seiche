//! The descriptor's builder against the in-process builders: forces built from
//! a [`ForceSetDescriptor`] equal what `LawInputs::composed` builds, for every
//! law, overlay and source, and for the compositions. "Equal" is the declared
//! terms and the forces count, then the bodies' positions after a run, bit for
//! bit (the engine is deterministic), since a force's data (masses, groups,
//! kinds, distances) is not in its terms.

use std::borrow::Cow;
use std::collections::HashMap;

use euclid::default::Point2D;
use seiche_model::{
    ForceSetDescriptor, GroupSource, PhysicsComposition, PhysicsDepthSource, PhysicsGrouping,
    PhysicsKindSource, PhysicsLaw, PhysicsMassSource, PhysicsOverlay,
};

use super::{LawInputs, LawSources, MeaningGroups, build_force_set};
use crate::{Force, NodeKey, Simulation};

fn key(i: usize) -> NodeKey {
    NodeKey::new(i)
}

/// Two islands (a six-cycle with a spur, a path) and an isolated node, on
/// three sites, with a cluster table and a meaning table that leaves one node
/// unknown.
fn inputs() -> LawInputs<'static> {
    let nodes: Vec<NodeKey> = (0..12).map(key).collect();
    let pairs = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 0),
        (2, 6),
        (1, 2),
        (7, 8),
        (8, 9),
        (9, 10),
    ];
    let edges: Vec<(NodeKey, NodeKey)> = pairs.iter().map(|&(a, b)| (key(a), key(b))).collect();
    let sites: HashMap<NodeKey, String> = nodes
        .iter()
        .map(|&k| (k, format!("site{}", k.index() % 3)))
        .collect();
    let clusters: Vec<(NodeKey, u32)> =
        nodes.iter().map(|&k| (k, (k.index() / 4) as u32)).collect();
    // Node 11 is unknown to the snapshot.
    let meaning: Vec<(NodeKey, u32)> = nodes[..11]
        .iter()
        .map(|&k| (k, (k.index() % 4) as u32))
        .collect();
    LawInputs::from_parts(nodes, edges, sites)
        .with_clusters(Some(Cow::Owned(clusters)))
        .with_meaning(Some(MeaningGroups {
            groups: Cow::Owned(meaning),
            unknown: 4,
        }))
}

fn run(
    forces: Vec<Box<dyn Force>>,
    ticks: usize,
    positions: &[(NodeKey, Point2D<f32>)],
) -> Vec<(usize, u32, u32)> {
    let mut sim = Simulation::new();
    sim.sync_nodes(positions.iter().copied());
    sim.sync_edges(inputs().edges().iter().copied());
    sim.set_forces(forces);
    for _ in 0..ticks {
        sim.tick(1.0 / 60.0);
    }
    let mut out: Vec<_> = sim
        .positions()
        .map(|(k, p)| (k.index(), p.x.to_bits(), p.y.to_bits()))
        .collect();
    out.sort();
    out
}

fn start() -> Vec<(NodeKey, Point2D<f32>)> {
    (0..12)
        .map(|i| {
            let a = i as f32 * 0.9;
            (
                key(i),
                Point2D::new(40.0 * a.cos() * (1.0 + i as f32 * 0.1), 40.0 * a.sin()),
            )
        })
        .collect()
}

fn terms_of(forces: &[Box<dyn Force>]) -> Vec<Vec<crate::Term>> {
    forces.iter().map(|f| f.terms().to_vec()).collect()
}

/// The descriptor's forces against the in-process ones, terms and then a run.
fn assert_equal(
    label: &str,
    law: PhysicsLaw,
    composition: Option<&PhysicsComposition>,
    overlays: &[PhysicsOverlay],
    sources: LawSources,
    ticks: usize,
) {
    let inputs = inputs();
    let direct = inputs.composed(law, composition, overlays, sources);
    let descriptor = inputs.describe(law, composition, overlays, sources);
    let built = build_force_set(&descriptor);
    assert_eq!(direct.len(), built.len(), "{label}: force count");
    assert_eq!(terms_of(&direct), terms_of(&built), "{label}: terms");
    if ticks > 0 {
        assert_eq!(
            run(direct, ticks, &start()),
            run(built, ticks, &start()),
            "{label}: positions"
        );
    }
}

fn sources() -> LawSources {
    LawSources {
        focus: Some(key(2)),
        ..LawSources::bare()
    }
}

#[test]
fn every_law_builds_the_same_forces_from_its_descriptor() {
    for law in PhysicsLaw::ALL {
        // Density settles in passes of a 64-cell grid; its terms and a few
        // ticks decide it.
        let ticks = if law == PhysicsLaw::Density { 3 } else { 30 };
        assert_equal(law.id(), law, None, &[], sources(), ticks);
    }
}

#[test]
fn every_overlay_builds_the_same_forces_over_every_law() {
    for law in PhysicsLaw::ALL {
        for overlay in PhysicsOverlay::ALL {
            let ticks = if law == PhysicsLaw::Density { 0 } else { 20 };
            assert_equal(
                &format!("{law:?} + {overlay:?}"),
                law,
                None,
                &[overlay],
                sources(),
                ticks,
            );
        }
        assert_equal(
            &format!("{law:?} + every overlay"),
            law,
            None,
            &PhysicsOverlay::ALL,
            sources(),
            if law == PhysicsLaw::Density { 0 } else { 10 },
        );
    }
}

#[test]
fn every_source_builds_the_same_forces_from_its_descriptor() {
    for kind in PhysicsKindSource::ALL {
        let s = LawSources {
            kind,
            groups: kind,
            ..sources()
        };
        assert_equal(
            &format!("Kinds by {kind:?}"),
            PhysicsLaw::Kinds,
            None,
            &[],
            s,
            30,
        );
        assert_equal(
            &format!("Group pull by {kind:?}"),
            PhysicsLaw::Springs,
            None,
            &[PhysicsOverlay::DomainCluster],
            s,
            30,
        );
    }
    for mass in PhysicsMassSource::ALL {
        let s = LawSources { mass, ..sources() };
        assert_equal(
            &format!("Orbit by {mass:?}"),
            PhysicsLaw::Orbit,
            None,
            &[],
            s,
            30,
        );
        assert_equal(
            &format!("hubs by {mass:?}"),
            PhysicsLaw::Springs,
            None,
            &[PhysicsOverlay::DegreeRepulsion, PhysicsOverlay::HubGravity],
            s,
            30,
        );
        assert_equal(
            &format!("Density by {mass:?}"),
            PhysicsLaw::Density,
            None,
            &[],
            s,
            0,
        );
    }
    for depth in PhysicsDepthSource::ALL {
        for focus in [None, Some(key(2)), Some(key(99))] {
            let s = LawSources {
                depth,
                focus,
                ..sources()
            };
            assert_equal(
                &format!("Depth by {depth:?} from {focus:?}"),
                PhysicsLaw::Springs,
                None,
                &[PhysicsOverlay::DepthGravity],
                s,
                30,
            );
        }
    }
    // Stress reads the hop distances and Skeleton the spanning tree.
    assert_equal(
        "Stress",
        PhysicsLaw::Stress,
        None,
        &[PhysicsOverlay::Skeleton],
        sources(),
        30,
    );
}

#[test]
fn compositions_build_the_same_forces_from_their_descriptor() {
    let mix = PhysicsComposition::Mix(vec![(PhysicsLaw::Charge, 2.0), (PhysicsLaw::Springs, 0.5)]);
    assert_equal(
        "mix",
        PhysicsLaw::Springs,
        Some(&mix),
        &[PhysicsOverlay::Tide],
        sources(),
        30,
    );
    for source in PhysicsKindSource::ALL {
        let grouped = PhysicsComposition::Grouped(PhysicsGrouping::charge_between(
            GroupSource::Channel(source),
        ));
        assert_equal(
            &format!("grouped by {source:?}"),
            PhysicsLaw::Springs,
            Some(&grouped),
            &[PhysicsOverlay::GravityLocus],
            sources(),
            30,
        );
    }
    let given: Vec<(NodeKey, u32)> = (0..12).map(|i| (key(i), (i % 3) as u32)).collect();
    let grouped = PhysicsComposition::Grouped(PhysicsGrouping {
        inner: PhysicsLaw::Energy,
        ..PhysicsGrouping::charge_between(GroupSource::Given(given))
    });
    assert_equal(
        "grouped given",
        PhysicsLaw::Springs,
        Some(&grouped),
        &[],
        sources(),
        30,
    );
}

#[test]
fn the_descriptor_reads_back_the_inputs_it_was_made_from() {
    let inputs = inputs();
    let descriptor = inputs.describe(PhysicsLaw::Orbit, None, &[], sources());
    let back = LawInputs::from_data(&descriptor.inputs).to_data();
    assert_eq!(back, descriptor.inputs);
    assert_eq!(
        descriptor,
        ForceSetDescriptor {
            inputs: back,
            ..descriptor.clone()
        }
    );
}

#[cfg(feature = "serde")]
#[test]
fn a_descriptor_builds_the_same_forces_after_its_wire_form() {
    let inputs = inputs();
    let composition = PhysicsComposition::Grouped(PhysicsGrouping::charge_between(
        GroupSource::Channel(PhysicsKindSource::Meaning),
    ));
    let descriptor = inputs.describe(
        PhysicsLaw::Orbit,
        Some(&composition),
        &[PhysicsOverlay::DomainCluster, PhysicsOverlay::DepthGravity],
        sources(),
    );
    let wire = serde_json::to_string(&descriptor).expect("serialize");
    let back: ForceSetDescriptor = serde_json::from_str(&wire).expect("deserialize");
    assert_eq!(back, descriptor, "{wire}");
    assert_eq!(
        run(build_force_set(&descriptor), 30, &start()),
        run(build_force_set(&back), 30, &start())
    );
}

#[test]
fn the_run_comparison_can_tell_forces_apart() {
    let inputs = inputs();
    let at = |law| {
        let d = inputs.describe(law, None, &[], sources());
        run(build_force_set(&d), 30, &start())
    };
    assert_ne!(at(PhysicsLaw::Springs), at(PhysicsLaw::Charge));
    assert_ne!(at(PhysicsLaw::Springs), at(PhysicsLaw::Still));
}

/// The descriptor sent to an offloaded backend is built on the actor's thread
/// and lands on the trajectory the inline backend builds, bit for bit.
#[cfg(feature = "actor")]
#[test]
fn an_offloaded_backend_builds_a_descriptor_to_the_inline_trajectory() {
    use std::sync::Arc;

    use crate::{LayoutView, Physics, TICK_DT};

    const TICKS: u32 = 30;
    let descriptor = inputs().describe(
        PhysicsLaw::Charge,
        None,
        &[PhysicsOverlay::Skeleton],
        LawSources::bare(),
    );
    let positions: Vec<(NodeKey, Point2D<f32>)> = (0..12)
        .map(|i| {
            (
                key(i),
                Point2D::new(i as f32 * 9.0, ((i * i) % 7) as f32 * 11.0),
            )
        })
        .collect();
    let finish = |offload: bool| {
        let mut sim = Simulation::new();
        sim.sync_nodes(positions.iter().copied());
        sim.sync_edges(inputs().edges().iter().copied());
        let mut physics = Physics::inline(sim, 0);
        assert!(physics.runs_inline());
        physics.set_force_set(descriptor.clone());
        if offload {
            physics.offload(Arc::new(|| {}));
            assert!(!physics.runs_inline());
        }
        physics.settle(TICKS);
        let mut view = LayoutView::default();
        let want = f64::from(TICKS) * f64::from(TICK_DT) - 1e-9;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
        while physics.field_clock().elapsed < want {
            assert!(std::time::Instant::now() < deadline, "the backend did not finish");
            physics.advance_frame(&mut view);
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let mut out: Vec<_> = view
            .positions()
            .map(|(k, p)| (k.index(), p.x.to_bits(), p.y.to_bits()))
            .collect();
        out.sort();
        out
    };
    let inline = finish(false);
    assert_eq!(inline.len(), 12);
    assert_eq!(finish(true), inline);
}
